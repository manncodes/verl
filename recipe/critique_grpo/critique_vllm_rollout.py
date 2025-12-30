# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Critique-GRPO vLLM Rollout.

This module extends the standard vLLM rollout with critique-based refinement
during sequence generation. For each batch:
1. Generate initial responses
2. Compute scores and generate critiques
3. Generate refined solutions
4. Return both initial and refined responses for training

Based on: https://arxiv.org/abs/2506.03106
"""

import copy
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig
from tensordict import TensorDict
from vllm import LLM, SamplingParams

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout

from .critique_prompts import generate_critique
from .refinement_prompts import generate_refinement, process_refinement_groups
from .reward_function import compute_math_score

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _pre_process_inputs(pad_token_id: int, prompt_token_ids: torch.Tensor) -> List[int]:
    """Remove left padding from prompt token ids."""
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)
    if len(non_pad_index) == 0:
        return []
    first_non_pad = non_pad_index[0][0]
    return prompt_token_ids[first_non_pad:].tolist()


def _pre_process_inputs_right_pad(pad_token_id: int, prompt_token_ids: torch.Tensor) -> List[int]:
    """Remove right padding from prompt token ids."""
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)
    if len(non_pad_index) == 0:
        return []
    last_non_pad = non_pad_index[-1][0]
    return prompt_token_ids[:last_non_pad + 1].tolist()


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int):
    """Repeat tensor/array interleaved."""
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    return np.repeat(value, repeats, axis=0)


class CritiqueVLLMRollout(BaseRollout):
    """
    vLLM Rollout with Critique-based Refinement.

    This class extends the standard vLLM rollout to:
    1. Generate initial responses
    2. Compute math verification scores
    3. Generate critiques for each response
    4. Generate refined solutions based on critiques
    5. Select best refinements for off-policy learning
    """

    def __init__(
        self,
        model_path: str,
        config: DictConfig,
        tokenizer,
        model_hf_config,
        **kwargs
    ):
        """Initialize the Critique vLLM Rollout.

        Args:
            model_path: Path to the model
            config: Configuration for rollout
            tokenizer: Tokenizer for the model
            model_hf_config: HuggingFace model config
            **kwargs: Additional arguments
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.end_token_id = tokenizer.eos_token_id

        # Critique-specific config
        self.critique_type = config.get("critique_type", "simple_gt")
        self.n_prefix = config.get("n_prefix", 1)
        self.max_refinement_len = config.get("max_refinement_length", 6144)

        # Validate config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size()

        max_num_batched_tokens = config.get("max_num_batched_tokens", 8192)

        # Check rope scaling
        rope_scaling_config = getattr(model_hf_config, 'rope_scaling', None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config"):
                if hasattr(model_hf_config.llm_config, "max_position_embeddings"):
                    max_position_embeddings = model_hf_config.llm_config.max_position_embeddings

            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")

            assert max_position_embeddings >= config.prompt_length + config.response_length

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        limit_mm_per_prompt = None
        if config.get("limit_images", None):
            limit_mm_per_prompt = {"image": config.get("limit_images")}

        # Initialize vLLM engine
        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            limit_mm_per_prompt=limit_mm_per_prompt,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
        )

        # Offload to reduce peak memory
        self.inference_engine.sleep(level=1)

        # Default sampling params
        kwargs = {
            "n": 1,
            "logprobs": 0,
            "max_tokens": config.response_length,
        }

        if vllm_version != "0.3.1":
            kwargs["detokenize"] = False

        # Add any config-specified sampling params
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        logger.info(f"Sampling params: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        """Temporarily update sampling parameters."""
        old_params = {}
        for key, value in kwargs.items():
            if hasattr(self.sampling_params, key):
                old_params[key] = getattr(self.sampling_params, key)
                setattr(self.sampling_params, key, value)
        try:
            yield
        finally:
            for key, value in old_params.items():
                setattr(self.sampling_params, key, value)

    def _process_critique_item(self, args):
        """Process a single item for critique generation."""
        i, response_ids, reward_model_data = args
        try:
            # Decode response
            response_str = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            # Get question and ground truth
            question = reward_model_data.get('question', '')
            gt = reward_model_data.get('ground_truth', '')

            # Compute score
            score_result = compute_math_score(response_str, gt)
            score = score_result.get("score", 0.0)

            # Create sample and generate critique
            sample = {
                "question": question,
                "response": response_str,
                "gt": gt,
                "score": score,
            }
            sample = generate_critique(sample, self.critique_type)

            # Generate refinement prompt
            _, refinement_ids = generate_refinement(sample, self.tokenizer)

            return i, refinement_ids, score

        except Exception as e:
            logger.error(f"Error processing critique item {i}: {e}")
            raise

    def _process_refinement_item(self, args):
        """Process a single refinement for scoring."""
        i, refinement_str, reward_model_data = args
        try:
            gt = reward_model_data.get('ground_truth', '')
            score_result = compute_math_score(refinement_str, gt)
            score = score_result.get("score", 0.0)

            return {
                "refinement": refinement_str,
                "score": score,
                "gt": gt
            }
        except Exception as e:
            logger.error(f"Error processing refinement item {i}: {e}")
            raise

    @GPUMemoryLogger(role="critique_vllm_rollout", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences with critique-based refinement.

        Args:
            prompts: Input prompts batch
            **kwargs: Additional generation arguments

        Returns:
            DataProto with generated responses and refinements
        """
        # Rebuild cache engine if needed
        if vllm_version in ("0.5.4", "0.6.3") and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch

        # Preprocess inputs
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)],
                dtype=object
            )

        # Prepare vLLM inputs
        vllm_inputs = [
            {"prompt_token_ids": list(raw_ids) if isinstance(raw_ids, np.ndarray) else raw_ids}
            for raw_ids in non_tensor_batch.pop("raw_prompt_ids")
        ]

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        is_train = 'tgt_input_ids' in prompts.batch

        # Configure sampling
        if not do_sample:
            sample_kwargs = {
                "best_of": 1, "top_p": 1.0, "top_k": -1,
                "min_p": 0.0, "temperature": 0, "n": 1
            }
        elif is_validate:
            sample_kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1
            }
        else:
            sample_kwargs = {}

        # Generate initial responses
        with self.update_sampling_params(**sample_kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )

        # Collect responses
        response = []
        for output in outputs:
            for sample_id in range(len(output.outputs)):
                response.append(output.outputs[sample_id].token_ids)

        # Handle training mode with critique/refinement
        prefix_mask = torch.tensor([])
        refinement_input_ids = None

        if is_train and do_sample:
            # Repeat non_tensor_batch for multiple samples
            if self.sampling_params.n > 1:
                for key in ['reward_model', 'target']:
                    if key in non_tensor_batch:
                        non_tensor_batch[key] = _repeat_interleave(
                            non_tensor_batch[key], self.sampling_params.n
                        )

            # Generate critiques and refinements in parallel
            init_responses = copy.deepcopy(response)

            args_list = []
            for i in range(len(init_responses)):
                rm_data = non_tensor_batch.get('reward_model', [{}])[i]
                if isinstance(rm_data, np.ndarray):
                    rm_data = rm_data.item() if rm_data.size == 1 else {}
                args_list.append((i, init_responses[i], rm_data))

            with ThreadPoolExecutor(max_workers=min(96, len(args_list))) as executor:
                critique_results = list(executor.map(self._process_critique_item, args_list))

            refinement_ids_list = [r[1] for r in critique_results]

            # Generate refinements
            refinement_inputs = [
                {"prompt_token_ids": ref_ids}
                for ref_ids in refinement_ids_list
            ]

            refinement_sampling = copy.deepcopy(self.sampling_params)
            refinement_sampling.n = 1

            with self.update_sampling_params(**sample_kwargs):
                refinement_outputs = self.inference_engine.generate(
                    prompts=refinement_inputs,
                    sampling_params=refinement_sampling,
                    use_tqdm=False,
                )

            # Score refinements
            refinement_strs = []
            for ref_out in refinement_outputs:
                for sample_id in range(len(ref_out.outputs)):
                    ref_str = self.tokenizer.decode(
                        ref_out.outputs[sample_id].token_ids,
                        skip_special_tokens=True
                    )
                    refinement_strs.append(ref_str)

            # Score refinements in parallel
            refine_args = []
            for i in range(len(refinement_strs)):
                rm_data = non_tensor_batch.get('reward_model', [{}])[i]
                if isinstance(rm_data, np.ndarray):
                    rm_data = rm_data.item() if rm_data.size == 1 else {}
                refine_args.append((i, refinement_strs[i], rm_data))

            with ThreadPoolExecutor(max_workers=min(96, len(refine_args))) as executor:
                refine_results = list(executor.map(self._process_refinement_item, refine_args))

            refinement_dicts = [
                {"refinement": r["refinement"], "score": r["score"], "gt": r["gt"]}
                for r in refine_results
            ]

            # Select best refinements
            selected_refinements, refinement_scores = process_refinement_groups(
                refinement_dicts, num_samples=self.sampling_params.n
            )

            # Tokenize selected refinements
            refinement_input_ids_list = []
            for ref in selected_refinements:
                ref_ids = self.tokenizer(
                    ref['refinement'],
                    add_special_tokens=False,
                    return_tensors='pt'
                )['input_ids'].to(device=idx.device)

                # Pad/truncate to max_refinement_len
                if ref_ids.size(1) < self.max_refinement_len:
                    padding = torch.full(
                        (1, self.max_refinement_len - ref_ids.size(1)),
                        self.tokenizer.pad_token_id,
                        device=idx.device
                    )
                    ref_ids = torch.cat([ref_ids, padding], dim=1)
                else:
                    ref_ids = ref_ids[:, :self.max_refinement_len]

                refinement_input_ids_list.append(ref_ids)

            if refinement_input_ids_list:
                refinement_input_ids = torch.cat(refinement_input_ids_list, dim=0)
            else:
                refinement_input_ids = torch.empty(
                    (0, self.max_refinement_len),
                    dtype=torch.long,
                    device=idx.device
                )

            # Build prefix mask and response with refinements
            tgt_list = [
                _pre_process_inputs_right_pad(self.pad_token_id, refinement_input_ids[i])
                for i in range(len(selected_refinements))
            ]

            # Add EOS token
            tgt_list = [
                tgt + [self.end_token_id] if len(tgt) > 0 else tgt
                for tgt in tgt_list
            ]

            # Expand to match n samples
            tgt_list = sum(
                [[tgt_list[i]] * self.sampling_params.n for i in range(len(tgt_list))],
                []
            )

            assert len(tgt_list) == self.sampling_params.n * batch_size

            # Build prefix ratios
            prefix_ratios = []
            for i in range(batch_size):
                if self.n_prefix > 0:
                    prefix_ratios.extend([1.0] * self.n_prefix)
                    prefix_ratios.extend([0.0] * (self.sampling_params.n - self.n_prefix))
                else:
                    prefix_ratios.extend([0.0] * self.sampling_params.n)

            # Build prefix list
            prefix_list = []
            for ratio, tgt_ids in zip(prefix_ratios, tgt_list):
                prefix_list.append(tgt_ids if ratio > 0 else [])

            # Build response with prefixes
            response = pad_2d_list_to_length(
                response, self.pad_token_id, max_length=self.config.response_length
            ).to(idx.device)

            resp_list = [
                _pre_process_inputs_right_pad(self.pad_token_id, resp)
                for resp in response
            ]

            concat_resp_list = []
            prefix_mask = torch.zeros(
                [len(resp_list), self.config.response_length],
                dtype=torch.bool
            ).to(idx.device)

            for i, (prefix_tgt, resp_ids) in enumerate(zip(prefix_list, resp_list)):
                prefix_len = min(len(prefix_tgt), self.config.response_length)
                prefix_mask[i, :prefix_len] = True

                if prefix_tgt:
                    concat_resp_list.append(torch.tensor(prefix_tgt))
                else:
                    concat_resp_list.append(torch.tensor(resp_ids))

            # Pad responses
            resp_max_len = max(len(resp) for resp in concat_resp_list)
            response_padded = torch.ones(len(concat_resp_list), resp_max_len).fill_(
                self.pad_token_id
            )
            for i, resp in enumerate(concat_resp_list):
                response_padded[i, :len(resp)] = resp.clone()

            response = response_padded.to(idx.device)[:, :self.config.response_length].to(
                prompts.batch["input_ids"].dtype
            )

        else:
            response = pad_2d_list_to_length(
                response, self.pad_token_id, max_length=self.config.response_length
            ).to(idx.device)

        # Handle multi-sample expansion
        if self.sampling_params.n > 1 and do_sample:
            idx = _repeat_interleave(idx, self.sampling_params.n)
            if refinement_input_ids is not None:
                refinement_input_ids = _repeat_interleave(
                    refinement_input_ids, self.sampling_params.n
                )
            attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
            position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
            batch_size = batch_size * self.sampling_params.n

            if "multi_modal_inputs" in non_tensor_batch:
                non_tensor_batch["multi_modal_inputs"] = _repeat_interleave(
                    non_tensor_batch["multi_modal_inputs"], self.sampling_params.n
                )

        # Build sequence
        seq = torch.cat([idx, response], dim=-1)

        # Compute position IDs for response
        response_length = response.size(1)
        delta_position_id = torch.arange(
            1, response_length + 1, device=position_ids.device
        ).unsqueeze(0).expand(batch_size, -1)

        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        # Compute attention mask
        response_attention_mask = get_response_mask(
            response_id=response,
            eos_token=eos_token_id,
            dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # Build output batch
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # Add refinement data if available
        if refinement_input_ids is not None:
            batch['tgt_input_ids'] = refinement_input_ids

        if prefix_mask.numel() > 0:
            batch['prefix_mask'] = prefix_mask

        # Free cache if needed
        if vllm_version in ("0.5.4", "0.6.3") and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
