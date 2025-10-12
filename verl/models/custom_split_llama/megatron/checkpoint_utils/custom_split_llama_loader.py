# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Weight loading utilities for Custom Split LLaMA model.

This module provides functionality to load weights from separate 8B and 70B
Llama checkpoints into a merged custom split model with Megatron parallelism.
"""

import time
from typing import Dict, Any

import torch
import torch.distributed as dist
from transformers import LlamaConfig

from verl.utils.device import get_device_id, get_torch_device


def load_custom_split_llama_weights(
    state_dict_8b: Dict[str, torch.Tensor],
    state_dict_70b: Dict[str, torch.Tensor],
    adapter_state_dict: Dict[str, torch.Tensor],
    wrapped_models,
    config,
    params_dtype,
    is_value_model=False,
):
    """
    Load weights from separate 8B and 70B checkpoints into Custom Split LLaMA model.

    Args:
        state_dict_8b: State dict from 8B Llama model (for first layers)
        state_dict_70b: State dict from 70B Llama model (for last layers)
        adapter_state_dict: State dict for adapter weights
        wrapped_models: List of wrapped model modules
        config: Model configuration with split model attributes
        params_dtype: Parameter dtype for loading
        is_value_model: Whether this is a value model (critic)
    """
    from megatron.core import DistributedDataParallel as LocalDDP
    from megatron.core import mpu
    from megatron.core.transformer.module import Float16Module
    from torch.nn.parallel import DistributedDataParallel as torchDDP

    from verl.utils.logger import print_rank_0
    from verl.utils.megatron_utils import unwrap_model

    start_time = time.time()

    # Get parallel ranks
    dp_rank = mpu.get_data_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()

    if not isinstance(wrapped_models, list | tuple):
        wrapped_models = [wrapped_models]

    # Get layer configuration
    num_layers_8b = config.num_layers_8
    num_layers_70b = config.num_layers_70

    # Unwrap models
    models = []
    for wrapped_model in wrapped_models:
        model = unwrap_model(wrapped_model, (torchDDP, LocalDDP, Float16Module))
        models.append(model)

    def _fetch_tp_shard_tensor(tensor, name, state_dict, chunk_dim=0):
        """Fetch and shard tensor for tensor parallelism"""
        if name in state_dict:
            full_weight = state_dict[name]
            tensor_chunk = torch.chunk(full_weight, tp_size, dim=chunk_dim)
            if tensor is not None:
                tensor.data.copy_(tensor_chunk[tp_rank])
        else:
            print_rank_0(f"Warning: {name} not found in state_dict")

    def _fetch_tp_shard_tensor_gate_up(tensor, gate_name, up_name, state_dict):
        """Fetch and fuse gate_up tensors for tensor parallelism"""
        if gate_name in state_dict and up_name in state_dict:
            gate_weight = state_dict[gate_name]
            up_weight = state_dict[up_name]
            # Concatenate gate and up weights
            fused_weight = torch.cat([gate_weight, up_weight], dim=0)
            # Shard for TP
            tensor_chunk = torch.chunk(fused_weight, tp_size, dim=0)
            if tensor is not None:
                tensor.data.copy_(tensor_chunk[tp_rank])
        else:
            print_rank_0(f"Warning: {gate_name} or {up_name} not found")

    def _fetch_tp_shard_tensor_qkv(tensor, q_name, k_name, v_name, state_dict):
        """Fetch and fuse qkv tensors for tensor parallelism"""
        if q_name in state_dict and k_name in state_dict and v_name in state_dict:
            q_weight = state_dict[q_name]
            k_weight = state_dict[k_name]
            v_weight = state_dict[v_name]
            # Concatenate qkv weights
            fused_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            # Shard for TP
            tensor_chunk = torch.chunk(fused_weight, tp_size, dim=0)
            if tensor is not None:
                tensor.data.copy_(tensor_chunk[tp_rank])
        else:
            print_rank_0(f"Warning: QKV weights not found")

    # Load weights for each model in the pipeline
    for model_idx, model in enumerate(models):
        # Load embedding layer (only on first PP rank)
        if pp_rank == 0:
            embed_weight_name = "model.embed_tokens.weight"
            if hasattr(model, 'embed_tokens'):
                _fetch_tp_shard_tensor(
                    model.embed_tokens.weight,
                    embed_weight_name,
                    state_dict_8b,
                    chunk_dim=0  # Shard along vocab dimension (dim 0)
                )

        # Load first layers (from 8B model)
        if hasattr(model, 'layers_first'):
            for layer_idx, layer in enumerate(model.layers_first):
                prefix = f"model.layers.{layer_idx}"

                # Attention weights (QKV fused)
                _fetch_tp_shard_tensor_qkv(
                    layer.self_attn.qkv_proj.weight,
                    f"{prefix}.self_attn.q_proj.weight",
                    f"{prefix}.self_attn.k_proj.weight",
                    f"{prefix}.self_attn.v_proj.weight",
                    state_dict_8b
                )

                # Attention output projection
                _fetch_tp_shard_tensor(
                    layer.self_attn.o_proj.weight,
                    f"{prefix}.self_attn.o_proj.weight",
                    state_dict_8b,
                    chunk_dim=1
                )

                # MLP gate_up fused
                _fetch_tp_shard_tensor_gate_up(
                    layer.mlp.gate_up_proj.weight,
                    f"{prefix}.mlp.gate_proj.weight",
                    f"{prefix}.mlp.up_proj.weight",
                    state_dict_8b
                )

                # MLP down projection
                _fetch_tp_shard_tensor(
                    layer.mlp.down_proj.weight,
                    f"{prefix}.mlp.down_proj.weight",
                    state_dict_8b,
                    chunk_dim=1
                )

                # Layer norms
                if f"{prefix}.input_layernorm.weight" in state_dict_8b:
                    layer.input_layernorm.weight.data.copy_(
                        state_dict_8b[f"{prefix}.input_layernorm.weight"]
                    )
                if f"{prefix}.post_attention_layernorm.weight" in state_dict_8b:
                    layer.post_attention_layernorm.weight.data.copy_(
                        state_dict_8b[f"{prefix}.post_attention_layernorm.weight"]
                    )

        # Load adapter layer
        if hasattr(model, 'adapter') and adapter_state_dict:
            _fetch_tp_shard_tensor(
                model.adapter.adapter_linear_1.weight,
                "adapter.adapter_linear_1.weight",
                adapter_state_dict,
                chunk_dim=0
            )
            _fetch_tp_shard_tensor(
                model.adapter.adapter_linear_2.weight,
                "adapter.adapter_linear_2.weight",
                adapter_state_dict,
                chunk_dim=1
            )

        # Load last layers (from 70B model)
        if hasattr(model, 'layers_last'):
            config_70b = LlamaConfig.from_pretrained(config.path70b)
            start_idx_70b = config_70b.num_hidden_layers - num_layers_70b

            for local_idx, layer in enumerate(model.layers_last):
                global_idx = start_idx_70b + local_idx
                prefix = f"model.layers.{global_idx}"

                # Attention weights (QKV fused)
                _fetch_tp_shard_tensor_qkv(
                    layer.self_attn.qkv_proj.weight,
                    f"{prefix}.self_attn.q_proj.weight",
                    f"{prefix}.self_attn.k_proj.weight",
                    f"{prefix}.self_attn.v_proj.weight",
                    state_dict_70b
                )

                # Attention output projection
                _fetch_tp_shard_tensor(
                    layer.self_attn.o_proj.weight,
                    f"{prefix}.self_attn.o_proj.weight",
                    state_dict_70b,
                    chunk_dim=1
                )

                # MLP gate_up fused
                _fetch_tp_shard_tensor_gate_up(
                    layer.mlp.gate_up_proj.weight,
                    f"{prefix}.mlp.gate_proj.weight",
                    f"{prefix}.mlp.up_proj.weight",
                    state_dict_70b
                )

                # MLP down projection
                _fetch_tp_shard_tensor(
                    layer.mlp.down_proj.weight,
                    f"{prefix}.mlp.down_proj.weight",
                    state_dict_70b,
                    chunk_dim=1
                )

                # Layer norms
                if f"{prefix}.input_layernorm.weight" in state_dict_70b:
                    layer.input_layernorm.weight.data.copy_(
                        state_dict_70b[f"{prefix}.input_layernorm.weight"]
                    )
                if f"{prefix}.post_attention_layernorm.weight" in state_dict_70b:
                    layer.post_attention_layernorm.weight.data.copy_(
                        state_dict_70b[f"{prefix}.post_attention_layernorm.weight"]
                    )

        # Load final norm (only on last PP rank)
        if pp_rank == pp_size - 1:
            if "model.norm.weight" in state_dict_70b:
                model.norm.weight.data.copy_(state_dict_70b["model.norm.weight"])

            # Load LM head or value head
            if is_value_model and hasattr(model, 'value_head'):
                # Value head initialization (can be random or from checkpoint)
                pass
            elif hasattr(model, 'lm_head'):
                _fetch_tp_shard_tensor(
                    model.lm_head.weight,
                    "lm_head.weight",
                    state_dict_70b,
                    chunk_dim=0
                )

    end_time = time.time()
    print_rank_0(f"Custom Split LLaMA weight loading completed in {end_time - start_time:.2f}s")


def load_hf_weights_to_custom_split_llama(
    model_path_8b: str,
    model_path_70b: str,
    adapter_checkpoint_path: str,
    wrapped_models,
    config,
    params_dtype,
    is_value_model=False,
):
    """
    Load HuggingFace checkpoint weights into Custom Split LLaMA model.

    Args:
        model_path_8b: Path to 8B Llama HuggingFace checkpoint
        model_path_70b: Path to 70B Llama HuggingFace checkpoint
        adapter_checkpoint_path: Path to adapter weights checkpoint
        wrapped_models: List of wrapped model modules
        config: Model configuration
        params_dtype: Parameter dtype
        is_value_model: Whether this is a value model
    """
    from transformers import AutoModelForCausalLM

    # Load 8B model state dict
    model_8b = AutoModelForCausalLM.from_pretrained(
        model_path_8b,
        torch_dtype=params_dtype,
        low_cpu_mem_usage=True
    )
    state_dict_8b = model_8b.state_dict()
    del model_8b

    # Load 70B model state dict
    model_70b = AutoModelForCausalLM.from_pretrained(
        model_path_70b,
        torch_dtype=params_dtype,
        low_cpu_mem_usage=True
    )
    state_dict_70b = model_70b.state_dict()
    del model_70b

    # Load adapter weights
    adapter_state_dict = {}
    if adapter_checkpoint_path:
        adapter_state_dict = torch.load(adapter_checkpoint_path, map_location='cpu')

    # Load weights
    load_custom_split_llama_weights(
        state_dict_8b=state_dict_8b,
        state_dict_70b=state_dict_70b,
        adapter_state_dict=adapter_state_dict,
        wrapped_models=wrapped_models,
        config=config,
        params_dtype=params_dtype,
        is_value_model=is_value_model,
    )
