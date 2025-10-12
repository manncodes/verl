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
Custom Split LLaMA vLLM implementation.

This file should be copied to vLLM's model directory or registered via vLLM's plugin system.
"""

import os
from collections import defaultdict
from typing import Iterable, List, Optional, Tuple, Union
import torch
from torch import nn
from transformers import LlamaConfig

from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                                RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.llama import LlamaDecoderLayer, LlamaForCausalLM
from vllm.model_executor.models.utils import make_empty_intermediate_tensors_factory


class AdapterLayer(nn.Module):
    """
    Custom adapter layer to bridge 8B to 70B hidden dimensions.
    This implementation from the previous fix is correct.
    """
    def __init__(self, config_8b, config_70b, mlp_adapter=False):
        super().__init__()
        self.mlp_adapter = mlp_adapter

        self.adapter_linear_1 = ColumnParallelLinear(
            config_8b.hidden_size,
            config_70b.hidden_size,
            bias=False,
        )

        self.adapter_linear_2 = RowParallelLinear(
            config_70b.hidden_size,
            config_70b.hidden_size,
            bias=False,
            input_is_parallel=True
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor,
                residual: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if residual is not None:
            input_states = hidden_states + residual
        else:
            input_states = hidden_states

        adapted_states, _ = self.adapter_linear_1(input_states)

        if self.mlp_adapter:
            adapted_states = torch.nn.functional.relu(adapted_states)

        adapted_states, _ = self.adapter_linear_2(adapted_states)

        return adapted_states, None


class CustomSplitLLamaForCausalLM(LlamaForCausalLM):
    """Custom split model combining 8B and 70B Llama components"""

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super(LlamaForCausalLM, self).__init__()

        self.config = vllm_config.model_config.hf_config

        config_8b = LlamaConfig.from_pretrained(self.config.path8b)
        config_70b = LlamaConfig.from_pretrained(self.config.path70b)

        config_8b._attn_implementation = "flash_attention_2"
        config_70b._attn_implementation = "flash_attention_2"

        self.num_layers_8b = self.config.num_layers_8
        self.num_layers_70b = self.config.num_layers_70
        self.mlp = hasattr(self.config, "mlp") and self.config.mlp

        self.embed_tokens = VocabParallelEmbedding(
            config_8b.vocab_size,
            config_8b.hidden_size
        )

        self.layers = nn.ModuleList()

        for i in range(self.num_layers_8b):
            self.layers.append(
                LlamaDecoderLayer(
                    config=config_8b,
                    cache_config=vllm_config.cache_config,
                    quant_config=vllm_config.quant_config,
                    prefix=f"{prefix}layers.{i}"
                )
            )

        # Adapter is at index `num_layers_8b`
        self.adapter_layer_idx = self.num_layers_8b
        self.layers.append(AdapterLayer(config_8b, config_70b, mlp_adapter=self.mlp))

        for i in range(self.num_layers_70b):
            # The vLLM layer index continues sequentially after the adapter
            vllm_layer_idx = self.num_layers_8b + 1 + i
            self.layers.append(
                LlamaDecoderLayer(
                    config=config_70b,
                    cache_config=vllm_config.cache_config,
                    quant_config=vllm_config.quant_config,
                    prefix=f"{prefix}layers.{vllm_layer_idx}"
                )
            )

        self.norm = RMSNorm(config_70b.hidden_size, eps=config_70b.rms_norm_eps)
        self.unpadded_vocab_size = self.config.vocab_size
        self.lm_head = ParallelLMHead(self.unpadded_vocab_size, config_70b.hidden_size)

        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config_70b.hidden_size
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, IntermediateTensors]:

        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)

        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states)

        return hidden_states

    def load_weights(self,
                     weights: Iterable[Tuple[str, torch.Tensor]],
                     target_dtype: Optional[torch.dtype] = None):

        params = dict(self.named_parameters())
        loaded_params = set()

        # Step 1: Group all incoming shards by their final destination parameter name.
        grouped_shards = defaultdict(lambda: defaultdict(list))

        for hf_name, shard in weights:
            # Determine the correct vLLM parameter name for this shard
            if hf_name.startswith("model."):
                param_name = hf_name[len("model."):]
            else:
                param_name = hf_name

            mapped_pname = None
            is_fused_part = False

            # --- Name Mapping Logic ---
            if param_name.startswith("layers_first."):
                parts = param_name.split('.')
                idx, tail = int(parts[1]), '.'.join(parts[2:])
                mapped_pname = f"layers.{idx}.{tail}"
            elif param_name.startswith("layers_last."):
                parts = param_name.split('.')
                idx_in_last, tail = int(parts[1]), '.'.join(parts[2:])
                vllm_idx = self.adapter_layer_idx + 1 + idx_in_last
                mapped_pname = f"layers.{vllm_idx}.{tail}"
            elif "adapter" in param_name:
                mapped_pname = f"layers.{self.adapter_layer_idx}.{param_name}"
            elif param_name.startswith(("embed_tokens.", "norm.", "lm_head.")):
                mapped_pname = param_name

            if not mapped_pname:
                continue

            # --- Grouping Logic for Fused Weights ---
            # If it's a Q, K, or V weight, group it under the 'qkv_proj' key.
            if mapped_pname.endswith("self_attn.q_proj.weight"):
                pname_base = mapped_pname.replace("q_proj.weight", "qkv_proj.weight")
                grouped_shards[pname_base]["q"].append(shard)
            elif mapped_pname.endswith("self_attn.k_proj.weight"):
                pname_base = mapped_pname.replace("k_proj.weight", "qkv_proj.weight")
                grouped_shards[pname_base]["k"].append(shard)
            elif mapped_pname.endswith("self_attn.v_proj.weight"):
                pname_base = mapped_pname.replace("v_proj.weight", "qkv_proj.weight")
                grouped_shards[pname_base]["v"].append(shard)
            # If it's a Gate or Up weight, group it under the 'gate_up_proj' key.
            elif mapped_pname.endswith("mlp.gate_proj.weight"):
                pname_base = mapped_pname.replace("gate_proj.weight", "gate_up_proj.weight")
                grouped_shards[pname_base]["gate"].append(shard)
            elif mapped_pname.endswith("mlp.up_proj.weight"):
                pname_base = mapped_pname.replace("up_proj.weight", "gate_up_proj.weight")
                grouped_shards[pname_base]["up"].append(shard)
            # Otherwise, it's a simple, non-fused weight.
            else:
                grouped_shards[mapped_pname]["__default__"].append(shard)

        # Step 2: Iterate through the grouped shards, coalesce, and load.
        for pname, shard_parts in grouped_shards.items():
            param = params.get(pname)
            if param is None:
                print(f"[WARNING] No parameter found for mapped name: {pname}")
                continue

            full_tensor = None
            # Handle Fused Tensors
            if pname.endswith("qkv_proj.weight"):
                q = torch.cat(shard_parts["q"], dim=0)
                k = torch.cat(shard_parts["k"], dim=0)
                v = torch.cat(shard_parts["v"], dim=0)
                full_tensor = torch.cat([q, k, v], dim=0)
            elif pname.endswith("gate_up_proj.weight"):
                gate = torch.cat(shard_parts["gate"], dim=0)
                up = torch.cat(shard_parts["up"], dim=0)
                full_tensor = torch.cat([gate, up], dim=0)
            # Handle Simple Tensors
            else:
                shards = shard_parts["__default__"]
                if len(shards) == 0: continue
                # Explicitly define concat axis based on parameter type
                if "o_proj" in pname or "down_proj" in pname:
                    full_tensor = torch.cat(shards, dim=1)
                else: # Covers embed_tokens, layernorms, adapter weights
                    full_tensor = torch.cat(shards, dim=0) if len(shards) > 1 else shards[0]

            try:
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, full_tensor)
                loaded_params.add(pname)
            except Exception as e:
                print(f"[ERROR] Failed to load {pname}: {e}")

        # Step 3: Final Summary
        print("\n--- Weight loading summary ---")
        total_params = len(params)
        print(f"Model has {total_params} parameters; loaded {len(loaded_params)}")
        if len(loaded_params) < total_params:
            print("WARNING: Some parameters were not loaded:")
            missing = sorted(list(set(params.keys()) - loaded_params))
            for name in missing:
                print(f"  - {name}")
        else:
            print("SUCCESS: All mapped parameters loaded")
        print("------------------------------")
