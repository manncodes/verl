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

"""PyTorch Custom Split LLaMA model with Megatron-style acceleration."""

from typing import Optional

import torch
import torch.utils.checkpoint
from megatron.core import ModelParallelConfig, mpu, tensor_parallel
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import CausalLMOutputWithPast

from verl.utils.megatron import sequence_parallel as sp_utils
from verl.utils.megatron import tensor_parallel as tp_utils
from verl.utils.megatron_utils import TransformerConfig, convert_config

# Import Llama decoder layers from the existing Llama model
from verl.models.llama.megatron.layers import (
    ParallelLlamaDecoderLayer,
    ParallelLlamaDecoderLayerRmPad,
    ParallelLlamaRMSNorm
)

from .layers.parallel_adapter import ParallelAdapter, ParallelAdapterRmPad

# Import for packed inputs (remove padding)
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input


class ParallelCustomSplitLLamaModel(nn.Module):
    """
    Custom Split LLaMA model combining 8B and 70B components with Megatron parallelism.

    Args:
        config: LlamaConfig with additional attributes for split configuration
        megatron_config: ModelParallelConfig for distributed training
    """

    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig):
        super().__init__()
        self.megatron_config = megatron_config

        # Load configurations for 8B and 70B models
        config_8b = LlamaConfig.from_pretrained(config.path8b)
        config_70b = LlamaConfig.from_pretrained(config.path70b)

        # Get layer counts
        self.num_layers_8b = config.num_layers_8
        self.num_layers_70b = config.num_layers_70
        self.mlp = config.mlp if hasattr(config, "mlp") else False

        # Embedding layer (from 8B architecture)
        embedding_kwargs = tp_utils.get_default_kwargs_for_parallel_embedding()
        if megatron_config is not None:
            tp_utils.update_kwargs_with_config(embedding_kwargs, megatron_config)

        self.embed_tokens = tensor_parallel.VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config_8b.hidden_size,
            **embedding_kwargs
        )

        # First set of layers (from 8B config)
        self.layers_first = nn.ModuleList(
            [ParallelLlamaDecoderLayer(config_8b, megatron_config)
             for _ in range(self.num_layers_8b)]
        )

        # Adapter layer
        self.adapter = ParallelAdapter(config_8b, config_70b, megatron_config, mlp_adapter=self.mlp)

        # Last set of layers (from 70B config)
        self.layers_last = nn.ModuleList(
            [ParallelLlamaDecoderLayer(config_70b, megatron_config)
             for _ in range(self.num_layers_70b)]
        )

        # Final normalization (from 70B architecture)
        self.norm = ParallelLlamaRMSNorm(config_70b, megatron_config)

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds):
        # Copied from Llama implementation
        combined_attention_mask = None
        if input_shape[-1] > 1:
            from verl.models.llama.megatron.modeling_llama_megatron import _make_causal_mask
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
            )

        if attention_mask is not None:
            from verl.models.llama.megatron.modeling_llama_megatron import _expand_mask
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype,
                                             tgt_len=input_shape[-1]).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_length = input_ids.shape
        inputs_embeds = self.embed_tokens(input_ids)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds
        )

        hidden_states = inputs_embeds

        # Process first layers (8B)
        for decoder_layer in self.layers_first:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        # Apply adapter
        hidden_states = self.adapter(hidden_states)

        # Process last layers (70B)
        for decoder_layer in self.layers_last:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states


class ParallelCustomSplitLLamaForCausalLM(nn.Module):
    """Custom Split LLaMA for Causal Language Modeling with Megatron parallelism."""

    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig):
        super().__init__()
        self.config = config
        self.megatron_config = megatron_config
        self.model = ParallelCustomSplitLLamaModel(config, megatron_config=megatron_config)
        self.vocab_size = config.vocab_size

        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        if megatron_config is not None:
            tp_utils.update_kwargs_with_config(column_kwargs, megatron_config)

        # Load 70B config for hidden size
        config_70b = LlamaConfig.from_pretrained(config.path70b)

        self.lm_head = tensor_parallel.ColumnParallelLinear(
            input_size=config_70b.hidden_size,
            output_size=config.vocab_size,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            **column_kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        hidden_states = outputs
        logits = self.lm_head(hidden_states)[0]
        logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)
        logits = logits.float()

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


class ParallelCustomSplitLLamaModelRmPad(nn.Module):
    """
    Custom Split LLaMA model with packed inputs (remove padding) for Megatron parallelism.

    Args:
        config: LlamaConfig with additional attributes for split configuration
        megatron_config: ModelParallelConfig for distributed training
    """

    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig):
        super().__init__()
        self.megatron_config = megatron_config

        # Load configurations for 8B and 70B models
        config_8b = LlamaConfig.from_pretrained(config.path8b)
        config_70b = LlamaConfig.from_pretrained(config.path70b)

        # Get layer counts
        self.num_layers_8b = config.num_layers_8
        self.num_layers_70b = config.num_layers_70
        self.mlp = config.mlp if hasattr(config, "mlp") else False

        # Embedding layer (from 8B architecture)
        embedding_kwargs = tp_utils.get_default_kwargs_for_parallel_embedding()
        if megatron_config is not None:
            tp_utils.update_kwargs_with_config(embedding_kwargs, megatron_config)

        self.embed_tokens = tensor_parallel.VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config_8b.hidden_size,
            **embedding_kwargs
        )

        # First set of layers (from 8B config)
        self.layers_first = nn.ModuleList(
            [ParallelLlamaDecoderLayerRmPad(config_8b, megatron_config)
             for _ in range(self.num_layers_8b)]
        )

        # Adapter layer
        self.adapter = ParallelAdapterRmPad(config_8b, config_70b, megatron_config, mlp_adapter=self.mlp)

        # Last set of layers (from 70B config)
        self.layers_last = nn.ModuleList(
            [ParallelLlamaDecoderLayerRmPad(config_70b, megatron_config)
             for _ in range(self.num_layers_70b)]
        )

        # Final normalization (from 70B architecture)
        self.norm = ParallelLlamaRMSNorm(config_70b, megatron_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        sequence_length: int = None,
        indices: torch.Tensor = None,
        cu_seqlens: torch.Tensor = None,
        max_seqlen_in_batch: int = None,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)  # (1, total_nnz) -> (1, total_nnz, hidden_size)

        # (1, total_nnz, hidden_size) -> (total_nnz, 1, hidden_size) -> (total_nnz // sp, 1, hidden_size)
        inputs_embeds = inputs_embeds.transpose(0, 1)
        if self.megatron_config.sequence_parallel:
            inputs_embeds = tensor_parallel.scatter_to_sequence_parallel_region(inputs_embeds)

        hidden_states = inputs_embeds

        # Process first layers (8B)
        for decoder_layer in self.layers_first:
            hidden_states = decoder_layer(
                hidden_states,
                position_ids=position_ids,
                sequence_length=sequence_length,
                indices=indices,
                cu_seqlens=cu_seqlens,
                max_seqlen_in_batch=max_seqlen_in_batch,
            )

        # Apply adapter
        hidden_states = self.adapter(hidden_states)

        # Process last layers (70B)
        for decoder_layer in self.layers_last:
            hidden_states = decoder_layer(
                hidden_states,
                position_ids=position_ids,
                sequence_length=sequence_length,
                indices=indices,
                cu_seqlens=cu_seqlens,
                max_seqlen_in_batch=max_seqlen_in_batch,
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states


class ParallelCustomSplitLLamaForCausalLMRmPad(nn.Module):
    """Custom Split LLaMA for Causal LM with packed inputs and Megatron parallelism."""

    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig):
        super().__init__()
        self.config = config
        self.megatron_config = megatron_config
        self.model = ParallelCustomSplitLLamaModelRmPad(config, megatron_config=megatron_config)
        self.vocab_size = config.vocab_size
        self._init_head(config)

    def _init_head(self, config):
        """Initialize the LM head. Can be overridden by Value model."""
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        if self.megatron_config is not None:
            assert column_kwargs.get("config", False), "must have ModelParallelConfig"
            tp_utils.update_kwargs_with_config(column_kwargs, self.megatron_config)

        # Load 70B config for hidden size
        config_70b = LlamaConfig.from_pretrained(config.path70b)

        self.lm_head = tensor_parallel.ColumnParallelLinear(
            input_size=config_70b.hidden_size,
            output_size=config.vocab_size,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            **column_kwargs,
        )

    def _forward_head(self, hidden_states):
        """Forward pass through the LM head. Can be overridden by Value model."""
        # all_gather from sequence parallel region is performed inside lm_head
        logits = self.lm_head(hidden_states)[0]
        logits = logits.float()  # (total_nnz_padded, 1, vocab_size // tp)
        logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)  # (total_nnz_padded, 1, vocab_size)
        return logits

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass with automatic padding removal and restoration.

        Args:
            input_ids: Input token IDs (batch_size, sequence_length)
            attention_mask: Attention mask (batch_size, sequence_length)
            position_ids: Position IDs (optional)
        """
        batch_size, sequence_length = input_ids.shape

        # Remove padding for efficient training
        input_ids, indices, cu_seqlens, max_seqlen_in_batch, *_ = unpad_input(
            input_ids.unsqueeze(dim=-1), attention_mask
        )  # (total_nnz, 1)

        # Pad input_ids to multiple of tp for sequence parallel
        if self.megatron_config.sequence_parallel:
            input_ids = sp_utils.pad_to_sequence_parallel(input_ids)

        input_ids = input_ids.transpose(0, 1)  # (1, total_nnz+pad)

        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            sequence_length=sequence_length,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen_in_batch=max_seqlen_in_batch,
        )

        logits = self._forward_head(hidden_states)

        # Remove padding from sequence parallel
        if self.megatron_config.sequence_parallel:
            total_nnz = cu_seqlens[-1]
            logits = logits[:total_nnz]  # (total_nnz)

        logits = torch.squeeze(logits, dim=1)  # remove the artificial batch dimension
        # Add removed padding back
        logits = pad_input(
            logits, indices, batch_size, seqlen=sequence_length
        )  # (batch_size, sequence_length, vocab_size)

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


# For value model (critic) support
class ParallelCustomSplitLLamaForValueRmPad(ParallelCustomSplitLLamaForCausalLMRmPad):
    """
    Custom Split LLaMA for value estimation (critic) with packed inputs.

    Inherits from CausalLM and overrides head initialization to use value head.
    """

    def _init_head(self, config):
        """Override to use value head instead of LM head."""
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        if self.megatron_config is not None:
            assert column_kwargs.get("config", False), "must have ModelParallelConfig"
            tp_utils.update_kwargs_with_config(column_kwargs, self.megatron_config)

        # Load 70B config for hidden size
        config_70b = LlamaConfig.from_pretrained(config.path70b)

        self.value_head = tensor_parallel.ColumnParallelLinear(
            input_size=config_70b.hidden_size,
            output_size=1,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            **column_kwargs,
        )

    def _forward_head(self, hidden_states):
        """Override to forward through value head with appropriate gathering."""
        # all_gather from sequence parallel region is performed inside value_head
        values = self.value_head(hidden_states)[0]
        values = values.float()  # (total_nnz_padded, 1, 1)

        # Gather from tensor parallel region
        if self.megatron_config.sequence_parallel:
            values = tensor_parallel.gather_from_sequence_parallel_region(values, to_model_parallel=False)
        else:
            values = tensor_parallel.gather_from_tensor_model_parallel_region(values)

        return values


# Pipeline parallel versions (PP)
class ParallelCustomSplitLLamaForCausalLMRmPadPP(ParallelCustomSplitLLamaForCausalLMRmPad):
    """Custom Split LLaMA with pipeline parallelism support."""
    pass


class ParallelCustomSplitLLamaForValueRmPadPP(ParallelCustomSplitLLamaForValueRmPad):
    """Custom Split LLaMA value model with pipeline parallelism support."""
    pass
