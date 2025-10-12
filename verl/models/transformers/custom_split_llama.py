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

import torch
import torch.nn as nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel, LlamaPreTrainedModel, LlamaDecoderLayer, LlamaRMSNorm
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from typing import Optional, Union, List, Tuple
from torch.nn import CrossEntropyLoss
from transformers.generation import GenerationMixin
from transformers.utils import logging
from torch.utils.checkpoint import checkpoint


logger = logging.get_logger(__name__)


class CustomSplitLLamaModel(LlamaModel):
    config_class = LlamaConfig
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)
        del self.layers

        # --- Step 1: Load configs and define layer counts ---
        config_8b = LlamaConfig.from_pretrained(config.path8b)
        config_70b = LlamaConfig.from_pretrained(config.path70b)

        config_8b._attn_implementation = "flash_attention_2"
        config_70b._attn_implementation = "flash_attention_2"

        self.mlp = config.mlp if hasattr(config, "mlp") else False
        self.num_layers_8b = config.num_layers_8
        self.num_layers_70b = config.num_layers_70

        # --- Step 2: Build the model architecture from scratch ---
        # Embedding layer (from 8B architecture)
        self.embed_tokens = nn.Embedding(config.vocab_size, config_8b.hidden_size, self.padding_idx)

        # First set of layers (indices 0 to 31, from 8B config)
        self.layers_first = nn.ModuleList(
            [LlamaDecoderLayer(config_8b, layer_idx=i) for i in range(self.num_layers_8b)]
        )

        # Adapter to bridge the hidden dimensions
        if self.mlp:
            self.adapter_linear_1 = nn.Linear(config_8b.hidden_size, config_70b.hidden_size, bias=False)
            self.adapter_linear_2 = nn.Linear(config_70b.hidden_size, config_70b.hidden_size, bias=False)
        else:
            self.adapter = nn.Linear(config_8b.hidden_size, config_70b.hidden_size, bias=False)

        # The original 70B model has 80 layers (0-79). We need the last 8,
        # which are indices 72 through 79. This is crucial for loading weights correctly.
        start_idx_70b = config_70b.num_hidden_layers - self.num_layers_70b # e.g., 80 - 8 = 72
        self.layers_last = nn.ModuleList(
            [LlamaDecoderLayer(config_70b, layer_idx=i)
             for i in range(start_idx_70b, config_70b.num_hidden_layers)]
        )

        self.norm = LlamaRMSNorm(config_70b.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config_70b.hidden_size, config.vocab_size, bias=False)
        self.gradient_checkpointing = True

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Handle cache initialization
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        # `cache_position` is ONLY needed when `use_cache` is True.
        # When `use_cache=False` (like in loglikelihood), we must pass None to avoid bugs.
        if use_cache:
            past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
            if cache_position is None:
                cache_position = torch.arange(
                    past_key_values_length, past_key_values_length + inputs_embeds.shape[1], device=inputs_embeds.device
                )
        else:
            past_key_values_length = 0
            cache_position = None

        # Handle position IDs
        if position_ids is None:
            if attention_mask is not None:
                # This correctly creates position_ids from the attention mask, which is necessary for packing.
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                if past_key_values_length > 0:
                    position_ids = position_ids[:, -inputs_embeds.shape[1] :]
            else:
                # Fallback for simple cases like text generation
                position_ids = torch.arange(
                    past_key_values_length, inputs_embeds.shape[1] + past_key_values_length, dtype=torch.long, device=inputs_embeds.device
                ).unsqueeze(0)

        use_checkpointing = self.gradient_checkpointing and self.training

        if use_cache:
            use_checkpointing = False

        hs = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # Process 8B layers (first half)
        for idx, layer in enumerate(self.layers_first):
            if output_hidden_states:
                all_hidden_states += (hs,)

            if use_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(hidden_states, attention_mask, position_ids, cache_position):
                        return module(
                            hidden_states,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_value=None, # Must be None for checkpointing
                            output_attentions=output_attentions,
                            use_cache=False,      # Must be False for checkpointing
                            cache_position=cache_position,
                        )
                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer),
                    hs,
                    attention_mask,
                    position_ids,
                    cache_position,
                    use_reentrant=False # Recommended for modern PyTorch
                )
            else:
                layer_outputs = layer(
                    hs,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hs = layer_outputs[0]

            if use_cache:
                # The layer returns updated cache
                past_key_values = layer_outputs[1]

            if output_attentions:
                all_self_attns += (layer_outputs[2 if use_cache else 1],)

        # Apply adapter
        if self.mlp:
            hs = torch.relu(self.adapter_linear_1(hs))
            hs = self.adapter_linear_2(hs)
        else:
            hs = self.adapter(hs)

        # Process 70B layers (last half)
        for idx, layer in enumerate(self.layers_last):
            if output_hidden_states:
                all_hidden_states += (hs,)

            if use_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(hidden_states, attention_mask, position_ids, cache_position):
                        return module(
                            hidden_states,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_value=None, # Must be None for checkpointing
                            output_attentions=output_attentions,
                            use_cache=False,      # Must be False for checkpointing
                            cache_position=cache_position,
                        )
                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer),
                    hs,
                    attention_mask,
                    position_ids,
                    cache_position,
                    use_reentrant=False # Recommended for modern PyTorch
                )
            else:
                layer_outputs = layer(
                    hs,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hs = layer_outputs[0]

            if use_cache:
                # The layer returns updated cache
                past_key_values = layer_outputs[1]

            if output_attentions:
                all_self_attns += (layer_outputs[2 if use_cache else 1],)

        hs = self.norm(hs)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hs,)

        return BaseModelOutputWithPast(
            last_hidden_state=hs,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class CustomSplitLLamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = CustomSplitLLamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = self.model.lm_head
        del self.model.lm_head

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits
        if num_logits_to_keep > 0:
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
        else:
            logits = self.lm_head(hidden_states)

        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0]:]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
