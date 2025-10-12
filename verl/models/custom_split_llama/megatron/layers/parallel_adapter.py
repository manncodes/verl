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

from typing import Optional

import torch
from megatron.core import ModelParallelConfig, mpu, tensor_parallel
from torch import nn
from transformers.models.llama.configuration_llama import LlamaConfig

from verl.utils.megatron import tensor_parallel as tp_utils


class ParallelAdapter(nn.Module):
    """
    Parallel adapter layer to bridge 8B to 70B hidden dimensions.
    Supports tensor parallelism for efficient distributed training.
    """

    def __init__(self, config_8b: LlamaConfig, config_70b: LlamaConfig,
                 megatron_config: ModelParallelConfig, mlp_adapter: bool = False):
        super().__init__()
        self.mlp_adapter = mlp_adapter
        self.megatron_config = megatron_config

        # First linear layer: project from 8B to 70B hidden size
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        if megatron_config is not None:
            tp_utils.update_kwargs_with_config(column_kwargs, megatron_config)

        self.adapter_linear_1 = tensor_parallel.ColumnParallelLinear(
            input_size=config_8b.hidden_size,
            output_size=config_70b.hidden_size,
            bias=False,
            gather_output=False,
            **column_kwargs,
        )

        # Second linear layer: maintain 70B hidden size
        row_kwargs = tp_utils.get_default_kwargs_for_row_parallel_linear()
        if megatron_config is not None:
            tp_utils.update_kwargs_with_config(row_kwargs, megatron_config)

        self.adapter_linear_2 = tensor_parallel.RowParallelLinear(
            input_size=config_70b.hidden_size,
            output_size=config_70b.hidden_size,
            bias=False,
            input_is_parallel=True,
            **row_kwargs,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # First projection
        hidden_states, _ = self.adapter_linear_1(hidden_states)

        # Optional activation
        if self.mlp_adapter:
            hidden_states = torch.nn.functional.relu(hidden_states)

        # Second projection
        hidden_states, _ = self.adapter_linear_2(hidden_states)

        return hidden_states


class ParallelAdapterRmPad(nn.Module):
    """
    Parallel adapter layer for packed inputs (removed padding).
    Supports sequence parallelism for efficient training with variable-length sequences.
    """

    def __init__(self, config_8b: LlamaConfig, config_70b: LlamaConfig,
                 megatron_config: ModelParallelConfig, mlp_adapter: bool = False):
        super().__init__()
        self.mlp_adapter = mlp_adapter
        self.megatron_config = megatron_config

        # First linear layer: project from 8B to 70B hidden size
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        if megatron_config is not None:
            tp_utils.update_kwargs_with_config(column_kwargs, megatron_config)

        self.adapter_linear_1 = tensor_parallel.ColumnParallelLinear(
            input_size=config_8b.hidden_size,
            output_size=config_70b.hidden_size,
            bias=False,
            gather_output=False,
            **column_kwargs,
        )

        # Second linear layer: maintain 70B hidden size
        row_kwargs = tp_utils.get_default_kwargs_for_row_parallel_linear()
        if megatron_config is not None:
            tp_utils.update_kwargs_with_config(row_kwargs, megatron_config)

        self.adapter_linear_2 = tensor_parallel.RowParallelLinear(
            input_size=config_70b.hidden_size,
            output_size=config_70b.hidden_size,
            bias=False,
            input_is_parallel=True,
            **row_kwargs,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # First projection
        hidden_states, _ = self.adapter_linear_1(hidden_states)

        # Optional activation
        if self.mlp_adapter:
            hidden_states = torch.nn.functional.relu(hidden_states)

        # Second projection
        hidden_states, _ = self.adapter_linear_2(hidden_states)

        return hidden_states
