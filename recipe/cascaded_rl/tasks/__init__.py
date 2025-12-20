# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 The verl Authors
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
"""Domain-specific tasks and reward functions for Cascaded RL."""

from .reward_functions import (
    RLHFRewardManager,
    IFRewardManager,
    MathRewardManager,
    CodeRewardManager,
    SWERewardManager,
    CascadeRewardManager,
    get_reward_manager_for_domain,
)

__all__ = [
    "RLHFRewardManager",
    "IFRewardManager",
    "MathRewardManager",
    "CodeRewardManager",
    "SWERewardManager",
    "CascadeRewardManager",
    "get_reward_manager_for_domain",
]
