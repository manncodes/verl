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
"""
Cascaded RL Training Recipe for Multi-Task Domain-Wise Reinforcement Learning.

This recipe implements the Nemotron-Cascade methodology for training
general-purpose reasoning models through sequential domain-wise RL:
    SFT → RLHF → IF-RL → Math-RL → Code-RL → SWE-RL

Key features:
- GRPO algorithm with on-policy training (no KL penalty)
- Domain-specific reward functions and verifiers
- Checkpoint transfer between training stages
- RL stages rarely degrade prior domain performance

References:
    - Nemotron-Cascade: https://arxiv.org/abs/2512.13607
    - Dolci-Think-RL: https://huggingface.co/datasets/allenai/Dolci-Think-RL-32B
"""

from .cascaded_trainer import (
    CascadedRLTrainer,
    CascadeRLConfig,
    StageConfig,
    create_default_cascade_stages,
)

from .tasks.reward_functions import (
    RLHFRewardManager,
    IFRewardManager,
    MathRewardManager,
    CodeRewardManager,
    SWERewardManager,
    CascadeRewardManager,
    get_reward_manager_for_domain,
)

__all__ = [
    # Trainer
    "CascadedRLTrainer",
    "CascadeRLConfig",
    "StageConfig",
    "create_default_cascade_stages",
    # Reward managers
    "RLHFRewardManager",
    "IFRewardManager",
    "MathRewardManager",
    "CodeRewardManager",
    "SWERewardManager",
    "CascadeRewardManager",
    "get_reward_manager_for_domain",
]
