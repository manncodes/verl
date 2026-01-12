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
GDPO (Group reward-Decoupled normalization Policy Optimization) Recipe.

GDPO addresses the issue where directly applying GRPO to normalize distinct rollout
reward combinations causes them to collapse into identical advantage values.

Key difference from GRPO:
- GRPO: Aggregate rewards first → Normalize the combined reward
    combined_reward = w1*r1 + w2*r2 + ...
    normalized_adv = (combined_reward - group_mean) / group_std

- GDPO: Normalize each reward independently → Then aggregate
    normalized_r1 = (r1 - group_mean(r1)) / group_std(r1)
    normalized_r2 = (r2 - group_mean(r2)) / group_std(r2)
    combined_adv = w1*normalized_r1 + w2*normalized_r2 + ...

See: https://arxiv.org/abs/2601.05242
"""
