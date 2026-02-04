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
Main entry point for structured output RL training with GRPO.

This recipe trains LLMs to produce valid structured outputs (JSON, etc.)
using Group Relative Policy Optimization (GRPO) with schema validation rewards.

Supports three reward modes:
- fine_grained: Weighted combination of JSON parsability, schema validity,
  field coverage, and content correctness scores.
- binary: 0/1 reward based on whether output passes schema validation.
- crane: CRANE-style hybrid decoding where the model reasons freely then
  produces constrained output within delimiters.

This entry point delegates to the standard verl PPO training pipeline
(verl.trainer.main_ppo.run_ppo) with a structured output-specific Hydra
config. The structured output reward manager is configured via:
  - reward_model.reward_manager=structured_output (selects StructuredOutputRewardManager)
  - reward_model.reward_kwargs (passes reward_mode, reasoning_delimiter, etc.)

Usage:
    python -m recipe.structured_output.main_structured_output \
        data.train_files=~/data/structured_output/structured_output_train.parquet \
        data.val_files=~/data/structured_output/structured_output_val.parquet \
        actor_rollout_ref.model.path=meta-llama/Llama-3.1-8B-Instruct
"""

import hydra

from verl.trainer.main_ppo import run_ppo


@hydra.main(config_path="config", config_name="structured_output_grpo", version_base=None)
def main(config):
    run_ppo(config)


if __name__ == "__main__":
    main()
