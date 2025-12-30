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
Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback

Implementation based on: https://arxiv.org/abs/2506.03106

This recipe combines natural language feedback (critiques) with numerical rewards
to train LLMs for improved reasoning capabilities.
"""

from .critique_prompts import generate_critique
from .refinement_prompts import generate_refinement, refine_prompt

__all__ = [
    "generate_critique",
    "generate_refinement",
    "refine_prompt",
]
