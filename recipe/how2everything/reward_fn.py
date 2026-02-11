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
Custom reward functions for How2Everything VeRL integration.

Provides two reward functions:
1. compute_score_how2: Async GenRM reward using How2Judge model
2. compute_score_how2_rule: Rule-based heuristic reward (no judge model needed)

Usage in VeRL config:
    custom_reward_function.path=recipe/how2everything/reward_fn.py
    custom_reward_function.name=compute_score_how2          # For GenRM
    custom_reward_function.name=compute_score_how2_rule     # For rule-only ablation

Reference: https://github.com/lilakk/how2everything
"""

import asyncio
import json
import logging
import os
import re

import aiohttp
from transformers import PreTrainedTokenizer

from recipe.how2everything.judge_prompt import (
    JUDGE_SAMPLING_PARAMS,
    build_judge_prompt,
    parse_judge_verdict,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ---------------------------------------------------------------------------
# HTTP helper for async GenRM calls (mirrors recipe/fapo/reward_fn_reasoning.py)
# ---------------------------------------------------------------------------


async def generate_aiohttp(router_address, prompt_ids, sampling_params):
    """Send a generation request to the GenRM vLLM/SGLang server.

    Args:
        router_address: Host:port of the reward model server.
        prompt_ids: Tokenized prompt as a list of int IDs.
        sampling_params: Dict of sampling parameters.

    Returns:
        Dict with 'output_ids' key on success, empty dict on failure.
    """
    payload = {
        "input_ids": prompt_ids,
        "sampling_params": sampling_params,
    }
    url = f"http://{router_address}/generate"
    try:
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        async with session.post(url, json=payload) as resp:
            output = await resp.text()
            try:
                output = json.loads(output)
                return output
            except Exception:
                logger.error(f"Failed to parse JSON response: {output}")
                return {}
    finally:
        await session.close()


# ---------------------------------------------------------------------------
# Variant A: Async GenRM reward using How2Judge
# ---------------------------------------------------------------------------


async def compute_score_how2(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str,
    reward_model_tokenizer: PreTrainedTokenizer,
):
    """Compute How2Score reward using the How2Judge generative reward model.

    This function is called by VeRL's reward manager for each generated response.
    It constructs a judge prompt comparing the generated procedure against the
    reference, sends it to the How2Judge model, and parses the verdict.

    Args:
        data_source: Dataset identifier (e.g., "how2everything/how2train").
        solution_str: The generated procedure text from the policy model.
        ground_truth: JSON-encoded reference procedure with keys:
            goal, resources, steps, n_steps.
        extra_info: Additional metadata (split, index, topic, etc.).
        reward_router_address: Host:port of the How2Judge vLLM server.
        reward_model_tokenizer: Tokenizer for the How2Judge model.

    Returns:
        Dict with keys: score (float), verdict (str), has_critical_failure (bool|None).
    """
    loop = asyncio.get_running_loop()

    # Parse ground truth
    try:
        ref = json.loads(ground_truth)
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"Failed to parse ground_truth JSON: {ground_truth[:200]}")
        return {"score": 0.0, "verdict": "error", "has_critical_failure": None}

    # Skip judge for empty or refusal responses → direct penalty
    stripped = solution_str.strip()
    if not stripped or len(stripped) < 20:
        return {"score": -1.0, "verdict": "fail", "has_critical_failure": True}

    # Build judge prompt
    judge_prompt = build_judge_prompt(
        goal=ref["goal"],
        resources=ref.get("resources", []),
        reference_steps=ref.get("steps", []),
        candidate_steps=solution_str,
    )

    # Tokenize for the How2Judge model
    prompt_ids = await loop.run_in_executor(
        None,
        lambda: reward_model_tokenizer.apply_chat_template(
            [{"role": "user", "content": judge_prompt}],
            tokenize=True,
            add_generation_prompt=True,
        ),
    )

    # Call GenRM
    grm_outputs = await generate_aiohttp(
        router_address=reward_router_address,
        prompt_ids=prompt_ids,
        sampling_params=JUDGE_SAMPLING_PARAMS,
    )

    # Parse judge response
    grm_response_ids = grm_outputs.get("output_ids", None)
    if grm_response_ids is not None:
        grm_response = await loop.run_in_executor(
            None,
            lambda: reward_model_tokenizer.decode(grm_response_ids, skip_special_tokens=True),
        )
        result = parse_judge_verdict(grm_response)
    else:
        logger.warning("How2Judge returned no output_ids, scoring as ambiguous")
        result = {"score": 0.0, "verdict": "error", "has_critical_failure": None}

    return {
        "score": result["score"],
        "verdict": result["verdict"],
        "has_critical_failure": result["has_critical_failure"],
    }


# ---------------------------------------------------------------------------
# Variant B: Rule-based heuristic reward (no judge model needed)
# ---------------------------------------------------------------------------


def compute_score_how2_rule(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
):
    """Heuristic reward for procedural generation without a judge model.

    Useful for ablation studies, debugging the data pipeline, or environments
    where the How2Judge model is not available.

    Scoring criteria (total 0.0 to 1.0):
    - +0.3: Response contains numbered steps
    - +0.3: Number of steps matches expected count (+/- 1)
    - +0.2: Response mentions key resources from the prompt
    - +0.2: Non-trivial response length (>50 chars)
    - -1.0: Empty or refusal response

    Args:
        data_source: Dataset identifier.
        solution_str: Generated procedure text.
        ground_truth: JSON-encoded reference procedure.
        extra_info: Optional metadata dict.

    Returns:
        Dict with keys: score (float), acc (bool).
    """
    # Parse ground truth
    try:
        ref = json.loads(ground_truth)
    except (json.JSONDecodeError, TypeError):
        return {"score": 0.0, "acc": False}

    # Penalize empty or refusal responses
    stripped = solution_str.strip()
    if not stripped:
        return {"score": -1.0, "acc": False}

    refusal_patterns = ["i cannot", "i can't", "i'm sorry", "i am unable", "as an ai"]
    if any(p in stripped.lower() for p in refusal_patterns):
        return {"score": -1.0, "acc": False}

    score = 0.0

    # Check for numbered steps
    numbered_steps = re.findall(r"^\s*\d+[\.\)]\s", solution_str, re.MULTILINE)
    if numbered_steps:
        score += 0.3

    # Check step count matches expected
    expected_n = ref.get("n_steps", 0)
    if expected_n > 0 and numbered_steps:
        if abs(len(numbered_steps) - expected_n) <= 1:
            score += 0.3

    # Check resource coverage
    resources = ref.get("resources", [])
    if resources:
        mentioned = sum(1 for r in resources if r.lower() in solution_str.lower())
        coverage = mentioned / len(resources)
        score += 0.2 * coverage

    # Base credit for non-trivial response
    if len(stripped) > 50:
        score += 0.2

    return {"score": score, "acc": score >= 0.8}
