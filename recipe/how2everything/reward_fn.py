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
1. compute_score_how2: Async GenRM reward using How2Judge model (requires rollout.mode=async)
2. compute_score_how2_rule: Sync rule-based heuristic reward (no judge model needed)

The async variant is called by NaiveRewardLoopManager.run_single() which provides
reward_router_address and reward_model_tokenizer as extra kwargs. This requires:
    actor_rollout_ref.rollout.mode=async
    reward_model.enable=True
    reward_model.enable_resource_pool=True

Usage in VeRL config:
    custom_reward_function.path=recipe/how2everything/reward_fn.py
    custom_reward_function.name=compute_score_how2          # For GenRM (async mode)
    custom_reward_function.name=compute_score_how2_rule     # For rule-only ablation (sync mode)

Reference: https://github.com/lilakk/how2everything
"""

import asyncio
import json
import logging
import os
import re

import aiohttp
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ---------------------------------------------------------------------------
# How2Score judge prompt template (upstream: prompts/judge.txt)
#
# The How2Judge model was trained to output structured JSON:
#   {"reasoning": str, "critical_failures": [{"failure": str, "L1_steps": [int], "L2_steps": [int]}]}
# Pass/fail is determined by whether critical_failures is empty.
#
# Template variables: {goal}, {reference_steps}, {steps}
# Note: resources are NOT part of the upstream judge prompt.
# ---------------------------------------------------------------------------

HOW2SCORE_JUDGE_TEMPLATE = (
    "You are evaluating whether a candidate procedure (L2) correctly achieves "
    "a stated goal, using a reference procedure (L1) as a reliable guide.\n\n"
    "[Goal]\n{goal}\n\n"
    "[Reference Procedure (L1)]\n{reference_steps}\n\n"
    "[Candidate Procedure (L2)]\n{steps}\n\n"
    'A "critical failure" is any issue that would prevent achieving the goal. '
    "This includes:\n"
    "- Steps that contradict the goal or diverge significantly from the reference\n"
    "- Internal inconsistencies, incoherence, or severe vagueness\n"
    "- Missing essential steps or unnecessary additions that would prevent success\n\n"
    "L1 reliably achieves the goal as written, but it may not be the only valid way "
    "to do so. Use it as a reliable reference, not the exclusive solution. "
    "Minor phrasing differences and additional practical steps that don't interfere "
    "with the outcome are acceptable.\n\n"
    "Return only valid json."
)

JUDGE_SAMPLING_PARAMS = {
    "max_new_tokens": 2048,
}


# ---------------------------------------------------------------------------
# Judge prompt construction and response parsing
# ---------------------------------------------------------------------------


def _format_steps(steps):
    """Format a list of steps into numbered lines."""
    if isinstance(steps, list):
        return "\n".join(f"{i + 1}. {step}" for i, step in enumerate(steps))
    return str(steps)


def _build_judge_prompt(goal, reference_steps, candidate_steps):
    """Build the How2Score judge prompt.

    Args:
        goal: The procedural goal string.
        reference_steps: List of reference steps (ground truth).
        candidate_steps: The candidate procedure text (model output).

    Returns:
        Formatted judge prompt string.
    """
    return HOW2SCORE_JUDGE_TEMPLATE.format(
        goal=goal,
        reference_steps=_format_steps(reference_steps),
        steps=candidate_steps,
    )


def _parse_judge_response(judge_output):
    """Parse the How2Judge model JSON output into a reward signal.

    The How2Judge model outputs JSON with:
        {"reasoning": str, "critical_failures": [{"failure": str, "L1_steps": [...], "L2_steps": [...]}]}

    Pass/fail is determined by whether critical_failures is empty.

    Args:
        judge_output: Raw text output from the How2Judge model.

    Returns:
        Dict with keys: score (float), has_failure (bool), n_failures (int), parse_failed (bool).
    """
    if not judge_output or not judge_output.strip():
        return {"score": 0.0, "has_failure": False, "n_failures": 0, "parse_failed": True}

    text = judge_output.strip()

    # Try to parse as JSON directly
    try:
        result = json.loads(text)
        failures = result.get("critical_failures", [])
        has_failure = len(failures) > 0
        return {
            "score": -1.0 if has_failure else 1.0,
            "has_failure": has_failure,
            "n_failures": len(failures),
            "parse_failed": False,
        }
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: try to extract JSON from within the text (model may add preamble)
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(0))
            failures = result.get("critical_failures", [])
            has_failure = len(failures) > 0
            return {
                "score": -1.0 if has_failure else 1.0,
                "has_failure": has_failure,
                "n_failures": len(failures),
                "parse_failed": False,
            }
        except (json.JSONDecodeError, TypeError):
            pass

    # Could not parse JSON -- treat as parse failure
    return {"score": 0.0, "has_failure": False, "n_failures": 0, "parse_failed": True}


# ---------------------------------------------------------------------------
# HTTP helper for async GenRM calls (mirrors recipe/fapo/reward_fn_reasoning.py)
# ---------------------------------------------------------------------------


async def generate_aiohttp(router_address, prompt_ids, sampling_params):
    """Send a generation request to the GenRM server."""
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
#
# Called by NaiveRewardLoopManager.run_single() which provides
# reward_router_address and reward_model_tokenizer as kwargs.
# Requires actor_rollout_ref.rollout.mode=async.
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

    Args:
        data_source: Dataset identifier (e.g., "how2everything/how2train").
        solution_str: The generated procedure text from the policy model.
        ground_truth: JSON-encoded reference procedure with keys: goal, steps.
        extra_info: Additional metadata (split, index, topic, etc.).
        reward_router_address: Host:port of the How2Judge server (injected by RewardLoopWorker).
        reward_model_tokenizer: Tokenizer for the How2Judge model (injected by RewardLoopWorker).

    Returns:
        Dict with keys: score, has_failure, n_failures, parse_failed.
    """
    loop = asyncio.get_running_loop()

    # Parse ground truth
    try:
        ref = json.loads(ground_truth)
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"Failed to parse ground_truth JSON: {ground_truth[:200]}")
        return {"score": 0.0, "has_failure": False, "n_failures": 0, "parse_failed": True}

    # Empty or trivially short responses get a direct penalty
    stripped = solution_str.strip()
    if not stripped or len(stripped) < 20:
        return {"score": -1.0, "has_failure": True, "n_failures": 1, "parse_failed": False}

    # Build judge prompt (uses only goal, reference_steps, candidate_steps -- no resources)
    judge_prompt = _build_judge_prompt(
        goal=ref["goal"],
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

    # Call GenRM server
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
        result = _parse_judge_response(grm_response)
    else:
        logger.warning("How2Judge returned no output_ids")
        result = {"score": 0.0, "has_failure": False, "n_failures": 0, "parse_failed": True}

    return result


# ---------------------------------------------------------------------------
# Variant B: Rule-based heuristic reward (no judge model needed)
#
# Called by NaiveRewardManager.__call__() in sync mode.
# Does NOT require async rollout mode or a reward model server.
# ---------------------------------------------------------------------------


def compute_score_how2_rule(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
):
    """Heuristic reward for procedural generation without a judge model.

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
    try:
        ref = json.loads(ground_truth)
    except (json.JSONDecodeError, TypeError):
        return {"score": 0.0, "acc": False}

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
