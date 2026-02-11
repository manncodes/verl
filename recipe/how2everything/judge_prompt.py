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
How2Score judge prompt template and verdict parser.

Implements the LLM-as-judge protocol from the how2everything project for
evaluating procedural instruction quality. The judge assesses whether a
candidate procedure contains "critical failures" that would prevent
achieving the stated goal.

Reference: https://github.com/lilakk/how2everything
"""

import re

# How2Score judge prompt template.
# L1 = reference procedure (ground truth), L2 = candidate procedure (model output).
HOW2SCORE_JUDGE_TEMPLATE = """\
You are evaluating whether a candidate procedure (L2) correctly achieves a stated goal, \
using a reference procedure (L1) as a reliable guide.

[Goal]
{goal}

[Resources]
{resources}

[Reference Procedure (L1)]
{reference_steps}

[Candidate Procedure (L2)]
{candidate_steps}

A "critical failure" is any issue that would prevent achieving the goal. This includes:
- Steps that contradict the goal or diverge significantly from the reference
- Internal inconsistencies, incoherence, or severe vagueness
- Missing essential steps or unnecessary additions that would prevent success

L1 reliably achieves the goal as written, but it may not be the only valid way to do so. \
Use it as a reliable reference, not the exclusive solution. \
Minor phrasing differences and additional practical steps that don't interfere \
with the outcome are acceptable.

First, provide detailed reasoning explaining your evaluation step by step. \
Then, list any critical failures found. If there are no critical failures, \
state "No critical failures found."

Finally, on the last line, output your verdict as exactly one of:
VERDICT: PASS
VERDICT: FAIL
"""

# Sampling parameters for the How2Judge generative reward model.
JUDGE_SAMPLING_PARAMS = {
    "max_new_tokens": 2048,
    "temperature": 0.0,
}


def format_steps(steps):
    """Format a list of steps into numbered lines for the judge prompt.

    Args:
        steps: List of step strings, or a single string.

    Returns:
        Numbered steps as a single string.
    """
    if isinstance(steps, list):
        return "\n".join(f"{i + 1}. {step}" for i, step in enumerate(steps))
    return str(steps)


def format_resources(resources):
    """Format a list of resources into a bracketed string.

    Args:
        resources: List of resource strings, or a single string.

    Returns:
        Bracketed, comma-separated resource string.
    """
    if isinstance(resources, list):
        return "[" + ", ".join(str(r) for r in resources) + "]"
    return str(resources)


def build_judge_prompt(goal, resources, reference_steps, candidate_steps):
    """Build the full How2Score judge prompt.

    Args:
        goal: The procedural goal string.
        resources: List of resources (or formatted string).
        reference_steps: List of reference steps (ground truth).
        candidate_steps: The candidate procedure text (model output).

    Returns:
        Formatted judge prompt string.
    """
    return HOW2SCORE_JUDGE_TEMPLATE.format(
        goal=goal,
        resources=format_resources(resources),
        reference_steps=format_steps(reference_steps),
        candidate_steps=candidate_steps,
    )


def parse_judge_verdict(judge_output):
    """Parse the How2Judge model output into a reward signal.

    Looks for a "VERDICT: PASS" or "VERDICT: FAIL" line in the judge's response.
    Returns a dict with the scalar score and metadata.

    Args:
        judge_output: Raw text output from the How2Judge model.

    Returns:
        Dict with keys:
            - score (float): 1.0 for PASS, -1.0 for FAIL, 0.0 for ambiguous
            - verdict (str): "pass", "fail", or "ambiguous"
            - has_critical_failure (bool or None): Whether critical failures were detected
    """
    if not judge_output or not judge_output.strip():
        return {"score": 0.0, "verdict": "ambiguous", "has_critical_failure": None}

    text = judge_output.strip()

    # Search for VERDICT line (case-insensitive, flexible whitespace)
    verdict_match = re.search(r"VERDICT\s*:\s*(PASS|FAIL)", text, re.IGNORECASE)

    if verdict_match:
        verdict_str = verdict_match.group(1).upper()
        if verdict_str == "PASS":
            return {"score": 1.0, "verdict": "pass", "has_critical_failure": False}
        else:
            return {"score": -1.0, "verdict": "fail", "has_critical_failure": True}

    # Fallback: look for strong signals in the text
    text_lower = text.lower()
    if "no critical failures found" in text_lower:
        return {"score": 1.0, "verdict": "pass", "has_critical_failure": False}
    if "critical failure" in text_lower:
        return {"score": -1.0, "verdict": "fail", "has_critical_failure": True}

    # Could not determine verdict
    return {"score": 0.0, "verdict": "ambiguous", "has_critical_failure": None}
