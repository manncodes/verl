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
Refinement prompt generation for Critique-GRPO.

This module creates refinement prompts that incorporate critiques to guide
the model in generating improved solutions.

Based on: https://arxiv.org/abs/2506.03106
"""

import logging
from typing import Any, Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)


# Default system prompt for refinement
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Based on the given question, previous solution "
    "attempt, and critique, please generate an improved solution. Think step by step."
)

# Qwen-style thinking prompt
QWEN_THINKING_PROMPT = (
    "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. "
    "You should think step-by-step."
)


def refine_prompt(
    question: str,
    response: str,
    critique: str,
    system_prompt: Optional[str] = None
) -> str:
    """Create a refinement prompt incorporating critique feedback.

    This function constructs a prompt that presents:
    1. The original question
    2. The previous (potentially incorrect) response
    3. The critique/feedback on the response

    The model is then expected to generate an improved solution.

    Args:
        question: The original question/problem
        response: The previous response/solution attempt
        critique: The critique/feedback on the response
        system_prompt: Optional custom system prompt

    Returns:
        Formatted refinement prompt string
    """
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    # Construct the refinement context
    refinement_context = f"""**Question:**
{question}

**Previous Attempt:**
{response}

**Critique:**
{critique}

**Improved Solution:**
"""

    return refinement_context


def create_refinement_messages(
    question: str,
    response: str,
    critique: str,
    system_prompt: Optional[str] = None,
    include_thinking_tag: bool = True
) -> List[Dict[str, str]]:
    """Create chat-formatted refinement messages.

    Args:
        question: The original question/problem
        response: The previous response/solution attempt
        critique: The critique/feedback on the response
        system_prompt: Optional custom system prompt
        include_thinking_tag: Whether to add <think> tag for Qwen-style models

    Returns:
        List of message dictionaries for chat format
    """
    if system_prompt is None:
        system_prompt = QWEN_THINKING_PROMPT if include_thinking_tag else DEFAULT_SYSTEM_PROMPT

    user_content = f"""**Question:**
{question}

**Previous Attempt:**
{response}

**Critique:**
{critique}

Please provide an improved solution to the question above."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    return messages


def generate_refinement(
    sample: Dict[str, Any],
    tokenizer: Any,
    system_prompt: Optional[str] = None,
    include_thinking_tag: bool = True,
    max_prompt_length: Optional[int] = None
) -> Tuple[str, List[int]]:
    """Generate a refinement prompt and tokenize it.

    This is the main entry point for creating refinement prompts from a sample
    that has already been critiqued.

    Args:
        sample: Dictionary containing:
            - "question": The original question
            - "response": The previous response to refine
            - "critique": The critique/feedback on the response
        tokenizer: HuggingFace tokenizer for encoding the prompt
        system_prompt: Optional custom system prompt
        include_thinking_tag: Whether to add <think> tag for Qwen-style models
        max_prompt_length: Maximum length for the tokenized prompt (truncates if exceeded)

    Returns:
        Tuple of (prompt_string, token_ids)

    Example:
        >>> sample = {
        ...     "question": "What is 2 + 2?",
        ...     "response": "2 + 2 = 5",
        ...     "critique": "The solution is incorrect, the ground truth is 4."
        ... }
        >>> prompt, token_ids = generate_refinement(sample, tokenizer)
    """
    question = sample.get("question", "")
    response = sample.get("response", "")
    critique = sample.get("critique", "")

    # Create chat messages
    messages = create_refinement_messages(
        question=question,
        response=response,
        critique=critique,
        system_prompt=system_prompt,
        include_thinking_tag=include_thinking_tag
    )

    # Apply chat template
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        logger.warning(f"Failed to apply chat template: {e}. Using simple format.")
        prompt = refine_prompt(question, response, critique, system_prompt)

    # Add thinking tag if needed (for Qwen-style models)
    if include_thinking_tag and "<think>" not in prompt:
        prompt = prompt + "<think>\n"

    # Tokenize
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)

    # Truncate if needed
    if max_prompt_length is not None and len(token_ids) > max_prompt_length:
        logger.warning(
            f"Refinement prompt exceeds max length ({len(token_ids)} > {max_prompt_length}). "
            "Truncating from the middle of the response."
        )
        # Keep the beginning and end, truncate the middle of the response
        token_ids = token_ids[:max_prompt_length]
        prompt = tokenizer.decode(token_ids, skip_special_tokens=False)

    return prompt, token_ids


def generate_refinement_batch(
    samples: List[Dict[str, Any]],
    tokenizer: Any,
    system_prompt: Optional[str] = None,
    include_thinking_tag: bool = True,
    max_prompt_length: Optional[int] = None,
    num_workers: int = 8
) -> List[Tuple[str, List[int]]]:
    """Generate refinement prompts for a batch of samples.

    Args:
        samples: List of sample dictionaries
        tokenizer: HuggingFace tokenizer
        system_prompt: Optional custom system prompt
        include_thinking_tag: Whether to add <think> tag
        max_prompt_length: Maximum length for tokenized prompts
        num_workers: Number of parallel workers

    Returns:
        List of (prompt_string, token_ids) tuples
    """
    from concurrent.futures import ThreadPoolExecutor

    def process_sample(sample):
        return generate_refinement(
            sample=sample,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            include_thinking_tag=include_thinking_tag,
            max_prompt_length=max_prompt_length
        )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_sample, samples))

    return results


def process_refinement_groups(
    refinement_dicts: List[Dict[str, Any]],
    num_samples: int = 7
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Process refinement groups and select the best refinement from each group.

    For each group of refinements (typically from the same prompt with different
    sampling), this function selects the best one based on the score.

    Args:
        refinement_dicts: List of refinement dictionaries containing:
            - 'refinement': The refined text
            - 'score': The score of the refinement (1.0 is correct, 0.0 is incorrect)
            - 'gt': The ground truth
        num_samples: Number of samples per group (rollout n)

    Returns:
        Tuple of:
            - List of selected refinements (one per group)
            - List of binary scores (1 if any correct refinement found, 0 otherwise)

    Example:
        >>> refinements = [
        ...     {"refinement": "2+2=4", "score": 1.0, "gt": "4"},
        ...     {"refinement": "2+2=5", "score": 0.0, "gt": "4"},
        ...     # ... more refinements
        ... ]
        >>> selected, scores = process_refinement_groups(refinements, num_samples=8)
    """
    selected_refinements = []
    refinement_scores = []

    # Split into groups of num_samples
    for i in range(0, len(refinement_dicts), num_samples):
        group = refinement_dicts[i:i + num_samples]

        if not group:
            continue

        # Try to find a perfect score (1.0) first
        perfect_refinements = [item for item in group if item.get('score', 0) == 1.0]

        if perfect_refinements:
            # If multiple perfect scores, pick the first one
            selected = perfect_refinements[0]
            refine_score = 1
        else:
            # Otherwise select the highest scoring refinement
            selected = max(group, key=lambda x: x.get('score', 0))
            refine_score = 0

        selected_refinements.append(selected)
        refinement_scores.append(refine_score)

    return selected_refinements, refinement_scores
