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
Critique generation functions for Critique-GRPO.

This module provides different types of critique generation:
- simple: Just indicates correct/incorrect
- simple_gt: Indicates correct/incorrect with ground truth
- text: Full LLM-generated critique with step-by-step error analysis

Based on: https://arxiv.org/abs/2506.03106
"""

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def create_incorrect_critique_prompt(answer: str, ground_truth: str, question: str) -> list[dict]:
    """Create a prompt for generating a critique of an incorrect solution.

    Args:
        answer: The student's answer to critique
        ground_truth: The correct solution
        question: The original question

    Returns:
        Chat prompt formatted as a list of message dictionaries
    """
    chat_prompt = [
        {
            "role": "system",
            "content": (
                "You are a science expert. Analyze a student's incorrect solution by comparing "
                "it with the correct solution. Identify any errors and provide a detailed "
                "step-by-step explanation of mistakes. Conclude with: 'Conclusion: wrong [END]'"
            )
        },
        {
            "role": "user",
            "content": f"""**Question:** {question}

**Student's Answer:**
{answer}

**Correct Solution:**
{ground_truth}

**Critique:**"""
        }
    ]
    return chat_prompt


def create_correct_critique_prompt(answer: str, ground_truth: str, question: str) -> list[dict]:
    """Create a prompt for generating a critique of a correct solution.

    Args:
        answer: The student's answer to critique
        ground_truth: The correct solution
        question: The original question

    Returns:
        Chat prompt formatted as a list of message dictionaries
    """
    chat_prompt = [
        {
            "role": "system",
            "content": (
                "You are a science expert. Analyze a student's solution by comparing it with "
                "the correct solution. Identify any potential errors and provide a detailed "
                "step-by-step explanation, if they exist. Conclude with: 'Conclusion: right [END]'"
            )
        },
        {
            "role": "user",
            "content": f"""**Question:** {question}

**Student's Answer:**
{answer}

**Correct Solution:**
{ground_truth}

**Critique:**"""
        }
    ]
    return chat_prompt


def call_critique_api(
    prompt: list[dict],
    api_endpoint: Optional[str] = None,
    model_name: str = "gpt-4o",
    max_tokens: int = 4096,
    temperature: float = 0.7,
    max_retries: int = 3
) -> Optional[str]:
    """Call an external API to generate a critique.

    Args:
        prompt: Chat prompt as list of message dictionaries
        api_endpoint: API endpoint URL (uses Azure OpenAI if None)
        model_name: Name of the model to use
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        max_retries: Number of retry attempts

    Returns:
        Generated critique text or None if failed
    """
    try:
        from openai import AzureOpenAI
    except ImportError:
        logger.warning("OpenAI package not installed. Text critique generation unavailable.")
        return None

    if not api_endpoint:
        logger.warning("No API endpoint configured for text critique generation.")
        return None

    client = AzureOpenAI(
        azure_endpoint=api_endpoint,
        api_key="",  # Should be configured via environment
        api_version="",
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.75,
                timeout=30
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to generate critique after {max_retries} attempts: {e}")
                return None
            time.sleep(10)

    return None


def generate_critique(
    sample: Dict[str, Any],
    critique_type: str = "simple_gt",
    api_endpoint: Optional[str] = None,
    model_name: str = "gpt-4o"
) -> Dict[str, Any]:
    """Generate a critique for a given sample based on the specified critique type.

    This is the main entry point for critique generation. It supports three types:

    1. "simple": Just indicates if the solution is correct or incorrect
    2. "simple_gt": Indicates correct/incorrect and includes the ground truth
    3. "text": Full LLM-generated critique with detailed error analysis

    Args:
        sample: Dictionary containing:
            - "question": The original question
            - "response": The model's response to critique
            - "gt": The ground truth answer
            - "target": The target solution (optional, used for text critiques)
            - "score": The correctness score (1.0 = correct, 0.0 = incorrect)
        critique_type: Type of critique to generate ("simple", "simple_gt", or "text")
        api_endpoint: API endpoint for text critique generation (only used for "text" type)
        model_name: Model name for API calls (only used for "text" type)

    Returns:
        Updated sample dictionary with "critique" field added

    Example:
        >>> sample = {
        ...     "question": "What is 2 + 2?",
        ...     "response": "2 + 2 = 5",
        ...     "gt": "4",
        ...     "score": 0.0
        ... }
        >>> result = generate_critique(sample, critique_type="simple_gt")
        >>> print(result["critique"])
        "The generated solution is incorrect, the ground truth is 4."
    """
    score = sample.get("score", 0.0)
    gt = sample.get("gt", "")

    if critique_type == "text":
        # Full text critique using external LLM
        if score > 0:
            # Correct solution - simple positive feedback
            critique = f"The generated solution is correct, the ground truth is {gt}."
        else:
            # Incorrect solution - generate detailed critique
            critique_prompt = create_incorrect_critique_prompt(
                answer=sample.get("response", ""),
                ground_truth=sample.get("target", gt),
                question=sample.get("question", "")
            )
            critique = call_critique_api(
                prompt=critique_prompt,
                api_endpoint=api_endpoint,
                model_name=model_name
            )
            # Fallback to simple_gt if API call fails
            if critique is None or isinstance(critique, dict):
                critique = f"The generated solution is incorrect, the ground truth is {gt}."

    elif critique_type == "simple_gt":
        # Simple critique with ground truth
        if score > 0:
            critique = f"The generated solution is correct, the ground truth is {gt}."
        else:
            critique = f"The generated solution is incorrect, the ground truth is {gt}."

    elif critique_type == "simple":
        # Simplest critique - just correct/incorrect
        if score > 0:
            critique = "The generated solution is correct."
        else:
            critique = "The generated solution is incorrect."

    else:
        raise ValueError(f"Unknown critique_type: {critique_type}. "
                        f"Expected one of: 'simple', 'simple_gt', 'text'")

    sample["critique"] = critique
    logger.debug(f"Generated critique ({critique_type}): {critique[:100]}...")

    return sample


def generate_critique_batch(
    samples: list[Dict[str, Any]],
    critique_type: str = "simple_gt",
    api_endpoint: Optional[str] = None,
    model_name: str = "gpt-4o",
    num_workers: int = 8
) -> list[Dict[str, Any]]:
    """Generate critiques for a batch of samples in parallel.

    Args:
        samples: List of sample dictionaries
        critique_type: Type of critique to generate
        api_endpoint: API endpoint for text critique generation
        model_name: Model name for API calls
        num_workers: Number of parallel workers for text critique generation

    Returns:
        List of updated sample dictionaries with critiques
    """
    from concurrent.futures import ThreadPoolExecutor

    if critique_type == "text" and api_endpoint:
        # Use parallel processing for API calls
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(
                lambda s: generate_critique(s, critique_type, api_endpoint, model_name),
                samples
            ))
        return results
    else:
        # Simple/simple_gt critiques are fast, no need for parallelization
        return [generate_critique(s, critique_type) for s in samples]
