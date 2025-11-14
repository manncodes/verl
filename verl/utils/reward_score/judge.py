"""
Pydantic-Based LLM Judge with Structured Outputs

Uses OpenAI's structured output feature to guarantee valid JSON schema.
Works with vLLM and other OpenAI-compatible APIs.

Features robust timeout handling to prevent hanging on stuck vLLM generations.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from openai import OpenAI
import time
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
import os

###############################################################
if "KUBERNETES_SERVICE_HOST" in os.environ and os.getenv("KUBERNETES_SERVICE_HOST") != "":
    # Append internal cluster domain to no_proxy lists
    if os.getenv("NO_PROXY"):
        os.environ["NO_PROXY"] += ",.svc.cluster.local"
    if os.getenv("no_proxy"):
        os.environ["no_proxy"] += ",.svc.cluster.local"
###############################################################

# Define the structured output schema
class InstructionCompliance(BaseModel):
    """Evaluation of instruction following."""
    score: int = Field(ge=0, le=10, description="Score from 0-10")


class TaskCompletion(BaseModel):
    """Evaluation of task completion."""
    score: int = Field(ge=0, le=10, description="Score from 0-10")


class ResponseQuality(BaseModel):
    """Evaluation of response quality."""
    score: int = Field(ge=0, le=10, description="Score from 0-10")


class OverallAssessment(BaseModel):
    """Overall judgment."""
    aggregate_score: float = Field(ge=0, le=10, description="Average of three scores")


class JudgeEvaluation(BaseModel):
    """Complete structured evaluation from LLM judge."""
    instruction_compliance: InstructionCompliance
    task_completion: TaskCompletion
    response_quality: ResponseQuality
    overall: OverallAssessment


# System prompt for the judge
STRUCTURED_JUDGE_SYSTEM_PROMPT = """You are an expert AI evaluator. Assess responses rigorously using this rubric:

## CRITICAL RULES
1. Be strict!
2. Missing required tags = automatic major deduction
3. Emphasis on overall response scoring, and not just instruction following

## INSTRUCTION COMPLIANCE (0-10)
Deduct points for violations of instructions and contrainsts.
- make sure it has one pair of <think>...</think> and <answer>...</answer> tag, else score 0-2 points
- the answer is in <answer> tags
- thinking trace is reasonable and not vague
- focus on not only the satisfaction of constraint but proper instruction following too
- if you detect reward hacking, score in 0-2 points
- Missing <think> tags: -4 points
- Missing <answer> tags: -4 points
- Wrong format (JSON/structure): -3 points
- Under word count: -3 points
- Uses prohibited words: -3 points
- Missing required phrases: -2 each

Scoring:
- 0-2: Multiple critical violations
- 3-5: Some violations
- 6-7: Minor issues only
- 8-9: All requirements met
- 10: Perfect compliance

## TASK COMPLETION (0-10)
Did it do what was asked?
- focus on not only the satisfaction of constraint but proper instruction following too
- if you detect reward hacking, score in 0-2 points.
- 0-2: Wrong task or inadequate
- 3-5: Attempted but incomplete
- 6-7: Completed minimally
- 8-9: Well completed
- 10: Excellently completed

## RESPONSE QUALITY (0-10)
Accuracy, coherence, effort:
- focus on how human would rate the response, with detailed thinking and explanation
- think tag should be something detailed and not just something vauge like "reasoning process"
- if it has ant flaws rate it below 5 points.
- if you see anything abnormal, punish it severely, 0 to 2 points
- 0-2: Poor quality, errors
- 3-5: Low quality, minimal effort
- 6-7: Adequate
- 8-9: High quality
- 10: Exceptional
"""


def create_structured_judge_prompt(prompt: str, response: str) -> str:
    """Create the user prompt for evaluation."""
    return f"""Evaluate this response:

<prompt>
{prompt}
</prompt>

<response>
{response}
</response>

Provide a rigorous evaluation with:
- Scores for each dimension (0-10)
"""


class StructuredJudge:
    """LLM Judge with guaranteed structured outputs using Pydantic.

    Features robust timeout handling to prevent hanging on stuck generations.
    """

    def __init__(
        self,
        base_url: str = "http://0.0.0.0:8000/v1",
        api_key: str = "dummy",
        model: str = None,
        max_workers: int = 128,
        timeout: float = 2.0,
        max_retries: int = 0
    ):
        """
        Initialize the structured judge.

        Args:
            base_url: OpenAI-compatible API endpoint
            api_key: API key (can be dummy for local vLLM)
            model: Model name (auto-detected if None)
            max_workers: Maximum concurrent requests for batched inference
            timeout: Timeout in seconds for each request
            max_retries: Maximum retry attempts for failed/timed out evaluations
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model or self.client.models.list().data[0].id
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_retries = max_retries
        self._thread_local = threading.local()
        print(f"Initialized StructuredJudge with model: {self.model}")
        print(f"Timeout: {self.timeout}s, Max retries: {self.max_retries}, Max workers: {self.max_workers}")

    def _get_client(self) -> OpenAI:
        """Get thread-local OpenAI client for concurrent requests."""
        if not hasattr(self._thread_local, 'client'):
            self._thread_local.client = OpenAI(
                base_url=self.client.base_url,
                api_key=self.client.api_key
            )
        return self._thread_local.client

    def _execute_evaluation(
        self,
        user_message: str,
        temperature: float
    ) -> JudgeEvaluation:
        """
        Internal method to execute the evaluation.
        This runs the actual API call and can timeout.
        """
        client = self._get_client()

        completion = client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": STRUCTURED_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            response_format=JudgeEvaluation,
            temperature=temperature
        )

        message = completion.choices[0].message
        if not message.parsed:
            raise ValueError("Failed to parse structured output")

        return message.parsed

    def evaluate(
        self,
        prompt: str,
        response: str,
        temperature: float = 0.0,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None
    ) -> Optional[JudgeEvaluation]:
        """
        Evaluate a response and return structured output with timeout support.

        Args:
            prompt: The original prompt/task
            response: The model's response to evaluate
            temperature: Sampling temperature (0 for deterministic)
            timeout: Override default timeout in seconds
            max_retries: Override default max retries

        Returns:
            JudgeEvaluation object with guaranteed schema, or None if timed out/failed
        """
        timeout_val = timeout if timeout is not None else self.timeout
        retries = max_retries if max_retries is not None else self.max_retries

        user_message = create_structured_judge_prompt(prompt, response)

        for attempt in range(retries + 1):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self._execute_evaluation,
                    user_message,
                    temperature
                )

                try:
                    result = future.result(timeout=timeout_val)
                    return result

                except concurrent.futures.TimeoutError:
                    # Cancel the future to clean up resources
                    future.cancel()

                    if attempt < retries:
                        # Exponential backoff
                        backoff = 0.5 * (2 ** attempt)
                        print(f"Timeout on attempt {attempt + 1}/{retries + 1}. "
                              f"Retrying after {backoff:.1f}s...")
                        time.sleep(backoff)
                        continue
                    else:
                        print(f"Evaluation timed out after {timeout_val}s "
                              f"(tried {retries + 1} times)")
                        return None

                except Exception as e:
                    if attempt < retries:
                        backoff = 0.5 * (2 ** attempt)
                        print(f"Error on attempt {attempt + 1}/{retries + 1}: {e}. "
                              f"Retrying after {backoff:.1f}s...")
                        time.sleep(backoff)
                        continue
                    else:
                        print(f"Evaluation failed after {retries + 1} attempts: {e}")
                        return None

        return None

    def evaluate_with_timeout(
        self,
        prompt: str,
        response: str,
        temperature: float = 0.0,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None
    ) -> Optional[JudgeEvaluation]:
        """
        Alias for evaluate() with timeout support.
        Kept for backward compatibility.
        """
        return self.evaluate(prompt, response, temperature, timeout, max_retries)

    def _evaluate_single_with_timeout(
        self,
        prompt: str,
        response: str,
        temperature: float,
        index: int,
        timeout: float,
        max_retries: int,
        pbar: Optional[tqdm] = None
    ) -> tuple[int, Optional[JudgeEvaluation]]:
        """
        Evaluate a single response with timeout and retry.

        Args:
            prompt: The original prompt
            response: The model's response
            temperature: Sampling temperature
            index: Original index for result ordering
            timeout: Timeout in seconds
            max_retries: Maximum retry attempts
            pbar: Optional progress bar to update

        Returns:
            Tuple of (index, evaluation or None)
        """
        user_message = create_structured_judge_prompt(prompt, response)

        for attempt in range(max_retries + 1):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self._execute_evaluation,
                    user_message,
                    temperature
                )

                try:
                    result = future.result(timeout=timeout)
                    if pbar:
                        pbar.update(1)
                    return index, result

                except concurrent.futures.TimeoutError:
                    future.cancel()

                    if attempt < max_retries:
                        backoff = 0.5 * (2 ** attempt)
                        if pbar:
                            pbar.set_postfix_str(f"Retry {attempt + 1}/{max_retries} for idx {index}")
                        time.sleep(backoff)
                        continue
                    else:
                        if pbar:
                            pbar.write(f"Timeout after {max_retries + 1} attempts for index {index}")
                            pbar.update(1)
                        return index, None

                except Exception as e:
                    if attempt < max_retries:
                        backoff = 0.5 * (2 ** attempt)
                        if pbar:
                            pbar.set_postfix_str(f"Retry {attempt + 1}/{max_retries} for idx {index}: {str(e)[:30]}")
                        time.sleep(backoff)
                        continue
                    else:
                        if pbar:
                            pbar.write(f"Failed after {max_retries + 1} attempts for index {index}: {e}")
                            pbar.update(1)
                        return index, None

        return index, None

    def evaluate_batch(
        self,
        prompts: List[str],
        responses: List[str],
        temperature: float = 0.0,
        show_progress: bool = True,
        desc: str = "Evaluating"
    ) -> List[Optional[JudgeEvaluation]]:
        """
        Evaluate multiple responses using concurrent requests with timeout.

        Args:
            prompts: List of original prompts
            responses: List of model responses
            temperature: Sampling temperature
            show_progress: Whether to show progress bar
            desc: Description for progress bar

        Returns:
            List of JudgeEvaluation objects (or None for failed/timed out items) in original order
        """
        evaluations = [None] * len(prompts)

        # Create progress bar
        pbar = tqdm(
            total=len(prompts),
            desc=desc,
            disable=not show_progress,
            unit="eval"
        ) if show_progress else None

        # Use outer ThreadPoolExecutor for concurrent evaluations
        # Each evaluation will use its own inner executor for timeout handling
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    self._evaluate_single_with_timeout,
                    prompt,
                    response,
                    temperature,
                    i,
                    self.timeout,
                    self.max_retries,
                    pbar
                ): i
                for i, (prompt, response) in enumerate(zip(prompts, responses))
            }

            # Process completions as they finish
            for future in as_completed(futures):
                try:
                    index, evaluation = future.result()
                    evaluations[index] = evaluation
                except Exception as e:
                    original_index = futures[future]
                    if pbar:
                        pbar.close()
                    print(f"\nUnexpected error evaluating index {original_index}: {e}")
                    evaluations[original_index] = None

        if pbar:
            pbar.close()

        # Count failures
        failed_count = sum(1 for e in evaluations if e is None)
        if failed_count > 0:
            print(f"\nWarning: {failed_count}/{len(prompts)} evaluations failed or timed out")

        return evaluations

    async def evaluate_batch_async(
        self,
        prompts: List[str],
        responses: List[str],
        temperature: float = 0.0,
        show_progress: bool = True,
        desc: str = "Evaluating"
    ) -> List[Optional[JudgeEvaluation]]:
        """
        Evaluate multiple responses using async/await with timeout and retry.

        Args:
            prompts: List of original prompts
            responses: List of model responses
            temperature: Sampling temperature
            show_progress: Whether to show progress bar
            desc: Description for progress bar

        Returns:
            List of JudgeEvaluation objects (or None for failed items) in original order
        """
        from openai import AsyncOpenAI

        async_client = AsyncOpenAI(
            base_url=str(self.client.base_url),
            api_key=self.client.api_key
        )

        async def evaluate_single_with_retry(
            prompt: str,
            response: str,
            index: int,
            semaphore: asyncio.Semaphore,
            pbar: Optional[tqdm] = None
        ) -> tuple[int, Optional[JudgeEvaluation]]:
            """Evaluate with timeout and retry logic."""
            user_message = create_structured_judge_prompt(prompt, response)

            for attempt in range(self.max_retries + 1):
                async with semaphore:
                    try:
                        # Apply timeout to the API call
                        completion = await asyncio.wait_for(
                            async_client.beta.chat.completions.parse(
                                model=self.model,
                                messages=[
                                    {"role": "system", "content": STRUCTURED_JUDGE_SYSTEM_PROMPT},
                                    {"role": "user", "content": user_message}
                                ],
                                response_format=JudgeEvaluation,
                                temperature=temperature
                            ),
                            timeout=self.timeout
                        )

                        message = completion.choices[0].message
                        if not message.parsed:
                            raise ValueError(f"Failed to parse structured output for index {index}")

                        if pbar:
                            pbar.update(1)

                        return index, message.parsed

                    except asyncio.TimeoutError:
                        if attempt < self.max_retries:
                            if pbar:
                                pbar.set_postfix_str(f"Retry {attempt + 1}/{self.max_retries} for idx {index}")
                            await asyncio.sleep(0.5 * (2 ** attempt))  # Exponential backoff
                            continue
                        else:
                            if pbar:
                                pbar.write(f"Timeout after {self.max_retries + 1} attempts for index {index}")
                                pbar.update(1)
                            return index, None

                    except Exception as e:
                        if attempt < self.max_retries:
                            if pbar:
                                pbar.set_postfix_str(f"Retry {attempt + 1}/{self.max_retries} for idx {index}: {str(e)[:30]}")
                            await asyncio.sleep(0.5 * (2 ** attempt))
                            continue
                        else:
                            if pbar:
                                pbar.write(f"Failed after {self.max_retries + 1} attempts for index {index}: {e}")
                                pbar.update(1)
                            return index, None

            return index, None

        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(self.max_workers)

        # Create progress bar
        pbar = tqdm(
            total=len(prompts),
            desc=desc,
            disable=not show_progress,
            unit="eval"
        ) if show_progress else None

        # Create tasks for all evaluations
        tasks = [
            evaluate_single_with_retry(prompt, response, i, semaphore, pbar)
            for i, (prompt, response) in enumerate(zip(prompts, responses))
        ]

        # Run all tasks
        results = await asyncio.gather(*tasks, return_exceptions=False)

        if pbar:
            pbar.close()

        # Sort by index to maintain original order
        evaluations = [None] * len(prompts)
        failed_indices = []

        for index, evaluation in results:
            if evaluation is not None:
                evaluations[index] = evaluation
            else:
                failed_indices.append(index)

        if failed_indices:
            print(f"\nWarning: Failed to evaluate {len(failed_indices)} items at indices: {failed_indices}")

        return evaluations
