"""
rubric.py
Rubric-Based LLM Judge for VERL Reward Computation

Based on research from:
- Prometheus (ICLR 2024): Fine-grained evaluation with custom score rubrics
- Rubrics as Rewards (RaR): Checklist-style criteria for RL training
- G-Eval: Chain-of-thought decomposition for evaluation
"""

from operator import index
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import asyncio
import time

from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm


#### E N V   S E T U P#########################################
import os

if "KUBERNETES_SERVICE_HOST" in os.environ and os.getenv("KUBERNETES_SERVICE_HOST") != "":
    if os.getenv("NO_PROXY"):
        os.environ["NO_PROXY"] += ",.svc.cluster.local"
    if os.getenv("no_proxy"):
        os.environ["no_proxy"] += ",.svc.cluster.local"


DEFAULT_LLM_JUDGE_URL = os.environ.get(
    "LLM_JUDGE_URL",
    "http://qpn744-vllm-gptoss120b-generate-svc.llm-pretraining.svc.cluster.local:8000/v1",
)
###############################################################

# =============================================================================
# Rubric Definitions
# =============================================================================

class RubricCriterion(BaseModel):
    """Single evaluation criterion with score descriptions."""
    name: str
    weight: float = 1.0
    score_descriptions: Dict[int, str]  # {1: "Poor...", 2: "Fair...", ..., 5: "Excellent..."}


class Rubric(BaseModel):
    """Complete evaluation rubric with multiple criteria."""
    name: str
    criteria: List[RubricCriterion]

    def to_prompt(self) -> str:
        lines = [f"[Evaluation Rubric: {self.name}]"]
        for c in self.criteria:
            lines.append(f"\n### {c.name} (weight: {c.weight})")
            for score, desc in sorted(c.score_descriptions.items()):
                lines.append(f"Score {score}: {desc}")
        return "\n".join(lines)


# =============================================================================
# Pre-defined Rubrics
# =============================================================================

HELPFULNESS_RUBRIC = Rubric(
    name="Helpfulness",
    criteria=[
        RubricCriterion(
            name="Helpfulness",
            weight=1.0,
            score_descriptions={
                1: "Response is irrelevant, incorrect, or refuses to help without justification.",
                2: "Response attempts to help but has major errors or omissions.",
                3: "Response is partially helpful with some correct information but incomplete.",
                4: "Response is helpful and mostly correct with minor issues.",
                5: "Response is highly helpful, accurate, complete, and well-structured.",
            }
        )
    ]
)

REASONING_RUBRIC = Rubric(
    name="Reasoning Quality",
    criteria=[
        RubricCriterion(
            name="Logical Correctness",
            weight=0.5,
            score_descriptions={
                1: "Reasoning contains fundamental logical errors or contradictions.",
                2: "Reasoning has significant gaps or flawed logic.",
                3: "Reasoning is mostly sound but with some unclear steps.",
                4: "Reasoning is clear and logically valid with minor issues.",
                5: "Reasoning is rigorous, clear, and fully justified.",
            }
        ),
        RubricCriterion(
            name="Solution Correctness",
            weight=0.5,
            score_descriptions={
                1: "Final answer is completely wrong.",
                2: "Final answer is wrong but approach shows some understanding.",
                3: "Final answer is partially correct or correct with wrong reasoning.",
                4: "Final answer is correct with minor presentation issues.",
                5: "Final answer is correct, well-presented, and verified.",
            }
        )
    ]
)

CODE_RUBRIC = Rubric(
    name="Code Quality",
    criteria=[
        RubricCriterion(
            name="Correctness",
            weight=0.6,
            score_descriptions={
                1: "Code does not run or produces completely wrong output.",
                2: "Code runs but fails most test cases.",
                3: "Code passes some test cases but has bugs.",
                4: "Code passes most test cases with minor edge case issues.",
                5: "Code passes all test cases and handles edge cases.",
            }
        ),
        RubricCriterion(
            name="Quality",
            weight=0.4,
            score_descriptions={
                1: "Code is unreadable, no structure, poor naming.",
                2: "Code is poorly organized with minimal documentation.",
                3: "Code is functional but could be cleaner.",
                4: "Code is well-organized, readable, with good practices.",
                5: "Code is exemplary: clean, efficient, well-documented.",
            }
        )
    ]
)

RESPONSE_STYLE_RUBRIC = Rubric(
    name="Response Style",
    criteria=[
        RubricCriterion(
            name="Clarity",
            weight=0.4,
            score_descriptions={
                1: "Response is confusing, disorganized, or incoherent.",
                2: "Response is understandable but poorly structured.",
                3: "Response is clear but could be more concise.",
                4: "Response is clear, well-structured, and appropriately detailed.",
                5: "Response is exceptionally clear, concise, and well-organized.",
            }
        ),
        RubricCriterion(
            name="Formatting",
            weight=0.3,
            score_descriptions={
                1: "Formatting is inappropriate or makes response harder to read.",
                2: "Formatting is inconsistent or excessive.",
                3: "Formatting is adequate but not optimal.",
                4: "Formatting enhances readability appropriately.",
                5: "Formatting is perfect for the content type.",
            }
        ),
        RubricCriterion(
            name="Tone",
            weight=0.3,
            score_descriptions={
                1: "Tone is inappropriate, rude, or unprofessional.",
                2: "Tone is awkward or mismatched to context.",
                3: "Tone is acceptable but generic.",
                4: "Tone is appropriate and engaging.",
                5: "Tone is perfectly calibrated to user and context.",
            }
        )
    ]
)

PREDEFINED_RUBRICS = {
    "helpfulness": HELPFULNESS_RUBRIC,
    "reasoning": REASONING_RUBRIC,
    "code": CODE_RUBRIC,
    "style": RESPONSE_STYLE_RUBRIC,
}


# =============================================================================
# Pydantic Output Schemas
# =============================================================================

class CriterionScore(BaseModel):
    """Score for a single criterion."""
    criterion: str
    score: int = Field(ge=0, le=5)
    feedback: str


class RubricEvaluation(BaseModel):
    """Complete rubric-based evaluation output."""
    criterion_scores: List[CriterionScore]
    overall_feedback: str
    overall_score: float = Field(ge=0.0, le=1.0)


# =============================================================================
# System Prompts
# =============================================================================

def build_rubric_system_prompt(rubric: Rubric, use_reference: bool = True) -> str:
    ref_instruction = """
- Compare the response to the reference answer when provided.
- The reference answer represents a score of 5; judge how close the response is.""" if use_reference else ""

    return f"""You are a strict evaluator assessing AI response quality.

Critical Rules:
- Score 0 exists. Use it for bad incomplete answers.
- An incomplete answer CANNOT score above 3 on ANY criterion.
- "Concise" is NOT a virtue if information is missing - that's just incomplete.
- Good formatting of empty content scores 0-1 on Presentation, not higher.
- Judge what the response SHOULD have contained.
- Be harsh. Most mediocre responses should score 2-3, not 4-5.{ref_instruction}

{rubric.to_prompt()}

Output JSON:
- criterion_scores: array of {{criterion, score, feedback}}
- overall_feedback: 1-2 sentence summary
- overall_score: weighted average normalized to 0-1"""

def build_rubric_user_prompt(
    instruction: str,
    response: str,
    reference: Optional[str] = None
) -> str:
    parts = [
        f"[Instruction]\n{instruction}",
        f"\n[Response to Evaluate]\n{response}",
    ]
    if reference:
        parts.append(f"\n[Reference Answer (Score 5)]\n{reference}")
    parts.append("\n[Your Evaluation]")
    return "\n".join(parts)


# =============================================================================
# Evaluation Result
# =============================================================================

@dataclass
class EvaluationResult:
    """Container for evaluation result with metadata."""
    index: int
    evaluation: Optional[RubricEvaluation] = None
    elapsed_ms: float = 0.0
    retries: int = 0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.evaluation is not None

    @property
    def reward(self) -> float:
        if self.evaluation is None:
            return 0.0
        return self.evaluation.overall_score


# =============================================================================
# RubricJudge Implementation
# =============================================================================

class RubricJudge:
    """
    Rubric-based LLM Judge for VERL reward computation.

    Supports:
    - Multi-criteria evaluation with configurable rubrics
    - Reference-based and reference-free evaluation
    - Weighted score aggregation
    - Async batch processing with global timeout
    """

    def __init__(
        self,
        base_url: str = os.environ.get("LLM_JUDGE_URL", DEFAULT_LLM_JUDGE_URL),
        api_key: str = "dummy",
        model: Optional[str] = None,
        rubric: str | Rubric = "helpfulness",
        use_reference: bool = True,
        max_concurrent: int = 128,
        timeout: float = 30.0,
        max_retries: int = 1,
        batch_timeout: float = 120.0,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_concurrent = max_concurrent
        self.batch_timeout = batch_timeout
        self.use_reference = use_reference

        # Resolve rubric
        if isinstance(rubric, str):
            self.rubric = PREDEFINED_RUBRICS.get(rubric, HELPFULNESS_RUBRIC)
        else:
            self.rubric = rubric

        self._sync_client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self._async_client: Optional[AsyncOpenAI] = None
        self.model = model or self._detect_model()

        self._system_prompt = build_rubric_system_prompt(self.rubric, use_reference)

        # Stats
        self._total_evaluations = 0
        self._total_failures = 0

        print(f"RubricJudge initialized: model={self.model}, rubric={self.rubric.name}")

    def _detect_model(self) -> str:
        try:
            models = self._sync_client.models.list()
            model_id = models.data[0].id
            print(f"[DEBUG] Detected model: {model_id}")
            return model_id
        except Exception as e:
            print(f"[DEBUG] Model detection failed: {type(e).__name__}: {e}")
            return "unknown"

    def _get_async_client(self) -> AsyncOpenAI:
        if self._async_client is None:
            print(f"using base url: {self.base_url}")
            self._async_client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._async_client

    # =========================================================================
    # Core Evaluation
    # =========================================================================

    async def _evaluate_single_async(
        self,
        instruction: str,
        response: str,
        index: int,
        semaphore: asyncio.Semaphore,
        reference: Optional[str] = None,
        temperature: float = 0.0,
    ) -> EvaluationResult:
        client = self._get_async_client()
        user_msg = build_rubric_user_prompt(instruction, response, reference)
        start = time.perf_counter()

        for attempt in range(self.max_retries + 1):
            try:
                async with semaphore:
                    completion = await asyncio.wait_for(
                        client.beta.chat.completions.parse(
                            model=self.model,
                            messages=[
                                {"role": "system", "content": self._system_prompt},
                                {"role": "user", "content": user_msg},
                            ],
                            response_format=RubricEvaluation,
                            temperature=temperature,
                        ),
                        timeout=self.timeout,
                    )

                    parsed = completion.choices[0].message.parsed
                    if parsed is None:
                        raise ValueError("Failed to parse structured output")

                    return EvaluationResult(
                        index=index,
                        evaluation=parsed,
                        elapsed_ms=(time.perf_counter() - start) * 1000,
                        retries=attempt,
                    )


            except asyncio.CancelledError:
                return EvaluationResult(
                    index=index,
                    elapsed_ms=(time.perf_counter() - start) * 1000,
                    retries=attempt,
                    error="Cancelled",
                )

            except Exception as e:
                print(f"[DEBUG] Eval {index} attempt {attempt} failed: {type(e).__name__}: {e}")
                if attempt == self.max_retries:
                    self._total_failures += 1
                    return EvaluationResult(
                        index=index,
                        elapsed_ms=(time.perf_counter() - start) * 1000,
                        retries=attempt,
                        error=str(e),
                    )
                await asyncio.sleep(0.1 * (2 ** attempt))

        return EvaluationResult(index=index, error="Unknown")

    # =========================================================================
    # Batch Evaluation
    # =========================================================================

    async def evaluate_batch_async(
        self,
        instructions: List[str],
        responses: List[str],
        references: Optional[List[str]] = None,
        temperature: float = 0.0,
        show_progress: bool = True,
    ) -> List[EvaluationResult]:
        n = len(instructions)
        if len(responses) != n:
            raise ValueError(f"Mismatch: {n} instructions vs {len(responses)} responses")
        if references and len(references) != n:
            raise ValueError(f"Mismatch: {n} instructions vs {len(references)} references")

        semaphore = asyncio.Semaphore(self.max_concurrent)
        results: Dict[int, EvaluationResult] = {}

        tasks = [
            asyncio.create_task(
                self._evaluate_single_async(
                    instructions[i],
                    responses[i],
                    i,
                    semaphore,
                    references[i] if references else None,
                    temperature,
                ),
                name=f"eval_{i}"
            )
            for i in range(n)
        ]

        pbar = tqdm(total=n, desc="Evaluating", disable=not show_progress)
        start = time.perf_counter()
        pending = set(tasks)

        try:
            while pending:
                remaining = max(0.1, self.batch_timeout - (time.perf_counter() - start))
                if remaining <= 0.1:
                    break

                done, pending = await asyncio.wait(
                    pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    try:
                        r = task.result()
                        results[r.index] = r
                        pbar.update(1)
                    except Exception as e:
                        idx = int(task.get_name().split("_")[1])
                        results[idx] = EvaluationResult(index=idx, error=str(e))
                        pbar.update(1)
        finally:
            pbar.close()

        # Cancel pending and fill missing
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        for i in range(n):
            if i not in results:
                results[i] = EvaluationResult(index=i, error="Timeout")

        self._total_evaluations += n
        return [results[i] for i in range(n)]

    def evaluate_batch(
        self,
        instructions: List[str],
        responses: List[str],
        references: Optional[List[str]] = None,
        temperature: float = 0.0,
        show_progress: bool = True,
    ) -> List[EvaluationResult]:
        try:
            asyncio.get_running_loop()
            # Already in async context - need thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self.evaluate_batch_async(
                        instructions, responses, references, temperature, show_progress
                    )
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(
                self.evaluate_batch_async(
                    instructions, responses, references, temperature, show_progress
                )
            )

    # =========================================================================
    # VERL Integration
    # =========================================================================

    def compute_rewards(
        self,
        prompts: List[str],
        responses: List[str],
        references: Optional[List[str]] = None,
        temperature: float = 0.0,
        show_progress: bool = True,
    ) -> List[float|dict]:
        """Compute normalized rewards [0, 1] for VERL training."""
        results = self.evaluate_batch(
            prompts, responses, references, temperature, show_progress
        )
        return results

    def get_statistics(self) -> dict:
        return {
            ""
            "total_evaluations": self._total_evaluations,
            "total_failures": self._total_failures,
            "failure_rate": self._total_failures / max(1, self._total_evaluations),
        }

    def close(self):
        if self._sync_client:
            self._sync_client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# =============================================================================
# Factory Functions
# =============================================================================

def create_rubric_judge(
    base_url: str = DEFAULT_LLM_JUDGE_URL,
    rubric: str | Rubric = "helpfulness",
    use_reference: bool = True,
    max_concurrent: int = 128,
    batch_timeout: float = 120.0,
) -> RubricJudge:
    """Create a rubric-based judge for VERL."""
    return RubricJudge(
        base_url=base_url,
        rubric=rubric,
        use_reference=use_reference,
        max_concurrent=max_concurrent,
        batch_timeout=batch_timeout,
    )


def create_custom_rubric(
    name: str,
    criteria: List[Dict[str, Any]],
) -> Rubric:
    """
    Create a custom rubric from a list of criterion dicts.

    Example:
        rubric = create_custom_rubric("MyRubric", [
            {
                "name": "Accuracy",
                "weight": 0.6,
                "scores": {
                    1: "Completely wrong",
                    2: "Mostly wrong",
                    3: "Partially correct",
                    4: "Mostly correct",
                    5: "Fully correct",
                }
            },
            ...
        ])
    """
    return Rubric(
        name=name,
        criteria=[
            RubricCriterion(
                name=c["name"],
                weight=c.get("weight", 1.0),
                score_descriptions=c["scores"],
            )
            for c in criteria
        ]
    )


# =============================================================================
# Template Rubrics
# =============================================================================

RESPONSE_QUALITY_RUBRIC = Rubric(
    name="Response Quality",
    criteria=[
        RubricCriterion(
            name="Usefulness",
            weight=0.50,
            score_descriptions={
                0: "No answer. Refuses, deflects, or says nothing ('there are many ways' without any).",
                1: "Nearly useless. Vaguely touches topic but provides nothing actionable.",
                2: "Barely useful. Some relevant info but missing most of what's needed.",
                3: "Somewhat useful. Addresses the question with gaps.",
                4: "Useful. Solid answer covering key points.",
                5: "Very useful. Comprehensive, accurate, fully satisfies the need.",
            }
        ),
        RubricCriterion(
            name="Efficiency",
            weight=0.25,
            score_descriptions={
                0: "Complete waste. No content, or so incomplete/verbose it's worthless.",
                1: "Very inefficient. Filler-heavy OR forces immediate follow-up.",
                2: "Inefficient. Too verbose OR too sparse.",
                3: "Adequate. Minor length issues.",
                4: "Efficient. Right-sized, no waste.",
                5: "Optimal. Maximum value per word.",
            }
        ),
        RubricCriterion(
            name="Presentation",
            weight=0.25,
            score_descriptions={
                0: "Broken or useless. Malformed output OR polished formatting of zero content.",
                1: "Poor. Wrong format, vague language, OR neat presentation of empty content.",
                2: "Weak. Format/language issues, OR presentable but inadequate content.",
                3: "Acceptable. Reasonable format and language.",
                4: "Good. Format enhances comprehension, precise language, content well-organized.",
                5: "Excellent. Format, precision, organization all serve a comprehensive answer.",
            }
        ),
    ]
)

response_judge = create_rubric_judge(rubric=RESPONSE_QUALITY_RUBRIC)


def _extract_prompts_from_extra_infos(
    extra_infos: Optional[list[dict]],
    data_sources: Optional[list[str]],
    num_samples: int,
) -> list[str]:
    """
    Extract prompts from extra_infos (list of dicts) with robust fallback.

    VERL passes extra_infos as a list of dictionaries, one per sample.
    This function extracts prompts from each dict, checking multiple possible keys.

    Args:
        extra_infos: List of dicts, one per sample. Each dict may contain prompt info.
        data_sources: Fallback list of strings if prompts not in extra_infos.
        num_samples: Number of samples (for generating placeholder prompts).

    Returns:
        List of prompt strings.
    """
    # Priority keys to check in each extra_info dict
    PROMPT_KEYS = ["original_prompt", "prompt", "question", "instruction", "input"]

    if extra_infos and len(extra_infos) > 0:
        # Find which key contains the prompt by checking the first dict
        prompt_key = None
        first_info = extra_infos[0] if isinstance(extra_infos[0], dict) else {}

        for key in PROMPT_KEYS:
            if key in first_info:
                prompt_key = key
                break

        if prompt_key:
            # Extract prompt from each dict in the list
            prompts = []
            for info in extra_infos:
                if isinstance(info, dict):
                    prompt = info.get(prompt_key, "")
                    # Handle case where prompt is a list of messages (chat format)
                    if isinstance(prompt, list):
                        # Extract content from message list
                        prompt = " ".join(
                            msg.get("content", "") if isinstance(msg, dict) else str(msg)
                            for msg in prompt
                        )
                    prompts.append(str(prompt) if prompt else "")
                else:
                    prompts.append("")
            return prompts

    # Fallback to data_sources if available
    if data_sources and len(data_sources) == num_samples:
        return list(data_sources)

    # Last resort: empty prompts (will likely cause poor evaluations)
    print("[WARNING] No prompts found in extra_infos or data_sources. Using empty prompts.")
    return [""] * num_samples


def compute_score_batch(
    solution_strs: list[str],
    ground_truths: list[Any],
    extra_infos: Optional[list[dict]] = None,
    data_sources: Optional[list[str]] = None,
    **kwargs,
) -> list[dict]:
    """
    Compute rubric-based reward scores for a batch of responses.

    This function is compatible with VERL's BatchRewardManager interface.

    Args:
        solution_strs: List of model response strings to evaluate.
        ground_truths: List of ground truth values (used as references if strings).
        extra_infos: List of dicts (one per sample) containing additional info.
                     Expected to contain prompt/question in keys like:
                     "original_prompt", "prompt", "question", "instruction", "input"
        data_sources: List of data source identifiers (fallback for prompts).
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        List of dicts with "score" key and evaluation details.
    """
    num_samples = len(solution_strs)

    # Extract prompts from the list of extra_info dicts
    prompts = _extract_prompts_from_extra_infos(extra_infos, data_sources, num_samples)

    # Use ground_truths as references if they are strings
    references = None
    if ground_truths and all(isinstance(gt, str) and gt for gt in ground_truths):
        references = ground_truths

    try:
        rewards = response_judge.compute_rewards(prompts, solution_strs, references)
        reward_dicts = [{"score": r.reward, **asdict(r)} for r in rewards]
        return reward_dicts
    except Exception as e:
        print(f"[ERROR] in compute_score_batch of response_judge: {e}")
        return [{"score": 0.0}] * num_samples
