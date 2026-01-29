"""
Rubric-Based LLM Judge for VERL Reward Computation.

Based on research from Prometheus (ICLR 2024) and Rubrics as Rewards (RaR).
"""

import asyncio
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

# Kubernetes proxy setup
if os.environ.get("KUBERNETES_SERVICE_HOST"):
    for key in ("NO_PROXY", "no_proxy"):
        if os.getenv(key):
            os.environ[key] += ",.svc.cluster.local"

DEFAULT_LLM_JUDGE_URL = os.environ.get(
    "LLM_JUDGE_URL",
    "http://qpn744-vllm-gptoss120b-generate-svc.llm-pretraining.svc.cluster.local:8000/v1",
)


# =============================================================================
# Rubric Definitions
# =============================================================================


class RubricCriterion(BaseModel):
    """Single evaluation criterion with score descriptions."""

    name: str
    weight: float = 1.0
    score_descriptions: Dict[int, str]


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
        return self.evaluation.overall_score if self.evaluation else 0.0


# =============================================================================
# Prompt Builders
# =============================================================================


def build_rubric_system_prompt(rubric: Rubric, use_reference: bool = True) -> str:
    ref_instruction = (
        "\n- Compare the response to the reference answer when provided."
        "\n- The reference answer represents a score of 5; judge how close the response is."
        if use_reference
        else ""
    )

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
    instruction: str, response: str, reference: Optional[str] = None
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
# RubricJudge Implementation
# =============================================================================


class RubricJudge:
    """Rubric-based LLM Judge for VERL reward computation."""

    def __init__(
        self,
        base_url: str = DEFAULT_LLM_JUDGE_URL,
        api_key: str = "dummy",
        model: Optional[str] = None,
        rubric: "Rubric" = None,
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
        self.rubric = rubric or RESPONSE_QUALITY_RUBRIC

        self._sync_client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self._async_client: Optional[AsyncOpenAI] = None
        self.model = model or self._detect_model()
        self._system_prompt = build_rubric_system_prompt(self.rubric, use_reference)

        self._total_evaluations = 0
        self._total_failures = 0

        print(f"RubricJudge initialized: model={self.model}, rubric={self.rubric.name}")

    def _detect_model(self) -> str:
        try:
            models = self._sync_client.models.list()
            return models.data[0].id
        except Exception as e:
            print(f"[DEBUG] Model detection failed: {type(e).__name__}: {e}")
            return "unknown"

    def _get_async_client(self) -> AsyncOpenAI:
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                base_url=self.base_url, api_key=self.api_key, timeout=self.timeout
            )
        return self._async_client

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
                if attempt == self.max_retries:
                    self._total_failures += 1
                    return EvaluationResult(
                        index=index,
                        elapsed_ms=(time.perf_counter() - start) * 1000,
                        retries=attempt,
                        error=str(e),
                    )
                await asyncio.sleep(0.1 * (2**attempt))

        return EvaluationResult(index=index, error="Unknown")

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
                name=f"eval_{i}",
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
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self.evaluate_batch_async(
                        instructions, responses, references, temperature, show_progress
                    ),
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(
                self.evaluate_batch_async(
                    instructions, responses, references, temperature, show_progress
                )
            )

    def compute_rewards(
        self,
        prompts: List[str],
        responses: List[str],
        references: Optional[List[str]] = None,
        temperature: float = 0.0,
        show_progress: bool = True,
    ) -> List[EvaluationResult]:
        """Compute normalized rewards [0, 1] for VERL training."""
        return self.evaluate_batch(prompts, responses, references, temperature, show_progress)

    def get_statistics(self) -> dict:
        return {
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
# Default Rubric (Response Quality)
# =============================================================================

RESPONSE_QUALITY_RUBRIC = Rubric(
    name="Response Quality",
    criteria=[
        RubricCriterion(
            name="Usefulness",
            weight=0.50,
            score_descriptions={
                0: "No answer. Refuses, deflects, or says nothing.",
                1: "Nearly useless. Vaguely touches topic but nothing actionable.",
                2: "Barely useful. Some relevant info but missing most of what's needed.",
                3: "Somewhat useful. Addresses the question with gaps.",
                4: "Useful. Solid answer covering key points.",
                5: "Very useful. Comprehensive, accurate, fully satisfies the need.",
            },
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
            },
        ),
        RubricCriterion(
            name="Presentation",
            weight=0.25,
            score_descriptions={
                0: "Broken or useless. Malformed output OR polished formatting of zero content.",
                1: "Poor. Wrong format, vague language, OR neat presentation of empty content.",
                2: "Weak. Format/language issues, OR presentable but inadequate content.",
                3: "Acceptable. Reasonable format and language.",
                4: "Good. Format enhances comprehension, precise language, well-organized.",
                5: "Excellent. Format, precision, organization all serve a comprehensive answer.",
            },
        ),
    ],
)


# =============================================================================
# VERL Interface
# =============================================================================

# Global judge instance (lazy initialized)
_response_judge: Optional[RubricJudge] = None


def _get_judge() -> RubricJudge:
    global _response_judge
    if _response_judge is None:
        _response_judge = RubricJudge(rubric=RESPONSE_QUALITY_RUBRIC)
    return _response_judge


def _extract_prompts(
    extra_infos,  # Can be list, numpy array, or None
    data_sources,  # Can be list, numpy array, or None
    num_samples: int,
) -> List[str]:
    """Extract prompts from extra_infos (list/array of dicts) with fallback."""
    PROMPT_KEYS = ["prompt", "original_prompt", "question", "instruction", "input"]

    # Handle numpy arrays and lists - check length directly to avoid numpy truth ambiguity
    if extra_infos is not None and len(extra_infos) > 0:
        first_info = extra_infos[0] if isinstance(extra_infos[0], dict) else {}
        for key in PROMPT_KEYS:
            if key in first_info:
                prompts = []
                for info in extra_infos:
                    prompt = info.get(key, "") if isinstance(info, dict) else ""
                    # Handle chat format (list of messages)
                    if isinstance(prompt, list):
                        prompt = " ".join(
                            m.get("content", "") if isinstance(m, dict) else str(m) for m in prompt
                        )
                    prompts.append(str(prompt) if prompt else "")
                return prompts

    if data_sources is not None and len(data_sources) == num_samples:
        return [str(s) for s in data_sources]

    print("[WARNING] No prompts found in extra_infos. Using empty prompts.")
    return [""] * num_samples


def compute_score_batch(
    solution_strs: List[str],
    ground_truths: List[Any],
    extra_infos: Optional[List[dict]] = None,
    data_sources: Optional[List[str]] = None,
    **kwargs,
) -> List[dict]:
    """
    Compute rubric-based reward scores for a batch of responses.

    Compatible with VERL's BatchRewardManager interface.
    """
    num_samples = len(solution_strs)
    prompts = _extract_prompts(extra_infos, data_sources, num_samples)

    # Use ground_truths as references if they are all non-empty strings
    references = None
    if ground_truths is not None and len(ground_truths) > 0:
        if all(isinstance(gt, str) and gt for gt in ground_truths):
            references = list(ground_truths)

    try:
        judge = _get_judge()
        results = judge.compute_rewards(prompts, solution_strs, references)
        # Build return dicts with only serializable values (no Pydantic models)
        reward_dicts = []
        for r in results:
            d = {
                "score": r.reward,
                "index": r.index,
                "elapsed_ms": r.elapsed_ms,
                "retries": r.retries,
                "error": r.error,
                "success": r.success,
            }
            # Add evaluation details if present (convert Pydantic to dict)
            if r.evaluation is not None:
                d["overall_feedback"] = r.evaluation.overall_feedback
                d["overall_score"] = r.evaluation.overall_score
                d["criterion_scores"] = [
                    {"criterion": cs.criterion, "score": cs.score, "feedback": cs.feedback}
                    for cs in r.evaluation.criterion_scores
                ]
            reward_dicts.append(d)
        return reward_dicts
    except Exception as e:
        print(f"[ERROR] compute_score_batch failed: {e}")
        return [{"score": 0.0}] * num_samples
