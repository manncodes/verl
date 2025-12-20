# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 The verl Authors
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
Domain-specific reward functions for Cascaded RL training.

Following Nemotron-Cascade:
- RLHF: Reward model based scoring
- IF-RL: Rule-based instruction following checks + optional RLHF signal
- Math-RL: Rule-based answer verification
- Code-RL: Execution-based verification (unit tests)
- SWE-RL: Hybrid lexical-semantic reward

References:
- Nemotron-Cascade: https://arxiv.org/abs/2512.13607
- AceReason-Nemotron for Math training strategy
"""

import re
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from verl import DataProto
from verl.utils.reward_score import default_compute_score


@dataclass
class RewardResult:
    """Result from a reward computation."""
    reward_tensor: torch.Tensor
    reward_extra_info: Optional[Dict[str, List]] = None


# =============================================================================
# RLHF Reward (Stage 1)
# =============================================================================

class RLHFRewardManager:
    """
    RLHF reward manager using a reward model.

    In Nemotron-Cascade, they use a 72B reward model trained on ~82K preference pairs.
    This implementation supports both external RM scores and integrated scoring.
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int = 0,
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_scale = reward_scale
        self.reward_bias = reward_bias

    def __call__(self, data: DataProto, return_dict: bool = False) -> Union[torch.Tensor, RewardResult]:
        """
        Compute RLHF rewards.

        If rm_scores are already in the batch (from a separate RM model),
        use those. Otherwise, this would need to be connected to a reward model.
        """
        batch_size = len(data)
        response_length = data.batch["responses"].shape[1]
        reward_tensor = torch.zeros(batch_size, response_length, dtype=torch.float32)

        # Check if RM scores are pre-computed
        if "rm_scores" in data.batch:
            # Use pre-computed RM scores (last token reward)
            rm_scores = data.batch["rm_scores"]
            # Apply scaling and bias
            scaled_scores = rm_scores * self.reward_scale + self.reward_bias
            # Place reward at last token
            response_mask = data.batch.get("response_mask", torch.ones_like(reward_tensor))
            last_token_idx = response_mask.sum(dim=1).long() - 1
            for i in range(batch_size):
                if last_token_idx[i] >= 0:
                    reward_tensor[i, last_token_idx[i]] = scaled_scores[i]
        else:
            # Fallback: use a constant reward (needs actual RM integration)
            print("Warning: No rm_scores found. Using constant reward.")
            response_mask = data.batch.get("response_mask", torch.ones_like(reward_tensor))
            last_token_idx = response_mask.sum(dim=1).long() - 1
            for i in range(batch_size):
                if last_token_idx[i] >= 0:
                    reward_tensor[i, last_token_idx[i]] = 0.5

        if return_dict:
            return RewardResult(
                reward_tensor=reward_tensor,
                reward_extra_info={"rlhf_score": reward_tensor.sum(-1).tolist()}
            )
        return reward_tensor


# =============================================================================
# Instruction Following Reward (Stage 2)
# =============================================================================

class IFRewardManager:
    """
    Instruction Following reward manager.

    Combines rule-based IF verification with optional RLHF signal.
    Following Nemotron-Cascade: rewards = IF_check + scaled_RLHF_signal
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int = 0,
        if_weight: float = 1.0,
        rlhf_weight: float = 0.1,  # Scaled RLHF signal
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.if_weight = if_weight
        self.rlhf_weight = rlhf_weight

        # Common IF constraints to check
        self.constraint_patterns = {
            "word_count": r"(?:use|write|include)\s+(?:exactly\s+)?(\d+)\s+words?",
            "sentence_count": r"(?:use|write)\s+(?:exactly\s+)?(\d+)\s+sentences?",
            "paragraph_count": r"(?:use|write)\s+(?:exactly\s+)?(\d+)\s+paragraphs?",
            "bullet_points": r"(?:use|include)\s+(?:exactly\s+)?(\d+)\s+bullet\s*points?",
            "no_letter": r"(?:do\s+not|don't|avoid)\s+(?:use|include)\s+(?:the\s+)?letter\s+['\"]?(\w)['\"]?",
            "must_include": r"(?:must|should)\s+(?:include|contain)\s+(?:the\s+(?:word|phrase)\s+)?['\"]([^'\"]+)['\"]",
            "format_json": r"(?:respond|answer|output)\s+(?:in|as|with)\s+json",
            "format_markdown": r"(?:use|format\s+(?:in|as|with))\s+markdown",
        }

    def _check_word_count(self, response: str, target: int, tolerance: float = 0.1) -> float:
        """Check if response has approximately the target word count."""
        word_count = len(response.split())
        lower = int(target * (1 - tolerance))
        upper = int(target * (1 + tolerance))
        if lower <= word_count <= upper:
            return 1.0
        # Partial credit based on distance
        distance = min(abs(word_count - lower), abs(word_count - upper))
        return max(0, 1 - distance / target)

    def _check_sentence_count(self, response: str, target: int) -> float:
        """Check sentence count."""
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        count = len(sentences)
        if count == target:
            return 1.0
        return max(0, 1 - abs(count - target) / max(target, 1))

    def _check_no_letter(self, response: str, letter: str) -> float:
        """Check if a specific letter is absent."""
        return 0.0 if letter.lower() in response.lower() else 1.0

    def _check_must_include(self, response: str, phrase: str) -> float:
        """Check if response includes a required phrase."""
        return 1.0 if phrase.lower() in response.lower() else 0.0

    def _check_json_format(self, response: str) -> float:
        """Check if response is valid JSON."""
        import json
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{[^{}]*\}|\[[^\[\]]*\]', response, re.DOTALL)
            if json_match:
                json.loads(json_match.group())
                return 1.0
            return 0.0
        except json.JSONDecodeError:
            return 0.0

    def _evaluate_if_constraints(self, prompt: str, response: str) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate instruction following constraints.

        Returns:
            Tuple of (score, details_dict)
        """
        scores = []
        details = {}

        prompt_lower = prompt.lower()

        # Check word count constraint
        match = re.search(self.constraint_patterns["word_count"], prompt_lower)
        if match:
            target = int(match.group(1))
            score = self._check_word_count(response, target)
            scores.append(score)
            details["word_count"] = {"target": target, "score": score}

        # Check sentence count
        match = re.search(self.constraint_patterns["sentence_count"], prompt_lower)
        if match:
            target = int(match.group(1))
            score = self._check_sentence_count(response, target)
            scores.append(score)
            details["sentence_count"] = {"target": target, "score": score}

        # Check no letter constraint
        match = re.search(self.constraint_patterns["no_letter"], prompt_lower)
        if match:
            letter = match.group(1)
            score = self._check_no_letter(response, letter)
            scores.append(score)
            details["no_letter"] = {"letter": letter, "score": score}

        # Check must include constraint
        match = re.search(self.constraint_patterns["must_include"], prompt_lower)
        if match:
            phrase = match.group(1)
            score = self._check_must_include(response, phrase)
            scores.append(score)
            details["must_include"] = {"phrase": phrase, "score": score}

        # Check JSON format
        if re.search(self.constraint_patterns["format_json"], prompt_lower):
            score = self._check_json_format(response)
            scores.append(score)
            details["format_json"] = {"score": score}

        # Calculate average score
        if scores:
            final_score = sum(scores) / len(scores)
        else:
            # No constraints detected, give neutral score
            final_score = 0.5

        return final_score, details

    def __call__(self, data: DataProto, return_dict: bool = False) -> Union[torch.Tensor, RewardResult]:
        """Compute IF rewards."""
        batch_size = len(data)
        response_length = data.batch["responses"].shape[1]
        reward_tensor = torch.zeros(batch_size, response_length, dtype=torch.float32)

        scores_list = []
        details_list = []

        for i in range(batch_size):
            # Decode prompt and response
            prompt_ids = data.batch["prompts"][i]
            response_ids = data.batch["responses"][i]

            prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            # Evaluate IF constraints
            if_score, details = self._evaluate_if_constraints(prompt, response)

            # Optional: add RLHF component
            rlhf_score = 0.0
            if "rm_scores" in data.batch:
                rlhf_score = data.batch["rm_scores"][i].item()

            # Combine scores
            final_score = self.if_weight * if_score + self.rlhf_weight * rlhf_score

            # Place reward at last token
            response_mask = data.batch.get("response_mask", torch.ones(batch_size, response_length))
            last_idx = response_mask[i].sum().long() - 1
            if last_idx >= 0:
                reward_tensor[i, last_idx] = final_score

            scores_list.append(final_score)
            details_list.append(details)

        if return_dict:
            return RewardResult(
                reward_tensor=reward_tensor,
                reward_extra_info={
                    "if_score": scores_list,
                    "if_details": details_list,
                }
            )
        return reward_tensor


# =============================================================================
# Math Reward (Stage 3)
# =============================================================================

class MathRewardManager:
    """
    Math reward manager using rule-based answer verification.

    Following Nemotron-Cascade and AceReason-Nemotron:
    - Uses symbolic answer checkers
    - Supports dynamic token-budget curricula
    - Handles various math answer formats
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int = 0,
        partial_credit: bool = True,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.partial_credit = partial_credit

    def _extract_answer(self, response: str) -> Optional[str]:
        """Extract the final answer from a response."""
        # Look for boxed answer (common in math)
        boxed_match = re.search(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', response)
        if boxed_match:
            return boxed_match.group(1).strip()

        # Look for "The answer is X" pattern
        answer_match = re.search(
            r'(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+([^\n.]+)',
            response,
            re.IGNORECASE
        )
        if answer_match:
            return answer_match.group(1).strip()

        # Look for "= X" at the end
        equals_match = re.search(r'=\s*([^\n=]+)\s*$', response)
        if equals_match:
            return equals_match.group(1).strip()

        return None

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        # Remove whitespace
        answer = answer.strip()

        # Remove common formatting
        answer = re.sub(r'\\text\{([^}]*)\}', r'\1', answer)
        answer = re.sub(r'\$', '', answer)

        # Normalize fractions
        answer = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', answer)

        # Normalize numbers
        answer = answer.replace(',', '')

        return answer.lower().strip()

    def _compare_answers(self, pred: str, target: str) -> float:
        """Compare predicted and target answers."""
        pred_norm = self._normalize_answer(pred)
        target_norm = self._normalize_answer(target)

        if pred_norm == target_norm:
            return 1.0

        # Try numeric comparison
        try:
            pred_num = float(eval(pred_norm.replace('^', '**')))
            target_num = float(eval(target_norm.replace('^', '**')))
            if abs(pred_num - target_num) < 1e-6:
                return 1.0
            if self.partial_credit:
                # Partial credit for close answers
                rel_error = abs(pred_num - target_num) / max(abs(target_num), 1e-10)
                if rel_error < 0.01:
                    return 0.8
                elif rel_error < 0.1:
                    return 0.5
        except:
            pass

        return 0.0

    def __call__(self, data: DataProto, return_dict: bool = False) -> Union[torch.Tensor, RewardResult]:
        """Compute math rewards."""
        batch_size = len(data)
        response_length = data.batch["responses"].shape[1]
        reward_tensor = torch.zeros(batch_size, response_length, dtype=torch.float32)

        scores = []
        acc_list = []

        for i in range(batch_size):
            response_ids = data.batch["responses"][i]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            # Get ground truth
            ground_truth = None
            if "reward_model" in data.non_tensor_batch:
                rm_info = data[i].non_tensor_batch.get("reward_model", {})
                ground_truth = rm_info.get("ground_truth")

            if ground_truth is None:
                # No ground truth, neutral score
                score = 0.5
                acc = 0
            else:
                # Extract and compare answer
                pred_answer = self._extract_answer(response)
                if pred_answer is None:
                    score = 0.0
                    acc = 0
                else:
                    score = self._compare_answers(pred_answer, str(ground_truth))
                    acc = 1 if score >= 0.99 else 0

            # Place reward at last token
            response_mask = data.batch.get("response_mask", torch.ones(batch_size, response_length))
            last_idx = response_mask[i].sum().long() - 1
            if last_idx >= 0:
                reward_tensor[i, last_idx] = score

            scores.append(score)
            acc_list.append(acc)

        if return_dict:
            return RewardResult(
                reward_tensor=reward_tensor,
                reward_extra_info={
                    "math_score": scores,
                    "acc": acc_list,
                }
            )
        return reward_tensor


# =============================================================================
# Code Reward (Stage 4)
# =============================================================================

class CodeRewardManager:
    """
    Code reward manager using execution-based verification.

    Following Nemotron-Cascade:
    - Uses unit test execution for verification
    - Supports timeout and resource limits
    - Higher temperature for exploration
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int = 0,
        timeout: float = 10.0,
        use_sandbox: bool = True,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.timeout = timeout
        self.use_sandbox = use_sandbox

    def _extract_code(self, response: str) -> Optional[str]:
        """Extract code from response."""
        # Look for code blocks
        code_match = re.search(r'```(?:python)?\n?(.*?)```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # Look for indented code
        lines = response.split('\n')
        code_lines = []
        for line in lines:
            if line.startswith('    ') or line.startswith('\t'):
                code_lines.append(line)
            elif code_lines and line.strip() == '':
                code_lines.append(line)

        if code_lines:
            return '\n'.join(code_lines)

        return None

    def _execute_code_with_tests(
        self,
        code: str,
        test_cases: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Execute code with test cases.

        Note: In production, this should use a proper sandbox environment.
        """
        import subprocess
        import tempfile
        import os

        passed = 0
        total = len(test_cases)
        results = []

        for i, test in enumerate(test_cases):
            test_input = test.get("input", "")
            expected_output = test.get("output", "")

            # Create test script
            test_code = f"""
{code}

# Test case
input_data = {repr(test_input)}
result = solution(input_data) if callable(solution) else None
print(result)
"""

            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(test_code)
                    temp_file = f.name

                result = subprocess.run(
                    ['python', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )

                actual_output = result.stdout.strip()
                test_passed = actual_output == str(expected_output).strip()

                if test_passed:
                    passed += 1

                results.append({
                    "test_id": i,
                    "passed": test_passed,
                    "expected": expected_output,
                    "actual": actual_output,
                    "error": result.stderr if result.stderr else None,
                })

                os.unlink(temp_file)

            except subprocess.TimeoutExpired:
                results.append({
                    "test_id": i,
                    "passed": False,
                    "error": "Timeout",
                })
            except Exception as e:
                results.append({
                    "test_id": i,
                    "passed": False,
                    "error": str(e),
                })

        score = passed / total if total > 0 else 0.0
        return score, {"passed": passed, "total": total, "results": results}

    def __call__(self, data: DataProto, return_dict: bool = False) -> Union[torch.Tensor, RewardResult]:
        """Compute code rewards."""
        batch_size = len(data)
        response_length = data.batch["responses"].shape[1]
        reward_tensor = torch.zeros(batch_size, response_length, dtype=torch.float32)

        scores = []
        pass_rates = []

        for i in range(batch_size):
            response_ids = data.batch["responses"][i]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            # Get test cases from ground truth
            test_cases = []
            if "reward_model" in data.non_tensor_batch:
                rm_info = data[i].non_tensor_batch.get("reward_model", {})
                ground_truth = rm_info.get("ground_truth", {})
                if isinstance(ground_truth, dict):
                    test_cases = ground_truth.get("test_cases", [])
                elif isinstance(ground_truth, list):
                    test_cases = ground_truth

            if not test_cases:
                # No test cases, use syntax check only
                code = self._extract_code(response)
                if code:
                    try:
                        compile(code, '<string>', 'exec')
                        score = 0.5  # Valid syntax, neutral score
                    except SyntaxError:
                        score = 0.0
                else:
                    score = 0.0
                pass_rate = 0.0
            else:
                # Execute with test cases
                code = self._extract_code(response)
                if code:
                    score, details = self._execute_code_with_tests(code, test_cases)
                    pass_rate = details["passed"] / details["total"] if details["total"] > 0 else 0.0
                else:
                    score = 0.0
                    pass_rate = 0.0

            # Place reward at last token
            response_mask = data.batch.get("response_mask", torch.ones(batch_size, response_length))
            last_idx = response_mask[i].sum().long() - 1
            if last_idx >= 0:
                reward_tensor[i, last_idx] = score

            scores.append(score)
            pass_rates.append(pass_rate)

        if return_dict:
            return RewardResult(
                reward_tensor=reward_tensor,
                reward_extra_info={
                    "code_score": scores,
                    "pass_rate": pass_rates,
                }
            )
        return reward_tensor


# =============================================================================
# SWE Reward (Stage 5)
# =============================================================================

class SWERewardManager:
    """
    SWE (Software Engineering) reward manager.

    Following Nemotron-Cascade:
    - Uses hybrid lexical-semantic reward (execution-free)
    - Focuses on patch quality for GitHub repair tasks
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int = 0,
        semantic_weight: float = 0.5,
        lexical_weight: float = 0.5,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight

    def _extract_patch(self, response: str) -> Optional[str]:
        """Extract patch/diff from response."""
        # Look for diff blocks
        diff_match = re.search(r'```diff\n?(.*?)```', response, re.DOTALL)
        if diff_match:
            return diff_match.group(1).strip()

        # Look for unified diff patterns
        diff_lines = []
        in_diff = False
        for line in response.split('\n'):
            if line.startswith('---') or line.startswith('+++'):
                in_diff = True
                diff_lines.append(line)
            elif in_diff and (line.startswith('+') or line.startswith('-') or line.startswith(' ') or line.startswith('@@')):
                diff_lines.append(line)
            elif in_diff and line.strip() == '':
                diff_lines.append(line)
            elif in_diff:
                break

        if diff_lines:
            return '\n'.join(diff_lines)

        return None

    def _compute_lexical_similarity(self, pred: str, target: str) -> float:
        """Compute lexical similarity between predicted and target patches."""
        if not pred or not target:
            return 0.0

        # Simple token overlap
        pred_tokens = set(pred.lower().split())
        target_tokens = set(target.lower().split())

        if not pred_tokens or not target_tokens:
            return 0.0

        intersection = len(pred_tokens & target_tokens)
        union = len(pred_tokens | target_tokens)

        return intersection / union if union > 0 else 0.0

    def _compute_semantic_similarity(self, pred: str, target: str) -> float:
        """
        Compute semantic similarity.

        In practice, this would use an embedding model.
        For now, we use a simple approximation.
        """
        # Check for key operations matching
        def extract_operations(patch: str) -> set:
            ops = set()
            for line in patch.split('\n'):
                if line.startswith('+') and not line.startswith('+++'):
                    ops.add(('add', line[1:].strip()[:50]))
                elif line.startswith('-') and not line.startswith('---'):
                    ops.add(('remove', line[1:].strip()[:50]))
            return ops

        pred_ops = extract_operations(pred) if pred else set()
        target_ops = extract_operations(target) if target else set()

        if not pred_ops or not target_ops:
            return 0.0

        intersection = len(pred_ops & target_ops)
        union = len(pred_ops | target_ops)

        return intersection / union if union > 0 else 0.0

    def __call__(self, data: DataProto, return_dict: bool = False) -> Union[torch.Tensor, RewardResult]:
        """Compute SWE rewards."""
        batch_size = len(data)
        response_length = data.batch["responses"].shape[1]
        reward_tensor = torch.zeros(batch_size, response_length, dtype=torch.float32)

        scores = []

        for i in range(batch_size):
            response_ids = data.batch["responses"][i]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            # Get target patch from ground truth
            target_patch = None
            if "reward_model" in data.non_tensor_batch:
                rm_info = data[i].non_tensor_batch.get("reward_model", {})
                target_patch = rm_info.get("ground_truth")

            if target_patch is None:
                score = 0.5  # No ground truth, neutral
            else:
                pred_patch = self._extract_patch(response)
                if pred_patch is None:
                    score = 0.0
                else:
                    # Compute hybrid score
                    lexical = self._compute_lexical_similarity(pred_patch, target_patch)
                    semantic = self._compute_semantic_similarity(pred_patch, target_patch)
                    score = (
                        self.lexical_weight * lexical +
                        self.semantic_weight * semantic
                    )

            # Place reward at last token
            response_mask = data.batch.get("response_mask", torch.ones(batch_size, response_length))
            last_idx = response_mask[i].sum().long() - 1
            if last_idx >= 0:
                reward_tensor[i, last_idx] = score

            scores.append(score)

        if return_dict:
            return RewardResult(
                reward_tensor=reward_tensor,
                reward_extra_info={"swe_score": scores}
            )
        return reward_tensor


# =============================================================================
# Unified Multi-Domain Reward Manager
# =============================================================================

class CascadeRewardManager:
    """
    Unified reward manager that routes to domain-specific reward functions.

    Uses the data_source field to determine which reward function to apply.
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int = 0,
        domain_managers: Optional[Dict[str, Any]] = None,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine

        # Initialize domain-specific managers
        self.domain_managers = domain_managers or {
            "rlhf": RLHFRewardManager(tokenizer, num_examine),
            "instruction_following": IFRewardManager(tokenizer, num_examine),
            "math": MathRewardManager(tokenizer, num_examine),
            "code": CodeRewardManager(tokenizer, num_examine),
            "swe": SWERewardManager(tokenizer, num_examine),
        }

        # Default manager for unknown domains
        self.default_compute_score = default_compute_score

    def __call__(self, data: DataProto, return_dict: bool = False) -> Union[torch.Tensor, RewardResult]:
        """
        Compute rewards by routing to appropriate domain manager.
        """
        batch_size = len(data)
        response_length = data.batch["responses"].shape[1]
        reward_tensor = torch.zeros(batch_size, response_length, dtype=torch.float32)

        extra_info: Dict[str, List] = {}

        for i in range(batch_size):
            # Get domain from data_source
            data_source = data[i].non_tensor_batch.get("data_source", "unknown")

            # Determine domain
            domain = self._determine_domain(data_source)

            if domain in self.domain_managers:
                # Use domain-specific manager
                manager = self.domain_managers[domain]
                single_data = data[i:i+1]  # Create single-item batch
                result = manager(single_data, return_dict=True)
                reward_tensor[i] = result.reward_tensor[0]

                # Collect extra info
                if result.reward_extra_info:
                    for key, val in result.reward_extra_info.items():
                        if key not in extra_info:
                            extra_info[key] = []
                        extra_info[key].extend(val if isinstance(val, list) else [val])
            else:
                # Use default scoring
                response = self.tokenizer.decode(
                    data.batch["responses"][i],
                    skip_special_tokens=True
                )
                ground_truth = data[i].non_tensor_batch.get("reward_model", {}).get("ground_truth")

                score = self.default_compute_score(
                    data_source=data_source,
                    solution_str=response,
                    ground_truth=ground_truth,
                )

                response_mask = data.batch.get("response_mask", torch.ones(batch_size, response_length))
                last_idx = response_mask[i].sum().long() - 1
                if last_idx >= 0:
                    reward_tensor[i, last_idx] = score

        if return_dict:
            return RewardResult(
                reward_tensor=reward_tensor,
                reward_extra_info=extra_info if extra_info else None
            )
        return reward_tensor

    def _determine_domain(self, data_source: str) -> str:
        """Determine the domain from data_source string."""
        data_source_lower = data_source.lower()

        # Math domains
        if any(x in data_source_lower for x in ["math", "gsm", "aime", "olympiad", "geometry"]):
            return "math"

        # Code domains
        if any(x in data_source_lower for x in ["code", "livecodebench", "humaneval", "mbpp", "apps"]):
            return "code"

        # SWE domains
        if any(x in data_source_lower for x in ["swe", "github", "patch", "bug"]):
            return "swe"

        # IF domains
        if any(x in data_source_lower for x in ["ifeval", "instruction", "following"]):
            return "instruction_following"

        # Default to RLHF
        return "rlhf"


def get_reward_manager_for_domain(domain: str, tokenizer, **kwargs):
    """Factory function to get the appropriate reward manager for a domain."""
    managers = {
        "rlhf": RLHFRewardManager,
        "instruction_following": IFRewardManager,
        "math": MathRewardManager,
        "code": CodeRewardManager,
        "swe": SWERewardManager,
    }

    if domain in managers:
        return managers[domain](tokenizer, **kwargs)

    # Return unified manager for unknown domains
    return CascadeRewardManager(tokenizer, **kwargs)
