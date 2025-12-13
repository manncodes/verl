# Copyright 2025 verl contributors
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
Prime Intellect Reward Manager for verl.

This module provides reward managers that integrate Prime Intellect's
environments and sandboxes with verl's RL training pipeline.

Features:
    - Seamless integration with Prime Intellect Environments Hub
    - Remote code execution via Prime Intellect Sandboxes
    - Support for verifiers environments and rubrics
    - Async batch processing for high throughput
    - Rate limiting for API-based reward functions

Usage:
    >>> from verl.workers.reward_manager import get_reward_manager_cls
    >>> manager_cls = get_reward_manager_cls("prime_intellect")
    >>> manager = manager_cls(tokenizer, num_examine=5)
    >>> rewards = manager(data)
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

logger = logging.getLogger(__name__)


def _default_prime_intellect_compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> float | dict:
    """Default compute score function using Prime Intellect services.

    This function routes to Prime Intellect's compute_score which supports
    both sandbox execution and verifiers environments.
    """
    from verl.utils.prime_intellect import compute_score

    return compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )


@register("prime_intellect")
class PrimeIntellectRewardManager(AbstractRewardManager):
    """Reward manager using Prime Intellect environments and sandboxes.

    This manager integrates Prime Intellect's services for reward computation:
    - Sandbox execution for code evaluation
    - Verifiers environments for custom RL tasks
    - Support for both sync and async evaluation

    Args:
        tokenizer: Tokenizer for decoding model outputs.
        num_examine: Number of samples to print for debugging.
        compute_score: Custom scoring function (defaults to Prime Intellect's).
        reward_fn_key: Key for accessing data source in batch.
        prime_intellect_api_key: API key for Prime Intellect services.
        timeout: Timeout for remote execution in seconds.
        docker_image: Default Docker image for sandbox execution.
        max_workers: Maximum number of parallel workers for scoring.

    Example:
        >>> manager = PrimeIntellectRewardManager(
        ...     tokenizer=tokenizer,
        ...     num_examine=5,
        ...     prime_intellect_api_key="your-api-key",
        ... )
        >>> rewards = manager(data)
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Callable | None = None,
        reward_fn_key: str = "data_source",
        prime_intellect_api_key: str | None = None,
        timeout: int = 30,
        docker_image: str = "python:3.11-slim",
        max_workers: int = 16,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_prime_intellect_compute_score
        self.reward_fn_key = reward_fn_key
        self.prime_intellect_api_key = prime_intellect_api_key
        self.timeout = timeout
        self.docker_image = docker_image
        self.max_workers = max_workers

        # Store extra kwargs for compute_score
        self.extra_compute_kwargs = {
            "prime_intellect_api_key": prime_intellect_api_key,
            "timeout": timeout,
            "docker_image": docker_image,
        }
        self.extra_compute_kwargs.update(kwargs)

    def _compute_single_score(
        self,
        data_source: str,
        solution_str: str,
        ground_truth: str,
        extra_info: dict | None = None,
    ) -> dict | float:
        """Compute score for a single sample."""
        try:
            return self.compute_score(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **self.extra_compute_kwargs,
            )
        except Exception as e:
            logger.error(f"Score computation failed: {e}")
            return {"score": 0.0, "error": str(e)}

    def __call__(
        self,
        data: DataProto,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        """Compute rewards for a batch of data.

        Args:
            data: Input data containing prompts and responses.
            return_dict: If True, return dict with tensor and extra info.

        Returns:
            Reward tensor or dict with tensor and metadata.
        """
        # If pre-computed rm_scores exist, return them directly
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {
                    key: data.non_tensor_batch[key] for key in reward_extra_keys
                }
                return {
                    "reward_tensor": data.batch["rm_scores"],
                    "reward_extra_info": reward_extra_info,
                }
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        # Prepare all items for parallel processing
        items_to_process = []
        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # Decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})

            # Add prompt to extra_info for environment evaluation
            extra_info["prompt"] = prompt_str

            items_to_process.append({
                "index": i,
                "data_source": data_source,
                "response_str": response_str,
                "ground_truth": ground_truth,
                "extra_info": extra_info,
                "prompt_str": prompt_str,
                "valid_response_length": valid_response_length.item(),
            })

        # Process items in parallel using thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for item in items_to_process:
                future = executor.submit(
                    self._compute_single_score,
                    item["data_source"],
                    item["response_str"],
                    item["ground_truth"],
                    item["extra_info"],
                )
                futures.append((item, future))

            # Collect results
            for item, future in futures:
                try:
                    score_result = future.result(timeout=self.timeout + 10)
                except Exception as e:
                    logger.error(f"Failed to get result: {e}")
                    score_result = {"score": 0.0, "error": str(e)}

                i = item["index"]
                data_source = item["data_source"]

                if isinstance(score_result, dict):
                    reward = score_result.get("score", 0.0)
                    for key, value in score_result.items():
                        reward_extra_info[key].append(value)
                else:
                    reward = float(score_result)

                reward_tensor[i, item["valid_response_length"] - 1] = reward

                # Debug printing
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print(f"[Prime Intellect] data_source: {data_source}")
                    print(f"[prompt] {item['prompt_str'][:200]}...")
                    print(f"[response] {item['response_str'][:200]}...")
                    print(f"[ground_truth] {item['ground_truth']}")
                    print(f"[score] {reward}")
                    if isinstance(score_result, dict):
                        for key, value in score_result.items():
                            if key != "score":
                                print(f"[{key}] {value}")
                    print("-" * 50)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
        return reward_tensor


@register("prime_intellect_async")
class AsyncPrimeIntellectRewardManager(AbstractRewardManager):
    """Async reward manager for high-throughput Prime Intellect evaluation.

    This manager uses async I/O for concurrent reward computation,
    suitable for large-scale training with many parallel rollouts.

    Features:
        - Async HTTP requests to Prime Intellect API
        - Concurrent sandbox execution
        - Rate limiting support
        - Automatic retry with backoff

    Args:
        tokenizer: Tokenizer for decoding model outputs.
        num_examine: Number of samples to print for debugging.
        compute_score: Custom async scoring function.
        reward_fn_key: Key for accessing data source.
        prime_intellect_api_key: API key.
        timeout: Timeout for remote execution.
        max_concurrent: Maximum concurrent requests.
        rate_limit_rpm: Rate limit in requests per minute.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Callable | None = None,
        reward_fn_key: str = "data_source",
        prime_intellect_api_key: str | None = None,
        timeout: int = 30,
        docker_image: str = "python:3.11-slim",
        max_concurrent: int = 100,
        rate_limit_rpm: int | None = None,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.prime_intellect_api_key = prime_intellect_api_key
        self.timeout = timeout
        self.docker_image = docker_image
        self.max_concurrent = max_concurrent
        self.rate_limit_rpm = rate_limit_rpm

        # Use async compute score
        if compute_score is None:
            from verl.utils.prime_intellect.score import compute_score_async

            self.compute_score = compute_score_async
        else:
            self.compute_score = compute_score

        self.extra_compute_kwargs = {
            "prime_intellect_api_key": prime_intellect_api_key,
            "timeout": timeout,
            "docker_image": docker_image,
        }
        self.extra_compute_kwargs.update(kwargs)

        # Async resources (initialized lazily)
        self._semaphore = None
        self._rate_limiter = None
        self._loop = None

    def _get_loop(self):
        """Get or create event loop."""
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

    def _ensure_async_resources(self):
        """Initialize async resources."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)

        if self._rate_limiter is None and self.rate_limit_rpm:
            # Simple token bucket rate limiter
            from verl.experimental.reward.reward_loop.limited import AsyncTokenBucket

            requests_per_second = self.rate_limit_rpm / 60.0
            self._rate_limiter = AsyncTokenBucket(
                rate_limit=requests_per_second,
                max_tokens=requests_per_second,
            )

    async def _compute_single_score_async(
        self,
        data_source: str,
        solution_str: str,
        ground_truth: str,
        extra_info: dict | None = None,
    ) -> dict | float:
        """Compute score for a single sample asynchronously."""
        self._ensure_async_resources()

        # Apply rate limiting
        if self._rate_limiter:
            await self._rate_limiter.acquire(1.0)

        async with self._semaphore:
            try:
                result = await asyncio.wait_for(
                    self.compute_score(
                        data_source=data_source,
                        solution_str=solution_str,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                        **self.extra_compute_kwargs,
                    ),
                    timeout=self.timeout,
                )
                return result
            except asyncio.TimeoutError:
                logger.warning(f"Score computation timed out for {data_source}")
                return {"score": 0.0, "timeout": True}
            except Exception as e:
                logger.error(f"Score computation failed: {e}")
                return {"score": 0.0, "error": str(e)}

    async def _process_batch_async(self, data: DataProto) -> tuple[torch.Tensor, dict]:
        """Process batch asynchronously."""
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # Prepare tasks
        tasks = []
        item_info = []

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            extra_info["prompt"] = prompt_str

            task = self._compute_single_score_async(
                data_source, response_str, ground_truth, extra_info
            )
            tasks.append(task)
            item_info.append({
                "index": i,
                "valid_response_length": valid_response_length.item(),
                "data_source": data_source,
            })

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for item, result in zip(item_info, results):
            i = item["index"]

            if isinstance(result, Exception):
                reward = 0.0
                reward_extra_info["error"].append(str(result))
            elif isinstance(result, dict):
                reward = result.get("score", 0.0)
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                reward = float(result)

            reward_tensor[i, item["valid_response_length"] - 1] = reward

        return reward_tensor, dict(reward_extra_info)

    def __call__(
        self,
        data: DataProto,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        """Compute rewards for a batch using async processing."""
        # If pre-computed rm_scores exist, return them directly
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {
                    key: data.non_tensor_batch[key] for key in reward_extra_keys
                }
                return {
                    "reward_tensor": data.batch["rm_scores"],
                    "reward_extra_info": reward_extra_info,
                }
            return data.batch["rm_scores"]

        # Run async processing
        loop = self._get_loop()
        reward_tensor, reward_extra_info = loop.run_until_complete(
            self._process_batch_async(data)
        )

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor
