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
Prime Intellect Reward Loop Manager for verl.

This module provides an async reward loop manager that integrates with
Prime Intellect's environments and sandboxes for high-throughput
reward computation during RL training.

Features:
    - Async batch processing with rate limiting
    - Integration with Prime Intellect Environments Hub
    - Support for both sandbox execution and verifiers environments
    - Token bucket rate limiting for API compliance
    - Concurrent request management

Usage:
    >>> from verl.experimental.reward.reward_loop import get_reward_loop_manager_cls
    >>> manager_cls = get_reward_loop_manager_cls("prime_intellect")
    >>> manager = manager_cls(config, tokenizer)
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os

from collections import defaultdict
from omegaconf import DictConfig
from transformers import AutoTokenizer

import torch

from verl import DataProto
from verl.experimental.reward.reward_loop import register as register_loop
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase
from verl.experimental.reward.reward_loop.limited import AsyncTokenBucket
from verl.workers.reward_manager import register as register_manager

logger = logging.getLogger(__name__)


@register_loop("prime_intellect")
@register_manager("prime_intellect_loop")
class PrimeIntellectRewardLoopManager(RewardLoopManagerBase):
    """Async reward loop manager for Prime Intellect environments.

    This manager provides high-performance async reward computation using
    Prime Intellect's sandbox execution and verifiers environments.

    Features:
        - Three-layer rate limiting (concurrency, RPM, TPM)
        - Global state sharing across workers
        - Async sandbox execution
        - Environment caching

    Configuration:
        reward_model:
            max_concurrent: 100  # Max parallel requests
            max_rpm: 500  # Max requests per minute
            timeout: 60  # Timeout in seconds
            docker_image: python:3.11-slim
            prime_intellect_api_key: <optional, uses PRIME_API_KEY env var>

    Args:
        config: DictConfig containing reward_model settings.
        tokenizer: Tokenizer for decoding responses.
        compute_score: Optional custom scoring function.
        **kwargs: Additional arguments.
    """

    # Class-level state for global rate limiting
    _semaphore = None
    _max_concurrent = None
    _rpm_limiter = None
    _max_rpm = None
    _timeout = None
    _docker_image = None
    _api_key = None
    _client = None
    _class_initialized = False

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer):
        """Initialize class state shared across all instances."""
        if cls._class_initialized:
            return

        super().init_class(config, tokenizer)

        # Get config values with defaults
        reward_config = config.get("reward_model", {})

        # Concurrency limiter
        cls._max_concurrent = reward_config.get("max_concurrent", 100)
        cls._semaphore = asyncio.Semaphore(cls._max_concurrent)

        # Request rate limiter (RPM)
        cls._max_rpm = reward_config.get("max_rpm", None)
        if cls._max_rpm is not None:
            requests_per_second = cls._max_rpm / 60.0
            cls._rpm_limiter = AsyncTokenBucket(
                rate_limit=requests_per_second,
                max_tokens=requests_per_second * 2,  # Allow small burst
            )
        else:
            cls._rpm_limiter = None

        # Other settings
        cls._timeout = reward_config.get("timeout", 60)
        cls._docker_image = reward_config.get("docker_image", "python:3.11-slim")
        cls._api_key = reward_config.get(
            "prime_intellect_api_key",
            os.environ.get("PRIME_API_KEY"),
        )

        # Initialize async client
        try:
            from verl.utils.prime_intellect import AsyncPrimeIntellectClient

            cls._client = AsyncPrimeIntellectClient(
                api_key=cls._api_key,
                max_concurrent=cls._max_concurrent,
            )
        except ImportError:
            logger.warning("Prime Intellect client not available")
            cls._client = None

        # Log configuration
        log_msg = "Prime Intellect reward loop configuration:\n"
        log_msg += f"  - Concurrency limit: {cls._max_concurrent}\n"
        if cls._max_rpm is not None:
            log_msg += f"  - Request rate limit: {cls._max_rpm} RPM\n"
        else:
            log_msg += "  - Request rate limit: unlimited\n"
        log_msg += f"  - Timeout: {cls._timeout}s\n"
        log_msg += f"  - Docker image: {cls._docker_image}\n"
        log_msg += f"  - API key: {'configured' if cls._api_key else 'not set'}\n"
        logger.info(log_msg)

        cls._class_initialized = True

    def __init__(
        self,
        config: DictConfig,
        tokenizer: AutoTokenizer,
        compute_score=None,
        **kwargs,
    ):
        super().__init__(config, tokenizer)

        # Use provided compute_score or default to Prime Intellect's
        if compute_score is None:
            from verl.utils.prime_intellect.score import compute_score_async

            self.compute_score = compute_score_async
        else:
            self.compute_score = compute_score

        self.is_async_compute_score = inspect.iscoroutinefunction(self.compute_score)

        # Extra kwargs for compute_score
        self.extra_compute_kwargs = {
            "prime_intellect_api_key": self._api_key,
            "timeout": self._timeout,
            "docker_image": self._docker_image,
        }
        self.extra_compute_kwargs.update(kwargs)

    async def _compute_reward(
        self,
        data_source: str,
        solution_str: str,
        ground_truth: str,
        extra_info: dict,
    ) -> dict | float:
        """Compute reward for a single sample."""
        if self.is_async_compute_score:
            return await self.compute_score(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **self.extra_compute_kwargs,
            )
        else:
            return await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=solution_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **self.extra_compute_kwargs,
                ),
            )

    async def run_single(self, data: DataProto) -> dict:
        """Process a single data item asynchronously.

        Args:
            data: DataProto containing a single item.

        Returns:
            dict: Contains reward_score and reward_extra_info.
        """
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]

        # Extract response
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # Extract metadata
        data_source = data_item.non_tensor_batch.get("data_source", "prime_intellect/code")
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})

        # Handle tool extra fields
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields)

        # Decode response
        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        reward_extra_info = {}

        # Apply rate limiting
        if self._rpm_limiter is not None:
            await self._rpm_limiter.acquire(1.0)

        async with self._semaphore:
            try:
                result = await asyncio.wait_for(
                    self._compute_reward(
                        data_source=data_source,
                        solution_str=response_str,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                    ),
                    timeout=self._timeout,
                )

                if isinstance(result, dict):
                    score = result.get("score", 0.0)
                    for key, value in result.items():
                        reward_extra_info[key] = value
                else:
                    score = float(result)
                    reward_extra_info["acc"] = score

                reward = score

            except asyncio.TimeoutError:
                logger.warning(
                    f"Prime Intellect reward computation timed out after {self._timeout}s "
                    f"for data_source={data_source}"
                )
                reward = 0.0
                reward_extra_info["timeout"] = True
                reward_extra_info["acc"] = 0.0

            except Exception as e:
                logger.error(
                    f"Prime Intellect reward computation failed for "
                    f"data_source={data_source}: {e}"
                )
                reward = 0.0
                reward_extra_info["error"] = str(e)
                reward_extra_info["acc"] = 0.0

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}

    def __call__(
        self,
        data: DataProto,
        return_dict: bool = False,
    ) -> torch.Tensor | dict:
        """Make the manager callable like traditional reward managers.

        Args:
            data: Input DataProto with batch of samples.
            return_dict: If True, return dict with tensor and extra info.

        Returns:
            Reward tensor or dict with tensor and metadata.
        """
        # Return pre-computed scores if available
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

        # Initialize reward tensor
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # Process batch asynchronously
        async def process_batch():
            tasks = []
            for i in range(len(data)):
                data_item = data[i : i + 1]
                tasks.append(self.run_single(data_item))
            return await asyncio.gather(*tasks)

        # Run async processing
        results = self.loop.run_until_complete(process_batch())

        # Aggregate results
        for i, result in enumerate(results):
            data_item = data[i]
            response_ids = data_item.batch["responses"]
            response_length = response_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()

            reward = result["reward_score"]
            reward_tensor[i, valid_response_length - 1] = reward

            if "reward_extra_info" in result:
                for key, value in result["reward_extra_info"].items():
                    reward_extra_info[key].append(value)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
        return reward_tensor


@register_loop("prime_intellect_env")
@register_manager("prime_intellect_env")
class PrimeIntellectEnvRewardLoopManager(PrimeIntellectRewardLoopManager):
    """Reward loop manager specialized for Prime Intellect verifiers environments.

    This manager is optimized for evaluating responses using verifiers
    environments from the Prime Intellect Environments Hub.

    Configuration:
        reward_model:
            environment_name: will/wordle  # Environment from hub
            max_concurrent: 50
            timeout: 30

    Args:
        config: DictConfig with reward_model.environment_name.
        tokenizer: Tokenizer for decoding.
        **kwargs: Additional arguments.
    """

    _environment = None
    _environment_name = None

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer):
        """Initialize with environment loading."""
        if cls._class_initialized:
            return

        # Call parent init
        super().init_class(config, tokenizer)

        # Load environment
        reward_config = config.get("reward_model", {})
        cls._environment_name = reward_config.get("environment_name", None)

        if cls._environment_name:
            try:
                from verl.utils.prime_intellect import load_environment

                cls._environment = load_environment(cls._environment_name)
                logger.info(f"Loaded Prime Intellect environment: {cls._environment_name}")
            except Exception as e:
                logger.error(f"Failed to load environment {cls._environment_name}: {e}")
                cls._environment = None

    async def _compute_reward(
        self,
        data_source: str,
        solution_str: str,
        ground_truth: str,
        extra_info: dict,
    ) -> dict | float:
        """Compute reward using the loaded environment."""
        if self._environment is None:
            # Fall back to parent implementation
            return await super()._compute_reward(
                data_source, solution_str, ground_truth, extra_info
            )

        try:
            result = await self._environment.evaluate_async(
                response=solution_str,
                prompt=extra_info.get("prompt", ""),
                answer=ground_truth,
                info=extra_info.get("info", {}),
            )

            return {
                "score": result.score,
                "passed": result.passed,
                "details": result.details,
                "environment": self._environment_name,
            }
        except Exception as e:
            logger.error(f"Environment evaluation failed: {e}")
            return {"score": 0.0, "error": str(e)}
