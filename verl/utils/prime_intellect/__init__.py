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
Prime Intellect integration module for verl.

This module provides seamless integration with Prime Intellect's environments
and sandboxes for reward computation in reinforcement learning training.

Prime Intellect offers:
- Remote code execution sandboxes with sub-second provisioning
- Community-powered RL environments hub
- Verifiers library for modular environment creation

Installation:
    pip install prime-sandboxes  # Lightweight SDK
    # or
    pip install prime  # Full CLI + SDK

Usage:
    from verl.utils.prime_intellect import (
        PrimeIntellectClient,
        AsyncPrimeIntellectClient,
        load_environment,
        compute_score,
    )

    # Initialize client
    client = PrimeIntellectClient(api_key="your-api-key")

    # Execute code in sandbox
    result = client.execute_code(code="print('Hello')", docker_image="python:3.11-slim")

    # Load a verifiers environment
    env = load_environment("will/wordle")
"""

from verl.utils.prime_intellect.client import (
    AsyncPrimeIntellectClient,
    PrimeIntellectClient,
    PrimeIntellectSandbox,
)
from verl.utils.prime_intellect.environments import (
    PrimeIntellectEnvironment,
    load_environment,
)
from verl.utils.prime_intellect.score import compute_score

__all__ = [
    "PrimeIntellectClient",
    "AsyncPrimeIntellectClient",
    "PrimeIntellectSandbox",
    "PrimeIntellectEnvironment",
    "load_environment",
    "compute_score",
]
