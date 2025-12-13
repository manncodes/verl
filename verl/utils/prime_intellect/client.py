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
Prime Intellect client for sandbox and environment operations.

This module provides both synchronous and asynchronous clients for interacting
with Prime Intellect's sandbox execution environment.

Example:
    >>> from verl.utils.prime_intellect import PrimeIntellectClient
    >>> client = PrimeIntellectClient(api_key="your-api-key")
    >>> sandbox = client.create_sandbox(docker_image="python:3.11-slim")
    >>> result = client.execute_command(sandbox.id, "python -c 'print(1+1)'")
    >>> print(result.stdout)  # "2"
    >>> client.delete_sandbox(sandbox.id)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Default Prime Intellect API endpoint
DEFAULT_API_BASE_URL = "https://api.primeintellect.ai"

# Default sandbox configuration
DEFAULT_DOCKER_IMAGE = "python:3.11-slim"
DEFAULT_CPU_CORES = 2
DEFAULT_MEMORY_GB = 4
DEFAULT_TIMEOUT = 30  # seconds


@dataclass
class ExecutionResult:
    """Result from executing a command in a sandbox.

    Attributes:
        stdout: Standard output from the command.
        stderr: Standard error from the command.
        exit_code: Exit code from the command (0 = success).
        execution_time_ms: Time taken to execute the command in milliseconds.
        timed_out: Whether the command timed out.
        metadata: Additional metadata from the execution.
    """

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    execution_time_ms: float = 0.0
    timed_out: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Whether the execution was successful (exit_code == 0)."""
        return self.exit_code == 0 and not self.timed_out


@dataclass
class PrimeIntellectSandbox:
    """Represents a Prime Intellect sandbox instance.

    Attributes:
        id: Unique identifier for the sandbox.
        name: Human-readable name for the sandbox.
        docker_image: Docker image used for the sandbox.
        cpu_cores: Number of CPU cores allocated.
        memory_gb: Amount of memory allocated in GB.
        status: Current status of the sandbox.
        created_at: Timestamp when the sandbox was created.
    """

    id: str
    name: str = ""
    docker_image: str = DEFAULT_DOCKER_IMAGE
    cpu_cores: int = DEFAULT_CPU_CORES
    memory_gb: int = DEFAULT_MEMORY_GB
    status: str = "pending"
    created_at: float = field(default_factory=time.time)


class PrimeIntellectClient:
    """Synchronous client for Prime Intellect sandbox operations.

    This client provides a simple interface for creating sandboxes, executing
    code, and managing sandbox lifecycle.

    Example:
        >>> client = PrimeIntellectClient(api_key="your-key")
        >>> result = client.execute_code("print('Hello, World!')")
        >>> print(result.stdout)
        Hello, World!

    Args:
        api_key: Prime Intellect API key. If not provided, reads from
            PRIME_API_KEY environment variable.
        api_base_url: Base URL for the Prime Intellect API.
        default_docker_image: Default Docker image for sandboxes.
        default_timeout: Default timeout for command execution in seconds.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base_url: str = DEFAULT_API_BASE_URL,
        default_docker_image: str = DEFAULT_DOCKER_IMAGE,
        default_timeout: int = DEFAULT_TIMEOUT,
    ):
        self.api_key = api_key or os.environ.get("PRIME_API_KEY")
        self.api_base_url = api_base_url.rstrip("/")
        self.default_docker_image = default_docker_image
        self.default_timeout = default_timeout
        self._session = None

        if not self.api_key:
            logger.warning(
                "No API key provided. Set PRIME_API_KEY environment variable "
                "or pass api_key to the constructor."
            )

    def _get_session(self):
        """Get or create the HTTP session."""
        if self._session is None:
            try:
                import requests

                self._session = requests.Session()
                self._session.headers.update(
                    {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    }
                )
            except ImportError:
                raise ImportError(
                    "requests library is required for sync client. "
                    "Install with: pip install requests"
                )
        return self._session

    def _make_request(
        self,
        method: str,
        endpoint: str,
        json: dict | None = None,
        timeout: int | None = None,
    ) -> dict:
        """Make an HTTP request to the Prime Intellect API."""
        session = self._get_session()
        url = f"{self.api_base_url}{endpoint}"
        timeout = timeout or self.default_timeout

        try:
            response = session.request(method, url, json=json, timeout=timeout)
            response.raise_for_status()
            return response.json() if response.content else {}
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise

    def create_sandbox(
        self,
        name: str | None = None,
        docker_image: str | None = None,
        cpu_cores: int = DEFAULT_CPU_CORES,
        memory_gb: int = DEFAULT_MEMORY_GB,
    ) -> PrimeIntellectSandbox:
        """Create a new sandbox instance.

        Args:
            name: Optional name for the sandbox.
            docker_image: Docker image to use. Defaults to python:3.11-slim.
            cpu_cores: Number of CPU cores to allocate.
            memory_gb: Amount of memory in GB to allocate.

        Returns:
            PrimeIntellectSandbox: The created sandbox instance.
        """
        docker_image = docker_image or self.default_docker_image
        name = name or f"verl-sandbox-{int(time.time())}"

        payload = {
            "name": name,
            "docker_image": docker_image,
            "cpu_cores": cpu_cores,
            "memory_gb": memory_gb,
        }

        response = self._make_request("POST", "/v1/sandboxes", json=payload)

        return PrimeIntellectSandbox(
            id=response.get("id", response.get("sandbox_id", "")),
            name=name,
            docker_image=docker_image,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            status=response.get("status", "pending"),
        )

    def wait_for_sandbox(
        self,
        sandbox_id: str,
        timeout: int = 60,
        poll_interval: float = 0.5,
    ) -> bool:
        """Wait for a sandbox to be ready.

        Args:
            sandbox_id: ID of the sandbox to wait for.
            timeout: Maximum time to wait in seconds.
            poll_interval: Time between status checks in seconds.

        Returns:
            bool: True if sandbox is ready, False if timeout.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = self._make_request("GET", f"/v1/sandboxes/{sandbox_id}")
                status = response.get("status", "")
                if status in ("ready", "running"):
                    return True
                if status in ("failed", "error", "terminated"):
                    logger.error(f"Sandbox {sandbox_id} failed with status: {status}")
                    return False
            except Exception as e:
                logger.warning(f"Error checking sandbox status: {e}")

            time.sleep(poll_interval)

        logger.warning(f"Timeout waiting for sandbox {sandbox_id}")
        return False

    def execute_command(
        self,
        sandbox_id: str,
        command: str,
        timeout: int | None = None,
        workdir: str | None = None,
    ) -> ExecutionResult:
        """Execute a command in a sandbox.

        Args:
            sandbox_id: ID of the sandbox to execute in.
            command: Command to execute.
            timeout: Timeout in seconds.
            workdir: Working directory for the command.

        Returns:
            ExecutionResult: Result of the command execution.
        """
        timeout = timeout or self.default_timeout

        payload = {
            "command": command,
            "timeout": timeout,
        }
        if workdir:
            payload["workdir"] = workdir

        start_time = time.time()
        try:
            response = self._make_request(
                "POST",
                f"/v1/sandboxes/{sandbox_id}/execute",
                json=payload,
                timeout=timeout + 10,  # Extra time for network
            )

            return ExecutionResult(
                stdout=response.get("stdout", ""),
                stderr=response.get("stderr", ""),
                exit_code=response.get("exit_code", 0),
                execution_time_ms=(time.time() - start_time) * 1000,
                timed_out=response.get("timed_out", False),
                metadata=response.get("metadata", {}),
            )
        except Exception as e:
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                exit_code=-1,
                execution_time_ms=(time.time() - start_time) * 1000,
                timed_out=True,
                metadata={"error": str(e)},
            )

    def execute_code(
        self,
        code: str,
        docker_image: str | None = None,
        timeout: int | None = None,
        language: str = "python",
    ) -> ExecutionResult:
        """Execute code in a temporary sandbox.

        This is a convenience method that creates a sandbox, executes code,
        and cleans up automatically.

        Args:
            code: Code to execute.
            docker_image: Docker image to use.
            timeout: Timeout in seconds.
            language: Programming language (for command construction).

        Returns:
            ExecutionResult: Result of the code execution.
        """
        docker_image = docker_image or self.default_docker_image
        timeout = timeout or self.default_timeout

        # Create sandbox
        sandbox = self.create_sandbox(docker_image=docker_image)

        try:
            # Wait for sandbox to be ready
            if not self.wait_for_sandbox(sandbox.id, timeout=30):
                return ExecutionResult(
                    stdout="",
                    stderr="Sandbox failed to start",
                    exit_code=-1,
                    timed_out=True,
                )

            # Construct command based on language
            if language == "python":
                command = f'python -c "{code}"' if "\n" not in code else f"python << 'EOF'\n{code}\nEOF"
            else:
                command = code

            # Execute command
            return self.execute_command(sandbox.id, command, timeout=timeout)
        finally:
            # Clean up
            try:
                self.delete_sandbox(sandbox.id)
            except Exception as e:
                logger.warning(f"Failed to delete sandbox: {e}")

    def delete_sandbox(self, sandbox_id: str) -> bool:
        """Delete a sandbox.

        Args:
            sandbox_id: ID of the sandbox to delete.

        Returns:
            bool: True if deletion was successful.
        """
        try:
            self._make_request("DELETE", f"/v1/sandboxes/{sandbox_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete sandbox {sandbox_id}: {e}")
            return False

    def upload_file(
        self,
        sandbox_id: str,
        local_path: str,
        remote_path: str,
    ) -> bool:
        """Upload a file to a sandbox.

        Args:
            sandbox_id: ID of the sandbox.
            local_path: Local path to the file.
            remote_path: Remote path in the sandbox.

        Returns:
            bool: True if upload was successful.
        """
        try:
            with open(local_path, "rb") as f:
                content = f.read()

            # Use base64 encoding for file content
            import base64

            encoded_content = base64.b64encode(content).decode("utf-8")

            payload = {
                "path": remote_path,
                "content": encoded_content,
                "encoding": "base64",
            }

            self._make_request(
                "POST", f"/v1/sandboxes/{sandbox_id}/files", json=payload
            )
            return True
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            return False

    def close(self):
        """Close the client and release resources."""
        if self._session:
            self._session.close()
            self._session = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AsyncPrimeIntellectClient:
    """Asynchronous client for Prime Intellect sandbox operations.

    This client provides an async interface for high-throughput sandbox
    operations, suitable for concurrent reward computation.

    Example:
        >>> async with AsyncPrimeIntellectClient(api_key="your-key") as client:
        ...     results = await asyncio.gather(*[
        ...         client.execute_code(f"print({i})")
        ...         for i in range(10)
        ...     ])

    Args:
        api_key: Prime Intellect API key.
        api_base_url: Base URL for the Prime Intellect API.
        default_docker_image: Default Docker image for sandboxes.
        default_timeout: Default timeout for command execution.
        max_concurrent: Maximum number of concurrent requests.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base_url: str = DEFAULT_API_BASE_URL,
        default_docker_image: str = DEFAULT_DOCKER_IMAGE,
        default_timeout: int = DEFAULT_TIMEOUT,
        max_concurrent: int = 100,
    ):
        self.api_key = api_key or os.environ.get("PRIME_API_KEY")
        self.api_base_url = api_base_url.rstrip("/")
        self.default_docker_image = default_docker_image
        self.default_timeout = default_timeout
        self.max_concurrent = max_concurrent
        self._session = None
        self._semaphore = None

        if not self.api_key:
            logger.warning(
                "No API key provided. Set PRIME_API_KEY environment variable "
                "or pass api_key to the constructor."
            )

    async def _get_session(self):
        """Get or create the async HTTP session."""
        if self._session is None:
            try:
                import aiohttp

                self._session = aiohttp.ClientSession(
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    }
                )
                self._semaphore = asyncio.Semaphore(self.max_concurrent)
            except ImportError:
                raise ImportError(
                    "aiohttp library is required for async client. "
                    "Install with: pip install aiohttp"
                )
        return self._session

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        json: dict | None = None,
        timeout: int | None = None,
    ) -> dict:
        """Make an async HTTP request to the Prime Intellect API."""
        session = await self._get_session()
        url = f"{self.api_base_url}{endpoint}"
        timeout_obj = None

        if timeout:
            try:
                import aiohttp

                timeout_obj = aiohttp.ClientTimeout(total=timeout)
            except ImportError:
                pass

        async with self._semaphore:
            try:
                async with session.request(
                    method, url, json=json, timeout=timeout_obj
                ) as response:
                    response.raise_for_status()
                    return await response.json() if response.content_length else {}
            except Exception as e:
                logger.error(f"API request failed: {e}")
                raise

    async def create_sandbox(
        self,
        name: str | None = None,
        docker_image: str | None = None,
        cpu_cores: int = DEFAULT_CPU_CORES,
        memory_gb: int = DEFAULT_MEMORY_GB,
    ) -> PrimeIntellectSandbox:
        """Create a new sandbox instance asynchronously."""
        docker_image = docker_image or self.default_docker_image
        name = name or f"verl-sandbox-{int(time.time())}"

        payload = {
            "name": name,
            "docker_image": docker_image,
            "cpu_cores": cpu_cores,
            "memory_gb": memory_gb,
        }

        response = await self._make_request("POST", "/v1/sandboxes", json=payload)

        return PrimeIntellectSandbox(
            id=response.get("id", response.get("sandbox_id", "")),
            name=name,
            docker_image=docker_image,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            status=response.get("status", "pending"),
        )

    async def wait_for_sandbox(
        self,
        sandbox_id: str,
        timeout: int = 60,
        poll_interval: float = 0.5,
    ) -> bool:
        """Wait for a sandbox to be ready asynchronously."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = await self._make_request(
                    "GET", f"/v1/sandboxes/{sandbox_id}"
                )
                status = response.get("status", "")
                if status in ("ready", "running"):
                    return True
                if status in ("failed", "error", "terminated"):
                    logger.error(f"Sandbox {sandbox_id} failed with status: {status}")
                    return False
            except Exception as e:
                logger.warning(f"Error checking sandbox status: {e}")

            await asyncio.sleep(poll_interval)

        logger.warning(f"Timeout waiting for sandbox {sandbox_id}")
        return False

    async def execute_command(
        self,
        sandbox_id: str,
        command: str,
        timeout: int | None = None,
        workdir: str | None = None,
    ) -> ExecutionResult:
        """Execute a command in a sandbox asynchronously."""
        timeout = timeout or self.default_timeout

        payload = {
            "command": command,
            "timeout": timeout,
        }
        if workdir:
            payload["workdir"] = workdir

        start_time = time.time()
        try:
            response = await self._make_request(
                "POST",
                f"/v1/sandboxes/{sandbox_id}/execute",
                json=payload,
                timeout=timeout + 10,
            )

            return ExecutionResult(
                stdout=response.get("stdout", ""),
                stderr=response.get("stderr", ""),
                exit_code=response.get("exit_code", 0),
                execution_time_ms=(time.time() - start_time) * 1000,
                timed_out=response.get("timed_out", False),
                metadata=response.get("metadata", {}),
            )
        except Exception as e:
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                exit_code=-1,
                execution_time_ms=(time.time() - start_time) * 1000,
                timed_out=True,
                metadata={"error": str(e)},
            )

    async def execute_code(
        self,
        code: str,
        docker_image: str | None = None,
        timeout: int | None = None,
        language: str = "python",
    ) -> ExecutionResult:
        """Execute code in a temporary sandbox asynchronously."""
        docker_image = docker_image or self.default_docker_image
        timeout = timeout or self.default_timeout

        # Create sandbox
        sandbox = await self.create_sandbox(docker_image=docker_image)

        try:
            # Wait for sandbox to be ready
            if not await self.wait_for_sandbox(sandbox.id, timeout=30):
                return ExecutionResult(
                    stdout="",
                    stderr="Sandbox failed to start",
                    exit_code=-1,
                    timed_out=True,
                )

            # Construct command based on language
            if language == "python":
                command = (
                    f'python -c "{code}"'
                    if "\n" not in code
                    else f"python << 'EOF'\n{code}\nEOF"
                )
            else:
                command = code

            # Execute command
            return await self.execute_command(sandbox.id, command, timeout=timeout)
        finally:
            # Clean up
            try:
                await self.delete_sandbox(sandbox.id)
            except Exception as e:
                logger.warning(f"Failed to delete sandbox: {e}")

    async def delete_sandbox(self, sandbox_id: str) -> bool:
        """Delete a sandbox asynchronously."""
        try:
            await self._make_request("DELETE", f"/v1/sandboxes/{sandbox_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete sandbox {sandbox_id}: {e}")
            return False

    async def close(self):
        """Close the client and release resources."""
        if self._session:
            await self._session.close()
            self._session = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Convenience function to get the appropriate client
def get_client(
    api_key: str | None = None,
    async_mode: bool = False,
    **kwargs,
) -> PrimeIntellectClient | AsyncPrimeIntellectClient:
    """Get a Prime Intellect client instance.

    Args:
        api_key: API key for Prime Intellect.
        async_mode: If True, return an async client.
        **kwargs: Additional arguments for the client constructor.

    Returns:
        Either a sync or async client based on async_mode.
    """
    if async_mode:
        return AsyncPrimeIntellectClient(api_key=api_key, **kwargs)
    return PrimeIntellectClient(api_key=api_key, **kwargs)
