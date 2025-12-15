import pandas as pd
import numpy as np
import json
import threading
import logging
import time
import subprocess
import requests
import signal
import atexit
from collections import Counter
from typing import Optional

# Enable logging to see sandbox errors
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Terminal colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


from rich.syntax import Syntax
from rich.console import Console
console = Console()

def print_code(code: str, max_lines: int = 30, title: str = "Code"):
    syntax = Syntax(code, "python", line_numbers=True, theme="monokai")
    console.print(syntax)


# Global variable to track VLLM process for cleanup
_vllm_process: Optional[subprocess.Popen] = None


def start_vllm_server(
    model: str,
    port: int = 8000,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 8192,
    dtype: str = "auto",
    wait_for_ready: bool = True,
    timeout: int = 300,
) -> subprocess.Popen:
    """Start a VLLM server as a subprocess.

    Args:
        model: HuggingFace model name or path
        port: Port to serve on
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: Fraction of GPU memory to use
        max_model_len: Maximum sequence length
        dtype: Data type (auto, float16, bfloat16)
        wait_for_ready: Wait for server to be ready before returning
        timeout: Timeout in seconds for server to become ready

    Returns:
        subprocess.Popen: The server process
    """
    global _vllm_process

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--dtype", dtype,
        "--trust-remote-code",
    ]

    print(f"{Colors.CYAN}Starting VLLM server...{Colors.ENDC}")
    print(f"{Colors.DIM}Command: {' '.join(cmd)}{Colors.ENDC}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN),
    )

    _vllm_process = process

    # Register cleanup handler
    def cleanup():
        if _vllm_process and _vllm_process.poll() is None:
            print(f"\n{Colors.YELLOW}Shutting down VLLM server...{Colors.ENDC}")
            _vllm_process.terminate()
            try:
                _vllm_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _vllm_process.kill()

    atexit.register(cleanup)

    if wait_for_ready:
        base_url = f"http://localhost:{port}"
        start_time = time.time()

        print(f"{Colors.CYAN}Waiting for VLLM server to be ready (timeout={timeout}s)...{Colors.ENDC}")

        while time.time() - start_time < timeout:
            # Check if process died
            if process.poll() is not None:
                # Read any output for debugging
                output = process.stdout.read() if process.stdout else ""
                raise RuntimeError(f"VLLM server died during startup. Output:\n{output}")

            try:
                response = requests.get(f"{base_url}/health", timeout=5)
                if response.status_code == 200:
                    print(f"{Colors.GREEN}VLLM server is ready!{Colors.ENDC}")
                    return process
            except requests.exceptions.ConnectionError:
                pass
            except requests.exceptions.Timeout:
                pass

            time.sleep(2)

        # Timeout - kill process and raise error
        process.terminate()
        raise TimeoutError(f"VLLM server did not become ready within {timeout} seconds")

    return process


def stop_vllm_server(process: subprocess.Popen, timeout: int = 10):
    """Stop the VLLM server gracefully."""
    global _vllm_process

    if process.poll() is not None:
        print(f"{Colors.YELLOW}VLLM server already stopped.{Colors.ENDC}")
        return

    print(f"{Colors.CYAN}Stopping VLLM server...{Colors.ENDC}")
    process.terminate()

    try:
        process.wait(timeout=timeout)
        print(f"{Colors.GREEN}VLLM server stopped gracefully.{Colors.ENDC}")
    except subprocess.TimeoutExpired:
        print(f"{Colors.YELLOW}Force killing VLLM server...{Colors.ENDC}")
        process.kill()
        process.wait()

    _vllm_process = None


def get_vllm_completion(
    prompt: str,
    vllm_url: str = "http://localhost:8000",
    model: str = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    stop: list = None,
    timeout: int = 120,
) -> str:
    """Get a completion from the VLLM server.

    Args:
        prompt: The prompt to send
        vllm_url: Base URL of the VLLM server
        model: Model name (if None, uses the first available model)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        stop: Stop sequences
        timeout: Request timeout

    Returns:
        str: The generated completion
    """
    # Get available models if not specified
    if model is None:
        models_response = requests.get(f"{vllm_url}/v1/models", timeout=10)
        models_data = models_response.json()
        model = models_data["data"][0]["id"]

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop or [],
    }

    response = requests.post(
        f"{vllm_url}/v1/completions",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()

    result = response.json()
    return result["choices"][0]["text"]


def get_vllm_chat_completion(
    messages: list,
    vllm_url: str = "http://localhost:8000",
    model: str = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    stop: list = None,
    timeout: int = 120,
) -> str:
    """Get a chat completion from the VLLM server.

    Args:
        messages: List of message dicts with 'role' and 'content'
        vllm_url: Base URL of the VLLM server
        model: Model name (if None, uses the first available model)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        stop: Stop sequences
        timeout: Request timeout

    Returns:
        str: The generated completion
    """
    # Get available models if not specified
    if model is None:
        models_response = requests.get(f"{vllm_url}/v1/models", timeout=10)
        models_data = models_response.json()
        model = models_data["data"][0]["id"]

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop or [],
    }

    response = requests.post(
        f"{vllm_url}/v1/chat/completions",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()

    result = response.json()
    return result["choices"][0]["message"]["content"]


def test_code_scoring(
    parquet_path: str,
    sample_n: int = 100,
    seed: int = 42,
    sandbox_fusion_url: str = "http://sandbox-fusion-code-rl-service.llm-pretraining.svc.cluster.local:8080/run_code",
    max_concurrent: int = 64,
    timeout: int = 10,
    verbose: bool = True,
    # VLLM parameters
    vllm_url: str = "http://localhost:8000",
    vllm_model: str = None,
    vllm_max_tokens: int = 2048,
    vllm_temperature: float = 0.0,
    use_vllm: bool = True,
    use_chat_api: bool = True,
) -> dict:
    """Test code scoring with sandbox fusion for code/code_stdio tasks only.

    Args:
        parquet_path: Path to the parquet file with test data
        sample_n: Number of samples to test
        seed: Random seed for sampling
        sandbox_fusion_url: URL of the sandbox fusion service
        max_concurrent: Maximum concurrent sandbox calls
        timeout: Sandbox timeout in seconds
        verbose: Whether to print detailed output
        vllm_url: URL of the VLLM server
        vllm_model: Model name for VLLM (None = auto-detect)
        vllm_max_tokens: Max tokens for VLLM generation
        vllm_temperature: Temperature for VLLM generation
        use_vllm: Whether to use VLLM for generating solutions (vs using existing outputs)
        use_chat_api: Whether to use chat completions API (vs completions API)
    """
    from verl.utils.reward_score.sandbox_fusion import compute_score as sandbox_compute_score
    from verl.utils.reward_score.dolci_think_rl_v2 import _convert_to_sandbox_format

    # Load data
    print(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # Filter for CODE tasks only
    def get_dataset_source(row):
        extra_info = row.get("extra_info", {})
        if extra_info is None:
            return "unknown"
        if hasattr(extra_info, 'tolist'):
            extra_info = extra_info.tolist()
        return extra_info.get("dataset_source", "unknown") if isinstance(extra_info, dict) else "unknown"

    df["_dataset_source"] = df.apply(get_dataset_source, axis=1)
    df_code = df[df["_dataset_source"].isin(["code", "code_stdio"])].copy()

    print(f"Found {len(df_code)} code/code_stdio samples out of {len(df)} total")

    if len(df_code) == 0:
        print("No code samples found!")
        return {"error": "No code samples"}

    # Sample
    if sample_n < len(df_code):
        print(f"Sampling {sample_n} rows...")
        df_code = df_code.sample(n=sample_n, random_state=seed).reset_index(drop=True)
    else:
        print(f"Using all {len(df_code)} code samples...")

    # Semaphore to limit concurrent API calls
    semaphore = threading.Semaphore(max_concurrent)

    # Results tracking
    results = []

    for idx, row in df_code.iterrows():
        sample_start_time = time.time()

        result = {
            "index": idx,
            "dataset_source": row["_dataset_source"],
            "score": 0.0,
            "num_test_cases": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "error_types": [],
            "status": "unknown",
            "elapsed_time": 0.0,
            "vllm_time": 0.0,
        }

        if verbose:
            print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
            print(f"{Colors.BOLD}Sample {idx + 1}/{len(df_code)} | Source: {row['_dataset_source']}{Colors.ENDC}")
            print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")

        # Extract prompt/question
        prompt = ""
        messages = []
        try:
            prompt_data = row.get("prompt")
            if prompt_data is not None:
                if hasattr(prompt_data, 'tolist'):
                    prompt_data = prompt_data.tolist()
                if isinstance(prompt_data, list) and len(prompt_data) > 0:
                    if isinstance(prompt_data[0], dict):
                        # This is chat format - store full messages
                        messages = prompt_data
                        prompt = prompt_data[0].get("content", "")
                    else:
                        prompt = str(prompt_data[0])
                elif isinstance(prompt_data, str):
                    prompt = prompt_data
        except Exception as e:
            prompt = f"[Error extracting prompt: {e}]"

        if verbose and prompt:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}PROMPT:{Colors.ENDC}")
            prompt_display = prompt
            print(f"{Colors.DIM}{prompt_display}{Colors.ENDC}")

        # Extract ground truth
        reward_model = row.get("reward_model")
        if reward_model is None:
            result["status"] = "no_reward_model"
            results.append(result)
            continue

        gt_raw = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None

        if gt_raw is None:
            result["status"] = "no_ground_truth"
            results.append(result)
            continue

        # Parse ground truth
        try:
            if isinstance(gt_raw, str):
                gt = json.loads(gt_raw)
            else:
                gt = gt_raw
        except json.JSONDecodeError as e:
            result["status"] = f"json_decode_error: {e}"
            results.append(result)
            continue

        # Convert to sandbox format (now includes fn_name)
        test_cases = _convert_to_sandbox_format(gt)

        if not test_cases:
            result["status"] = "invalid_test_cases"
            results.append(result)
            continue

        # Show test cases info
        if verbose:
            num_inputs = len(test_cases.get("inputs", []))
            num_asserts = len(test_cases.get("assert_case", []))
            fn_name = test_cases.get("fn_name", "N/A")
            print(f"\n{Colors.YELLOW}{Colors.BOLD}TEST CASES:{Colors.ENDC} {num_inputs} inputs, {num_asserts} asserts")
            print(f"{Colors.CYAN}   Expected function name: {Colors.BOLD}{fn_name}{Colors.ENDC}")

            if test_cases.get("inputs") and test_cases["inputs"][0]:
                print(f"{Colors.DIM}First input: {str(test_cases['inputs'][0])[:-1]}...{Colors.ENDC}")
            if test_cases.get("assert_case") and test_cases["assert_case"][0]:
                print(f"{Colors.DIM}First assert: {str(test_cases['assert_case'][0])[:-1]}...{Colors.ENDC}")

        # Get solution - either from VLLM or from existing outputs
        if use_vllm:
            # Generate solution using VLLM
            if verbose:
                print(f"\n{Colors.CYAN}Generating solution with VLLM...{Colors.ENDC}")

            vllm_start = time.time()
            try:
                if use_chat_api and messages:
                    solution = get_vllm_chat_completion(
                        messages=messages,
                        vllm_url=vllm_url,
                        model=vllm_model,
                        max_tokens=vllm_max_tokens,
                        temperature=vllm_temperature,
                    )
                else:
                    solution = get_vllm_completion(
                        prompt=prompt,
                        vllm_url=vllm_url,
                        model=vllm_model,
                        max_tokens=vllm_max_tokens,
                        temperature=vllm_temperature,
                    )
                result["vllm_time"] = time.time() - vllm_start

                if verbose:
                    print(f"{Colors.GREEN}VLLM response received in {result['vllm_time']:.2f}s{Colors.ENDC}")

            except Exception as e:
                result["status"] = f"vllm_error: {e}"
                result["vllm_time"] = time.time() - vllm_start
                results.append(result)
                if verbose:
                    print(f"{Colors.RED}VLLM Error: {e}{Colors.ENDC}")
                continue
        else:
            # Use existing outputs from the parquet file
            try:
                outputs = row.get("outputs")
                if outputs is None:
                    result["status"] = "no_outputs"
                    results.append(result)
                    continue

                if hasattr(outputs, 'tolist'):
                    outputs = outputs.tolist()

                if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                    solution = outputs[0]
                elif isinstance(outputs, str):
                    solution = outputs
                else:
                    solution = str(outputs)

                if hasattr(solution, 'item'):
                    solution = solution.item()
                if not isinstance(solution, str):
                    solution = str(solution)

            except Exception as e:
                result["status"] = f"output_extraction_error: {e}"
                results.append(result)
                continue

        # Print the solution code
        if verbose:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}SOLUTION:{Colors.ENDC}")
            print_code(solution, max_lines=40, title="Model Output")

        # Count test cases
        num_cases = len(test_cases.get("inputs", []))
        result["num_test_cases"] = num_cases

        # Run sandbox
        if verbose:
            print(f"\n{Colors.CYAN}Running sandbox with {num_cases} test cases (timeout={timeout}s)...{Colors.ENDC}")

        sandbox_start = time.time()
        try:
            score, metadata_list = sandbox_compute_score(
                sandbox_fusion_url=sandbox_fusion_url,
                concurrent_semaphore=semaphore,
                memory_limit_mb=1024,
                completion=solution,
                test_cases=test_cases,
                continuous=True,
                timeout=timeout,
            )
            sandbox_elapsed = time.time() - sandbox_start

            result["score"] = score
            result["status"] = "completed"

            # Analyze metadata
            for meta in metadata_list:
                if isinstance(meta, dict):
                    status = meta.get("status", "unknown")
                    if status == "success":
                        result["passed"] += 1
                    elif status == "wrong_answer":
                        result["failed"] += 1
                    else:
                        result["errors"] += 1
                        result["error_types"].append(status)

            if verbose:
                score_color = Colors.GREEN if score > 0.5 else (Colors.YELLOW if score > 0 else Colors.RED)
                print(f"\n{Colors.BOLD}RESULT:{Colors.ENDC}")
                print(f"  Score: {score_color}{score:.4f}{Colors.ENDC}")
                print(f"  Passed: {Colors.GREEN}{result['passed']}{Colors.ENDC} | "
                      f"Failed: {Colors.RED}{result['failed']}{Colors.ENDC} | "
                      f"Errors: {Colors.YELLOW}{result['errors']}{Colors.ENDC}")
                print(f"  Sandbox Time: {sandbox_elapsed:.2f}s")

                if result["error_types"]:
                    print(f"  Error Types: {Colors.RED}{Counter(result['error_types'])}{Colors.ENDC}")

        except Exception as e:
            sandbox_elapsed = time.time() - sandbox_start
            result["status"] = f"sandbox_exception: {e}"
            result["errors"] = num_cases
            if verbose:
                print(f"{Colors.RED}Sandbox Exception: {e}{Colors.ENDC}")

        result["elapsed_time"] = time.time() - sample_start_time
        results.append(result)

        if verbose:
            print(f"{Colors.DIM}Total sample time: {result['elapsed_time']:.2f}s{Colors.ENDC}")

    # Aggregate statistics
    df_results = pd.DataFrame(results)
    completed = df_results[df_results["status"] == "completed"]

    summary = {
        "total_samples": len(df_results),
        "completed": len(completed),
        "failed_to_run": len(df_results) - len(completed),
        "failure_reasons": dict(Counter(df_results[df_results["status"] != "completed"]["status"])),
        "mean_score": completed["score"].mean() if len(completed) > 0 else 0.0,
        "median_score": completed["score"].median() if len(completed) > 0 else 0.0,
        "min_score": completed["score"].min() if len(completed) > 0 else 0.0,
        "max_score": completed["score"].max() if len(completed) > 0 else 0.0,
        "perfect_scores": (completed["score"] == 1.0).sum() if len(completed) > 0 else 0,
        "zero_scores": (completed["score"] == 0.0).sum() if len(completed) > 0 else 0,
        "total_test_cases": df_results["num_test_cases"].sum(),
        "total_passed": df_results["passed"].sum(),
        "total_failed": df_results["failed"].sum(),
        "total_errors": df_results["errors"].sum(),
        "error_types": dict(Counter([e for r in results for e in r["error_types"]])),
        "total_time": df_results["elapsed_time"].sum(),
        "avg_time_per_sample": df_results["elapsed_time"].mean(),
        "total_vllm_time": df_results["vllm_time"].sum() if "vllm_time" in df_results.columns else 0.0,
        "avg_vllm_time": df_results["vllm_time"].mean() if "vllm_time" in df_results.columns else 0.0,
    }

    # Print final summary
    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}FINAL SUMMARY{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"Total Samples:        {summary['total_samples']}")
    print(f"Completed:            {summary['completed']}")
    print(f"Failed to Run:        {summary['failed_to_run']}")
    print(f"Mean Score:           {summary['mean_score']:.4f}")
    print(f"Total Time:           {summary['total_time']:.2f}s")
    print(f"Avg Time/Sample:      {summary['avg_time_per_sample']:.2f}s")

    if use_vllm:
        print(f"Total VLLM Time:      {summary['total_vllm_time']:.2f}s")
        print(f"Avg VLLM Time:        {summary['avg_vllm_time']:.2f}s")

    if summary['error_types']:
        print(f"\nError Breakdown: {summary['error_types']}")

    return {"summary": summary, "results": results, "df_results": df_results}


def main():
    """Main function to run the test with VLLM server."""
    import argparse

    parser = argparse.ArgumentParser(description="Test code scoring with VLLM")
    parser.add_argument("--parquet-path", type=str, required=True, help="Path to parquet file")
    parser.add_argument("--model", type=str, default=None, help="HuggingFace model to serve with VLLM")
    parser.add_argument("--sample-n", type=int, default=10, help="Number of samples to test")
    parser.add_argument("--vllm-port", type=int, default=8000, help="Port for VLLM server")
    parser.add_argument("--vllm-url", type=str, default=None, help="Existing VLLM server URL (skip starting server)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max-model-len", type=int, default=8192, help="Max model length")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--timeout", type=int, default=10, help="Sandbox timeout")
    parser.add_argument("--max-concurrent", type=int, default=128, help="Max concurrent sandbox calls")
    parser.add_argument("--use-existing-outputs", action="store_true", help="Use outputs from parquet instead of VLLM")
    parser.add_argument("--no-chat", action="store_true", help="Use completions API instead of chat API")
    parser.add_argument("--sandbox-url", type=str,
                       default="http://sandbox-fusion-code-rl-service.llm-pretraining.svc.cluster.local:8080/run_code",
                       help="Sandbox fusion URL")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    vllm_process = None
    vllm_url = args.vllm_url

    try:
        # Start VLLM server if needed
        if not args.use_existing_outputs and vllm_url is None:
            if args.model is None:
                raise ValueError("--model is required when using VLLM (unless --vllm-url or --use-existing-outputs is provided)")

            vllm_process = start_vllm_server(
                model=args.model,
                port=args.vllm_port,
                tensor_parallel_size=args.tensor_parallel_size,
                max_model_len=args.max_model_len,
            )
            vllm_url = f"http://localhost:{args.vllm_port}"

        # Run the test
        result = test_code_scoring(
            parquet_path=args.parquet_path,
            sample_n=args.sample_n,
            max_concurrent=args.max_concurrent,
            timeout=args.timeout,
            verbose=not args.quiet,
            vllm_url=vllm_url or f"http://localhost:{args.vllm_port}",
            vllm_max_tokens=args.max_tokens,
            vllm_temperature=args.temperature,
            use_vllm=not args.use_existing_outputs,
            use_chat_api=not args.no_chat,
            sandbox_fusion_url=args.sandbox_url,
            seed=args.seed,
        )

        return result

    finally:
        # Clean up VLLM server
        if vllm_process is not None:
            stop_vllm_server(vllm_process)


if __name__ == "__main__":
    main()
