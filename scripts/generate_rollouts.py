#!/usr/bin/env python3
"""
Simple Rollout Generation Script

Just generate responses - no reward calculation!
Save rollouts to disk for later analysis.

Usage:
    python scripts/generate_rollouts.py \
        --model_path <path> \
        --data_path <path> \
        --output_path ./rollouts.parquet \
        --num_samples 8 \
        --use_vllm \
        --vllm_base_url http://localhost:8000/v1
"""

import argparse
import asyncio
import json
import logging
from os import environ, getenv
from pathlib import Path
from typing import List

import pandas as pd
import torch
from openai import AsyncOpenAI
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class VLLMRolloutGenerator:
    """Generate rollouts using VLLM server (async)."""

    def __init__(self, base_url: str, model_name: str, tokenizer, max_concurrent: int = 100):
        # Remove proxy vars
        for env_var in ["https_proxy", "http_proxy", "HTTPS_PROXY", "HTTP_PROXY"]:
            if getenv(env_var):
                del environ[env_var]

        self.base_url = base_url
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.client = AsyncOpenAI(api_key="EMPTY", base_url=base_url)
        self.max_concurrent = max_concurrent
        log.info(f"Initialized VLLM client: {base_url}, max_concurrent={max_concurrent}")

    async def _generate_one(self, prompt: str, temperature: float, top_p: float, max_tokens: int) -> str:
        """Generate one response."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            log.error(f"Generation error: {e}")
            return ""

    async def _generate_batch_async(self, prompts: List[str], temperature: float,
                                   top_p: float, max_tokens: int) -> List[str]:
        """Generate all prompts concurrently."""
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def generate_with_sem(prompt: str) -> str:
            async with semaphore:
                return await self._generate_one(prompt, temperature, top_p, max_tokens)

        tasks = [generate_with_sem(p) for p in prompts]
        return await asyncio.gather(*tasks)

    def generate_batch(self, prompts: List[str], temperature: float = 0.7,
                      top_p: float = 0.9, max_tokens: int = 512) -> List[str]:
        """Generate responses for batch of prompts."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._generate_batch_async(prompts, temperature, top_p, max_tokens)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self._generate_batch_async(prompts, temperature, top_p, max_tokens)
                )
        except RuntimeError:
            return asyncio.run(
                self._generate_batch_async(prompts, temperature, top_p, max_tokens)
            )


class HFRolloutGenerator:
    """Generate rollouts using HuggingFace model."""

    def __init__(self, model_path: str, tokenizer):
        self.tokenizer = tokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"

        log.info(f"Loading model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        if device != "cuda":
            self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        log.info(f"Model loaded on {device}")

    def generate_batch(self, prompts: List[str], temperature: float = 0.7,
                      top_p: float = 0.9, max_tokens: int = 512) -> List[str]:
        """Generate responses for batch of prompts."""
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_length = inputs['input_ids'].shape[1]
        responses = []
        for output in outputs:
            response = self.tokenizer.decode(output[prompt_length:], skip_special_tokens=True)
            responses.append(response)

        return responses


def load_data(data_path: str, prompt_key: str = "prompt") -> pd.DataFrame:
    """Load data from parquet file."""
    log.info(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    log.info(f"Loaded {len(df)} samples")
    return df


def extract_prompts(df: pd.DataFrame, prompt_key: str = "prompt") -> List[str]:
    """Extract text prompts from dataframe."""
    prompts = []

    for _, row in df.iterrows():
        prompt_data = row.get(prompt_key, row.get("input", ""))

        # Handle different prompt formats
        if isinstance(prompt_data, list):
            # Chat format
            prompt = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                               for msg in prompt_data])
        else:
            prompt = str(prompt_data)

        prompts.append(prompt)

    return prompts


def generate_rollouts(args):
    """Main rollout generation logic."""
    # Load tokenizer
    log.info(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize generator
    if args.use_vllm:
        generator = VLLMRolloutGenerator(
            base_url=args.vllm_base_url,
            model_name=args.vllm_model_name or args.model_path,
            tokenizer=tokenizer,
            max_concurrent=args.vllm_max_concurrent,
        )
    else:
        generator = HFRolloutGenerator(
            model_path=args.model_path,
            tokenizer=tokenizer,
        )

    # Load data
    df = load_data(args.data_path, args.prompt_key)
    prompts = extract_prompts(df, args.prompt_key)

    # Generate rollouts
    log.info(f"Generating {args.num_samples} responses per prompt...")

    all_results = []
    batch_size = args.batch_size

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating rollouts"):
        batch_prompts = prompts[i:i + batch_size]
        batch_indices = list(range(i, min(i + batch_size, len(prompts))))

        # Generate num_samples responses for each prompt
        for sample_idx in range(args.num_samples):
            responses = generator.generate_batch(
                batch_prompts,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )

            # Store results
            for prompt, response, data_idx in zip(batch_prompts, responses, batch_indices):
                result = {
                    "prompt": prompt,
                    "response": response,
                    "sample_idx": sample_idx,
                    "data_idx": data_idx,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                }

                # Add original data fields
                for col in df.columns:
                    if col not in ["prompt", "response"]:
                        result[f"original_{col}"] = df.iloc[data_idx][col]

                all_results.append(result)

    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(all_results)

    if args.output_format == "parquet":
        results_df.to_parquet(output_path, index=False)
        log.info(f"Saved {len(results_df)} rollouts to {output_path}")
    elif args.output_format == "jsonl":
        with open(output_path, "w") as f:
            for _, row in results_df.iterrows():
                f.write(json.dumps(row.to_dict()) + "\n")
        log.info(f"Saved {len(results_df)} rollouts to {output_path}")

    # Print summary
    log.info("\n" + "="*80)
    log.info("ROLLOUT GENERATION COMPLETE")
    log.info("="*80)
    log.info(f"Total prompts: {len(prompts)}")
    log.info(f"Samples per prompt: {args.num_samples}")
    log.info(f"Total rollouts: {len(results_df)}")
    log.info(f"Output: {output_path}")
    log.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Generate rollouts (prompts + responses) without reward calculation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model args
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model or model name")
    parser.add_argument("--trust_remote_code", action="store_true",
                       help="Trust remote code in model")

    # VLLM args
    parser.add_argument("--use_vllm", action="store_true",
                       help="Use VLLM server for generation")
    parser.add_argument("--vllm_base_url", type=str, default="http://localhost:8000/v1",
                       help="VLLM server URL")
    parser.add_argument("--vllm_model_name", type=str, default=None,
                       help="Model name on VLLM server")
    parser.add_argument("--vllm_max_concurrent", type=int, default=100,
                       help="Max concurrent requests to VLLM")

    # Data args
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to input data (parquet)")
    parser.add_argument("--prompt_key", type=str, default="prompt",
                       help="Key for prompt in data")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for generation")

    # Generation args
    parser.add_argument("--num_samples", type=int, default=8,
                       help="Number of responses per prompt")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Nucleus sampling parameter")
    parser.add_argument("--max_tokens", type=int, default=512,
                       help="Max tokens to generate")

    # Output args
    parser.add_argument("--output_path", type=str, default="./rollouts.parquet",
                       help="Output path for rollouts")
    parser.add_argument("--output_format", type=str, default="parquet",
                       choices=["parquet", "jsonl"],
                       help="Output format")

    args = parser.parse_args()
    generate_rollouts(args)


if __name__ == "__main__":
    main()
