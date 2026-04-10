# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Prepare training data for prompt distillation.

This script generates teacher-labeled data for off-policy (SFT-based) prompt distillation.
A teacher model receives a detailed system prompt along with each user input and generates
high-quality responses. The outputs are saved as Parquet files WITHOUT the system prompt,
so the student model learns to replicate the teacher's behavior without needing the prompt.

Usage:
    python prepare_data.py \
        --model_path <teacher_model> \
        --system_prompt_file <path_to_system_prompt.txt> \
        --input_file <input_data.parquet> \
        --output_file <output_data.parquet> \
        --input_key question \
        --batch_size 64 \
        --temperature 0.7 \
        --max_new_tokens 512
"""

import argparse
import os

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Generate teacher-labeled data for prompt distillation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the teacher model")
    parser.add_argument(
        "--system_prompt_file", type=str, default=None, help="Path to a text file containing the system prompt"
    )
    parser.add_argument("--system_prompt", type=str, default=None, help="Inline system prompt string")
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to input data (Parquet or JSON/JSONL)"
    )
    parser.add_argument("--output_file", type=str, required=True, help="Path to save output Parquet file")
    parser.add_argument("--input_key", type=str, default="question", help="Column name for user inputs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of responses per input")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size for vLLM")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85, help="GPU memory utilization for vLLM")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code for model loading")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum number of input samples to process (-1 for all)")
    return parser.parse_args()


def load_system_prompt(args):
    """Load system prompt from file or inline argument."""
    if args.system_prompt_file is not None:
        with open(args.system_prompt_file) as f:
            return f.read().strip()
    elif args.system_prompt is not None:
        return args.system_prompt
    else:
        raise ValueError("Either --system_prompt_file or --system_prompt must be provided")


def load_input_data(input_file, input_key, max_samples=-1):
    """Load input data from Parquet or JSON/JSONL file."""
    if input_file.endswith(".parquet"):
        df = pd.read_parquet(input_file)
    elif input_file.endswith(".json"):
        df = pd.read_json(input_file)
    elif input_file.endswith(".jsonl"):
        df = pd.read_json(input_file, lines=True)
    else:
        raise ValueError(f"Unsupported file format: {input_file}. Use .parquet, .json, or .jsonl")

    if input_key not in df.columns:
        raise ValueError(f"Column '{input_key}' not found in input data. Available columns: {list(df.columns)}")

    if max_samples > 0 and max_samples < len(df):
        df = df.head(max_samples)

    return df


def build_teacher_prompts(tokenizer, system_prompt, user_inputs):
    """Build chat-formatted prompts with system prompt for the teacher model."""
    prompts = []
    for user_input in user_inputs:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        prompt_str = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompts.append(prompt_str)
    return prompts


def main():
    args = parse_args()

    # Load system prompt
    system_prompt = load_system_prompt(args)
    print(f"System prompt ({len(system_prompt)} chars):\n{system_prompt[:200]}...")

    # Load input data
    df = load_input_data(args.input_file, args.input_key, args.max_samples)
    user_inputs = df[args.input_key].tolist()
    print(f"Loaded {len(user_inputs)} input samples")

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    llm = LLM(
        args.model_path,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        n=args.n_samples,
    )

    # Build teacher prompts (with system prompt)
    teacher_prompts = build_teacher_prompts(tokenizer, system_prompt, user_inputs)

    # Generate in batches
    all_responses = [[] for _ in range(args.n_samples)]
    num_batches = -(-len(teacher_prompts) // args.batch_size)

    for batch_idx in range(num_batches):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, len(teacher_prompts))
        batch_prompts = teacher_prompts[start:end]

        print(f"Generating batch {batch_idx + 1}/{num_batches} ({len(batch_prompts)} samples)...")
        outputs = llm.generate(batch_prompts, sampling_params)

        for output in outputs:
            for sample_idx in range(args.n_samples):
                response_text = output.outputs[sample_idx].text
                all_responses[sample_idx].append(response_text)

    # Build output dataframe
    # The student's training data does NOT include the system prompt
    output_data = {args.input_key: user_inputs}

    if args.n_samples == 1:
        output_data["response"] = all_responses[0]
    else:
        # Store multiple responses as a list per row
        output_data["response"] = list(zip(*all_responses))

    # Preserve any extra columns from the original data
    for col in df.columns:
        if col != args.input_key and col not in output_data:
            output_data[col] = df[col].tolist()

    output_df = pd.DataFrame(output_data)

    # Save as Parquet
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    output_df.to_parquet(args.output_file, index=False)
    print(f"Saved {len(output_df)} samples to {args.output_file}")

    # Print sample
    print("\n--- Sample output ---")
    print(f"Input: {user_inputs[0][:200]}")
    if args.n_samples == 1:
        print(f"Response: {all_responses[0][0][:200]}")
    else:
        print(f"Response[0]: {all_responses[0][0][:200]}")


if __name__ == "__main__":
    main()
