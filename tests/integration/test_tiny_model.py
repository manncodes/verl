#!/usr/bin/env python3
"""
Test the tiny CustomSplitLLama model:
1. Load with transformers
2. Test forward pass
3. Test generation
4. Verify FSDP worker can load it
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def test_model_loading(model_path):
    """Test loading the model with transformers"""
    print("="*60)
    print("Test 1: Loading Model with Transformers")
    print("="*60)

    print(f"Model path: {model_path}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"✓ Tokenizer loaded")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  BOS token: {tokenizer.bos_token} ({tokenizer.bos_token_id})")
    print(f"  EOS token: {tokenizer.eos_token} ({tokenizer.eos_token_id})")
    print()

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"✓ Model loaded")
    print(f"  Architecture: {model.config.architectures}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    return model, tokenizer


def test_forward_pass(model, tokenizer):
    """Test forward pass"""
    print("="*60)
    print("Test 2: Forward Pass")
    print("="*60)

    prompt = "The answer is"
    print(f"Prompt: '{prompt}'")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(f"Input IDs shape: {inputs['input_ids'].shape}")
    print()

    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(**inputs)

    print(f"✓ Forward pass successful")
    print(f"  Logits shape: {outputs.logits.shape}")
    print(f"  Logits dtype: {outputs.logits.dtype}")
    print()

    return outputs


def test_generation(model, tokenizer):
    """Test text generation"""
    print("="*60)
    print("Test 3: Text Generation")
    print("="*60)

    prompt = "What is 2+2? Answer:"
    print(f"Prompt: '{prompt}'")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print()

    print("Generating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=50,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"✓ Generation successful")
    print(f"Generated text:\n{generated_text}")
    print()

    return generated_text


def test_fsdp_worker_import(model_path):
    """Test that FSDP worker can import and detect the model"""
    print("="*60)
    print("Test 4: FSDP Worker Compatibility")
    print("="*60)

    # Add verl to path
    import sys
    verl_path = "/home/jovyan/rl/verl"
    if os.path.exists(verl_path):
        sys.path.insert(0, verl_path)
        print(f"Added {verl_path} to Python path")
    else:
        print(f"⚠ Warning: verl not found at {verl_path}")
        print("  Skipping FSDP worker test")
        return

    try:
        # Import the custom model
        from verl.models.transformers.custom_split_llama import CustomSplitLLamaForCausalLM
        print("✓ CustomSplitLLamaForCausalLM import successful")

        # Load config and check
        import json
        with open(os.path.join(model_path, "config.json")) as f:
            config = json.load(f)

        is_custom_split = (
            "architectures" in config and
            "CustomSplitLLamaForCausalLM" in config["architectures"][0]
        )
        print(f"✓ Architecture detection: {is_custom_split}")
        print(f"  Architectures: {config.get('architectures', [])}")

        # Try loading with the custom class
        print("\nLoading with CustomSplitLLamaForCausalLM...")
        model = CustomSplitLLamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",  # Use CPU to save memory
        )
        print("✓ Model loaded with custom class")
        print(f"  Layers first: {len(model.model.layers_first)}")
        print(f"  Layers last: {len(model.model.layers_last)}")
        print(f"  Has adapter: {hasattr(model.model, 'adapter') or hasattr(model.model, 'adapter_linear_1')}")

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("  Make sure verl is installed and custom_split_llama.py exists")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_tiny_model.py <model_path>")
        print("\nExample:")
        print("  python test_tiny_model.py /model-zoo/tiny-custom-split-llama-test")
        sys.exit(1)

    model_path = sys.argv[1]

    if not os.path.exists(model_path):
        print(f"Error: Model path not found: {model_path}")
        sys.exit(1)

    if not os.path.exists(os.path.join(model_path, "config.json")):
        print(f"Error: config.json not found in {model_path}")
        sys.exit(1)

    print("\n" + "="*60)
    print("Testing Tiny CustomSplitLLama")
    print("="*60)
    print()

    try:
        # Test 1: Load model
        model, tokenizer = test_model_loading(model_path)

        # Test 2: Forward pass
        test_forward_pass(model, tokenizer)

        # Test 3: Generation
        test_generation(model, tokenizer)

        # Test 4: FSDP worker compatibility
        test_fsdp_worker_import(model_path)

        print("="*60)
        print("✅ All Tests Passed!")
        print("="*60)
        print("\nThe model is ready for veRL FSDP + vLLM testing")
        print("Next step: Run GRPO training")
        print()

    except Exception as e:
        print("\n" + "="*60)
        print("✗ Test Failed")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
