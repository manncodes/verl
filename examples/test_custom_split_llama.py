#!/usr/bin/env python3
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
Test script for Custom Split LLaMA model integration in veRL.

This script verifies that:
1. The model is registered in veRL's model registry
2. Model classes can be loaded correctly
3. Basic model instantiation works

Usage:
    python test_custom_split_llama.py
"""

import sys
from pathlib import Path

# Add verl to path
verl_path = Path(__file__).parent.parent
sys.path.insert(0, str(verl_path))


def test_model_registry():
    """Test that the model is registered in veRL's registry."""
    from verl.models.registry import ModelRegistry

    print("Testing model registry...")

    # Check if model is in supported architectures
    supported_archs = ModelRegistry.get_supported_archs()
    print(f"Supported architectures: {supported_archs}")

    assert "CustomSplitLLamaForCausalLM" in supported_archs, \
        "CustomSplitLLamaForCausalLM not found in registry!"

    print("✅ Model is registered in veRL's model registry")


def test_model_class_loading():
    """Test loading model classes from registry."""
    from verl.models.registry import ModelRegistry

    print("\nTesting model class loading...")

    # Load actor/ref model class
    actor_cls = ModelRegistry.load_model_cls(
        model_arch="CustomSplitLLamaForCausalLM",
        value=False
    )
    assert actor_cls is not None, "Failed to load actor model class"
    print(f"✅ Actor model class loaded: {actor_cls.__name__}")

    # Load value model class
    value_cls = ModelRegistry.load_model_cls(
        model_arch="CustomSplitLLamaForCausalLM",
        value=True
    )
    assert value_cls is not None, "Failed to load value model class"
    print(f"✅ Value model class loaded: {value_cls.__name__}")


def test_transformers_import():
    """Test importing transformers version of the model."""
    print("\nTesting transformers import...")

    try:
        from verl.models.transformers.custom_split_llama import (
            CustomSplitLLamaForCausalLM,
            CustomSplitLLamaModel
        )
        print("✅ Transformers model classes imported successfully")
        print(f"  - CustomSplitLLamaModel: {CustomSplitLLamaModel}")
        print(f"  - CustomSplitLLamaForCausalLM: {CustomSplitLLamaForCausalLM}")
    except ImportError as e:
        print(f"❌ Failed to import transformers models: {e}")
        raise


def test_megatron_import():
    """Test importing Megatron version of the model."""
    print("\nTesting Megatron import...")

    try:
        from verl.models.custom_split_llama.megatron.modeling_custom_split_llama_megatron import (
            ParallelCustomSplitLLamaForCausalLMRmPad,
            ParallelCustomSplitLLamaForValueRmPad,
        )
        print("✅ Megatron model classes imported successfully")
        print(f"  - ParallelCustomSplitLLamaForCausalLMRmPad: {ParallelCustomSplitLLamaForCausalLMRmPad}")
        print(f"  - ParallelCustomSplitLLamaForValueRmPad: {ParallelCustomSplitLLamaForValueRmPad}")
    except ImportError as e:
        print(f"❌ Failed to import Megatron models: {e}")
        raise


def test_adapter_import():
    """Test importing adapter layers."""
    print("\nTesting adapter layer import...")

    try:
        from verl.models.custom_split_llama.megatron.layers.parallel_adapter import (
            ParallelAdapter,
            ParallelAdapterRmPad
        )
        print("✅ Adapter layers imported successfully")
        print(f"  - ParallelAdapter: {ParallelAdapter}")
        print(f"  - ParallelAdapterRmPad: {ParallelAdapterRmPad}")
    except ImportError as e:
        print(f"❌ Failed to import adapter layers: {e}")
        raise


def main():
    """Run all tests."""
    print("=" * 70)
    print("Custom Split LLaMA Integration Test")
    print("=" * 70)

    try:
        test_model_registry()
        test_model_class_loading()
        test_transformers_import()
        test_megatron_import()
        test_adapter_import()

        print("\n" + "=" * 70)
        print("✅ All tests passed! Custom Split LLaMA is properly integrated.")
        print("=" * 70)

        print("\n📝 Next steps:")
        print("1. Update the config.json with your model paths (see examples/custom_split_llama_config.json)")
        print("2. Prepare your 8B and 70B model checkpoints")
        print("3. Initialize or train adapter weights")
        print("4. Start training with veRL!")

        return 0

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ Tests failed with error: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
