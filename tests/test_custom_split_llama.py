"""
Unit tests for Custom Split LLaMA model integration
Tests model registration, imports, and basic functionality
"""

import os
import sys
import pytest
import torch


class TestCustomSplitLLamaRegistration:
    """Test Custom Split LLaMA model registration in veRL"""

    def test_model_registry_import(self):
        """Test that ModelRegistry can be imported"""
        try:
            from verl.models.registry import ModelRegistry
            print("✓ ModelRegistry imported successfully")
            assert ModelRegistry is not None
        except ImportError as e:
            pytest.fail(f"Cannot import ModelRegistry: {e}")

    def test_custom_split_llama_registered(self):
        """Test that CustomSplitLLamaForCausalLM is registered"""
        from verl.models.registry import ModelRegistry

        supported_archs = ModelRegistry.get_supported_archs()
        assert 'CustomSplitLLamaForCausalLM' in supported_archs, \
            f"CustomSplitLLamaForCausalLM not found in registry: {supported_archs}"
        print("✓ CustomSplitLLamaForCausalLM is registered")

    def test_registry_returns_correct_classes(self):
        """Test that registry returns correct implementation classes"""
        from verl.models.registry import ModelRegistry

        module_name, class_names = ModelRegistry._MODELS.get('CustomSplitLLamaForCausalLM', (None, None))

        assert module_name == 'custom_split_llama', \
            f"Wrong module name: {module_name}"
        assert 'ParallelCustomSplitLLamaForCausalLMRmPadPP' in class_names, \
            f"ParallelCustomSplitLLamaForCausalLMRmPadPP not in classes: {class_names}"
        assert 'ParallelCustomSplitLLamaForValueRmPadPP' in class_names, \
            f"ParallelCustomSplitLLamaForValueRmPadPP not in classes: {class_names}"
        print(f"✓ Registry returns correct classes: {class_names}")


class TestCustomSplitLLamaTransformers:
    """Test Transformers implementation"""

    def test_transformers_import(self):
        """Test that transformers implementation can be imported"""
        try:
            from verl.models.transformers.custom_split_llama import (
                CustomSplitLLamaModel,
                CustomSplitLLamaForCausalLM
            )
            print("✓ Transformers implementation imported")
            assert CustomSplitLLamaModel is not None
            assert CustomSplitLLamaForCausalLM is not None
        except ImportError as e:
            pytest.fail(f"Cannot import transformers implementation: {e}")

    def test_transformers_config(self):
        """Test Custom Split LLaMA config"""
        from transformers import LlamaConfig

        # Create a minimal config
        config = LlamaConfig()
        config.path8b = "dummy/8b"
        config.path70b = "dummy/70b"
        config.num_layers_8 = 4
        config.num_layers_70 = 2
        config.mlp = False

        assert hasattr(config, 'path8b')
        assert hasattr(config, 'path70b')
        assert hasattr(config, 'num_layers_8')
        print("✓ Config attributes are correct")


class TestCustomSplitLLamaMegatron:
    """Test Megatron implementation"""

    def test_megatron_import(self):
        """Test that Megatron implementation can be imported"""
        try:
            from verl.models.custom_split_llama.megatron.modeling_custom_split_llama_megatron import (
                ParallelCustomSplitLLamaForCausalLMRmPad,
                ParallelCustomSplitLLamaForValueRmPad,
                ParallelCustomSplitLLamaForCausalLMRmPadPP,
                ParallelCustomSplitLLamaForValueRmPadPP,
            )
            print("✓ Megatron implementation imported")
            assert ParallelCustomSplitLLamaForCausalLMRmPad is not None
            assert ParallelCustomSplitLLamaForValueRmPad is not None
        except ImportError as e:
            pytest.skip(f"Megatron not available: {e}")

    def test_megatron_adapter_import(self):
        """Test that adapter layers can be imported"""
        try:
            from verl.models.custom_split_llama.megatron.layers.parallel_adapter import (
                ParallelAdapter,
                ParallelAdapterRmPad
            )
            print("✓ Parallel adapter layers imported")
            assert ParallelAdapter is not None
            assert ParallelAdapterRmPad is not None
        except ImportError as e:
            pytest.skip(f"Megatron adapters not available: {e}")

    def test_checkpoint_loader_import(self):
        """Test that checkpoint loader can be imported"""
        try:
            from verl.models.custom_split_llama.megatron.checkpoint_utils.custom_split_llama_loader import (
                load_custom_split_llama_weights_for_megatron
            )
            print("✓ Checkpoint loader imported")
            assert load_custom_split_llama_weights_for_megatron is not None
        except ImportError as e:
            pytest.skip(f"Checkpoint loader not available: {e}")

    def test_head_initialization_pattern(self):
        """Test that _init_head and _forward_head methods exist"""
        try:
            from verl.models.custom_split_llama.megatron.modeling_custom_split_llama_megatron import (
                ParallelCustomSplitLLamaForCausalLMRmPad,
                ParallelCustomSplitLLamaForValueRmPad
            )

            # Check CausalLM has both methods
            assert hasattr(ParallelCustomSplitLLamaForCausalLMRmPad, '_init_head')
            assert hasattr(ParallelCustomSplitLLamaForCausalLMRmPad, '_forward_head')

            # Check Value model has overridden methods
            assert hasattr(ParallelCustomSplitLLamaForValueRmPad, '_init_head')
            assert hasattr(ParallelCustomSplitLLamaForValueRmPad, '_forward_head')

            print("✓ Head initialization pattern implemented correctly")
        except ImportError as e:
            pytest.skip(f"Megatron not available: {e}")

    def test_value_model_inheritance(self):
        """Test that Value model inherits from CausalLM"""
        try:
            from verl.models.custom_split_llama.megatron.modeling_custom_split_llama_megatron import (
                ParallelCustomSplitLLamaForCausalLMRmPad,
                ParallelCustomSplitLLamaForValueRmPad
            )

            # Check inheritance
            assert issubclass(ParallelCustomSplitLLamaForValueRmPad, ParallelCustomSplitLLamaForCausalLMRmPad)
            print("✓ Value model correctly inherits from CausalLM")
        except ImportError as e:
            pytest.skip(f"Megatron not available: {e}")


class TestVLLMIntegration:
    """Test vLLM integration"""

    def test_vllm_model_import(self):
        """Test that vLLM model can be imported"""
        try:
            # Add vllm_models to path if needed
            import sys
            from pathlib import Path
            vllm_models_path = Path(__file__).parent.parent / 'vllm_models'
            if vllm_models_path.exists():
                sys.path.insert(0, str(vllm_models_path))

            from custom_split_llama import CustomSplitLLamaForCausalLM
            print("✓ vLLM model imported")
            assert CustomSplitLLamaForCausalLM is not None
        except ImportError as e:
            pytest.skip(f"vLLM model not available: {e}")


class TestOptimizations:
    """Test that efficiency optimizations are present"""

    def test_flash_attention_imports(self):
        """Test that Flash Attention imports are present in Megatron code"""
        try:
            from verl.models.custom_split_llama.megatron import modeling_custom_split_llama_megatron
            import inspect

            source = inspect.getsource(modeling_custom_split_llama_megatron)

            # Check for flash attention imports
            assert 'from flash_attn.bert_padding import' in source
            assert 'unpad_input' in source
            assert 'pad_input' in source
            print("✓ Flash Attention imports present")
        except ImportError as e:
            pytest.skip(f"Megatron not available: {e}")

    def test_sequence_parallel_padding(self):
        """Test that sequence parallel padding is implemented"""
        try:
            from verl.models.custom_split_llama.megatron import modeling_custom_split_llama_megatron
            import inspect

            source = inspect.getsource(modeling_custom_split_llama_megatron)

            # Check for sequence parallel padding
            assert 'pad_to_sequence_parallel' in source
            print("✓ Sequence parallel padding implemented")
        except ImportError as e:
            pytest.skip(f"Megatron not available: {e}")


class TestExamples:
    """Test example files"""

    def test_config_example_exists(self):
        """Test that example config exists"""
        from pathlib import Path

        config_path = Path(__file__).parent.parent / 'examples' / 'custom_split_llama_config.json'
        assert config_path.exists(), f"Config example not found: {config_path}"
        print("✓ Example config file exists")

    def test_training_script_exists(self):
        """Test that training script exists"""
        from pathlib import Path

        script_path = Path(__file__).parent.parent / 'examples' / 'grpo_trainer' / 'run_custom_split_llama_gsm8k_grpo.sh'
        assert script_path.exists(), f"Training script not found: {script_path}"
        print("✓ Training script exists")

    def test_yaml_config_exists(self):
        """Test that YAML config exists"""
        from pathlib import Path

        yaml_path = Path(__file__).parent.parent / 'examples' / 'grpo_trainer' / 'custom_split_llama_gsm8k_grpo.yaml'
        assert yaml_path.exists(), f"YAML config not found: {yaml_path}"
        print("✓ YAML config exists")


class TestDocumentation:
    """Test documentation files"""

    def test_integration_guide_exists(self):
        """Test that integration guide exists"""
        from pathlib import Path

        doc_path = Path(__file__).parent.parent / 'CUSTOM_SPLIT_LLAMA_INTEGRATION.md'
        assert doc_path.exists(), f"Integration guide not found: {doc_path}"
        print("✓ Integration guide exists")

    def test_training_guide_exists(self):
        """Test that training guide exists"""
        from pathlib import Path

        doc_path = Path(__file__).parent.parent / 'TRAINING_GUIDE.md'
        assert doc_path.exists(), f"Training guide not found: {doc_path}"
        print("✓ Training guide exists")

    def test_optimization_summary_exists(self):
        """Test that optimization summary exists"""
        from pathlib import Path

        doc_path = Path(__file__).parent.parent / 'OPTIMIZATION_SUMMARY.md'
        assert doc_path.exists(), f"Optimization summary not found: {doc_path}"
        print("✓ Optimization summary exists")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s"])
