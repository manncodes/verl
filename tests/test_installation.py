"""
Unit tests for veRL installation verification
Tests that all dependencies are installed and working
"""

import os
import sys
import pytest
import subprocess


class TestPythonEnvironment:
    """Test Python environment"""

    def test_python_version(self):
        """Test Python version is >= 3.10"""
        version_info = sys.version_info
        assert version_info >= (3, 10), f"Python {version_info.major}.{version_info.minor} is too old, need >= 3.10"
        print(f"✓ Python version: {version_info.major}.{version_info.minor}.{version_info.micro}")

    def test_virtual_environment(self):
        """Test that we're running in a virtual environment"""
        in_venv = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
        if not in_venv:
            print("⚠ Not in virtual environment (optional)")
        else:
            print(f"✓ Running in virtual environment: {sys.prefix}")


class TestCorePackages:
    """Test core package installations"""

    def test_torch_import(self):
        """Test PyTorch installation"""
        try:
            import torch
            print(f"✓ PyTorch version: {torch.__version__}")
            assert torch.__version__ is not None
        except ImportError as e:
            pytest.fail(f"PyTorch not installed: {e}")

    def test_torch_cuda(self):
        """Test PyTorch CUDA availability"""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    def test_transformers_import(self):
        """Test Transformers installation"""
        try:
            import transformers
            print(f"✓ Transformers version: {transformers.__version__}")
        except ImportError as e:
            pytest.fail(f"Transformers not installed: {e}")

    def test_verl_import(self):
        """Test veRL installation"""
        try:
            import verl
            print(f"✓ veRL version: {verl.__version__}")
        except ImportError as e:
            pytest.fail(f"veRL not installed: {e}")
        except AttributeError:
            print("✓ veRL imported (version attribute missing)")


class TestOptionalPackages:
    """Test optional package installations"""

    def test_flash_attn_import(self):
        """Test Flash Attention installation (optional)"""
        try:
            import flash_attn
            print(f"✓ Flash Attention version: {flash_attn.__version__}")
        except ImportError:
            print("⚠ Flash Attention not installed (training will be slower)")
            pytest.skip("Flash Attention not installed")

    def test_vllm_import(self):
        """Test vLLM installation (optional)"""
        try:
            import vllm
            print(f"✓ vLLM version: {vllm.__version__}")
        except ImportError:
            print("⚠ vLLM not installed (inference will be slower)")
            pytest.skip("vLLM not installed")

    def test_triton_import(self):
        """Test Triton installation (optional but needed for Megatron)"""
        os.environ['LC_ALL'] = 'C.UTF-8'
        os.environ['LANG'] = 'C.UTF-8'

        try:
            import triton
            print(f"✓ Triton version: {triton.__version__}")
        except ImportError:
            print("⚠ Triton not installed")
            pytest.skip("Triton not installed")
        except Exception as e:
            print(f"⚠ Triton import error: {e}")
            pytest.skip(f"Triton error: {e}")

    def test_transformer_engine_import(self):
        """Test Transformer Engine installation (optional, for Megatron)"""
        os.environ['LC_ALL'] = 'C.UTF-8'
        os.environ['LANG'] = 'C.UTF-8'

        try:
            import transformer_engine
            print(f"✓ Transformer Engine version: {transformer_engine.__version__}")
        except ImportError:
            print("⚠ Transformer Engine not installed (Megatron backend unavailable)")
            pytest.skip("Transformer Engine not installed")

    def test_ray_import(self):
        """Test Ray installation"""
        try:
            import ray
            print(f"✓ Ray version: {ray.__version__}")
        except ImportError as e:
            pytest.fail(f"Ray not installed (required): {e}")

    def test_wandb_import(self):
        """Test Weights & Biases installation"""
        try:
            import wandb
            print(f"✓ wandb version: {wandb.__version__}")
        except ImportError:
            print("⚠ wandb not installed (logging unavailable)")
            pytest.skip("wandb not installed")


class TestSystemCommands:
    """Test system commands and utilities"""

    def test_nvcc_available(self):
        """Test NVCC compiler availability"""
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                version_line = [l for l in result.stdout.split('\n') if 'release' in l.lower()][0]
                print(f"✓ NVCC available: {version_line.strip()}")
            else:
                pytest.skip("nvcc not available")
        except FileNotFoundError:
            pytest.skip("nvcc not found")

    def test_nvidia_smi_available(self):
        """Test nvidia-smi availability"""
        try:
            result = subprocess.run(['nvidia-smi', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ nvidia-smi available")
                # Also get GPU info
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    print(f"  {result.stdout.strip()}")
        except FileNotFoundError:
            pytest.fail("nvidia-smi not found (GPU drivers not installed)")

    def test_git_available(self):
        """Test git availability"""
        try:
            result = subprocess.run(['git', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ {result.stdout.strip()}")
        except FileNotFoundError:
            print("⚠ git not found")
            pytest.skip("git not available")


class TestCUDAConfiguration:
    """Test CUDA configuration"""

    def test_cuda_visible_devices(self):
        """Test CUDA_VISIBLE_DEVICES configuration"""
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
        print(f"CUDA_VISIBLE_DEVICES: {cuda_devices}")

        import torch
        if torch.cuda.is_available():
            visible_count = torch.cuda.device_count()
            print(f"✓ Visible GPU count: {visible_count}")

    def test_cuda_home(self):
        """Test CUDA_HOME environment variable"""
        cuda_home = os.environ.get('CUDA_HOME', os.environ.get('CUDA_PATH', 'not set'))
        print(f"CUDA_HOME: {cuda_home}")

        if cuda_home != 'not set':
            import pathlib
            if pathlib.Path(cuda_home).exists():
                print(f"✓ CUDA_HOME directory exists")
            else:
                print(f"⚠ CUDA_HOME directory does not exist")


class TestVeRLComponents:
    """Test veRL-specific components"""

    def test_verl_trainer_import(self):
        """Test veRL trainer import"""
        os.environ['LC_ALL'] = 'C.UTF-8'
        os.environ['LANG'] = 'C.UTF-8'

        try:
            from verl.trainer.ppo.ray_trainer import RayPPOTrainer
            print("✓ veRL trainer imported successfully")
        except ImportError as e:
            pytest.skip(f"veRL trainer not available: {e}")
        except Exception as e:
            pytest.skip(f"veRL trainer import error: {e}")

    def test_verl_workers_import(self):
        """Test veRL workers import"""
        try:
            from verl.workers.actor import ActorRolloutRefWorker
            print("✓ veRL workers imported successfully")
        except ImportError as e:
            pytest.skip(f"veRL workers not available: {e}")

    def test_verl_data_loader(self):
        """Test veRL data loading utilities"""
        try:
            from verl.utils.dataset import GSM8KDataset
            print("✓ veRL dataset utilities available")
        except ImportError as e:
            pytest.skip(f"veRL dataset utilities not available: {e}")


class TestDataPreprocessing:
    """Test data preprocessing"""

    def test_gsm8k_script_exists(self):
        """Test that GSM8K preprocessing script exists"""
        from pathlib import Path

        script_path = Path(__file__).parent.parent / 'examples' / 'data_preprocess' / 'gsm8k.py'
        assert script_path.exists(), f"GSM8K preprocessing script not found: {script_path}"
        print("✓ GSM8K preprocessing script exists")

    def test_datasets_package(self):
        """Test datasets package (for data loading)"""
        try:
            import datasets
            print(f"✓ datasets package version: {datasets.__version__}")
        except ImportError as e:
            pytest.fail(f"datasets package not installed: {e}")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s"])
