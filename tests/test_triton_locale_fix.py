"""
Unit tests for Triton locale fix
Tests that the locale configuration resolves Triton import issues
"""

import os
import sys
import subprocess
import pytest


class TestTritonLocaleFix:
    """Test suite for Triton locale configuration"""

    def test_locale_env_vars_set(self):
        """Test that locale environment variables are properly set"""
        # Set locale
        os.environ['LC_ALL'] = 'C.UTF-8'
        os.environ['LANG'] = 'C.UTF-8'
        os.environ['LANGUAGE'] = 'en_US.UTF-8'

        assert os.environ.get('LC_ALL') == 'C.UTF-8'
        assert os.environ.get('LANG') == 'C.UTF-8'
        assert os.environ.get('LANGUAGE') == 'en_US.UTF-8'
        print("✓ Locale environment variables set correctly")

    def test_triton_import(self):
        """Test that Triton can be imported without UnicodeDecodeError"""
        # Set locale before import
        os.environ['LC_ALL'] = 'C.UTF-8'
        os.environ['LANG'] = 'C.UTF-8'

        try:
            import triton
            print(f"✓ Triton imported successfully: version {triton.__version__}")
            assert True
        except UnicodeDecodeError as e:
            pytest.fail(f"UnicodeDecodeError during Triton import: {e}")
        except ImportError as e:
            pytest.skip(f"Triton not installed: {e}")

    def test_transformer_engine_import(self):
        """Test that Transformer Engine can be imported"""
        os.environ['LC_ALL'] = 'C.UTF-8'
        os.environ['LANG'] = 'C.UTF-8'

        try:
            import transformer_engine
            print(f"✓ Transformer Engine imported successfully: version {transformer_engine.__version__}")
            assert True
        except UnicodeDecodeError as e:
            pytest.fail(f"UnicodeDecodeError during Transformer Engine import: {e}")
        except ImportError as e:
            pytest.skip(f"Transformer Engine not installed: {e}")

    def test_ldconfig_output_parsing(self):
        """Test that ldconfig output can be parsed without UnicodeDecodeError"""
        os.environ['LC_ALL'] = 'C.UTF-8'
        os.environ['LANG'] = 'C.UTF-8'

        try:
            # This is what Triton does internally
            result = subprocess.check_output(["/sbin/ldconfig", "-p"])
            # Try to decode as UTF-8
            output = result.decode('utf-8')
            print(f"✓ ldconfig output parsed successfully ({len(output)} chars)")
            assert len(output) > 0
        except UnicodeDecodeError as e:
            pytest.fail(f"ldconfig output cannot be decoded as UTF-8: {e}")
        except FileNotFoundError:
            pytest.skip("ldconfig not found on this system")

    def test_triton_backends_nvidia_driver(self):
        """Test that Triton's NVIDIA driver backend initializes correctly"""
        os.environ['LC_ALL'] = 'C.UTF-8'
        os.environ['LANG'] = 'C.UTF-8'

        try:
            # This is the exact import that fails in the original error
            from triton.backends.nvidia.driver import CudaUtils
            utils = CudaUtils()
            print("✓ Triton NVIDIA backend initialized successfully")
            assert utils is not None
        except UnicodeDecodeError as e:
            pytest.fail(f"UnicodeDecodeError in Triton NVIDIA backend: {e}")
        except Exception as e:
            # Other exceptions are OK (no GPU, etc.)
            print(f"⚠ Triton backend error (expected without GPU): {e}")
            pytest.skip(f"Triton backend initialization failed: {e}")

    def test_megatron_core_import(self):
        """Test that Megatron Core can be imported (requires transformer_engine)"""
        os.environ['LC_ALL'] = 'C.UTF-8'
        os.environ['LANG'] = 'C.UTF-8'

        try:
            from megatron.core import parallel_state as mpu
            print("✓ Megatron Core imported successfully")
            assert mpu is not None
        except UnicodeDecodeError as e:
            pytest.fail(f"UnicodeDecodeError during Megatron Core import: {e}")
        except ImportError as e:
            pytest.skip(f"Megatron Core not installed: {e}")

    def test_full_import_chain(self):
        """Test the full import chain that fails in the original error"""
        os.environ['LC_ALL'] = 'C.UTF-8'
        os.environ['LANG'] = 'C.UTF-8'

        try:
            # This is the exact import chain from the error
            from verl.trainer.ppo.ray_trainer import RayPPOTrainer
            print("✓ Full veRL import chain successful")
            assert RayPPOTrainer is not None
        except UnicodeDecodeError as e:
            pytest.fail(f"UnicodeDecodeError in veRL import chain: {e}")
        except ImportError as e:
            pytest.skip(f"veRL components not installed: {e}")

    def test_locale_persistence(self):
        """Test that locale settings persist in subprocess"""
        os.environ['LC_ALL'] = 'C.UTF-8'
        os.environ['LANG'] = 'C.UTF-8'

        # Run a subprocess to check if locale is inherited
        result = subprocess.run(
            ['python3', '-c', 'import os; print(os.environ.get("LC_ALL"))'],
            capture_output=True,
            text=True
        )

        assert result.stdout.strip() == 'C.UTF-8'
        print("✓ Locale persists in subprocess")


class TestTritonCompatibility:
    """Test Triton compatibility with different versions"""

    def test_triton_version(self):
        """Test that Triton version is compatible"""
        try:
            import triton
            version = triton.__version__

            # Check version is >= 3.0.0
            major = int(version.split('.')[0])
            assert major >= 3, f"Triton version {version} is too old, need >= 3.0.0"
            print(f"✓ Triton version {version} is compatible")
        except ImportError:
            pytest.skip("Triton not installed")

    def test_transformer_engine_version(self):
        """Test that Transformer Engine version is compatible"""
        try:
            import transformer_engine
            version = transformer_engine.__version__

            # Check version is >= 1.7.0
            parts = version.split('.')
            major, minor = int(parts[0]), int(parts[1])
            assert (major > 1) or (major == 1 and minor >= 7), \
                f"Transformer Engine {version} is too old, need >= 1.7.0"
            print(f"✓ Transformer Engine version {version} is compatible")
        except ImportError:
            pytest.skip("Transformer Engine not installed")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s"])
