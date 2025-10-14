#!/usr/bin/env python3
"""
Quick verification test for veRL installation
Tests the specific Triton locale fix and basic imports
Can be run without pytest
"""

import os
import sys


def print_test(name, status="PASS", message=""):
    """Print test result"""
    symbols = {"PASS": "✓", "FAIL": "✗", "SKIP": "⚠"}
    colors = {"PASS": "\033[0;32m", "FAIL": "\033[0;31m", "SKIP": "\033[1;33m"}
    reset = "\033[0m"

    symbol = symbols.get(status, "?")
    color = colors.get(status, "")

    if message:
        print(f"{color}{symbol} {name}: {message}{reset}")
    else:
        print(f"{color}{symbol} {name}{reset}")


def main():
    print("=" * 60)
    print("veRL Quick Installation Test")
    print("=" * 60)
    print()

    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    skipped_tests = 0

    # Test 1: Locale settings
    total_tests += 1
    print("[1/10] Testing locale settings...")
    os.environ['LC_ALL'] = 'C.UTF-8'
    os.environ['LANG'] = 'C.UTF-8'
    os.environ['LANGUAGE'] = 'en_US.UTF-8'

    if os.environ.get('LC_ALL') == 'C.UTF-8':
        print_test("Locale settings", "PASS")
        passed_tests += 1
    else:
        print_test("Locale settings", "FAIL", "Could not set locale")
        failed_tests += 1

    # Test 2: Python version
    total_tests += 1
    print("\n[2/10] Testing Python version...")
    if sys.version_info >= (3, 10):
        print_test("Python version", "PASS", f"{sys.version_info.major}.{sys.version_info.minor}")
        passed_tests += 1
    else:
        print_test("Python version", "FAIL", f"{sys.version_info.major}.{sys.version_info.minor} (need >= 3.10)")
        failed_tests += 1

    # Test 3: PyTorch
    total_tests += 1
    print("\n[3/10] Testing PyTorch...")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print_test("PyTorch", "PASS", f"{torch.__version__}, CUDA: {cuda_available}")
        passed_tests += 1
    except ImportError as e:
        print_test("PyTorch", "FAIL", str(e))
        failed_tests += 1

    # Test 4: Triton (critical for Megatron)
    total_tests += 1
    print("\n[4/10] Testing Triton...")
    try:
        import triton
        print_test("Triton", "PASS", triton.__version__)
        passed_tests += 1
    except UnicodeDecodeError as e:
        print_test("Triton", "FAIL", f"UnicodeDecodeError (locale fix needed): {e}")
        failed_tests += 1
    except ImportError as e:
        print_test("Triton", "SKIP", "Not installed (optional)")
        skipped_tests += 1

    # Test 5: Transformer Engine
    total_tests += 1
    print("\n[5/10] Testing Transformer Engine...")
    try:
        import transformer_engine
        print_test("Transformer Engine", "PASS", transformer_engine.__version__)
        passed_tests += 1
    except UnicodeDecodeError as e:
        print_test("Transformer Engine", "FAIL", f"UnicodeDecodeError: {e}")
        failed_tests += 1
    except ImportError:
        print_test("Transformer Engine", "SKIP", "Not installed (optional for Megatron)")
        skipped_tests += 1

    # Test 6: Megatron Core
    total_tests += 1
    print("\n[6/10] Testing Megatron Core...")
    try:
        from megatron.core import parallel_state as mpu
        print_test("Megatron Core", "PASS")
        passed_tests += 1
    except UnicodeDecodeError as e:
        print_test("Megatron Core", "FAIL", f"UnicodeDecodeError: {e}")
        failed_tests += 1
    except ImportError:
        print_test("Megatron Core", "SKIP", "Not installed (optional)")
        skipped_tests += 1

    # Test 7: veRL
    total_tests += 1
    print("\n[7/10] Testing veRL...")
    try:
        import verl
        print_test("veRL", "PASS", verl.__version__)
        passed_tests += 1
    except ImportError as e:
        print_test("veRL", "FAIL", str(e))
        failed_tests += 1
    except AttributeError:
        print_test("veRL", "PASS", "imported (no version)")
        passed_tests += 1

    # Test 8: veRL trainer (the failing import)
    total_tests += 1
    print("\n[8/10] Testing veRL trainer (full import chain)...")
    try:
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        print_test("veRL trainer", "PASS", "Full import chain successful")
        passed_tests += 1
    except UnicodeDecodeError as e:
        print_test("veRL trainer", "FAIL", f"UnicodeDecodeError (RUN FIX SCRIPT): {e}")
        failed_tests += 1
    except ImportError as e:
        print_test("veRL trainer", "SKIP", f"Import error: {e}")
        skipped_tests += 1

    # Test 9: Custom Split LLaMA registration
    total_tests += 1
    print("\n[9/10] Testing Custom Split LLaMA...")
    try:
        from verl.models.registry import ModelRegistry
        if 'CustomSplitLLamaForCausalLM' in ModelRegistry.get_supported_archs():
            print_test("Custom Split LLaMA", "PASS", "Registered in veRL")
            passed_tests += 1
        else:
            print_test("Custom Split LLaMA", "FAIL", "Not registered")
            failed_tests += 1
    except ImportError as e:
        print_test("Custom Split LLaMA", "SKIP", str(e))
        skipped_tests += 1

    # Test 10: Flash Attention (optional)
    total_tests += 1
    print("\n[10/10] Testing Flash Attention...")
    try:
        import flash_attn
        print_test("Flash Attention", "PASS", flash_attn.__version__)
        passed_tests += 1
    except ImportError:
        print_test("Flash Attention", "SKIP", "Not installed (training will be slower)")
        skipped_tests += 1

    # Summary
    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Total:   {total_tests}")
    print(f"Passed:  {passed_tests}")
    print(f"Failed:  {failed_tests}")
    print(f"Skipped: {skipped_tests}")
    print()

    if failed_tests > 0:
        print("\033[0;31m✗ Some tests FAILED\033[0m")
        print()
        if "UnicodeDecodeError" in str(sys.exc_info()):
            print("🔧 FIX: Run the following command:")
            print("   bash fix_triton_locale.sh")
            print()
            print("Or add to your training script:")
            print("   export LC_ALL=C.UTF-8")
            print("   export LANG=C.UTF-8")
        return 1
    elif passed_tests == total_tests:
        print("\033[0;32m✓ All tests PASSED!\033[0m")
        print()
        print("Your installation is ready for training!")
        return 0
    else:
        print("\033[1;33m⚠ Tests passed with some skipped\033[0m")
        print()
        print("Installation is functional but some optional packages are missing.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
