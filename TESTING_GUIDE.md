# veRL Testing Guide

Complete guide for testing veRL installation, Custom Split LLaMA integration, and the Triton locale fix.

## Quick Start - Test Your Installation

### Fastest: Quick Test (No dependencies)

```bash
python3 quick_test.py
```

**Duration:** ~10 seconds
**No pytest required!**

This runs 10 essential tests and tells you exactly what's working and what needs fixing.

### Comprehensive: Full Test Suite

```bash
bash run_tests.sh
```

**Duration:** ~1-2 minutes
**Requires:** pytest

Runs all test modules and generates detailed reports.

## Test Files

### 1. Quick Test Script (`quick_test.py`)

**Standalone verification script - no pytest needed**

Tests:
- ✅ Locale configuration (C.UTF-8)
- ✅ Python version (>= 3.10)
- ✅ PyTorch + CUDA
- ✅ Triton (locale fix verification)
- ✅ Transformer Engine
- ✅ Megatron Core
- ✅ veRL installation
- ✅ veRL trainer (full import chain)
- ✅ Custom Split LLaMA registration
- ✅ Flash Attention

**Run:**
```bash
python3 quick_test.py
```

**Example output when successful:**
```
============================================================
veRL Quick Installation Test
============================================================

[1/10] Testing locale settings...
✓ Locale settings

[2/10] Testing Python version...
✓ Python version: 3.10

...

[10/10] Testing Flash Attention...
✓ Flash Attention: 2.5.0

============================================================
Test Summary
============================================================
Total:   10
Passed:  10
Failed:  0
Skipped: 0

✓ All tests PASSED!
```

**Example output with locale issue:**
```
[8/10] Testing veRL trainer (full import chain)...
✗ veRL trainer: UnicodeDecodeError (RUN FIX SCRIPT)

🔧 FIX: Run the following command:
   bash fix_triton_locale.sh
```

### 2. Triton Locale Fix Tests (`tests/test_triton_locale_fix.py`)

**Unit tests for the specific UnicodeDecodeError fix**

Tests:
- Locale environment variables are set correctly
- Triton can be imported without UnicodeDecodeError
- Transformer Engine can be imported
- ldconfig output can be parsed
- Triton NVIDIA backend initializes
- Megatron Core can be imported
- Full veRL import chain works
- Package versions are compatible

**Run:**
```bash
pytest tests/test_triton_locale_fix.py -v -s
```

**Key test (the one that was failing):**
```python
def test_full_import_chain(self):
    """Test the full import chain that fails in the original error"""
    os.environ['LC_ALL'] = 'C.UTF-8'
    os.environ['LANG'] = 'C.UTF-8'

    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    # This import would fail with UnicodeDecodeError before the fix
```

### 3. Custom Split LLaMA Tests (`tests/test_custom_split_llama.py`)

**Tests for Custom Split LLaMA integration**

Test categories:
- Model registration in veRL
- Transformers implementation
- Megatron implementation
- Adapter layers
- Checkpoint loader
- Head initialization pattern (_init_head/_forward_head)
- Value model inheritance
- vLLM integration
- Optimization implementations (Flash Attention, sequence parallel)
- Example files
- Documentation

**Run:**
```bash
pytest tests/test_custom_split_llama.py -v -s
```

### 4. Installation Tests (`tests/test_installation.py`)

**Comprehensive environment verification**

Test categories:
- Python environment
- Core packages (PyTorch, transformers, veRL, ray)
- Optional packages (Flash Attention, vLLM, triton, transformer_engine)
- System commands (nvcc, nvidia-smi, git)
- CUDA configuration
- veRL components
- Data preprocessing utilities

**Run:**
```bash
pytest tests/test_installation.py -v -s
```

## Test Runner Script (`run_tests.sh`)

**Automated test runner with reporting**

Runs all test modules in sequence:
1. Installation tests
2. Triton locale fix tests
3. Custom Split LLaMA tests
4. Full test suite with coverage

Creates timestamped output directory with logs:
```
test_results_20251013_210000/
├── installation_tests.log
├── triton_tests.log
├── custom_split_llama_tests.log
├── all_tests.log
└── report.html (if pytest-html installed)
```

**Run:**
```bash
bash run_tests.sh
```

**Example output:**
```
================================================
veRL Installation and Integration Tests
================================================

[1/4] Running Installation Tests
======================================
...
✓ Installation Tests: PASSED

[2/4] Running Triton Locale Fix Tests
======================================
...
✓ Triton Locale Tests: PASSED

[3/4] Running Custom Split LLaMA Tests
======================================
...
✓ Custom Split LLaMA Tests: PASSED

[4/4] Running All Tests with Coverage
======================================
...
✓ All Tests: PASSED

Test logs saved to: test_results_20251013_210000/

Results: 4 passed, 0 failed
```

## Fix Script (`fix_triton_locale.sh`)

**Automated fix for Triton UnicodeDecodeError**

What it does:
1. Sets locale environment variables (C.UTF-8)
2. Makes locale settings permanent (~/.bashrc)
3. Reinstalls transformer-engine and triton with compatible versions
4. Verifies the fix

**Run:**
```bash
bash fix_triton_locale.sh
```

**Expected output:**
```
=== Fixing Triton Locale Issue ===

Step 1: Setting locale environment variables...
✓ Locale variables set

Step 2: Reinstalling transformer-engine and triton...
✓ Packages reinstalled

Step 3: Verifying installation...
✓ Triton imported successfully
✓ Transformer Engine imported successfully

=== Fix Applied Successfully ===

Now run your training script with:
  export LC_ALL=C.UTF-8
  export LANG=C.UTF-8
  bash grpo.sh
```

## Common Test Scenarios

### Scenario 1: Fresh Installation

```bash
# 1. Install veRL
bash install_verl_pods.sh

# 2. Quick verification
python3 quick_test.py

# 3. If locale error appears:
bash fix_triton_locale.sh

# 4. Verify fix
python3 quick_test.py
```

### Scenario 2: Debugging Training Failure

```bash
# 1. Run quick test
python3 quick_test.py

# 2. If UnicodeDecodeError:
bash fix_triton_locale.sh

# 3. If other errors, run detailed tests
bash run_tests.sh

# 4. Check specific test output
cat test_results_*/installation_tests.log
```

### Scenario 3: CI/CD Pipeline

```yaml
# .github/workflows/test.yml
- name: Run Tests
  run: |
    export LC_ALL=C.UTF-8
    export LANG=C.UTF-8
    bash run_tests.sh
```

### Scenario 4: Development

```bash
# Run tests on file save
pytest tests/ --watch

# Run specific test during development
pytest tests/test_custom_split_llama.py::TestCustomSplitLLamaRegistration::test_model_registry_import -v

# Run with debugger
pytest tests/ --pdb
```

## Troubleshooting

### Issue: UnicodeDecodeError during tests

**Solution:**
```bash
# Run the fix script
bash fix_triton_locale.sh

# Or manually:
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
pytest tests/
```

### Issue: pytest not found

**Solution:**
```bash
pip install pytest pytest-xdist
```

### Issue: Tests skip with "not installed"

This is normal for optional packages. Example:
```
⚠ Flash Attention: SKIP (Not installed, training will be slower)
```

This means the package isn't critical but recommended.

### Issue: CUDA tests fail

If you don't have GPU:
```bash
# Run only CPU tests
pytest tests/ -k "not cuda and not gpu"
```

### Issue: Import errors in tests

```bash
# Ensure you're in virtual environment
source .venv/bin/activate

# Ensure veRL is installed
pip install -e .

# Re-run tests
python3 quick_test.py
```

## Test Reports

### Viewing HTML Reports

If pytest-html is installed:
```bash
# Install
pip install pytest-html

# Run tests with HTML report
bash run_tests.sh

# Open report
open test_results_*/report.html
```

### Coverage Reports

```bash
# Install coverage
pip install pytest-cov

# Run with coverage
pytest tests/ --cov=verl --cov-report=html

# View
open htmlcov/index.html
```

## Running Tests Before Training

**Recommended workflow:**

```bash
# 1. Quick verification (10 seconds)
python3 quick_test.py

# If all passed:
# 2. Run training
bash examples/grpo_trainer/run_custom_split_llama_gsm8k_grpo.sh /path/to/model

# If tests failed:
# 2. Fix issues
bash fix_triton_locale.sh

# 3. Re-test
python3 quick_test.py

# 4. Run training
bash examples/grpo_trainer/run_custom_split_llama_gsm8k_grpo.sh /path/to/model
```

## Test Maintenance

### Adding New Tests

1. Create test file in `tests/` directory:
```python
# tests/test_my_feature.py
import pytest

class TestMyFeature:
    def test_basic_functionality(self):
        # Test code
        assert True
```

2. Run the new test:
```bash
pytest tests/test_my_feature.py -v
```

3. Add to test runner if needed (run_tests.sh already runs all tests)

### Updating Tests After Changes

```bash
# Run all tests
bash run_tests.sh

# Fix any failures
# Update tests as needed
# Re-run to verify
```

## Integration with Training Scripts

Add to the top of your training script (e.g., grpo.sh):

```bash
#!/bin/bash
# Locale fix for Triton
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export LANGUAGE=en_US.UTF-8

# Quick verification before training
echo "Running quick verification..."
python3 quick_test.py || exit 1

# Continue with training...
python3 -m verl.trainer.main_ppo ...
```

## Performance Benchmarking

Time tests to identify slow operations:

```bash
# Show 10 slowest tests
pytest tests/ --durations=10

# Profile specific test
pytest tests/test_installation.py --profile
```

## References

- [Triton Documentation](https://github.com/openai/triton)
- [PyTest Documentation](https://docs.pytest.org/)
- [veRL Documentation](https://verl.readthedocs.io/)
- [Custom Split LLaMA Guide](CUSTOM_SPLIT_LLAMA_INTEGRATION.md)
- [Installation Guide for Pods](PODS_INSTALLATION.md)

## Support

If tests continue to fail after following this guide:

1. Run `python3 quick_test.py` and save output
2. Run `bash run_tests.sh` and check logs in `test_results_*/`
3. Collect system information:
   ```bash
   python3 --version
   nvcc --version
   nvidia-smi
   cat /etc/os-release
   ```
4. Create issue with all logs and information

---

**Quick Commands Summary:**

```bash
# Quick test (10 seconds)
python3 quick_test.py

# Fix locale issue
bash fix_triton_locale.sh

# Full test suite
bash run_tests.sh

# Specific test module
pytest tests/test_triton_locale_fix.py -v

# Test with your script
export LC_ALL=C.UTF-8 && export LANG=C.UTF-8 && bash grpo.sh
```
