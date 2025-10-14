#!/bin/bash
# Comprehensive test runner for veRL installation and Custom Split LLaMA
# Run all tests and generate detailed report

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}veRL Installation and Integration Tests${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Set locale for Triton tests
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export LANGUAGE=en_US.UTF-8

# Activate virtual environment
if [ -d ".venv" ]; then
    echo "Activating .venv..."
    source .venv/bin/activate
elif [ -d "verl_env" ]; then
    echo "Activating verl_env..."
    source verl_env/bin/activate
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "Using active environment: $VIRTUAL_ENV"
else
    echo -e "${YELLOW}Warning: No virtual environment detected${NC}"
fi

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "Installing pytest..."
    pip install pytest pytest-xdist
fi

# Create tests directory if it doesn't exist
mkdir -p tests

# Test output directory
OUTPUT_DIR="test_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo ""
echo -e "${GREEN}[1/4] Running Installation Tests${NC}"
echo "======================================"
pytest tests/test_installation.py -v -s --tb=short 2>&1 | tee "$OUTPUT_DIR/installation_tests.log"
INSTALL_EXIT=$?

echo ""
echo -e "${GREEN}[2/4] Running Triton Locale Fix Tests${NC}"
echo "======================================"
pytest tests/test_triton_locale_fix.py -v -s --tb=short 2>&1 | tee "$OUTPUT_DIR/triton_tests.log"
TRITON_EXIT=$?

echo ""
echo -e "${GREEN}[3/4] Running Custom Split LLaMA Tests${NC}"
echo "======================================"
pytest tests/test_custom_split_llama.py -v -s --tb=short 2>&1 | tee "$OUTPUT_DIR/custom_split_llama_tests.log"
LLAMA_EXIT=$?

echo ""
echo -e "${GREEN}[4/4] Running All Tests with Coverage${NC}"
echo "======================================"
pytest tests/ -v --tb=short --maxfail=5 2>&1 | tee "$OUTPUT_DIR/all_tests.log"
ALL_EXIT=$?

# Generate summary
echo ""
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Function to print result
print_result() {
    local name=$1
    local exit_code=$2

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ $name: PASSED${NC}"
    elif [ $exit_code -eq 5 ]; then
        echo -e "${YELLOW}⚠ $name: SKIPPED (some tests)${NC}"
    else
        echo -e "${RED}✗ $name: FAILED${NC}"
    fi
}

print_result "Installation Tests" $INSTALL_EXIT
print_result "Triton Locale Tests" $TRITON_EXIT
print_result "Custom Split LLaMA Tests" $LLAMA_EXIT
print_result "All Tests" $ALL_EXIT

echo ""
echo "Test logs saved to: $OUTPUT_DIR/"

# Count results
TOTAL_PASSED=0
TOTAL_FAILED=0

for exit_code in $INSTALL_EXIT $TRITON_EXIT $LLAMA_EXIT; do
    if [ $exit_code -eq 0 ] || [ $exit_code -eq 5 ]; then
        ((TOTAL_PASSED++))
    else
        ((TOTAL_FAILED++))
    fi
done

echo ""
echo "Results: $TOTAL_PASSED passed, $TOTAL_FAILED failed"

# Generate HTML report (if pytest-html is available)
if command -v pytest &> /dev/null && pip show pytest-html &> /dev/null; then
    echo ""
    echo "Generating HTML report..."
    pytest tests/ --html="$OUTPUT_DIR/report.html" --self-contained-html --tb=short &> /dev/null || true
    if [ -f "$OUTPUT_DIR/report.html" ]; then
        echo -e "${GREEN}✓ HTML report: $OUTPUT_DIR/report.html${NC}"
    fi
fi

echo ""
echo -e "${BLUE}================================================${NC}"

# Exit with appropriate code
if [ $TOTAL_FAILED -gt 0 ]; then
    echo -e "${RED}Some tests failed. Check logs for details.${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed or skipped!${NC}"
    exit 0
fi
