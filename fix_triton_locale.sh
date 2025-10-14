#!/bin/bash
# Fix for Triton UnicodeDecodeError when running veRL
# Error: "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc4"

set -e

echo "=== Fixing Triton Locale Issue ==="
echo ""

# Activate environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "verl_env" ]; then
    source verl_env/bin/activate
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "Using active environment: $VIRTUAL_ENV"
else
    echo "Warning: No virtual environment found, installing globally"
fi

echo "Step 1: Setting locale environment variables..."

# Export locale settings
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export LANGUAGE=en_US.UTF-8

# Make permanent
cat >> ~/.bashrc << 'EOF'

# Fix for Triton locale issues
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export LANGUAGE=en_US.UTF-8
EOF

echo "✓ Locale variables set"

echo ""
echo "Step 2: Reinstalling transformer-engine and triton..."

# Uninstall transformer-engine and triton
pip uninstall -y transformer-engine triton || true

# Reinstall with compatible versions
pip install --no-cache-dir triton==3.0.0
pip install --no-cache-dir transformer-engine==1.7.0

echo "✓ Packages reinstalled"

echo ""
echo "Step 3: Verifying installation..."

# Test import
python3 -c "
import os
os.environ['LC_ALL'] = 'C.UTF-8'
os.environ['LANG'] = 'C.UTF-8'
import triton
print('✓ Triton imported successfully')
import transformer_engine
print('✓ Transformer Engine imported successfully')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Fix Applied Successfully ==="
    echo ""
    echo "Now run your training script with:"
    echo "  export LC_ALL=C.UTF-8"
    echo "  export LANG=C.UTF-8"
    echo "  bash grpo.sh"
    echo ""
    echo "Or add to your grpo.sh at the top:"
    echo "  export LC_ALL=C.UTF-8"
    echo "  export LANG=C.UTF-8"
else
    echo ""
    echo "=== Verification Failed ==="
    echo "Please check the error messages above"
    exit 1
fi
