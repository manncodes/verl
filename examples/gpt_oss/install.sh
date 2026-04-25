#!/bin/bash
# One-shot env install for the gpt-oss launch + correctness check.
#
# Strategy: bootstrap uv, create a 3.12 venv, then resolve every dependency
# needed by examples/gpt_oss/* in a single `uv pip install` pass (verl is a
# setuptools project with `dependencies = "dynamic"`, so `uv sync` proper is
# not usable; one bulk install is the closest equivalent).
#
# Usage (from repo root):
#     bash examples/gpt_oss/install.sh
#     source .venv/bin/activate
#
# Then run training:
#     bash examples/gpt_oss/launch_train_gpt_oss_20b.sh
#
# Knobs:
#     PYTHON_VERSION=3.12         (uv-managed Python)
#     VENV_DIR=.venv              (where to create the venv)
#     EXTRAS="sglang,gpu,math,test"  (verl extras to install)
#     SKIP_FLASH_ATTN=0           (set 1 to skip flash-attn build)
#     SKIP_PRE_COMMIT=0           (set 1 to skip pre-commit hook install)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_VERSION=${PYTHON_VERSION:-3.12}
VENV_DIR=${VENV_DIR:-.venv}
EXTRAS=${EXTRAS:-sglang,gpu,math,test}
SKIP_FLASH_ATTN=${SKIP_FLASH_ATTN:-0}
SKIP_PRE_COMMIT=${SKIP_PRE_COMMIT:-0}

log() { printf '[install] %s\n' "$*"; }

# ---- 1. uv ---------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    log "uv not found, bootstrapping from astral.sh"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv installs to ~/.local/bin or ~/.cargo/bin depending on platform
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
log "uv version: $(uv --version)"

# ---- 2. venv -------------------------------------------------------------
if [ ! -d "${VENV_DIR}" ]; then
    log "creating ${VENV_DIR} (python ${PYTHON_VERSION})"
    uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
else
    log "reusing existing venv at ${VENV_DIR}"
fi
# shellcheck disable=SC1090,SC1091
source "${VENV_DIR}/bin/activate"

# ---- 3. one-shot install -------------------------------------------------
# verl[sglang] pulls torch==2.9.1 + sglang==0.5.8 (the gpt-oss-supported combo).
# verl[gpu] pulls liger-kernel + flash-attn (set SKIP_FLASH_ATTN=1 if no CUDA
# toolchain or no compatible prebuilt wheel is available).
# verl[math] adds math-verify for the gsm8k reward.
# verl[test] adds pytest + pre-commit for dev workflow.
# transformers>=4.46 is required for Mxfp4Config (gpt-oss MXFP4 dequantization).

# flash-attn does NOT declare torch as a build dependency (despite needing it
# at build time), so a single-pass `uv pip install` fails inside uv's isolated
# build env with "ModuleNotFoundError: No module named 'torch'". Workaround:
# install torch + build helpers FIRST, then disable build isolation just for
# flash-attn on the second pass.

if [ "${SKIP_FLASH_ATTN}" = "1" ]; then
    log "SKIP_FLASH_ATTN=1, dropping flash-attn from the install set"
    # Strip 'gpu' extra so flash-attn is not pulled in.
    EXTRAS_EFFECTIVE=${EXTRAS//gpu,/}
    EXTRAS_EFFECTIVE=${EXTRAS_EFFECTIVE//,gpu/}
    EXTRAS_EFFECTIVE=${EXTRAS_EFFECTIVE//gpu/}
    log "installing verl[${EXTRAS_EFFECTIVE}] + gpt-oss runtime deps (single pass)"
    uv pip install \
        -e ".[${EXTRAS_EFFECTIVE}]" \
        "transformers>=4.46" \
        "datasets>=3.0" \
        "hf-transfer" \
        "accelerate"
else
    log "pass 1/2: installing torch + flash-attn build deps"
    # Pin torch to whatever sglang extra wants; the second pass will re-resolve
    # but having it present satisfies flash-attn's build-time `import torch`.
    uv pip install \
        "torch==2.9.1" \
        "packaging>=20.0" \
        wheel \
        setuptools \
        ninja

    log "pass 2/2: installing verl[${EXTRAS}] + gpt-oss runtime deps"
    # --no-build-isolation-package flash-attn lets flash-attn see the torch we
    # just installed instead of getting an empty isolated build env.
    # datasets>=3.0 because verl's setup.py is unpinned and the resolver
    # otherwise picks 2.14.x, which calls the removed pyarrow.PyExtensionType
    # against the modern pyarrow that sglang pulls in.
    uv pip install \
        --no-build-isolation-package flash-attn \
        -e ".[${EXTRAS}]" \
        "transformers>=4.46" \
        "datasets>=3.0" \
        "hf-transfer" \
        "accelerate"
fi

# ---- 4. pre-commit -------------------------------------------------------
if [ "${SKIP_PRE_COMMIT}" != "1" ] && [ -f .pre-commit-config.yaml ]; then
    log "installing pre-commit hooks"
    pre-commit install || log "pre-commit install failed (non-fatal)"
fi

# ---- 5. smoke check ------------------------------------------------------
log "verifying imports"
python - <<'PY'
import importlib, sys
required = ["torch", "transformers", "datasets", "verl", "ray", "hydra"]
optional = ["sglang", "flash_attn"]
missing = []
for mod in required:
    try:
        importlib.import_module(mod)
        print(f"  ok  {mod}")
    except Exception as exc:
        missing.append(f"  MISS {mod}: {exc}")
for mod in optional:
    try:
        importlib.import_module(mod)
        print(f"  ok  {mod} (optional)")
    except Exception as exc:
        print(f"  --  {mod} (optional, skipped): {type(exc).__name__}")
try:
    from transformers import Mxfp4Config  # noqa: F401
    print("  ok  Mxfp4Config")
except Exception as exc:
    missing.append(f"  MISS Mxfp4Config: {exc}")
if missing:
    print("\n".join(missing), file=sys.stderr)
    sys.exit(1)
PY

log ""
log "done. Activate with: source ${VENV_DIR}/bin/activate"
log "Then run:           bash examples/gpt_oss/launch_train_gpt_oss_20b.sh"
log ""
log "Notes:"
log "  * 'sglang has no extra named srt/openai' is benign — sglang 0.5.8 dropped"
log "    those extras; verl's setup.py request is harmless and sglang installs."
log "  * If flash_attn fails to import (ABI mismatch with newer CUDA drivers),"
log "    the recipe still works: gpt-oss uses attn_implementation=eager by default."
