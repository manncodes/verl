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
# verl[gpu] pulls liger-kernel + flash-attn (we drop flash-attn separately if
# the toolchain isn't available, see SKIP_FLASH_ATTN below).
# verl[math] adds math-verify for the gsm8k reward.
# verl[test] adds pytest + pre-commit for dev workflow.
# transformers>=4.46 is required for Mxfp4Config (gpt-oss MXFP4 dequantization).
log "installing verl[${EXTRAS}] + gpt-oss runtime deps"

INSTALL_TARGETS=(
    "-e" ".[${EXTRAS}]"
    "transformers>=4.46"
    "hf-transfer"      # faster HF downloads for the 20B checkpoint
    "accelerate"
)

if [ "${SKIP_FLASH_ATTN}" = "1" ]; then
    log "SKIP_FLASH_ATTN=1, dropping flash-attn from the install set"
    # remove flash-attn from the resolution by using sglang extra without gpu extra
    EXTRAS_NO_FA=${EXTRAS//gpu,/}
    EXTRAS_NO_FA=${EXTRAS_NO_FA//,gpu/}
    EXTRAS_NO_FA=${EXTRAS_NO_FA//gpu/}
    INSTALL_TARGETS=(
        "-e" ".[${EXTRAS_NO_FA}]"
        "transformers>=4.46"
        "hf-transfer"
        "accelerate"
    )
fi

# Single resolve+install pass — uv's equivalent of `uv sync`.
uv pip install "${INSTALL_TARGETS[@]}"

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

log "done. Activate with: source ${VENV_DIR}/bin/activate"
log "Then run:           bash examples/gpt_oss/launch_train_gpt_oss_20b.sh"
