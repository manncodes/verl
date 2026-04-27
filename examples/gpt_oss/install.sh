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
#     INSTALL_SONIC_MOE=0         (set 1 to install Dao-AILab/sonic-moe; Hopper/Blackwell only)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_VERSION=${PYTHON_VERSION:-3.12}
VENV_DIR=${VENV_DIR:-.venv}
EXTRAS=${EXTRAS:-sglang,gpu,math,test}
SKIP_FLASH_ATTN=${SKIP_FLASH_ATTN:-0}
SKIP_PRE_COMMIT=${SKIP_PRE_COMMIT:-0}
INSTALL_SONIC_MOE=${INSTALL_SONIC_MOE:-0}

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
        "accelerate" \
        "cachetools" \
        "nvidia-ml-py" \
        "mathruler"
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
    # cachetools / nvidia-ml-py / mathruler are eagerly imported by verl
    # (agent_loop, profiler, reward score) but missing from setup.py — they
    # crash training ~4min in (after FSDP wrap) without these explicit pins.
    uv pip install \
        --no-build-isolation-package flash-attn \
        -e ".[${EXTRAS}]" \
        "transformers>=4.46" \
        "datasets>=3.0" \
        "hf-transfer" \
        "accelerate" \
        "cachetools" \
        "nvidia-ml-py" \
        "mathruler"
fi

# ---- 3b. sonic-moe (opt-in, Hopper/Blackwell only) -----------------------
# https://github.com/Dao-AILab/sonic-moe — grouped-GEMM MoE kernels for H100/H200/B200.
# Off by default: gpt-oss uses a *clamped* GELU-style SwiGLU (alpha=1.702,
# limit=7.0) that sonic-moe does not implement natively, so the
# `examples/gpt_oss/sonic_moe_patch.py` adapter wraps the kernel with the
# clamps. Treat as experimental — run `python examples/gpt_oss/test_sonic_moe.py`
# after install to verify forward parity against HF eager before enabling it
# in the launcher (`USE_SONIC_MOE=1`).
if [ "${INSTALL_SONIC_MOE}" = "1" ]; then
    log "installing sonic-moe (experimental; runs only on Hopper/Blackwell)"
    # `uv pip install sonic-moe` against the PyPI sdist (0.1.2.post1, Apr 2026)
    # has been observed to resolve+exit-0 without actually installing the
    # package on this venv (only transitive deps move). The README's
    # source install path is more reliable, so use git directly. Pin to main
    # rather than a tag because the project doesn't tag releases yet and
    # CuTeDSL kernel APIs are still in flux.
    #
    # The [cu13] extra is the critical bit: bare `nvidia-cutlass-dsl` is
    # metadata-only and `import cutlass` will fail inside sonic-moe's quack
    # transitive dep ("ModuleNotFoundError: No module named 'cutlass'"). The
    # cu13 variant ships the actual python module and matches a driver-13
    # H100 box; if the box is on driver 12, swap to cu12 below.
    SONIC_MOE_EXTRA=${SONIC_MOE_EXTRA:-cu13}
    log "  using sonic-moe extra: [${SONIC_MOE_EXTRA}] (override with SONIC_MOE_EXTRA=cu12 if on CUDA 12 driver)"
    uv pip install --refresh \
        "sonic-moe[${SONIC_MOE_EXTRA}] @ git+https://github.com/Dao-AILab/sonic-moe.git@main"
    # Verify both the package itself AND its transitive `cutlass` import
    # are usable; one failed the other in the previous round and we want
    # the exact failure mode to surface here, not in the parity test.
    if ! python -c "import cutlass" >/dev/null 2>&1; then
        log "ERROR: 'import cutlass' fails — nvidia-cutlass-dsl[${SONIC_MOE_EXTRA}] didn't ship the module."
        log "       try: SONIC_MOE_EXTRA=cu12 INSTALL_SONIC_MOE=1 bash examples/gpt_oss/install.sh"
        log "       or:  uv pip install -v 'nvidia-cutlass-dsl[cu12]>=4.4.2'    (then re-run import check)"
        python -c "import cutlass" 2>&1 | tail -10
        exit 1
    fi
    if ! python -c "import sonicmoe" >/dev/null 2>&1; then
        log "ERROR: 'import cutlass' works but 'import sonicmoe' still fails."
        log "       full error:"
        python -c "import sonicmoe" 2>&1 | tail -20
        exit 1
    fi
    log "sonic-moe import OK: $(python -c 'import sonicmoe; print(getattr(sonicmoe, \"__version__\", \"unknown\"))')"
fi

# ---- 4. pre-commit -------------------------------------------------------
if [ "${SKIP_PRE_COMMIT}" != "1" ] && [ -f .pre-commit-config.yaml ]; then
    log "installing pre-commit hooks"
    pre-commit install || log "pre-commit install failed (non-fatal)"
fi

# ---- 5. smoke check ------------------------------------------------------
# We check both top-level packages AND the deep verl import chain that the
# launcher actually exercises — the agent loop module pulls in cachetools
# (and friends) on every training launch, so a missing transitive there
# crashes ~4 minutes into the run, after FSDP wrap. Catching it here
# instead saves a lot of GPU time on a misconfigured install.
log "verifying imports"
python - <<'PY'
import importlib, sys

required_top = ["torch", "transformers", "datasets", "verl", "ray", "hydra"]
optional_top = ["sglang", "flash_attn", "sonicmoe"]

# Deep verl chains that get hit on the FSDP+sglang+gsm8k launch path. If any
# of these fail with a missing third-party dep, install.sh needs to add it.
required_chains = [
    "verl.experimental.agent_loop",     # pulls cachetools, regex, pydantic, ...
    "verl.utils.tracking",              # pulls orjson, wandb
    "verl.workers.engine.fsdp.transformer_impl",  # pulls accelerate, peft, ...
    "verl.workers.rollout.sglang_rollout.async_sglang_server",  # pulls sglang internals
    "verl.trainer.main_ppo",            # pulls hydra, omegaconf, ray, ...
]

missing = []

for mod in required_top:
    try:
        importlib.import_module(mod)
        print(f"  ok  {mod}")
    except Exception as exc:
        missing.append(f"  MISS {mod}: {exc}")

for mod in optional_top:
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

print("verifying verl deep import chains (catches missing transitive deps)")
for chain in required_chains:
    try:
        importlib.import_module(chain)
        print(f"  ok  {chain}")
    except ModuleNotFoundError as exc:
        missing.append(f"  MISS {chain}: {exc}")
    except Exception as exc:
        # Other errors (CUDA-unavailable inside an import, etc.) we just warn —
        # they're environmental, not missing-dep.
        print(f"  ??  {chain} raised {type(exc).__name__}: {exc} (likely env, not a missing dep)")

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
log "  * If you set INSTALL_SONIC_MOE=1, run examples/gpt_oss/test_sonic_moe.py"
log "    on the H100 box to verify forward parity before turning on USE_SONIC_MOE=1."
