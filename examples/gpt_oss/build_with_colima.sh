#!/bin/bash
# One-shot: bring up colima, clone verl (if needed), build the gpt-oss image.
#
# Usage (from anywhere):
#     curl -sLO https://raw.githubusercontent.com/manncodes/verl/main/examples/gpt_oss/build_with_colima.sh
#     bash build_with_colima.sh
#
# Or from an existing verl checkout:
#     bash examples/gpt_oss/build_with_colima.sh
#
# Knobs (env vars):
#     VERL_REPO=https://github.com/manncodes/verl.git
#     VERL_REF=main                        (branch / tag / sha)
#     VERL_DIR=./verl                      (clone target if not in a verl checkout)
#     IMAGE_TAG=verl-gpt-oss:local
#     DOCKERFILE=examples/gpt_oss/Dockerfile
#     COLIMA_CPU=8                         (vCPUs for the colima VM)
#     COLIMA_MEMORY=24                     (GB)
#     COLIMA_DISK=120                      (GB)
#     COLIMA_ARCH=x86_64                   (force x86 emulation on Apple Silicon
#                                           — required: gpt-oss base image is
#                                           amd64 + CUDA)
#     SKIP_COLIMA_START=0                  (set 1 if colima already configured)
set -euo pipefail

VERL_REPO=${VERL_REPO:-https://github.com/manncodes/verl.git}
VERL_REF=${VERL_REF:-main}
VERL_DIR=${VERL_DIR:-./verl}
IMAGE_TAG=${IMAGE_TAG:-verl-gpt-oss:local}
DOCKERFILE=${DOCKERFILE:-examples/gpt_oss/Dockerfile}
COLIMA_CPU=${COLIMA_CPU:-8}
COLIMA_MEMORY=${COLIMA_MEMORY:-24}
COLIMA_DISK=${COLIMA_DISK:-120}
COLIMA_ARCH=${COLIMA_ARCH:-x86_64}
SKIP_COLIMA_START=${SKIP_COLIMA_START:-0}

log() { printf '[build] %s\n' "$*"; }

# ---- 1. ensure colima + docker CLI ---------------------------------------
ensure_tool() {
    local tool=$1
    if command -v "${tool}" >/dev/null 2>&1; then
        return 0
    fi
    log "${tool} not found"
    if command -v brew >/dev/null 2>&1; then
        log "installing ${tool} via brew"
        brew install "${tool}"
    else
        cat <<EOF >&2
[build] cannot auto-install ${tool} (no brew on PATH).
        Install manually:
          macOS:  brew install ${tool}
          Linux:  see https://github.com/abiosoft/colima#installation
EOF
        exit 1
    fi
}

ensure_tool colima
ensure_tool docker

log "colima version: $(colima version | head -n1)"
log "docker  version: $(docker --version)"

# ---- 2. start colima -----------------------------------------------------
if [ "${SKIP_COLIMA_START}" != "1" ]; then
    if colima status >/dev/null 2>&1; then
        log "colima already running"
    else
        log "starting colima (cpu=${COLIMA_CPU}, memory=${COLIMA_MEMORY}G, disk=${COLIMA_DISK}G, arch=${COLIMA_ARCH})"
        # --arch x86_64 is important on Apple Silicon: the sglang base image
        # is amd64 + CUDA, so we either emulate x86 in the VM or pass
        # --platform linux/amd64 to docker (the former is faster end-to-end
        # because layers cache properly).
        colima start \
            --cpu "${COLIMA_CPU}" \
            --memory "${COLIMA_MEMORY}" \
            --disk "${COLIMA_DISK}" \
            --arch "${COLIMA_ARCH}" \
            --runtime docker
    fi
fi

# Make sure docker CLI is talking to colima (not Docker Desktop).
if command -v docker >/dev/null 2>&1; then
    if docker context inspect colima >/dev/null 2>&1; then
        docker context use colima >/dev/null
        log "docker context: colima"
    fi
fi

# ---- 3. resolve repo (clone if not already in one) -----------------------
# If we're already inside a verl checkout (the Dockerfile we ship is present),
# build from here. Otherwise clone.
if [ -f "${DOCKERFILE}" ] && [ -f "setup.py" ] && [ -f "pyproject.toml" ]; then
    BUILD_CTX="$(pwd)"
    log "building from current verl checkout: ${BUILD_CTX}"
else
    if [ ! -d "${VERL_DIR}/.git" ]; then
        log "cloning ${VERL_REPO} (ref=${VERL_REF}) -> ${VERL_DIR}"
        git clone "${VERL_REPO}" "${VERL_DIR}"
        (cd "${VERL_DIR}" && git checkout "${VERL_REF}")
    else
        log "reusing existing checkout at ${VERL_DIR}"
        (cd "${VERL_DIR}" && git fetch origin "${VERL_REF}" && git checkout "${VERL_REF}")
    fi
    BUILD_CTX="$(cd "${VERL_DIR}" && pwd)"
fi

# ---- 4. build ------------------------------------------------------------
log "docker build -f ${DOCKERFILE} -t ${IMAGE_TAG} ${BUILD_CTX}"
docker build \
    --platform linux/amd64 \
    -f "${BUILD_CTX}/${DOCKERFILE}" \
    -t "${IMAGE_TAG}" \
    "${BUILD_CTX}"

log "built image: ${IMAGE_TAG}"
log ""
log "Run training (needs --gpus all on a CUDA host; colima on macOS has no GPU passthrough):"
log "    docker run --rm -it --gpus all \\"
log "        -v \$HOME/.cache/huggingface:/root/.cache/huggingface \\"
log "        -v \$HOME/models:/root/models \\"
log "        -v \$HOME/data:/root/data \\"
log "        ${IMAGE_TAG} \\"
log "        bash examples/gpt_oss/launch_train_gpt_oss_20b.sh"
