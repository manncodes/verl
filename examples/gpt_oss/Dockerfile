# gpt-oss training image: sglang stable base + verl + Mxfp4-capable transformers.
#
# Built by examples/gpt_oss/build_with_colima.sh; the build context is expected
# to be the verl repo root, which gets COPY'd in.
FROM lmsysorg/sglang:v0.5.9

ARG PIP_NO_CACHE_DIR=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PYTHONUNBUFFERED=1

# transformers>=4.46 ships Mxfp4Config (required to dequantize gpt-oss MXFP4
# weights to bf16 — see examples/gpt_oss/prepare_model.py).
RUN pip install --no-cache-dir \
        "transformers>=4.46" \
        "hf-transfer" \
        "accelerate" \
        "math-verify" \
        "mathruler"

COPY . /opt/verl
WORKDIR /opt/verl

# verl[sglang] resolves against the sglang already baked into the base image,
# so we only need the verl-side bindings and gpt-oss runtime deps. Build
# isolation stays on for everything except (implicitly) packages without a
# real build step.
RUN pip install --no-cache-dir -e ".[math,test]"

ENV PYTHONPATH=/opt/verl

# Default: drop into a shell. Override on `docker run` to launch training:
#   docker run --gpus all -it verl-gpt-oss:local \
#       bash examples/gpt_oss/launch_train_gpt_oss_20b.sh
CMD ["/bin/bash"]
