# syntax=docker/dockerfile:1.7
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ARG VLLM_BRANCH=feat/longcat-next
ARG VLLM_OMNI_BRANCH=feat/longcat-next-integration
ARG VLLM_REPO=https://github.com/gangula-karthik/vllm.git
ARG VLLM_OMNI_REPO=https://github.com/gangula-karthik/vllm-omni.git

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

ENV VIRTUAL_ENV=/root/venvs/dev
ENV UV_CACHE_DIR=/root/.cache/uv
# BuildKit cache mounts below persist uv's download cache across builds
# without baking it into any image layer -- copy (not hardlink) mode is
# required because the cache mount is a different filesystem than the venv.
ENV UV_LINK_MODE=copy
RUN uv venv --python 3.11 ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# vllm first, on its own layer: it changes far less often than vllm-omni
# during active development, so keeping it ahead of the vllm-omni clone
# means iterating on vllm-omni alone reuses this entire layer from cache
# instead of reinstalling vllm every time.
# --torch-backend=auto relies on detecting a live GPU/driver to pick the CUDA
# wheel; that detection fails silently under cross-arch emulation (no GPU
# visible in the build sandbox) and falls back to CPU-only torch, which then
# breaks flash-attn. Pin the CUDA backend explicitly instead.
RUN --mount=type=cache,target=/root/.cache/uv \
    git clone --depth 1 -b ${VLLM_BRANCH} ${VLLM_REPO} /opt/src/vllm \
    && cd /opt/src/vllm \
    && VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=cu130

# flash-attn: prebuilt wheel from astral's GPU wheel index (exact match:
# flash-attn 2.8.3.post1, cu13.0, torch 2.11, cp311). Skips the from-source
# nvcc/ptxas dance for flash-attn itself, which also sidesteps a cutlass
# template-compile OOM seen under cross-arch (QEMU) emulation.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install flash-attn \
    --index-url https://wheels.astral.sh/simple/cu130/ \
    --index-strategy unsafe-best-match

# nvidia-cuda-nvcc + lib64/libcudart symlinks are needed at RUNTIME (not
# build time): flashinfer JIT-compiles its sampling kernel lazily on the
# first real inference call on the GPU pod, and its build scripts hardcode
# -L.../lib64 and -lcudart/-lcuda unversioned.
# nvcc 13.0.88 matches torch's pinned nvidia-cuda-runtime==13.0.96 headers;
# its own bundled ptxas has a PTX-ISA version-skew bug (emits .version 9.3,
# only accepts up to 9.0), so swap in 13.3.73's ptxas binary while keeping
# 13.0.88's headers/frontend.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install nvidia-cuda-nvcc==13.0.88 \
    && cp ${VIRTUAL_ENV}/lib/python3.11/site-packages/nvidia/cu13/bin/ptxas /root/ptxas_13.0.88.bak \
    && uv pip install nvidia-cuda-nvcc==13.3.73 \
    && cp ${VIRTUAL_ENV}/lib/python3.11/site-packages/nvidia/cu13/bin/ptxas /root/ptxas_13.3.73.bak \
    && uv pip install nvidia-cuda-nvcc==13.0.88 \
    && cp /root/ptxas_13.3.73.bak ${VIRTUAL_ENV}/lib/python3.11/site-packages/nvidia/cu13/bin/ptxas \
    && ln -sfn ${VIRTUAL_ENV}/lib/python3.11/site-packages/nvidia/cu13/lib \
               ${VIRTUAL_ENV}/lib/python3.11/site-packages/nvidia/cu13/lib64 \
    && ln -sf ${VIRTUAL_ENV}/lib/python3.11/site-packages/nvidia/cu13/lib/libcudart.so.13 \
              ${VIRTUAL_ENV}/lib/python3.11/site-packages/nvidia/cu13/lib/libcudart.so

ENV CUDA_HOME=${VIRTUAL_ENV}/lib/python3.11/site-packages/nvidia/cu13
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV FLASH_ATTN_CUDA_ARCHS=80

# vllm-omni last: the layer most likely to need a rebuild on any given
# iteration, so everything above it (uv, vllm, flash-attn, nvcc) stays
# cached and this is the only step that actually re-runs.
RUN --mount=type=cache,target=/root/.cache/uv \
    git clone --depth 1 -b ${VLLM_OMNI_BRANCH} ${VLLM_OMNI_REPO} /opt/src/vllm-omni \
    && cd /opt/src/vllm-omni \
    && uv pip install --editable . --no-deps \
    && uv pip install -r requirements/cuda.txt

WORKDIR /opt/src/vllm-omni
# Do not override CMD: the base image's /start.sh is what launches sshd —
# a custom CMD here (e.g. ["/bin/bash"]) silently disables it and the pod
# never accepts SSH connections.
CMD ["/start.sh"]
