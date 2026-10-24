FROM vllm/vllm-openai:v0.29.0

WORKDIR /app

# Install the pinned Python dependency overlay.
# Preserve existing vLLM, PyTorch and CUDA packages.
COPY wheelhouse/ /tmp/wheelhouse/

RUN python3 -m pip install \
    --no-index \
    --no-deps \
    --disable-pip-version-check \
    /tmp/wheelhouse/*.whl \
    && rm -rf /tmp/wheelhouse

# Install the exact upstream vLLM-Omni revision.
COPY src/ /app/vllm-omni/

ARG VLLM_OMNI_VERSION_OVERRIDE=0.0.0.dev0+gf1a6e7ce

RUN cd /app/vllm-omni \
    && VLLM_OMNI_VERSION_OVERRIDE="${VLLM_OMNI_VERSION_OVERRIDE}" \
    python3 -m pip install \
      --no-index \
      --no-deps \
      --no-build-isolation \
      --disable-pip-version-check \
      .

ENTRYPOINT []
