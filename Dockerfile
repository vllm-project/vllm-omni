ARG VLLM_BASE_IMAGE=vllm/vllm-openai
ARG VLLM_BASE_TAG=v0.11.0
FROM ${VLLM_BASE_IMAGE}:${VLLM_BASE_TAG}
ARG APP_DIR=/workspace/vllm-omni
WORKDIR ${APP_DIR}

COPY . .

# Install vllm-omni into the same uv-managed Python environment used by the base image.
RUN uv pip install --python "$(python3 -c 'import sys; print(sys.executable)')" --no-cache-dir .

RUN ln -sf /usr/bin/python3 /usr/bin/python

ENTRYPOINT ["vllm", "serve", "--omni"]
