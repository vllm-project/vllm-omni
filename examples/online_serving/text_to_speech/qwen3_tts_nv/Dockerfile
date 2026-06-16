FROM nvcr.io/nvidia/tritonserver:26.02-py3

# 1. System dependency for git-based installs
RUN apt-get update && \
    apt-get install -y git sox libsox-fmt-all

# 2. Install upstream vLLM first so it pulls in torch and the core runtime stack
RUN pip install --no-cache-dir "vllm==0.22.0"

# 3. Install vLLM-Omni from the fork/branch on top of upstream vLLM
RUN git clone --single-branch --branch vklimkov/qwen3tts_nv \
        https://github.com/vklimkov-nvidia/vllm-omni.git /tmp/vllm-omni && \
    cd /tmp/vllm-omni && \
    VLLM_OMNI_TARGET_DEVICE=cuda pip install --no-cache-dir . && \
    cd / && rm -rf /tmp/vllm-omni

# 5. Extra python requirements
RUN pip install --no-cache-dir \
    onnxscript \
    librosa \
    sox \
    onnx-graphsurgeon \
    "tritonclient[grpc]"
RUN pip uninstall -y numpy || true
RUN rm -rf \
    /usr/local/lib/python3.12/dist-packages/numpy \
    /usr/local/lib/python3.12/dist-packages/numpy.libs \
    /usr/local/lib/python3.12/dist-packages/numpy-*.dist-info
RUN pip install --no-cache-dir numpy==2.3.5

WORKDIR /workspace
