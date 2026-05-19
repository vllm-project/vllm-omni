FROM nvcr.io/nvidia/tritonserver:26.02-py3

# 1. System dependency for git-based installs
RUN apt-get update && \
    apt-get install -y git sox libsox-fmt-all

# 2. Install upstream vLLM first so it pulls in torch and the core runtime stack
RUN pip install --no-cache-dir "vllm==0.21.0"

# 3. Install vLLM-Omni from the fork/branch on top of upstream vLLM
RUN git clone --single-branch --branch vklimkov/qwen3tts_nv \
        https://github.com/vklimkov-nvidia/vllm-omni.git /tmp/vllm-omni && \
    cd /tmp/vllm-omni && \
    VLLM_OMNI_TARGET_DEVICE=cuda pip install --no-cache-dir . && \
    cd / && rm -rf /tmp/vllm-omni

# 5. Install the Qwen3-TTS pieces not provided by vLLM/vLLM-Omni,
#    and pin the transformers version required by Qwen3-TTS.
RUN pip install --no-cache-dir \
    onnxscript \
    librosa \
    sox \
    onnx-graphsurgeon \
    "tritonclient[grpc]"

WORKDIR /workspace
