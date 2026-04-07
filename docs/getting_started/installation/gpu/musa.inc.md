# --8<-- [start:requirements]

- GPU: Moore Threads GPU with MUSA SDK installed (validated on MTT S5000)

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

vLLM-Omni for MUSA requires building from source. Pre-built wheels are not currently available.

!!! note
    MUSA platform requires vLLM-MUSA to be installed first.

# --8<-- [start:pre-built-wheels]

# --8<-- [end:pre-built-wheels]

# --8<-- [start:build-wheel-from-source]

#### Prerequisites

- **MUSA SDK**: Download from [MUSA SDK Download](https://developer.mthreads.com/sdk/download/musa)
- **torchada**: CUDA→MUSA compatibility layer for PyTorch (`pip install torchada`)
- **mthreads-ml-py**: MTML Python bindings (`pip install mthreads-ml-py`)
- **MATE**: MUSA AI Tensor Engine ([GitHub](https://github.com/MooreThreads/mate))

#### Installation of vLLM-MUSA

```bash
git clone https://github.com/MooreThreads/vllm-musa.git
cd vllm-musa
git checkout v0.18.0-dev
pip install . --no-build-isolation -v
```

#### Installation of vLLM-Omni

```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
VLLM_OMNI_TARGET_DEVICE=musa pip install -e . --no-build-isolation
```

For Gradio demos:

```bash
pip install -e '.[demo]' --no-build-isolation
```

#### Environment Variables

```bash
export MUSA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_MUSA_CUSTOM_OP_USE_NATIVE=false
```

# --8<-- [end:build-wheel-from-source]

# --8<-- [start:build-docker]

# --8<-- [end:build-docker]

# --8<-- [start:pre-built-images]

# --8<-- [end:pre-built-images]
