# SageAttention

SageAttention backends provide lossy low-precision attention for diffusion
models. Validate output quality against `TORCH_SDPA` at the same seed before
using either backend in production.

## `SAGE_ATTN`

`SAGE_ATTN` uses SageAttention 2.2 with INT8-quantized attention and FP16
accumulation.

### Installation

Install SageAttention into the same environment as vLLM-Omni:

```bash
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention
export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32
pip install . --no-build-isolation
```

Verify the installation:

```bash
python -c "import sageattention; print(sageattention.__file__)"
```

Select it globally:

```bash
vllm-omni serve <model> --diffusion-attention-backend SAGE_ATTN
```

## `SAGE_ATTN_3`

`SAGE_ATTN_3` uses the SageAttention3 Blackwell implementation.

### SageAttention3 installation

```bash
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention/sageattention3_blackwell
python setup.py install
```

Verify the installation:

```bash
python -c "import sageattn3; print(sageattn3.__file__)"
```

```bash
vllm-omni serve <model> --diffusion-attention-backend SAGE_ATTN_3
```

`SAGE_ATTN_3` requires CUDA, an importable `sageattn3`, and a Blackwell-class
GPU. Its kernel assumes the query-head count equals the key/value-head count.
GQA and MQA diffusion calls therefore fall back to PyTorch SDPA for
correctness.

For common configuration and platform routing, see the
[attention backend overview](../attention_backends.md).
