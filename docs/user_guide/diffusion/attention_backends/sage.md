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

`SAGE_ATTN_3` requires CUDA, an importable `sageattn3`, and one of the
architectures the kernel is built for. `sageattention3_blackwell/setup.py`
emits one gencode per supported compute capability and rejects anything else:

| Compute capability | Gencode | Example GPUs |
| --- | --- | --- |
| 10.0 | `sm_100a` | B200, GB200 |
| 12.0 | `sm_120a` | RTX PRO 6000, RTX 50-series |
| 12.1 | `sm_121a` | Consumer Blackwell refresh |

Not every Blackwell GPU qualifies. `sm_103` (B300 / GB300) is absent by
design: the kernel takes the SM120 warp-level `mma.sync` path, while SM103
requires the tcgen05 implementation. Selecting `SAGE_ATTN_3` on an
unsupported GPU logs a warning and falls back to `TORCH_SDPA`.

Its kernel also assumes the query-head count equals the key/value-head count.
GQA and MQA diffusion calls therefore fall back to PyTorch SDPA for
correctness.

For common configuration and platform routing, see the
[attention backend overview](../attention_backends.md).
