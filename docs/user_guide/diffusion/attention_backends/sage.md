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

`SAGE_ATTN_3` requires CUDA, an importable `sageattn3`, and a supported
Blackwell GPU (`sm_120` or `sm_121`; `sm_100`/`sm_103` are rejected at runtime
by upstream `sageattention3_blackwell`). Its kernel assumes the query-head count
equals the key/value-head count; GQA and MQA diffusion calls therefore fall back
to PyTorch SDPA for correctness. Both `SAGE_ATTN` and `SAGE_ATTN_3` support
mask-free trailing-padding slicing via `valid_kv_length`.

### `SAGE_ATTN_3` options (`sage3`)

On long non-causal sequences with `per_block_mean=True` (the upstream default),
`sageattn3_blackwell` allocates an FP4 `delta_s` table whose TMA batch stride
overflows `int32` at extreme sequence/head sizes unless Q-chunking (such as
`thu-ml/SageAttention#394`) is built into `sageattn3`. Configure `sage3` in
`--diffusion-attention-config` to tune these parameters:

```bash
vllm-omni serve <model> \
  --diffusion-attention-config '{"default":{"backend":"SAGE_ATTN_3","sage3":{"per_block_mean":true,"q_chunk_size":4096}}}'
```

- `per_block_mean` (`bool`, default `true`): disable (`false`) only after
  validating visual quality at the same seed, as disabling per-block Q-mean
  centering significantly increases quantization error on long video models.
- `q_chunk_size` (`int | null`, positive multiple of `128`): requires a
  `sageattn3` build exposing `q_chunk_size`.

For common configuration and platform routing, see the
[attention backend overview](../attention_backends.md).
