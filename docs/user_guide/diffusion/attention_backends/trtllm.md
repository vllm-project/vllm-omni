# TRTLLM Attention

`TRTLLM_ATTN` runs FlashInfer's trtllm-gen FMHA kernels. It supports dense
BF16 attention plus optional Skip-Softmax sparsity or SAGE quantization.

For common selection and per-role configuration, see the
[attention backend overview](../attention_backends.md).

## Requirements

Selecting `TRTLLM_ATTN` requires:

- datacenter Blackwell, sm_100 or sm_103;
- `head_dim=128`;
- FlashInfer; and
- a model-declared packed or mask-free compatible path.

It is not supported on workstation Blackwell (sm_120/sm_121). An incompatible
explicit selection raises instead of silently changing the backend.

## Dense mode

Select `TRTLLM_ATTN` without `skip_softmax` or `quant` to run dense BF16:

```bash
vllm-omni serve <model> --diffusion-attention-backend TRTLLM_ATTN
```

## Skip-Softmax

Skip-Softmax trades some fidelity for speed by skipping low-contribution
attention work. See the [feature design](../../../design/feature/skip_softmax.md)
for the algorithm and implementation contract.

| Key | Valid values | Meaning |
| --- | --- | --- |
| `target_sparsity` | finite, `[0, 1]` | Point on a model-calibrated curve |
| `threshold` | finite, `>= 0` | Direct threshold; mutually exclusive with `target_sparsity` |
| `disabled_until_timestep` | finite, `[0, 1]` | Keep early high-noise steps dense until normalized `t <= D` |

```bash
vllm-omni serve Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --diffusion-attention-config '{"default":{"backend":"TRTLLM_ATTN",\
    "skip_softmax":{"target_sparsity":0.65,"disabled_until_timestep":0.86}}}'
```

```python
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, SkipSoftmaxSpec

config = AttentionConfig(
    default=AttentionSpec(
        backend="TRTLLM_ATTN",
        skip_softmax=SkipSoftmaxSpec(
            target_sparsity=0.65,
            disabled_until_timestep=0.86,
        ),
    ),
)
```

Start with `target_sparsity=0.65` and `disabled_until_timestep=0.86`. Increase
sparsity only after comparing quality against dense output at the same seed.
The benefit grows with sequence length because only attention is accelerated.

## SAGE quantization

TRTLLM SAGE quantizes Q/K per block and V per channel so both attention
matrix multiplications use low precision.

| Key | Valid values | Meaning |
| --- | --- | --- |
| `dtype_qk` | `int8`, `fp8_e4m3` | Q/K quantization dtype; absent means dense |
| `q_block_size` | `1`, `4`, `16` | Q scale block size; default `1` |
| `k_block_size` | `1`, `4`, `16` | K scale block size; default `16` |

V is always quantized per channel to FP8 E4M3 and K-smoothing is internal.
Only block sizes with compiled kernels are accepted.

```bash
vllm-omni serve Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --diffusion-attention-config '{"default":{"backend":"TRTLLM_ATTN",\
    "quant":{"dtype_qk":"fp8_e4m3","q_block_size":1,"k_block_size":16}}}'
```

```python
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, AttnQuantSpec

config = AttentionConfig(
    default=AttentionSpec(
        backend="TRTLLM_ATTN",
        quant=AttnQuantSpec(
            dtype_qk="fp8_e4m3",
            q_block_size=1,
            k_block_size=16,
        ),
    ),
)
```

This path requires FlashInfer 0.6.16rc1 or newer. FP8 Q/K kernels are
available on sm_100 and sm_103; INT8 Q/K kernels are compiled for sm_100 only.
The shared `AttnQuantSpec` is backend typed: values intended for
`FLASHINFER_ATTN` are rejected here and vice versa.
