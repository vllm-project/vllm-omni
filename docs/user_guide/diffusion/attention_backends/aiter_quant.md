# AITER Quantized Attention

`AITER_QUANT_ATTN` runs AITER MHA v4 low-precision attention kernels on
supported AMD Instinct GPUs. See the
[AITER MHA v4 documentation](https://github.com/ROCm/aiter/blob/main/aiter/ops/mha_v4.md)
for details about the upstream kernels and quantization recipes.

For common selection and per-role configuration, see the
[attention backend overview](../attention_backends.md).

## Requirements

Selecting `AITER_QUANT_ATTN` requires:

- a ROCm environment;
- a gfx942 or gfx950 GPU;
- `head_dim=128`; and
- an AITER build containing `aiter.ops.mha_v4`.

An incompatible explicit selection raises an error instead of silently
falling back to another backend.

## Supported formats

| Architecture | Formats |
| --- | --- |
| gfx942 | `fp8`, `i8fp8` |
| gfx950 | `bf16`, `fp8`, `i8fp8`, `mxfp8`, `mxfp4`, `mxfp6`, `f6f4` |

The default format is `fp8`.

## Configuration

Omitting `aiter_quant` selects the default FP8 recipe:

```bash
vllm-omni serve <model> \
  --diffusion-attention-backend AITER_QUANT_ATTN
```

Choose a format with structured attention configuration:

```bash
vllm-omni serve Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --omni \
  --diffusion-attention-config \
  '{"default":{"backend":"AITER_QUANT_ATTN","aiter_quant":{"format":"mxfp4"}}}'
```

The same configuration is available through the Python API:

```python
from vllm_omni.diffusion.data import (
    AiterQuantSpec,
    AttentionConfig,
    AttentionSpec,
    OmniDiffusionConfig,
)

config = OmniDiffusionConfig(
    diffusion_attention_config=AttentionConfig(
        default=AttentionSpec(
            backend="AITER_QUANT_ATTN",
            aiter_quant=AiterQuantSpec(format="mxfp4"),
        ),
    ),
    ...,
)
```

## Limitations

The current MHA v4 integration supports:

- dense, non-causal attention;
- head dimension 128;
- GQA ratios 1, 2, 4, 8, and 16; and
- the standard attention scale `1 / sqrt(128)`.

Attention masks are not supported. Use `TORCH_SDPA` or another compatible
backend for masked or causal attention.

Ulysses sequence parallelism is supported because its Q/K/V exchange completes
before the selected local attention backend runs. Ring attention uses a
separate ring kernel and does not execute `AITER_QUANT_ATTN`.

## Compilation and troubleshooting

The quantization and packed-layout boundaries are registered as custom
operators for `torch.compile`. To isolate compilation-related behavior, disable
compilation with:

```bash
vllm-omni serve <model> --enforce-eager ...
```

If backend initialization reports that `aiter.ops.mha_v4` is missing, update
or rebuild AITER. If a format is rejected, verify the active GPU architecture
against the format matrix above.

## Quality considerations

Low-precision formats trade numerical accuracy for attention throughput.
Compare output against `TORCH_SDPA` with the same prompt, seed, and sampling
settings before deploying a quantized format. Performance and quality depend
on the model, sequence length, GPU architecture, and selected recipe.
