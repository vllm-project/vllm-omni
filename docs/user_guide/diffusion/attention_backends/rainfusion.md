# RainFusion Attention

`RAINFUSION_ATTN` runs MindIE-SD RainFusion (`rf_v2`) block-sparse video
attention on Ascend NPU. It pools 128-token key blocks, ranks them per query
block, and attends to the highest-scoring blocks after arranging video tokens
in `(t, h, w)` order.

Only the video segment is sparse. Prefix rows such as text, visual conditions,
and audio, plus first-frame blocks, remain dense. Unsupported calls—including
warmup steps, skipped layers, missing video geometry, and video segments below
32 blocks—delegate to `FLASH_ATTN`, so compatible models can select RainFusion
globally.

## Configuration

| Key | Valid values | Meaning |
| --- | --- | --- |
| `sparsity` | finite, `[0, 1]` | Nominal dropped-key-block fraction; default `0.8`; `0` disables sparsity |
| `start_step` | integer, `>= 0` | Number of early denoise steps kept dense |
| `end_step` | integer, `>= 0` | Number of final denoise steps kept dense (tail fallback) |
| `precision` | `"bf16"`, `"fp8"`, `"mix"` | Kernel precision mode; default `"bf16"`. Requires MindIE-SD with `sparse_attention(precision=...)` |
| `exec_kernel` | `"rainfusion"`, `"eagle_bsa"` | Which kernel executes the rf_v2 pattern; unset lets MindIE-SD decide (the CANN RainFusion operator). Requires MindIE-SD with `sparse_attention(exec_kernel=...)` |
| `skip_layers` | selector such as `"0-3,38"` | DiT blocks kept dense |

```bash
vllm-omni serve MiniMaxAI/MiniMax-H3 \
  --diffusion-attention-config '{"default":{"backend":"RAINFUSION_ATTN",\
    "block_sparse":{"sparsity":0.8,"start_step":0}}}'
```

```python
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, BlockSparseSpec

config = AttentionConfig(
    default=AttentionSpec(
        backend="RAINFUSION_ATTN",
        block_sparse=BlockSparseSpec(
            sparsity=0.8,
            start_step=0,
            skip_layers="0-1",
        ),
    ),
)
```

Tune `start_step`, then `sparsity`, then `skip_layers`. Increasing the dense
early-step window is usually the cheapest way to recover global structure;
use layer exclusions only after a same-seed dense comparison identifies
sensitive blocks.

## Requirements and compatibility

RainFusion requires Ascend NPU and `mindiesd`. Selecting it on another
platform raises. It is incompatible with ring sequence parallelism because
the kernel needs the complete key sequence for ranking; use Ulysses sequence
parallelism with `ring_degree=1`.

The `precision` knob requires a MindIE-SD release whose `sparse_attention`
accepts `precision=`; older releases accept it through `**kwargs` but
silently ignore it and run the BF16 path. `RAINFUSION_ATTN` raises at runtime
when a non-`bf16` precision is requested without that support, so pin a
compatible MindIE-SD release when using `precision="mix"` or `"fp8"`.

`exec_kernel` behaves the same way: MindIE-SD must expose
`sparse_attention(exec_kernel=...)`, or `RAINFUSION_ATTN` raises rather than
running its default operator under an explicit request for another one.
Switching the executor does not change the sparse pattern, the spatial
reorder, or the inverse reorder -- both routes execute the mask rf_v2 already
computed -- so the choice is about the implementation, not the pattern. Measure
the two kernels on the workload at hand before pinning one: `eagle_bsa` is
MindIE-SD's in-repo `eagle_block_sparse_attention` kernel, built for
`ascend910b`/`ascend910_93` with fp16/bf16 queries, and it is not uniformly
faster than the CANN RainFusion operator it replaces.

`exec_kernel` only applies to the bf16 rf_v2 path. A non-`bf16` precision
selects rf_v3, and the hybrid reference-KV path runs `hybrid_attention`; both
raise if `exec_kernel` is set instead of silently ignoring it.

The two executors consume the same mask and reorder but are different kernels,
and they are not bit-identical. On our Ascend 910 T2VA workload (int8, 50
steps) the Eagle output measured 34.7 dB PSNR against the RainFusion operator,
below our 45 dB gate, so treat this as a measurement knob rather than an
assumption of equivalent output.

```bash
vllm-omni serve MiniMaxAI/MiniMax-H3 \
  --diffusion-attention-config '{"default":{"backend":"RAINFUSION_ATTN",\
    "block_sparse":{"sparsity":0.8,"start_step":0,"exec_kernel":"eagle_bsa"}}}'
```

## Geometry handling

RainFusion handles arbitrary video grids by rearranging video spatially and,
when necessary, promoting a real-video suffix to the always-kept prefix. It
does not pad because `rf_v2` does not consume a padding attention mask.

For MiniMax-H3 at 1344x768, grid `(62, 24, 42)` produces 62,496 video rows.
The implementation promotes 2,976 rows and leaves 59,520 sparse rows
(`465 x 128`) so the mask and kernel tiling remain aligned.

Spatial grids aligned to 8x8 are still preferable. A latent height or width
not divisible by 8 creates an always-kept suffix and reduces realized
sparsity. At the image level, that means width and height divisible by 256.
Protected video tails require a compatible MindIE-SD release.

For common configuration and selector behavior, see the
[attention backend overview](../attention_backends.md) and the
[backend selection design](../../../design/feature/attention_backend_selection.md).

## Wan2.2 T2V quantization on Ascend

`RAINFUSION_ATTN` accepts `diffusion_kv_cache_dtype: fp8` or `mxfp4` for
Wan2.2 T2V A14B.
MindIE-SD must expose `sparse_attention` with `rf_v3` and an explicit `precision`
argument. Omni passes the selected precision directly to this API; the installed
native operators must support that precision. Unsupported inputs or an
incompatible Python API raise an actionable error; Omni does not change BSA
precision automatically. Update `diffusion_kv_cache_dtype` to a precision
supported by the installed MindIE-SD, or unset it. Native execution errors also
propagate without retry.
BSA FP8 and MXFP4 retain MindIE-SD's default rotation policy.

`diffusion_kv_cache_skip_layers`/`diffusion_kv_cache_skip_steps` explicitly
select floating-point BSA and keep attention sparse. The existing `block_sparse`
warmup/layer exclusions and short-sequence threshold instead select Dense
attention, using the configured Dense precision. Leave `block_sparse.precision`
at its default when using `diffusion_kv_cache_dtype` as the shared Dense/BSA
precision setting.

Configure BSA quantization in the model's existing deployment YAML with
`backend: RAINFUSION_ATTN` and stage-level `diffusion_kv_cache_dtype: fp8` or
`mxfp4`. The precision and skip selectors are siblings of
`diffusion_attention_config`; they are not per-role `quant` fields.
For 40 denoising steps, `diffusion_kv_cache_skip_steps: "0,1,38,39"` and
`diffusion_kv_cache_skip_layers: "0,39"` are recommended starting points. These
fixed step indices do not automatically follow a changed step count. Selected
forwards remain floating-point BSA; unsupported BSA precision is an error rather
than an automatic fallback. See
[quantized attention](../../quantization/quantized_kvcache.md#wan22-t2v-quantized-attention-on-ascend)
for the shared configuration structure and index semantics.
