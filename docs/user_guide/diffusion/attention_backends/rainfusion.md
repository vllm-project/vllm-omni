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
