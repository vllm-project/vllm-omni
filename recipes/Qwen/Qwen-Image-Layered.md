# Qwen-Image-Layered for layered image editing on 2x RTX 5880 48GB

## Summary

- Vendor: Qwen
- Model: `Qwen/Qwen-Image-Layered`
- Task: Layered image editing (outputs multiple RGBA layers from a single input image)
- Mode: Offline inference
- Maintainer: Community

## When to use this recipe

Use this recipe when you do **not** have access to a large-memory GPU (e.g.,
80 GB A100) but have two 48 GB GPUs available (e.g., RTX 5880, RTX A6000, or
L40S). The model weights are approximately 45 GB, which exceeds the capacity of
a single 48 GB card once runtime overhead is considered. This recipe shows a
tested configuration using tensor parallelism across two GPUs combined with CPU
offloading to stay within memory limits.

## References

- Related example under `examples/`:
  [`examples/offline_inference/image_to_image/image_edit.py`](../../examples/offline_inference/image_to_image/image_edit.py)
- Related issue or discussion:
  [#2905](https://github.com/vllm-project/vllm-omni/pull/2905)

## Hardware Support

## GPU

### 2x RTX 5880 48GB (tensor parallelism)

#### Environment

- OS: Ubuntu 22.04
- Python: 3.11
- Driver / runtime: NVIDIA Driver 570.172.18, CUDA 12.8
- vLLM-Omni version or commit: `a683b1dd` (main)

#### Command

Run from the repository root:

```bash
python examples/offline_inference/image_to_image/image_edit.py \
    --model Qwen/Qwen-Image-Layered \
    --image input.jpg \
    --prompt "" \
    --output layered \
    --num-inference-steps 50 \
    --layers 2 \
    --color-format RGBA \
    --tensor-parallel-size 2 \
    --enable-cpu-offload
```

Key parameters:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `--tensor-parallel-size 2` | 2 | Split the DiT transformer across 2 GPUs via tensor parallelism |
| `--enable-cpu-offload` | flag | Automatically swap the text encoder (Qwen2.5-VL, ~30 GB) to CPU when the transformer runs, and vice versa |
| `--layers 2` | 2 | Number of layers to decompose the input image into (default 4; outputs exactly `layers` images). No hard min/max constraint in code; tested with 2 and 4 |

#### Verification

After the command finishes, check for the output files:

```bash
ls -lh layered_0.png layered_1.png
```

Expected: two RGBA PNG files (480 × 864) corresponding to the requested
layers — one for the foreground subject and one for the background. The input
image used for testing was 720 × 1280.

#### Notes

- Memory usage: Model weights load at ~15.7 GiB per GPU with CPU offloading
  enabled. Peak GPU memory during inference is ~34.2 GB reserved per GPU.
- Key flags: `--tensor-parallel-size 2` is **required** — the full model
  (~45 GB) does not fit on a single 48 GB card. `--enable-cpu-offload` is
  strongly recommended to keep peak memory well within the 48 GB limit.
- Without `--enable-cpu-offload`: the text encoder stays resident on GPU,
  raising per-GPU usage to ~41.8 GiB and leaving very little headroom for the
  denoising loop and VAE decode.
- VAE tiling: If you encounter memory pressure during VAE decode, add
  `--vae-use-tiling` to reduce VAE peak memory by decoding in tiles.

#### Performance (5-run average, 2x RTX 5880, 50 steps, 720×1280 input)

##### `--layers 2`

| Phase | Time | Notes |
|-------|------|-------|
| Pre-processing (VAE encode) | ~32 ms | Image → latent |
| Text encoding (incl. CPU↔GPU offload) | ~24 s | Qwen2.5-VL text encoder moved from CPU to GPU, executed, then moved back |
| Denoising (50 steps) | ~41 s | ~0.83 s/step |
| VAE decode + post-processing | < 1 s | Latent → output images |
| **End-to-end total** | **~65.6 s** | |

##### `--layers 4`

| Phase | Time | Notes |
|-------|------|-------|
| Pre-processing (VAE encode) | ~38 ms | Image → latent |
| Text encoding (incl. CPU↔GPU offload) | ~24 s | Same as layers 2 (unaffected by layer count) |
| Denoising (50 steps) | ~61 s | ~1.23 s/step |
| VAE decode + post-processing | < 1 s | Latent → output images |
| **End-to-end total** | **~85.9 s** | |

##### Comparison

| Metric | layers=2 | layers=4 | Delta |
|--------|----------|----------|-------|
| End-to-end total | ~65.6 s | ~85.9 s | +31% |
| Denoising per step | ~0.83 s | ~1.23 s | +48% |
| Peak GPU memory | 34.23 GB | 34.23 GB | — |

- Peak GPU memory: **34.23 GB** reserved per GPU (consistent across all runs
  and both layer counts — memory is dominated by model weights, not layer count).
- The text-encoding phase is dominated by the CPU↔GPU weight transfer
  of the ~30 GB Qwen2.5-VL encoder, not by compute itself.
- Increasing `--layers` mainly impacts the denoising phase (larger latent
  tensor), while text encoding and VAE stages remain essentially unchanged.
