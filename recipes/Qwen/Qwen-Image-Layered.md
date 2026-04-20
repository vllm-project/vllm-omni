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
    --cfg-scale 4.0 \
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
| `--layers 2` | 2 | Number of output layers (RGBA) |
| `--cfg-scale 4.0` | 4.0 | True classifier-free guidance scale (requires `--negative-prompt` to take effect) |

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
- Generation time: ~65 seconds end-to-end on 2x RTX 5880 with 50 inference
  steps (including model loading and warmup).
