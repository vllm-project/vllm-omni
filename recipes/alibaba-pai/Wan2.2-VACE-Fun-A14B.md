# Wan2.2 VACE-Fun

> All-in-one video creation & editing (Wan2.2 VACE-Fun 14B, original / non-diffusers checkpoint)

## Summary

- Vendor: alibaba-pai
- Model: `alibaba-pai/Wan2.2-VACE-Fun-A14B`
- Task: VACE video generation & editing (T2V / I2V / R2V / V2V / inpainting)
- Mode: Offline inference
- Maintainer: Community

## When to use this recipe

Use this recipe to run the original (PAI) release of Wan2.2-VACE-Fun-A14B with
vLLM-Omni. This checkpoint ships in its native (non-diffusers) layout — two MoE
expert subfolders (`high_noise_model/` + `low_noise_model/`), a nested
`google/umt5-xxl` tokenizer, and `.pth` text-encoder / VAE. `Wan22VACEPipeline`
loads it directly: the original weight-key names and the transformer config are
converted to diffusers form in memory at load time, so no separate offline
conversion of the checkpoint is required.

The model is a two-expert (MoE) architecture — a high-noise and a low-noise
transformer switched at a noise-level boundary during denoising. The same
`Wan22VACEPipeline` already serves the single-expert diffusers **Wan2.1-VACE**
models (`Wan-AI/Wan2.1-VACE-*-diffusers`); for this checkpoint it loads both
experts and selects one per denoising step at the boundary. The original weight
keys are remapped to the pipeline's diffusers-format VACE transformer on load.

## References

- Upstream model card: <https://huggingface.co/alibaba-pai/Wan2.2-VACE-Fun-A14B>
- Related example: [`examples/offline_inference/vace/vace_video_generation.md`](../../examples/offline_inference/vace/vace_video_generation.md)
- Related issue: [vllm-project/vllm-omni#4206](https://github.com/vllm-project/vllm-omni/issues/4206)
- PR: [vllm-project/vllm-omni#4249](https://github.com/vllm-project/vllm-omni/pull/4249)

## Hardware Support

## GPU

### 1x NVIDIA A100 80GB

#### Environment

- OS: Linux
- Python: 3.12.13
- Driver: NVIDIA driver with CUDA 12.x runtime (validated on CUDA 12.9)
- vLLM version: 0.22.0
- vLLM-Omni version or commit: 0.22.x — built from PR #4249's branch (`main` aligned to vLLM v0.22.0)

#### Prerequisites

None. For `--mode i2v`, provide a reference image via `--image`.

#### Command

**Image-to-video (validated on 1x A100 80GB):**

```bash
python examples/offline_inference/vace/vace_video_generation.py \
  --model alibaba-pai/Wan2.2-VACE-Fun-A14B \
  --mode i2v --image ./i2v_input.JPG \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard" \
  --height 480 --width 832 --num-frames 81 --num-inference-steps 30 \
  --seed 42 --guidance-scale 5.0 \
  --vae-use-tiling --enforce-eager \
  --output vace_orig_full.mp4
```

See [`examples/offline_inference/vace/vace_video_generation.md`](../../examples/offline_inference/vace/vace_video_generation.md)
for the other VACE modes (T2V / R2V / V2V / inpainting) and their input flags.

#### Verification

The run produces a coherent `vace_orig_full.mp4`. Check:

1. The output video is written to the `--output` path.
2. Both experts load with no "weights not loaded" errors (the original keys are
   remapped to diffusers form on load).
3. The video is temporally coherent — the two experts are switched at the
   boundary; a single-expert run looks temporally unstable.

#### Notes

- **Key flags:**
  - `--mode <t2v|i2v|...>` — selects the VACE task by which inputs are provided.
  - `--guidance-scale 5.0` — single CFG scale, applied to both experts.
  - `--vae-use-tiling` — tiled VAE decoding to reduce peak VAE memory.
  - `--enforce-eager` — optional; skips torch.compile.
- **Two-expert (MoE) denoising:**
  - The boundary ratio defaults to `0.875` (matching the reference). The
    high-noise transformer runs above the boundary timestep and the low-noise
    transformer below it; both experts are loaded and driven across the denoise.
- **Known limitations:**
  - Validated on NVIDIA only; other platforms are not covered by this recipe.
