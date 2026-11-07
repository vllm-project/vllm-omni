# Helios for video generation

## Summary

- Vendor: Helios
- Model: `BestWishYsh/Helios-Base`, `BestWishYsh/Helios-Mid`, `BestWishYsh/Helios-Distilled`
- Task: Text-to-video generation
- Mode: Offline inference with the shared text_to_video example
- Maintainer: Community

## When to use this recipe

Use this recipe when you want a known-good starting point for running Helios
video generation with vLLM-Omni. The concrete command below focuses on
`BestWishYsh/Helios-Base` text-to-video generation on one NVIDIA H20 GPU via the
shared `text_to_video.py` example. The same example also supports Helios-Mid and
Helios-Distilled through the generic `--extra-body` flag (see below).
Image-to-video and video-to-video require image/video conditioning inputs and
are not covered by the text-to-video example.

## References

- Upstream repository: <https://github.com/PKU-YuanGroup/Helios>
- Model weights:
  <https://huggingface.co/BestWishYsh/Helios-Base>
- Related example under `examples/`:
  [`examples/offline_inference/text_to_video/text_to_video.md`](../../examples/offline_inference/text_to_video/text_to_video.md)

## Hardware Support

## GPU

### 1x NVIDIA H20

#### Environment

- OS: Ubuntu Linux x86_64
- Python: 3.12.12
- Driver / runtime: NVIDIA driver `580.126.20`, CUDA `13.0`
- Hardware: 1x NVIDIA H20 GPU from an 8x H20 host
- vLLM version: `0.19.0`
- vLLM-Omni version or commit: `a3903810`

#### Command

Run the baseline Helios-Base text-to-video example from the repository root:

```bash
cd examples/offline_inference/text_to_video

python text_to_video.py \
  --model BestWishYsh/Helios-Base \
  --prompt "A dynamic time-lapse video showing the rapidly moving scenery from the window of a speeding train." \
  --guidance-scale 5.0 \
  --output helios_t2v_base.mp4
```

To use cache-dit acceleration, run on a vLLM-Omni checkout that includes Helios cache-dit support and enable the cache backend:

```bash
cd examples/offline_inference/text_to_video

python text_to_video.py \
  --cache-backend cache_dit \
  --enable-cache-dit-summary \
  --model BestWishYsh/Helios-Base \
  --prompt "A dynamic time-lapse video showing the rapidly moving scenery from the window of a speeding train." \
  --guidance-scale 5.0 \
  --output helios_t2v_base.mp4
```

#### Verification

The script should print the resolved generation configuration, total generation
time, and output path. A successful run ends with output similar to:

```text
Total generation time: <seconds> seconds (<milliseconds> ms)
Saved generated video to helios_t2v_base.mp4
```

#### Important flags and deploy config

- `--model BestWishYsh/Helios-Base` selects the base Helios checkpoint used by
  this recipe.
- `--guidance-scale 5.0` matches the Helios-Base recommendation. Use
  `--guidance-scale 1.0` for Helios-Distilled.
- `--cache-backend cache_dit` enables the cache-dit acceleration path.
- `--enable-cache-dit-summary` prints cache-dit summary information after
  diffusion forward passes.
- No separate deploy config is required for this offline recipe; the shared
  `text_to_video.py` example configures the pipeline through its arguments.
- The VAE decodes in float32 by default. To run it in bfloat16 instead, set
  `"vae_dtype": "bfloat16"` in the `model_config` supplied to `Omni` /
  `AsyncOmni` (not a per-request `extra_body` option). On one A100-PCIE-40GB
  this cuts a 33-frame 384x640 chunk decode from 1253 ms to 665 ms (~1.9x) at
  PSNR 59.0 dB / SSIM 0.9997 versus float32, and lowers VAE peak memory from
  6.2 GiB to 4.0 GiB.
- `vae_patch_parallel_size > 1` now also distributes the Helios VAE decode
  across the diffusion group (tiling is enabled automatically), like the other
  Wan-VAE video models.
- Helios-specific knobs (declared in `vllm_omni/model_extras/helios.py`) are
  passed via the generic `--extra-body` JSON flag:
    - Helios-Mid: `--extra-body '{"is_enable_stage2": true, "pyramid_num_inference_steps_list": [20, 20, 20], "use_cfg_zero_star": true, "use_zero_init": true, "zero_steps": 1}'`
    - Helios-Distilled: `--extra-body '{"is_enable_stage2": true, "pyramid_num_inference_steps_list": [2, 2, 2], "is_amplify_first_chunk": true}'`

#### Known limitations

- Helios generates video in 33-frame chunks. For best performance, set
  `--num-frames` to a multiple of `33`; non-multiple values are rounded up to
  the nearest multiple of `33`.
