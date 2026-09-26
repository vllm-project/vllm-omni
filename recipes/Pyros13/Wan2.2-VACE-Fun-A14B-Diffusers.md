# Wan2.2 VACE-Fun Diffusers

> All-in-one video creation and editing with Wan2.2 VACE-Fun A14B in diffusers format.

## Summary

- Vendor: Pyros13
- Model: `Pyros13/Wan2.2-VACE-Fun-A14B-Diffusers`
- Base model: `alibaba-pai/Wan2.2-VACE-Fun-A14B`
- Task: VACE video generation and editing (T2V / I2V / R2V / V2V / inpainting)
- Mode: Offline inference and OpenAI-compatible online serving
- Maintainer: Community

## When To Use This Recipe

Use this recipe to run Wan2.2-VACE-Fun-A14B through the standard diffusers-style
`Wan22VACEPipeline` path in vLLM-Omni. The checkpoint is a diffusers conversion of
`alibaba-pai/Wan2.2-VACE-Fun-A14B`, with `model_index.json`, `transformer/`,
`transformer_2/`, `text_encoder/`, `tokenizer/`, `vae/`, and `scheduler/`
subfolders. Because the model already uses the expected diffusers layout, vLLM-Omni
does not need any original-format key remapping or `.pth` component loaders.

The model is a two-expert architecture. The high-noise `transformer` and low-noise
`transformer_2` are switched by the Wan2.2 boundary during denoising.

## References

- Model card: <https://huggingface.co/Pyros13/Wan2.2-VACE-Fun-A14B-Diffusers>
- Base model card: <https://huggingface.co/alibaba-pai/Wan2.2-VACE-Fun-A14B>
- Related example: [`examples/offline_inference/vace/vace_video_generation.md`](../../examples/offline_inference/vace/vace_video_generation.md)
- Related issue: [vllm-project/vllm-omni#4206](https://github.com/vllm-project/vllm-omni/issues/4206)
- PR: [vllm-project/vllm-omni#4667](https://github.com/vllm-project/vllm-omni/pull/4667)

## Offline Inference

```bash
python examples/offline_inference/vace/vace_video_generation.py \
  --model Pyros13/Wan2.2-VACE-Fun-A14B-Diffusers \
  --mode i2v --image ./i2v_input.JPG \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard" \
  --height 480 --width 832 --num-frames 81 --num-inference-steps 30 \
  --seed 42 --guidance-scale 5.0 \
  --vae-use-tiling --enforce-eager \
  --output vace_fun_i2v.mp4
```

See [`examples/offline_inference/vace/vace_video_generation.md`](../../examples/offline_inference/vace/vace_video_generation.md)
for other VACE modes and their input flags.

## Online Serving

### Server

```bash
vllm serve Pyros13/Wan2.2-VACE-Fun-A14B-Diffusers --omni \
  --model-class-name Wan22VACEPipeline \
  --vae-use-tiling \
  --enforce-eager \
  --port 8091
```

For multi-GPU sequence-parallel serving, add the same parallelism flags you would
use offline, for example `--ulysses-degree 4`. The explicit `--model-class-name`
is optional for this diffusers checkpoint, but keeping it in examples makes the
selected pipeline obvious.

### Client

```bash
no_proxy=127.0.0.1 \
curl -X POST http://127.0.0.1:8091/v1/videos/sync \
  -F "prompt=Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard" \
  -F "input_reference=@/absolute/path/to/i2v_input.JPG" \
  -F "width=832" \
  -F "height=480" \
  -F "num_frames=81" \
  -F "fps=16" \
  -F "num_inference_steps=30" \
  -F "guidance_scale=5.0" \
  --output vace_fun_i2v_serve.mp4
```

For video-to-video or inpainting-style requests, upload the reference video with
`input_reference=@/absolute/path/to/input.mp4` or use `video_reference` with a URL
or JSON-safe data URI.

## Cache-DiT / TaylorSeer Notes

`Wan22VACEPipeline` uses the shared Wan2.2 Cache-DiT enabler for both the
high-noise and low-noise transformers. The adapter intentionally wraps only the
main denoising `blocks`. VACE `vace_blocks` form a conditioning branch: each step
they combine the current latent with the VACE context to produce hints that are
then injected into selected main blocks. Recomputing that branch preserves the
control signal; caching the main backbone still gives the acceleration target.

## Validation

Validated on NVIDIA B300 with `mkt_part_000.mp4`, 736x1280, 61 frames, 30
inference steps, `seed=1`, and `boundary_ratio=0.875`.

The Pyros13 diffusers checkpoint produced a byte-identical MP4 to the
`alibaba-pai/Wan2.2-VACE-Fun-A14B` original-format checkpoint under the same
vLLM-Omni pipeline and inputs:

- SHA256: `2e2db74a612a009b0d6ad6d739e039883f0de84ce5cdf2676324daf78fb05ef8`
- Frame shape: 61 x 736 x 1280
- MAE / RMSE / max pixel diff: 0 / 0 / 0
- PSNR: infinite
- Mean SSIM: 1.0
