# LTX-2 Family

> LTX-2 and LTX-2.3 text-to-video and image-to-video generation with synchronized audio

## Summary

- Vendor: Lightricks
- Models: `Lightricks/LTX-2`, `diffusers/LTX-2.3-Diffusers`
- Tasks: Text-to-video (T2V) and image-to-video (I2V)
- Mode: Offline inference or online serving with the OpenAI-compatible video API
- Maintainer: Community, @oglok

## Model And Pipeline Selection

Select the pipeline explicitly. T2V and I2V use the same checkpoint but
different public pipeline classes.

| Model | Checkpoint | Task | `--model-class-name` | Request batching |
|---|---|---|---|---|
| LTX-2 | `Lightricks/LTX-2` | One-stage T2V | `LTX2Pipeline` | Yes |
| LTX-2 | `Lightricks/LTX-2` | One-stage I2V | `LTX2ImageToVideoPipeline` | Yes |
| LTX-2 distilled | `rootonchair/LTX-2-19b-distilled` | Two-stage T2V | `LTX2TwoStagesPipeline` | No |
| LTX-2 distilled | `rootonchair/LTX-2-19b-distilled` | Two-stage I2V | `LTX2ImageToVideoTwoStagesPipeline` | No |
| LTX-2.3 | `diffusers/LTX-2.3-Diffusers` | One-stage T2V | `LTX23Pipeline` | Yes |
| LTX-2.3 | `diffusers/LTX-2.3-Diffusers` | One-stage I2V | `LTX23ImageToVideoPipeline` | Yes |

LTX-2.3 requires a Diffusers-format checkpoint. The upstream
`Lightricks/LTX-2.3` repository contains raw safetensors and is not directly
loadable by these pipeline classes.

## One-Stage Recipe Defaults

The selected pipeline defines the following model-specific one-stage recipe
defaults. The shared offline scripts apply these values when their corresponding
flags are omitted.

| Parameter | LTX-2 | LTX-2.3 |
|---|---:|---:|
| Width | 768 | 768 |
| Height | 512 | 512 |
| Frames | 121 | 121 |
| Frame rate | 24 | 24 |
| Denoise steps | 40 | 30 |
| Video CFG | 3.0 | 3.0 |
| Audio CFG | 7.0 | 7.0 |
| Video/audio STG | 1.0 | 1.0 |
| Video/audio modality guidance | 3.0 | 3.0 |
| Video/audio rescale | 0.7 | 0.7 |
| Video/audio STG blocks | `[29]` | `[28]` |

!!! warning "Set `num_frames` explicitly for online requests"

    The shared `/v1/videos` API initializes `num_frames` to `1`, so an online
    request that omits this field generates one frame rather than using the LTX
    recipe value of `121`. The LTX offline scripts default to `121`. Set
    `num_frames` explicitly in production requests so the behavior does not
    depend on the entrypoint.

The default negative prompt is also supplied by the model recipe. Passing the
top-level `guidance_scale` overrides both the video and audio CFG values. For
example, `guidance_scale=4.0` changes the defaults from video/audio `3.0/7.0`
to `4.0/4.0`.

## Serving

### Start A T2V Server

LTX-2:

```bash
vllm serve Lightricks/LTX-2 \
  --omni \
  --model-class-name LTX2Pipeline \
  --stage-init-timeout 600
```

LTX-2.3:

```bash
vllm serve diffusers/LTX-2.3-Diffusers \
  --omni \
  --model-class-name LTX23Pipeline \
  --stage-init-timeout 600
```

### Start An I2V Server

Use the corresponding image-to-video pipeline class:

```bash
vllm serve diffusers/LTX-2.3-Diffusers \
  --omni \
  --model-class-name LTX23ImageToVideoPipeline \
  --stage-init-timeout 600
```

Replace the model with `Lightricks/LTX-2` and the class with
`LTX2ImageToVideoPipeline` for LTX-2.

### T2V Request

This request keeps the selected pipeline's default guidance recipe:

```bash
curl -X POST http://localhost:8000/v1/videos/sync \
  -F "prompt=A cinematic close-up of ocean waves at golden hour." \
  -F "negative_prompt=worst quality, inconsistent motion, blurry, jittery, distorted" \
  -F "size=768x512" \
  -F "num_frames=121" \
  -F "fps=24" \
  -F "seed=42" \
  -o ltx_t2v.mp4
```

### I2V Request

The I2V pipeline requires exactly one initial image per request:

```bash
curl -X POST http://localhost:8000/v1/videos/sync \
  -F "prompt=A plush toy astronaut gently waving while the camera slowly pushes in." \
  -F "negative_prompt=worst quality, inconsistent motion, blurry, jittery, distorted" \
  -F "input_reference=@/absolute/path/to/reference.png" \
  -F "size=768x512" \
  -F "num_frames=121" \
  -F "fps=24" \
  -F "seed=42" \
  -o ltx_i2v.mp4
```

Use `image_reference` for a URL or JSON-safe image reference. Do not send
`input_reference` and `image_reference` together.

## Custom Guidance

LTX supports independent video and audio CFG, spatio-temporal guidance (STG),
cross-modality guidance, and guidance rescaling. Pass these model-specific
parameters in the JSON `extra_params` form field for online serving or through
`--extra-body` for the shared offline scripts.

### Parameters

| Canonical parameter | Type | Default | Effect | Accepted alias |
|---|---|---|---|---|
| `video_cfg_scale` | float | 3.0 | Video text CFG; `1.0` disables video CFG | `video_cfg_guidance_scale` |
| `audio_cfg_scale` | float | 7.0 | Audio text CFG; `1.0` disables audio CFG | `audio_cfg_guidance_scale` |
| `video_stg_scale` | float | 1.0 | Video STG strength; `0.0` disables video STG | `video_stg_guidance_scale` |
| `audio_stg_scale` | float | 1.0 | Audio STG strength; `0.0` disables audio STG | `audio_stg_guidance_scale` |
| `video_modality_scale` | float | 3.0 | Audio-to-video guidance; `1.0` disables it | `a2v_guidance_scale` |
| `audio_modality_scale` | float | 3.0 | Video-to-audio guidance; `1.0` disables it | `v2a_guidance_scale` |
| `video_rescale_scale` | float | 0.7 | Video guidance rescale; `0.0` disables it | None |
| `audio_rescale_scale` | float | 0.7 | Audio guidance rescale; `0.0` disables it | None |
| `video_stg_blocks` | int or list[int] | `[29]` / `[28]` | Transformer blocks perturbed for video STG | None |
| `audio_stg_blocks` | int or list[int] | `[29]` / `[28]` | Transformer blocks perturbed for audio STG | None |

The STG block default is `[29]` for LTX-2 and `[28]` for LTX-2.3. STG runs
only when its scale is nonzero and its block list is nonempty. If both a
canonical parameter and its alias are provided, the canonical parameter wins.

The standard `guidance_scale` request field has higher precedence than
`video_cfg_scale` and `audio_cfg_scale`: when provided, it sets both modalities
to the same CFG value. Omit it when using independent video/audio CFG values.

### Online Request

```bash
curl -X POST http://localhost:8000/v1/videos/sync \
  -F "prompt=Floating crystal islands in a cosmic starry sky, slow camera rotation." \
  -F "negative_prompt=low quality, blurry, noise, watermark, text" \
  -F "size=768x512" \
  -F "num_frames=121" \
  -F "fps=24" \
  -F "num_inference_steps=30" \
  -F "seed=42" \
  -F 'extra_params={"video_cfg_scale":3.0,"audio_cfg_scale":7.0,"video_stg_scale":1.0,"audio_stg_scale":1.0,"video_modality_scale":3.0,"audio_modality_scale":3.0,"video_rescale_scale":0.7,"audio_rescale_scale":0.7,"video_stg_blocks":[28],"audio_stg_blocks":[28]}' \
  -o ltx23_custom_guidance.mp4
```

### Offline Request

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model diffusers/LTX-2.3-Diffusers \
  --model-class-name LTX23Pipeline \
  --prompt "Floating crystal islands in a cosmic starry sky, slow camera rotation." \
  --negative-prompt "low quality, blurry, noise, watermark, text" \
  --width 768 --height 512 \
  --num-frames 121 --frame-rate 24 --fps 24 \
  --num-inference-steps 30 --seed 42 \
  --extra-body '{"video_cfg_scale":3.0,"audio_cfg_scale":7.0,"video_stg_scale":1.0,"audio_stg_scale":1.0,"video_modality_scale":3.0,"audio_modality_scale":3.0,"video_rescale_scale":0.7,"audio_rescale_scale":0.7,"video_stg_blocks":[28],"audio_stg_blocks":[28]}' \
  --output ltx23_custom_guidance.mp4
```

For I2V, use `examples/offline_inference/image_to_video/image_to_video.py`,
select the corresponding I2V pipeline class, add `--image`, and pass the same
`--extra-body` JSON.

### Python API

```python
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

sampling_params = OmniDiffusionSamplingParams(
    width=768,
    height=512,
    num_frames=121,
    frame_rate=24.0,
    num_inference_steps=30,
    seed=42,
    extra_args={
        "video_cfg_scale": 3.0,
        "audio_cfg_scale": 7.0,
        "video_stg_scale": 1.0,
        "audio_stg_scale": 1.0,
        "video_modality_scale": 3.0,
        "audio_modality_scale": 3.0,
        "video_rescale_scale": 0.7,
        "audio_rescale_scale": 0.7,
        "video_stg_blocks": [28],
        "audio_stg_blocks": [28],
    },
)
```

!!! warning "Request batching requires identical LTX guidance"

    One-stage LTX pipelines support request-level batching, but every request
    in one fused batch must resolve to exactly the same video and audio
    guidance configuration. This includes all CFG, STG, modality, rescale, and
    STG-block values. If requests with different effective LTX guidance are
    fused, the pipeline rejects the entire batch instead of executing separate
    sub-batches.

    Keep concurrent requests on one server guidance-homogeneous, or set
    `--max-num-seqs 1` when clients need independent guidance configurations.
    Shape, frame-rate, denoise-step, output-count, and standard CFG differences
    are already handled by the common request scheduler, but LTX-specific
    guidance is validated by the pipeline.

!!! warning "CFG parallel supports the CFG-only plan"

    LTX `--cfg-parallel-size 2` supports the positive/negative CFG-only plan
    without guidance rescale. Disable STG and modality guidance and set both
    rescale values to `0.0` before enabling CFG parallel. The full default
    guidance recipe is not CFG-parallel compatible.

## Standard Generation Parameters

| Online form field | Offline flag | Description |
|---|---|---|
| `prompt` | `--prompt` | Positive text prompt |
| `negative_prompt` | `--negative-prompt` | Negative text prompt; omitted values use the recipe default |
| `size` or `width`/`height` | `--width`, `--height` | Output dimensions; both dimensions must be divisible by 32 |
| `num_frames` | `--num-frames` | Output frame count and one input to audio-duration calculation; online default `1`, LTX offline default `121` |
| `fps` | `--fps`, `--frame-rate` | Output and model frame rate; LTX audio duration follows `num_frames / frame_rate` |
| `num_inference_steps` | `--num-inference-steps` | Denoise step count |
| `guidance_scale` | `--guidance-scale` | Optional common override for both video and audio CFG |
| `seed` | `--seed` | Per-request random seed |
| `extra_params` | `--extra-body` | JSON object containing the LTX-specific guidance parameters above |

## Operational Notes

- LTX-2 one-stage was previously validated on one H200 141GB, loading and
  peaking at approximately 73.5 GiB. Re-measure on the exact commit and
  hardware being deployed.
- LTX-2.3 combines a 22B transformer, Gemma text encoder, video VAE, audio VAE,
  and vocoder. Start validation on a 96GB-class GPU or use CPU/layerwise
  offload on smaller devices.
- `--stage-init-timeout 600` leaves enough time for large checkpoint loading
  and optional compilation warmup.
- For formal performance runs, use
  `tests/dfx/perf/tests/test_ltx2_vllm_omni.json` with
  `tests/dfx/perf/scripts/run_diffusion_benchmark.py`. Its cases include JSON
  marks for hardware, full-model, and diffusion filtering.
- The generated audio sample rate comes from the loaded audio/vocoder
  components; it is not a per-request guidance parameter.
- Two-stage LTX-2 support is currently limited to the distilled checkpoint.
  Its second denoise stage uses a fixed positive-only three-step schedule and
  does not support request-level batching.

## References

- LTX-2 checkpoint: <https://huggingface.co/Lightricks/LTX-2>
- LTX-2.3 raw checkpoints: <https://huggingface.co/Lightricks/LTX-2.3>
- LTX-2.3 Diffusers checkpoint: <https://huggingface.co/diffusers/LTX-2.3-Diffusers>
- [Online video generation guide](../../docs/user_guide/examples/online_serving/text_to_video.md)
- [Request-level batching](../../docs/user_guide/diffusion/request_batching.md)
- [Text-to-video offline example](../../examples/offline_inference/text_to_video/text_to_video.md)
- [Image-to-video offline example](../../examples/offline_inference/image_to_video/README.md)
