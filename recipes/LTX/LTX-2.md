# LTX-2 Family

> LTX-2 and LTX-2.3 text-to-video and image-to-video generation with synchronized audio

## Summary

- Vendor: Lightricks
- Models: `Lightricks/LTX-2`, `diffusers/LTX-2.3-Diffusers`
- Tasks: Text-to-video (T2V) and image-to-video (I2V)
- Mode: Offline inference or online serving with the OpenAI-compatible video API
- Maintainer: Community, @oglok

## Model And Pipeline Selection

`LTX2Pipeline` is the unified one-stage entry. Checkpoint metadata selects the
LTX-2 or LTX-2.3 component profile and recipe; omitting an image selects T2V,
while providing one initial image selects I2V.

| Model | Checkpoint | Task | `--model-class-name` | Request batching |
|---|---|---|---|---|
| LTX-2 | `Lightricks/LTX-2` | One-stage T2V/I2V | `LTX2Pipeline` | Yes |
| LTX-2 distilled | `rootonchair/LTX-2-19b-distilled` | Two-stage T2V | `LTX2TwoStagesPipeline` | No |
| LTX-2 distilled | `rootonchair/LTX-2-19b-distilled` | Two-stage I2V | `LTX2ImageToVideoTwoStagesPipeline` | No |
| LTX-2.3 | `diffusers/LTX-2.3-Diffusers` | One-stage T2V/I2V | `LTX2Pipeline` | Yes |

Both supported one-stage checkpoint repositories declare `LTX2Pipeline` in
`model_index.json`, so `--model-class-name` is optional. Version selection does
not depend on the repository or directory name: `model_version` is used when
present, otherwise the LTX-2.3 BWE vocoder declaration is the discriminator.
Unknown conversions use the LTX-2 defaults, matching the official fallback.

LTX-2.3 requires a Diffusers-format checkpoint. The upstream
`Lightricks/LTX-2.3` repository contains raw safetensors and is not directly
loadable by these pipeline classes.

## Breaking API Migration

`LTX2Pipeline` is the only supported one-stage class. Its direct Python
API accepts only `req` positionally; every optional `forward` argument is
keyword-only. Positional calls must be migrated as follows:

```python
# No longer supported
pipe(req, prompt)
pipe(req, image, prompt)

# Supported
pipe(req, prompt=prompt)
pipe(req, image=image, prompt=prompt)
```

Passing a second positional argument now raises `TypeError` immediately. This
is an intentional compatibility break for direct pipeline callers; the offline
scripts and serving API are unaffected because they already use named fields.

The consolidation also removes three old registry names. There are no aliases;
configs and commands that still select one of these names fail registry
resolution and must migrate to `LTX2Pipeline`:

| Removed name | Replacement |
|---|---|
| `LTX23Pipeline` | `LTX2Pipeline`; the checkpoint selects LTX-2.3 and no image selects T2V |
| `LTX2ImageToVideoPipeline` | `LTX2Pipeline` with `image=` |
| `LTX23ImageToVideoPipeline` | `LTX2Pipeline` with `image=`; the checkpoint selects LTX-2.3 |

These are two intentional breaking changes: optional direct-call arguments are
now keyword-only, and the three version/task-specific registry names are no
longer accepted. The shared offline and serving entrypoints already use named
fields and checkpoint metadata, so only direct Python callers and explicit
`--model-class-name` overrides require migration.

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

### Start A Server

LTX-2:

```bash
vllm serve Lightricks/LTX-2 \
  --omni \
  --stage-init-timeout 600
```

LTX-2.3:

```bash
vllm serve diffusers/LTX-2.3-Diffusers \
  --omni \
  --stage-init-timeout 600
```

The same server accepts T2V and I2V requests. An I2V request is selected by its
initial image rather than by a different pipeline class.

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
  --prompt "Floating crystal islands in a cosmic starry sky, slow camera rotation." \
  --negative-prompt "low quality, blurry, noise, watermark, text" \
  --width 768 --height 512 \
  --num-frames 121 --frame-rate 24 --fps 24 \
  --num-inference-steps 30 --seed 42 \
  --extra-body '{"video_cfg_scale":3.0,"audio_cfg_scale":7.0,"video_stg_scale":1.0,"audio_stg_scale":1.0,"video_modality_scale":3.0,"audio_modality_scale":3.0,"video_rescale_scale":0.7,"audio_rescale_scale":0.7,"video_stg_blocks":[28],"audio_stg_blocks":[28]}' \
  --output ltx23_custom_guidance.mp4
```

For I2V, use `examples/offline_inference/image_to_video/image_to_video.py`, add
`--image`, and pass the same `--extra-body` JSON.

### Python API

Build a `DiffusionRequestBatch`, pass it as the only positional argument, and
put normal per-request values in `OmniDiffusionSamplingParams`. The same call
selects T2V or I2V from the presence of an image.

```python
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
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

req = DiffusionRequestBatch(
    [
        OmniDiffusionRequest(
            prompt={"prompt": "Cherry blossoms moving in a light breeze"},
            sampling_params=sampling_params,
            request_id="ltx-example",
        )
    ]
)

t2v_output = pipe(req)
i2v_output = pipe(req, image=image)
```

Direct keyword arguments are fallbacks for low-level callers. For example,
`pipe(req, prompt=prompt, width=768)` is valid, but request-batch prompt and
sampling fields take precedence where the table below says “request first.”

#### Complete `forward` Surface

| Argument | Type and default | Meaning, precedence, and constraints |
|---|---|---|
| `req` | `DiffusionRequestBatch`, required | The only positional argument. Holds prompts and one `OmniDiffusionSamplingParams` per request. |
| `image` | image or batch, `None` | Initial frame. A direct image takes precedence over images embedded in `req`; no image selects T2V. Each I2V prompt accepts exactly one image, and a fused batch cannot mix T2V and I2V. |
| `prompt` | `str \| list[str]`, `None` | Positive-text fallback. Nonempty `req.prompts` take precedence. Mutually exclusive with `prompt_embeds`. |
| `negative_prompt` | `str \| list[str]`, `None` | Negative-text fallback. Per-request negative prompts win, then this value, then the model recipe default. Mutually exclusive with `negative_prompt_embeds`. |
| `height` | `int`, `None` | Output height. Request sampling value wins, then this fallback, then the recipe default. Must be divisible by 32. |
| `width` | `int`, `None` | Output width with the same precedence as `height`; must be divisible by 32. |
| `num_frames` | `int`, `None` | Number of output video frames. Request value wins, then this fallback, then the recipe default. It also determines audio duration with `frame_rate`. |
| `frame_rate` | `float`, `None` | Model frame rate. Request `frame_rate` wins over request `fps`, then this fallback, then the recipe default. |
| `num_inference_steps` | `int`, `None` | Denoise-step fallback. Request value wins, then this value, then the recipe default; values are clamped to at least 2. A custom `sigmas` schedule determines the actual executed steps. |
| `sigmas` | `list[float]`, `None` | Final scheduler boundaries. Request `sigmas` win over this fallback. All requests in a fused batch must use the same list. See [Custom Sigma Schedules](#custom-sigma-schedules). |
| `timesteps` | `list[int]`, `None` | Generic compatibility slot. LTX accepts only `None`; use `sigmas` instead. |
| `guidance_scale` | `float`, `None` | Common CFG fallback that sets both video and audio CFG. An explicitly provided request `guidance_scale` wins; omit both when using independent modality CFG values. |
| `guidance_rescale` | `float`, `None` | Generic compatibility slot. Only `None` or `0.0` is accepted; use `video_rescale_scale` and `audio_rescale_scale`. |
| `noise_scale` | `float`, `0.0` | Initial-noise compatibility slot. LTX accepts only `0.0`. |
| `num_videos_per_prompt` | `int`, `1` | Direct output-count fallback. Positive request `num_outputs_per_prompt` takes precedence. |
| `generator` | generator or list, `None` | Explicit RNG source. When omitted, request generators or seeds are collated. A list must match the effective output batch. |
| `latents` | `torch.Tensor`, `None` | Initial video latents. Request tensors take precedence and are collated. Packed `[B, S, C]` and validated unpacked video layouts are accepted by one-stage pipelines. |
| `audio_latents` | `torch.Tensor`, `None` | Initial audio latents. Request `audio_latents` take precedence and are collated. |
| `prompt_embeds` | `torch.Tensor`, `None` | Precomputed positive conditioning that bypasses text encoding. Request prompt payload wins; requires `prompt_attention_mask` and cannot be combined with `prompt`. |
| `negative_prompt_embeds` | `torch.Tensor`, `None` | Precomputed negative conditioning. Request payload wins; requires `negative_prompt_attention_mask` and cannot be combined with `negative_prompt`. |
| `prompt_attention_mask` | `torch.Tensor`, `None` | Mask paired with `prompt_embeds`; request `prompt_attention_mask` or `attention_mask` wins. |
| `negative_prompt_attention_mask` | `torch.Tensor`, `None` | Mask paired with `negative_prompt_embeds`; request `negative_prompt_attention_mask` or `negative_attention_mask` wins. |
| `decode_timestep` | float or list, `0.0` | Timestep-conditioned video-VAE decode value. Request value wins. A list may have length 1, prompt batch size, or effective output batch size. |
| `decode_noise_scale` | float or list, `None` | Decode-noise amount with the same list rules. Request value wins. When omitted, it follows `decode_timestep`; the defaults therefore add no noise. |
| `output_type` | `str`, `"np"` | Request value wins. `"np"` decodes video and audio; `"latent"` returns video/audio latents without VAE or vocoder decode. |
| `return_dict` | `bool`, `True` | Generic compatibility slot. LTX accepts only `True` and returns the standard `DiffusionOutput`. |
| `attention_kwargs` | `dict`, `None` | Generic compatibility slot. Public per-call attention kwargs are unsupported; configure the attention backend at engine startup. |
| `max_sequence_length` | `int`, `None` | Maximum prompt-token length. Request value wins, then this fallback, then the loaded tokenizer limit. |

For ordinary engine use, put values in `req` rather than duplicating them as
direct fallbacks. Prompt/sampling fields, custom sigmas, latents, embeddings,
decode controls, and output type are resolved per request; recipe defaults are
used only after both request and direct fallback values are absent.

### Custom Sigma Schedules

One-stage LTX requests can override the recipe's default sigma schedule through
the Python API. Set final scheduler boundary values directly on
`OmniDiffusionSamplingParams.sigmas`. Each non-terminal value produces one
denoise step; a terminal `0.0` is appended when omitted. The custom schedule,
not `num_inference_steps`, determines the executed step count.

```python
sampling_params = OmniDiffusionSamplingParams(
    num_inference_steps=4,
    sigmas=[1.0, 0.75, 0.5, 0.25],
)
```

A request-level schedule takes precedence over the recipe schedule, and all
requests in one fused batch must use identical `sigmas`. The `/v1/videos` form
API and bundled offline CLI do not currently expose this field; use the Python
request API when supplying a custom schedule.

### Request-Object Field Mapping

Most `forward` names are identical on `OmniDiffusionSamplingParams`. The
exceptions and prompt-payload fields are:

| `forward` concept | Request-object field |
|---|---|
| `num_videos_per_prompt` | `num_outputs_per_prompt` |
| `frame_rate` | `frame_rate`, falling back to `fps` |
| RNG | `generator`, otherwise `seed` |
| `image` | Prompt `multi_modal_data.image` or `additional_information.image` |
| `prompt`, `negative_prompt` | Prompt dictionary fields |
| Embeddings and masks | Prompt fields or `additional_information` |
| LTX modality guidance | Sampling `extra_args` entries listed under [Custom Guidance](#custom-guidance) |

These generic diffusion controls remain in the shared request surface for
API consistency but are intentionally rejected by LTX when non-default:

| Rejected control | Use instead |
|---|---|
| `timesteps` or `flow_shift` | Provide the final `sigmas` schedule |
| Common `guidance_rescale` | `video_rescale_scale` and `audio_rescale_scale` |
| Nonzero `noise_scale` | LTX one-stage always starts from its standard noise state |
| Public `attention_kwargs` | Configure the attention backend at engine startup |
| `return_dict=False` | Consume the standard `DiffusionOutput` |

!!! warning "Request batching requires identical LTX guidance"

    One-stage LTX pipelines support request-level batching, but every request
    in one fused batch must resolve to exactly the same video and audio
    guidance configuration. This includes all CFG, STG, modality, rescale, and
    STG-block values. If requests with different effective LTX guidance are
    fused, the pipeline rejects the entire batch instead of executing separate
    sub-batches. Custom sigma schedules are also a batch invariant: every
    request must provide the same `sigmas` list, or all requests must omit it.

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
  does not support request-level batching. Official LTX two-stage/HQ execution
  is outside the scope of this implementation.

## References

- LTX-2 checkpoint: <https://huggingface.co/Lightricks/LTX-2>
- LTX-2.3 raw checkpoints: <https://huggingface.co/Lightricks/LTX-2.3>
- LTX-2.3 Diffusers checkpoint: <https://huggingface.co/diffusers/LTX-2.3-Diffusers>
- [Online video generation guide](../../docs/user_guide/examples/online_serving/text_to_video.md)
- [Request-level batching](../../docs/user_guide/diffusion/request_batching.md)
- [Text-to-video offline example](../../examples/offline_inference/text_to_video/text_to_video.md)
- [Image-to-video offline example](../../examples/offline_inference/image_to_video/README.md)
