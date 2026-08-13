# LTX-2 Family

> LTX-2 and LTX-2.3 text-to-video and image-to-video generation with synchronized audio

## Pipelines

| `--model-class-name` | Task | Required checkpoint repositories |
|---|---|---|
| `LTX2Pipeline` | LTX-2 one-stage T2V/I2V | `Lightricks/LTX-2` |
| `LTX2TwoStagePipeline` | LTX-2 ordinary two-stage T2V/I2V | `Lightricks/LTX-2` |
| `LTX2DistilledPipeline` | LTX-2 full-distilled two-stage T2V/I2V | `rootonchair/LTX-2-19b-distilled` |
| `LTX2Pipeline` | LTX-2.3 one-stage T2V/I2V | `diffusers/LTX-2.3-Diffusers` |
| `LTX2TwoStagePipeline` | LTX-2.3 ordinary two-stage T2V/I2V | `diffusers/LTX-2.3-Diffusers`<br>`Lightricks/LTX-2.3` |
| `LTX2DistilledPipeline` | LTX-2.3 full-distilled two-stage T2V/I2V | `diffusers/LTX-2.3-Distilled-Diffusers`<br>`Lightricks/LTX-2.3` |

Repositories in the table are download units. A full pipeline repository
contains the Transformer, text encoder, connectors, VAEs, vocoder, scheduler,
and tokenizer; an additional repository supplies LoRA or upsampler sidecars.
The runtime first searches the model root for sidecars under their official
filenames, then downloads them from the matching Lightricks Hub repository.

`LTX2Pipeline` is the unified one-stage entry. Checkpoint metadata selects the
LTX-2 or LTX-2.3 profile; omitting an image selects T2V, while one initial
image selects I2V. Both one-stage repositories declare this class, so
`--model-class-name` is optional. LTX-2.3 requires the Diffusers checkpoint;
the raw `Lightricks/LTX-2.3` safetensors repository is not directly loadable.

`LTX2TwoStagePipeline` samples the regular model at half resolution, upsamples,
then refines with the distilled LoRA. `LTX2DistilledPipeline` uses a fully
merged distilled Transformer in both stages. Both entries support T2V and I2V;
select their class explicitly.

## API Migration

Only `req` may be passed positionally to `LTX2Pipeline`; every optional
`forward` argument is keyword-only:

```python
# No longer supported
pipe(req, prompt)
pipe(req, image, prompt)

# Supported
pipe(req, prompt=prompt)
pipe(req, image=image, prompt=prompt)
```

The consolidation also removes these registry names without aliases:

| Removed name | Replacement |
|---|---|
| `LTX23Pipeline` | `LTX2Pipeline`; checkpoint metadata selects LTX-2.3 |
| `LTX2ImageToVideoPipeline` | `LTX2Pipeline` with `image=` |
| `LTX23ImageToVideoPipeline` | `LTX2Pipeline` with `image=`; checkpoint metadata selects LTX-2.3 |
| `LTX2TwoStagesPipeline` | `LTX2DistilledPipeline` |
| `LTX2ImageToVideoTwoStagesPipeline` | `LTX2DistilledPipeline` with `image=` |

Passing any second positional argument now raises `TypeError`. These changes
affect direct Python callers and explicit `--model-class-name` overrides;
offline and serving entrypoints already use named fields and are unaffected.

## One-Stage Defaults

| Parameter | LTX-2 | LTX-2.3 |
|---|---:|---:|
| Width × height | 768 × 512 | 768 × 512 |
| Frames / frame rate | 121 / 24 | 121 / 24 |
| Denoise steps | 40 | 30 |
| Video/audio CFG | 3.0 / 7.0 | 3.0 / 7.0 |
| Video/audio STG | 1.0 / 1.0 | 1.0 / 1.0 |
| Video/audio modality guidance | 3.0 / 3.0 | 3.0 / 3.0 |
| Video/audio rescale | 0.7 / 0.7 | 0.7 / 0.7 |
| Video/audio STG blocks | `[29]` / `[29]` | `[28]` / `[28]` |

The model recipe also supplies the default negative prompt. Top-level
`guidance_scale` overrides both video and audio CFG values. The online video
API defaults `num_frames` to `1`, so set it explicitly; the offline LTX scripts
default to `121`.

## Two-Stage Defaults

| Parameter | Ordinary | Full-distilled |
|---|---:|---:|
| Final width × height | 1536 × 1024 | 1536 × 1024 |
| Stage 1 width × height | 768 × 512 | 768 × 512 |
| Frames / frame rate | 121 / 24 | 121 / 24 |
| Stage 1 / Stage 2 steps | 40 (LTX-2) or 30 (LTX-2.3) / 3 | 8 / 3 |
| Guidance | Stage 1 guided; Stage 2 positive-only | Fixed positive-only |

API dimensions are final dimensions and must be divisible by 64. All LTX
requests require `num_frames = 8k+1`. Ordinary Stage 1 uses the LTX-2 or
LTX-2.3 one-stage defaults shown above. Distilled schedules are fixed, so
`num_inference_steps`, when supplied, must be `8`. Both entries reject custom
sigmas and input latents.

Ordinary two-stage uses layer-fused LoRA for an unquantized BF16 Transformer
and automatically switches to dynamic LoRA when quantization is enabled.

## Serving

Start either one-stage checkpoint:

```bash
vllm serve Lightricks/LTX-2 --omni --stage-init-timeout 600
```

```bash
vllm serve diffusers/LTX-2.3-Diffusers --omni --stage-init-timeout 600
```

Start a two-stage checkpoint with its explicit class:

```bash
vllm serve rootonchair/LTX-2-19b-distilled --omni \
  --model-class-name LTX2DistilledPipeline --stage-init-timeout 600
# LTX-2.3 full-distilled; the v1.1 x2 upsampler is resolved when absent
vllm serve diffusers/LTX-2.3-Distilled-Diffusers --omni \
  --model-class-name LTX2DistilledPipeline --stage-init-timeout 600
# Ordinary LTX-2.3 two-stage
vllm serve diffusers/LTX-2.3-Diffusers --omni \
  --model-class-name LTX2TwoStagePipeline \
  --enable-layerwise-offload \
  --stage-init-timeout 600
```

The same server handles T2V and I2V. A T2V request using the selected recipe's
default guidance is:

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

For I2V, provide exactly one initial image:

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

Use `image_reference` for a URL or JSON-safe image reference. Do not provide it
together with `input_reference`.

## Guidance

One-stage and ordinary Stage 1 support independent video/audio CFG,
spatio-temporal guidance (STG), cross-modality guidance, and rescaling.
Distilled stages and ordinary Stage 2 are fixed positive-only.

| Parameter | Default | Effect | Alias |
|---|---:|---|---|
| `video_cfg_scale` | 3.0 | Video text CFG; `1.0` disables it | `video_cfg_guidance_scale` |
| `audio_cfg_scale` | 7.0 | Audio text CFG; `1.0` disables it | `audio_cfg_guidance_scale` |
| `video_stg_scale` | 1.0 | Video STG; `0.0` disables it | `video_stg_guidance_scale` |
| `audio_stg_scale` | 1.0 | Audio STG; `0.0` disables it | `audio_stg_guidance_scale` |
| `video_modality_scale` | 3.0 | Audio-to-video guidance; `1.0` disables it | `a2v_guidance_scale` |
| `audio_modality_scale` | 3.0 | Video-to-audio guidance; `1.0` disables it | `v2a_guidance_scale` |
| `video_rescale_scale` | 0.7 | Video guidance rescale; `0.0` disables it | — |
| `audio_rescale_scale` | 0.7 | Audio guidance rescale; `0.0` disables it | — |
| `video_stg_blocks` | `[29]` / `[28]` | Perturbed video transformer blocks | — |
| `audio_stg_blocks` | `[29]` / `[28]` | Perturbed audio transformer blocks | — |

The STG block defaults are shown as LTX-2 / LTX-2.3. A canonical name wins
over its alias. Explicit top-level `guidance_scale` wins over both CFG fields;
omit it for independent video/audio CFG.

Pass these fields in online `extra_params` or offline `--extra-body`:

```bash
# Add to the curl request above
-F 'extra_params={"video_cfg_scale":3.0,"audio_cfg_scale":7.0}'

# Add to text_to_video.py or image_to_video.py
--extra-body '{"video_cfg_scale":3.0,"audio_cfg_scale":7.0}'
```

### Guidance Parallelism

For LTX, `cfg_parallel_size` is the number of ranks used to execute the
complete guidance plan in parallel. Despite the legacy `cfg` name, it covers
text CFG, STG, and cross-modality guidance passes; guidance rescaling is
applied after all pass predictions are gathered.

The default one-stage recipe and ordinary Stage 1 have four Transformer passes
per denoise step: `cond`, `uncond`, `ptb` (STG), and `mod`
(cross-modality). The useful balanced configurations are therefore:

| `--cfg-parallel-size` | Passes per rank | Guidance-slot utilization | Notes |
|---:|---:|---:|---|
| `1` | 4 | 100% | Single-rank fused guidance batch |
| `2` | 2 | 100% | Recommended two-rank configuration |
| `4` | 1 | 100% | One guidance pass per rank |

Other positive sizes are accepted. When the pass count is not divisible by
`cfg_parallel_size`, ranks are padded to an equal number of execution slots
and LTX emits a warning with the expected utilization. For example, four
passes on three ranks use six slots and have 66.7% guidance-slot utilization.

Start an LTX-2.3 server with two-way guidance parallelism:

```bash
vllm serve diffusers/LTX-2.3-Diffusers --omni \
  --cfg-parallel-size 2 --stage-init-timeout 600
```

The total device count is the product of `cfg_parallel_size` and the other
configured parallel dimensions. Positive-only distilled phases do not benefit
from `cfg_parallel_size > 1`.

### Python API

Normal per-request values belong in `OmniDiffusionSamplingParams`. Pass the
resulting `DiffusionRequestBatch` as the only positional argument:

```python
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

params = OmniDiffusionSamplingParams(
    width=768,
    height=512,
    num_frames=121,
    frame_rate=24.0,
    num_inference_steps=30,
    seed=42,
    extra_args={"video_cfg_scale": 3.0, "audio_cfg_scale": 7.0},
)
req = DiffusionRequestBatch([
    OmniDiffusionRequest(
        prompt={"prompt": "Cherry blossoms moving in a light breeze"},
        sampling_params=params,
        request_id="ltx-example",
    )
])

t2v_output = pipe(req)
i2v_output = pipe(req, image=image)
```

Direct keyword arguments are low-level fallbacks, for example
`pipe(req, prompt=prompt, width=768)`. Request fields take precedence unless
noted below.

### Complete `forward` Surface

| Argument | Type/default | Meaning and constraints |
|---|---|---|
| `req` | `DiffusionRequestBatch`, required | Only positional argument; contains prompts and per-request sampling parameters. |
| `image` | image or batch, `None` | Direct value wins over request images; no image selects T2V. I2V accepts one image per prompt, and a batch cannot mix T2V/I2V. |
| `prompt` | string or list, `None` | Positive-text fallback; request prompts win. Mutually exclusive with `prompt_embeds`. |
| `negative_prompt` | string or list, `None` | Fallback after request values and before the recipe default; mutually exclusive with negative embeddings. |
| `height` | `int`, `None` | Request → direct value → recipe default; divisible by 32 one-stage or 64 two-stage. |
| `width` | `int`, `None` | Same precedence and alignment as `height`. |
| `num_frames` | `int`, `None` | Request → direct value → recipe default; must be `8k+1` and also determines audio duration with `frame_rate`. |
| `frame_rate` | `float`, `None` | Request `frame_rate` → request `fps` → direct value → recipe default. |
| `num_inference_steps` | `int`, `None` | Request → direct value → recipe default; minimum 2 one-stage and ordinary Stage 1, fixed at 8 for distilled Stage 1. Custom one-stage `sigmas` determine actual steps. |
| `sigmas` | list of float, `None` | One-stage only. Request values win; every request in a fused batch must use the same schedule. |
| `timesteps` | list of int, `None` | Compatibility slot; LTX accepts only `None`. Use `sigmas`. |
| `guidance_scale` | `float`, `None` | One-stage and ordinary Stage 1 common video/audio CFG fallback; an explicit request value wins. Distilled phases use fixed positive-only guidance. |
| `guidance_rescale` | `float`, `None` | Accepts only `None` or `0.0`; use the modality rescale fields. |
| `noise_scale` | `float`, `0.0` | Compatibility slot; LTX accepts only `0.0`. |
| `num_videos_per_prompt` | `int`, `1` | Output-count fallback; positive request `num_outputs_per_prompt` wins. |
| `generator` | generator or list, `None` | Explicit RNG; otherwise request generators/seeds are collated. Lists must match the effective output batch. |
| `latents` | tensor, `None` | One-stage only. Request tensors win; packed `[B, S, C]` and validated unpacked video layouts are accepted. |
| `audio_latents` | tensor, `None` | One-stage only. Request tensors win and are collated. |
| `prompt_embeds` | tensor, `None` | Precomputed positive conditioning; requires `prompt_attention_mask` and cannot accompany `prompt`. |
| `negative_prompt_embeds` | tensor, `None` | One-stage and ordinary Stage 1 conditioning; requires its mask and cannot accompany `negative_prompt`. |
| `prompt_attention_mask` | tensor, `None` | Mask for positive embeddings; request `prompt_attention_mask` or `attention_mask` wins. |
| `negative_prompt_attention_mask` | tensor, `None` | Mask for negative embeddings; request negative mask fields win. |
| `decode_timestep` | float or list, `0.0` | Video-VAE decode timestep; request value wins. Lists may match 1, prompt batch, or output batch. |
| `decode_noise_scale` | float or list, `None` | Same list rules; request value wins. Defaults to `decode_timestep`. |
| `output_type` | string, `"np"` | Request value wins. `"np"` decodes output; `"latent"` skips VAE/vocoder decode. |
| `return_dict` | `bool`, `True` | Compatibility slot; only `True` is accepted. |
| `attention_kwargs` | dict, `None` | Per-call values are unsupported; configure attention at engine startup. |
| `max_sequence_length` | `int`, `None` | Request → direct value → tokenizer limit. |

Request-object naming differs only in a few places: `num_videos_per_prompt`
maps to `num_outputs_per_prompt`; images and prompt text/embeddings live in the
request prompt payload; LTX guidance fields live in sampling `extra_args`.

### Recipe-Specific Request Capabilities

| Override | One-stage | Ordinary two-stage | Distilled two-stage |
|---|---|---|---|
| Guidance | Supported | Stage 1 only; Stage 2 is positive-only | Fixed positive-only |
| Negative prompt/embeddings | Supported | Supported by Stage 1 | Rejected |
| `num_inference_steps` | Supported | Controls Stage 1; Stage 2 uses 3 | Fixed at 8 for Stage 1; Stage 2 uses 3 |
| Custom `sigmas` | Supported | Rejected | Rejected; both phases use fixed schedules |
| Video/audio latents | Supported | Rejected | Rejected |

These capability checks apply equally to direct `forward` keywords and
values in `OmniDiffusionSamplingParams`; unsupported values fail instead of
being silently ignored.

### Custom Sigma Schedules

One-stage Python requests may set final scheduler boundaries directly:

```python
params = OmniDiffusionSamplingParams(sigmas=[1.0, 0.75, 0.5, 0.25])
```

Each nonterminal value produces one denoise step, and a terminal `0.0` is
appended when omitted. This schedule overrides `num_inference_steps`. All
requests in a fused batch must use the same list. The video form API and
bundled offline CLI do not currently expose `sigmas`.

### Constraints

- LTX rejects non-default `timesteps`, `flow_shift`, `guidance_rescale`,
  `noise_scale`, `attention_kwargs`, and `return_dict=False`. Use final
  `sigmas`, modality rescale fields, startup attention configuration, and the
  standard `DiffusionOutput` respectively.
- Fused one-stage requests must resolve to identical LTX guidance and sigma
  schedules. Keep concurrent requests guidance-homogeneous or use
  `--max-num-seqs 1`.
- `--cfg-parallel-size` shards the complete LTX guidance plan, including STG,
  modality guidance, and rescale-compatible prediction gathering.
- Sequence parallelism may pad audio latents. Pure Ulysses masks the padding;
  Ring cannot, so audio length must be SP-divisible. Use `ring_degree=1` or a
  divisible request shape.
- Cache-DiT is one-stage only; multi-stage configurations are rejected.

## Operational Notes

- LTX-2 one-stage previously loaded and peaked at about 73.5 GiB on one H200
  141GB; remeasure on the deployed commit and hardware.
- LTX-2.3 includes a 22B transformer, Gemma encoder, two VAEs, and a vocoder.
  Start on a 96GB-class GPU or use CPU/layerwise offload on smaller devices.
- The distilled upsampler uses the configured pipeline dtype and participates
  in component discovery/device placement; it remains resident during denoise
  offload because it is used only at the phase boundary.
- The output audio sample rate comes from the loaded components and is not a
  request parameter.
- For benchmarks, use `tests/dfx/perf/tests/test_ltx2_vllm_omni.json` with
  `tests/dfx/perf/scripts/run_diffusion_benchmark.py`.
- Ordinary and full-distilled two-stage T2V/I2V are supported; HQ execution
  remains out of scope.

## References

- <https://huggingface.co/Lightricks/LTX-2>
- <https://huggingface.co/Lightricks/LTX-2.3>
- <https://huggingface.co/diffusers/LTX-2.3-Diffusers>
- <https://huggingface.co/diffusers/LTX-2.3-Distilled-Diffusers>
- [Online video generation](../../docs/user_guide/examples/online_serving/text_to_video.md)
- [Diffusion execution modes](../../docs/user_guide/diffusion/execution_modes.md)
- [T2V offline example](../../examples/offline_inference/text_to_video/text_to_video.md)
- [I2V offline example](../../examples/offline_inference/image_to_video/README.md)
