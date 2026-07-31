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

`LTX2Pipeline` is the unified one-stage entry. Checkpoint metadata selects the
LTX version, and an optional initial image selects I2V instead of T2V. The
class name is normally inferred. LTX-2.3 requires the Diffusers checkpoint;
raw `Lightricks/LTX-2.3` safetensors are sidecars, not a loadable pipeline.

`LTX2TwoStagePipeline` samples the regular model at half resolution, upsamples,
then refines with the distilled LoRA. `LTX2DistilledPipeline` instead uses the
fully merged distilled Transformer in both stages and never loads that LoRA.
Always select the distilled class explicitly. Both entries support T2V and I2V.

For local sidecars, set `VLLM_OMNI_LTX_ARTIFACTS_DIR` to a directory containing
the required files with their original names. This is an authoritative
override: missing files fail startup. Otherwise, the runtime searches the model
root and the matching Lightricks repository. LTX ignores per-component
`model_paths` overrides.

## Diffusion Feature Matrix

Support is pipeline-specific; a model-level mark does not imply support in
every multi-stage entry.

| Diffusion feature | `LTX2Pipeline` | `LTX2TwoStagePipeline` | `LTX2DistilledPipeline` |
|---|:---:|:---:|:---:|
| T2V / I2V | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| Request-level batching | ✅ | ❌ | ❌ |
| Custom sigma schedule | ✅ | ❌ | ❌ |
| Request video/audio latents | ✅ | ❌ | ❌ |
| CFG | ✅ | ⚠️ Stage 1 | — positive-only |
| STG | ✅ | ⚠️ Stage 1 | — positive-only |
| Cross-modality guidance / rescale | ✅ | ⚠️ Stage 1 | — positive-only |
| Tensor parallel | ✅ | ✅ | ✅ |
| Ulysses sequence parallel | ⚠️ strict mode | ⚠️ strict mode | ⚠️ strict mode |
| Ring sequence parallel | ⚠️ | ⚠️ | ⚠️ |
| CFG parallel | ✅ full guidance plan | ⚠️ Stage 1, full guidance plan | — positive-only |
| HSDP | ✅ | ✅ | ✅ |
| Pipeline parallel | ❌ | ❌ | ❌ |
| Expert parallel | — | — | — |
| Module-wise CPU offload | ✅ | ✅ | ✅ |
| Layerwise CPU offload | ✅ | ✅ | ✅ |
| VAE patch parallel decode | ✅ | ✅ | ✅ |
| Quantization | ⚠️ | ⚠️ | ⚠️ |
| Internal distilled LoRA | — | ✅ layer-fused (default)/dynamic/resident | — merged weights |
| Cache-DiT | ✅ | ❌ | ❌ |
| TeaCache | ❌ | ❌ | ❌ |
| Step execution | ❌ | ❌ | ❌ |

Legend: ✅ supported; ⚠️ supported with the restriction shown below; ❌
explicitly unsupported; — not applicable.

- Ring SP requires audio latent length divisible by the SP size because it
  cannot consume the padding mask; otherwise use pure Ulysses.
- LTX accepts strict Ulysses only. `advanced_uaa` is rejected. Its 32 attention
  heads support common strict degrees such as 2, 4, and 8.
- LoRA determines quantization compatibility: `dynamic` supports a quantized
  base, `layer_fused` requires unquantized BF16, and `resident` requires dense
  source weights (serialized quantized checkpoints are rejected).
- Cache-DiT is one-stage only. Multi-stage requests fail at startup instead of
  using an uncached or stale Stage 2.

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

A second positional argument raises `TypeError`. Offline and serving
entrypoints already use named fields and are unaffected.

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

The recipe supplies the default negative prompt. Top-level `guidance_scale`
overrides both CFG values. The online API defaults `num_frames` to `1`, so set
it explicitly; the offline examples default to `121`.

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

Ordinary two-stage defaults to `layer_fused`: each affected BF16 weight is
materialized only during its Stage 2 layer call, matching official weight-space
fusion without a second DiT. Set `VLLM_OMNI_LTX_TWO_STAGE_LORA_MODE=dynamic`
for a quantized base; it computes `base(x) + lora_b(lora_a(x))` and is not
bitwise-equivalent to fusion. `resident` keeps a second pre-merged DiT.
Unsupported dtype, quantization, checkpoint format, or mode fails at startup.

## Serving

Start a one-stage checkpoint:

```bash
vllm serve Lightricks/LTX-2 --omni --stage-init-timeout 600
# or
vllm serve diffusers/LTX-2.3-Diffusers --omni --stage-init-timeout 600
```

Select a two-stage class explicitly:

```bash
vllm serve rootonchair/LTX-2-19b-distilled --omni \
  --model-class-name LTX2DistilledPipeline --stage-init-timeout 600
# LTX-2.3 full-distilled; the v1.1 x2 upsampler is resolved when absent
vllm serve diffusers/LTX-2.3-Distilled-Diffusers --omni \
  --model-class-name LTX2DistilledPipeline --stage-init-timeout 600
# Ordinary LTX-2.3 two-stage with local sidecars
VLLM_OMNI_LTX_ARTIFACTS_DIR=/data/models/LTX-2.3-sidecars \
vllm serve diffusers/LTX-2.3-Diffusers --omni \
  --model-class-name LTX2TwoStagePipeline --stage-init-timeout 600
```

All entries handle T2V and I2V. A T2V request is:

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

For I2V, add exactly one initial image to the request above:

```bash
-F "input_reference=@/absolute/path/to/reference.png"
```

Use `image_reference` for a URL or JSON-safe reference, but not together with
`input_reference`.

## Guidance

One-stage and ordinary Stage 1 support independent video/audio CFG, STG,
cross-modality guidance, and rescaling. Distilled stages and ordinary Stage 2
are fixed positive-only.

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

STG block defaults are LTX-2 / LTX-2.3. Canonical names override aliases;
top-level `guidance_scale` overrides both CFG fields.

Pass these fields in online `extra_params` or offline `--extra-body`:

```bash
# Add to the curl request above
-F 'extra_params={"video_cfg_scale":3.0,"audio_cfg_scale":7.0}'

# Add to text_to_video.py or image_to_video.py
--extra-body '{"video_cfg_scale":3.0,"audio_cfg_scale":7.0}'
```

### Guidance Parallelism

`cfg_parallel_size` distributes the complete guidance plan—text CFG, STG, and
cross-modality passes. Rescaling happens after predictions are gathered. The
default plan has four Transformer passes per step: `cond`, `uncond`, `ptb`,
and `mod`.

| `--cfg-parallel-size` | Passes per rank | Guidance-slot utilization | Notes |
|---:|---:|---:|---|
| `1` | 4 | 100% | Single-rank fused guidance batch |
| `2` | 2 | 100% | Recommended two-rank configuration |
| `4` | 1 | 100% | One guidance pass per rank |

Other positive sizes are accepted but may waste padded execution slots; LTX
warns with the expected utilization.

Start an LTX-2.3 server with two-way guidance parallelism:

```bash
vllm serve diffusers/LTX-2.3-Diffusers --omni \
  --cfg-parallel-size 2 --stage-init-timeout 600
```

Device count is the product of all parallel dimensions. Positive-only
distilled pipelines do not benefit from `cfg_parallel_size > 1`.

### Python API

Put per-request values in `OmniDiffusionSamplingParams`, then pass the
`DiffusionRequestBatch` as the only positional argument:

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

Direct keywords such as `pipe(req, prompt=prompt, width=768)` are low-level
fallbacks. Request fields take precedence unless noted below.

### Complete `forward` Surface

| Argument | Type/default | Meaning and constraints |
|---|---|---|
| `req` | `DiffusionRequestBatch`, required | Only positional argument; contains prompts and sampling parameters. |
| `image` | image or batch, `None` | Direct value wins. No image selects T2V; I2V needs one image per prompt. Batches cannot mix T2V/I2V. |
| `prompt` | string or list, `None` | Positive-text fallback; exclusive with `prompt_embeds`. |
| `negative_prompt` | string or list, `None` | Negative-text fallback; exclusive with negative embeddings. |
| `height`, `width` | `int`, `None` | Request → keyword → recipe default; divisible by 32 one-stage or 64 two-stage. |
| `num_frames` | `int`, `None` | Request → keyword → default; must be `8k+1`. With `frame_rate`, sets audio duration. |
| `frame_rate` | `float`, `None` | Request `frame_rate` → request `fps` → direct value → recipe default. |
| `num_inference_steps` | `int`, `None` | Controls one-stage or ordinary Stage 1; distilled Stage 1 is fixed at 8. `sigmas` overrides it. |
| `sigmas` | list of float, `None` | One-stage only; fused requests must share a schedule. |
| `timesteps` | list of int, `None` | Must be `None`; use `sigmas`. |
| `guidance_scale` | `float`, `None` | Common video/audio CFG fallback for one-stage and ordinary Stage 1. |
| `guidance_rescale` | `float`, `None` | Accepts only `None` or `0.0`; use the modality rescale fields. |
| `noise_scale` | `float`, `0.0` | Must be `0.0`. |
| `num_videos_per_prompt` | `int`, `1` | Output-count fallback; request `num_outputs_per_prompt` wins. |
| `generator` | generator or list, `None` | Explicit RNG; lists must match the output batch. |
| `latents`, `audio_latents` | tensor, `None` | One-stage only; request tensors win. Video accepts packed `[B, S, C]` or validated unpacked layouts. |
| `prompt_embeds` | tensor, `None` | Positive conditioning; requires its mask and excludes `prompt`. |
| `negative_prompt_embeds` | tensor, `None` | One-stage negative conditioning; requires its mask and excludes `negative_prompt`. |
| `prompt_attention_mask` | tensor, `None` | Positive embedding mask; request mask wins. |
| `negative_prompt_attention_mask` | tensor, `None` | Negative embedding mask; request mask wins. |
| `decode_timestep` | float or list, `0.0` | Video-VAE decode timestep; lists may match 1, prompt batch, or output batch. |
| `decode_noise_scale` | float or list, `None` | Same list rules; defaults to `decode_timestep`. |
| `output_type` | string, `"np"` | `"np"` decodes; `"latent"` skips VAE/vocoder decode. |
| `return_dict` | `bool`, `True` | Only `True` is accepted. |
| `attention_kwargs` | dict, `None` | Unsupported per call; configure attention at startup. |
| `max_sequence_length` | `int`, `None` | Request → direct value → tokenizer limit. |

In requests, `num_videos_per_prompt` maps to `num_outputs_per_prompt`; images
and prompt data live in the prompt payload; LTX guidance lives in `extra_args`.

### Recipe-Specific Request Capabilities

| Override | One-stage | Ordinary two-stage | Distilled two-stage |
|---|---|---|---|
| Guidance | Supported | Stage 1 only; Stage 2 is positive-only | Fixed positive-only |
| Negative prompt/embeddings | Supported | Supported by Stage 1 | Rejected |
| `num_inference_steps` | Supported | Controls Stage 1; Stage 2 uses 3 | Fixed at 8 for Stage 1; Stage 2 uses 3 |
| Custom `sigmas` | Supported | Rejected | Rejected; both phases use fixed schedules |
| Video/audio latents | Supported | Rejected | Rejected |

Checks apply equally to `forward` keywords and sampling parameters;
unsupported values fail instead of being ignored.

### Custom Sigma Schedules

One-stage Python requests may set final scheduler boundaries directly:

```python
params = OmniDiffusionSamplingParams(sigmas=[1.0, 0.75, 0.5, 0.25])
```

Each nonterminal value is one denoise step; terminal `0.0` is appended when
omitted. This overrides `num_inference_steps`. Fused requests must share the
list. The video form API and bundled offline CLI do not expose `sigmas`.

### Constraints

- LTX rejects non-default `timesteps`, `flow_shift`, `guidance_rescale`,
  `noise_scale`, `attention_kwargs`, and `return_dict=False`. Use final
  `sigmas`, modality rescale fields, startup attention configuration, and
  standard `DiffusionOutput`.
- Fused one-stage requests must resolve to identical LTX guidance and sigma
  schedules; otherwise use `--max-num-seqs 1`.
- Sequence parallelism may pad audio latents. Pure Ulysses masks the padding;
  Ring cannot, so audio length must be SP-divisible. Use `ring_degree=1` or a
  divisible request shape.

## Operational Notes

- LTX-2 one-stage previously peaked near 73.5 GiB on an H200; remeasure on your
  commit and hardware. LTX-2.3 generally needs a 96GB-class GPU or offload.
- The upsampler uses the pipeline dtype and remains resident during denoise
  offload because it runs only at the phase boundary.
- Output audio sample rate comes from the loaded components.
- For benchmarks, use `tests/dfx/perf/tests/test_ltx2_vllm_omni.json` with
  `tests/dfx/perf/scripts/run_diffusion_benchmark.py`.

## References

- <https://huggingface.co/Lightricks/LTX-2>
- <https://huggingface.co/Lightricks/LTX-2.3>
- <https://huggingface.co/diffusers/LTX-2.3-Diffusers>
- <https://huggingface.co/diffusers/LTX-2.3-Distilled-Diffusers>
- [Online video generation](../../docs/user_guide/examples/online_serving/text_to_video.md)
- [Diffusion execution modes](../../docs/user_guide/diffusion/execution_modes.md)
- [T2V offline example](../../examples/offline_inference/text_to_video/text_to_video.md)
- [I2V offline example](../../examples/offline_inference/image_to_video/README.md)
