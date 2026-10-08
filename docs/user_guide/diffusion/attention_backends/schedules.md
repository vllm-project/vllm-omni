# Attention Schedules

An attention schedule selects a named attention profile at zero-based denoising
step indices. Profiles are prepared at startup; requests select ranges, not new
backends. Steps outside a range use the base attention configuration.

Wan2.2 T2V, I2V, S2V and VACE publish progress in request mode.
MiniMax-H3 and HunyuanImage-3.0 publish it in request and step modes.
Other pipelines reject configured profiles at startup. Backend and model
requirements still apply; selecting a profile does not prevent kernel fallbacks.

## Format

The server configuration contains `profiles` and `default`:

- `profiles` maps names to complete attention configurations with `default`
  and optional `per_role` specs, as in `--diffusion-attention-config`.
- Profiles do not inherit the base configuration or backend/quantization
  environment variables. An unspecified role uses the platform default.
- Profile names start with an ASCII letter, followed by letters, digits,
  underscores or hyphens. All profiles are validated, even unused ones.
- The top-level `default` is a list of ranges. An empty configuration with
  no profiles is equivalent to not configuring schedules.

Each range has exactly `start`, `end` and `profile`:

| Field | Meaning |
| --- | --- |
| `start` | Integer >= 0; first included step. |
| `end` | Integer > `start`; first excluded step, or `null` for the sequence's end. Required. |
| `profile` | A profile declared at startup. |

Ranges must be ordered and non-overlapping. Adjacent ranges are allowed;
an open-ended range must be last. Gaps use the base configuration.
Ranges are not clipped: they must fit the actual timestep sequence, which may
differ from `num_inference_steps`. CFG evaluations share one step index.
Each S2V clip and each H3 seed/continuation window restarts at step 0.
Prompt encoding and decoding run with the base configuration.

## Server configuration

This example keeps the base backend for steps 0–9 and selects Skip-Softmax
from step 10 onward. It requires a model/platform that supports
[TRTLLM attention](trtllm.md#requirements); it is not suitable for HunyuanImage-3.0.

```bash
vllm-omni serve <model> \
  --diffusion-attention-backend TRTLLM_ATTN \
  --diffusion-attention-schedule '{
    "profiles": {
      "sparse": {
        "default": {
          "backend": "TRTLLM_ATTN",
          "skip_softmax": {"threshold": 0.05}
        }
      }
    },
    "default": [{"start": 10, "end": null, "profile": "sparse"}]
  }'
```

Dotted flags are also accepted: set
`--diffusion-attention-schedule.profiles.sparse.default.backend TRTLLM_ATTN`
and pass `--diffusion-attention-schedule.default` the ranges as one JSON list.
Set all backend options within the profile, including `quant`,
`fastvideo_vsa_topk` or Skip-Softmax settings when needed.

## Stage configuration

Set `diffusion_attention_schedule` on the diffusion stage in a deploy YAML,
using the same object as the server flag. Set its base configuration separately
with `diffusion_attention_config`. In the bundled disaggregated H3 and
HunyuanImage-3.0 deployments, the diffusion stage is `stage_id: 1`.

CLI overrides deep-merge profile mappings into the stage configuration.
A CLI `default` list replaces the stage list rather than appending ranges.

## Per-request configuration

| `attention_schedule` value | Effect |
| --- | --- |
| Omitted or `null` | Inherit defaults. |
| `[]` | Disable scheduling for this request. |
| Non-empty ranges | Replace defaults; do not merge. |

Video endpoints first apply a stage's `default_sampling_params`, if present;
an omitted or `null` request value preserves that stage default.

Put the field in the endpoint's model-specific extras:

- `/v1/videos` and `/v1/videos/sync`: JSON-encoded `extra_params` form field.
- `/v1/images/generations`: `extra_params`, only for a single diffusion stage;
  multi-stage image deployments ignore it. Use chat for the bundled Hunyuan deployment.
- Diffusion `/v1/chat/completions`: `extra_args` at the root or inside
  `extra_body`, but not both. With the OpenAI client, use
  `extra_body={"extra_args": {"attention_schedule": [...]}}`.
- Offline: `OmniDiffusionSamplingParams(attention_schedule=[...])`.

For the example server, an override can be
`{"attention_schedule":[{"start":20,"end":null,"profile":"sparse"}]}`.
Speech, realtime video, image edits and audio generation have no dedicated
schedule request handling.

## Batching and errors

Requests co-batch only with equal normalized request schedules after stage
defaults. Omitted schedules and explicit server-equivalent ranges remain
distinct batch keys. Scheduled H3 and Hunyuan step batches run one transformer
forward per request so each uses its own progress.

Malformed or unknown-profile schedules return HTTP 400 before dispatch.
Out-of-bounds ranges fail before denoising and retain the client error through
the runner and multiprocess executor. Asynchronous `/v1/videos` reports these
errors on the job record rather than the creation response.

## Compatibility limits

- Full compile scope and configured cache backends are rejected with profiles,
  even with empty default ranges. Regional compilation keeps attention eager
  on every request, including `[]`; other block operations remain compiled.
  Different candidate output layouts may trigger recompilation.
- H3 rejects non-empty schedules with request-scoped Cache-DiT (`quality=high`)
  or `latent_refine`. Disable the schedule or the conflicting option.
  Step mode rejects these H3 options regardless of schedule.
- Auto-padding sequence parallelism requires mask-capable profiles.
  Ring and scheduler-managed paged KV require profiles equivalent to the base
  backend and options. AllGather-KV rejects TRTLLM and sparse RainFusion.
  Configured KV-cache quantization requires dtype support in every candidate.
- Hunyuan image attention requires mask support. Wan VSA profiles require a
  VSA base that constructs its gate. H3 DiT profiles must exclude packed padding;
  VSA profiles require the VSA gate or a VSA base, and gated FastH3 layers accept only VSA.
- Model-owned kernels reject profiles. TRTLLM `target_sparsity` needs a
  per-layer calibration curve unless the layer is explicitly ignored.

Startup warm-up does not exercise profiles. Warm up the base and each profile
before timing. CPU tests use stand-in kernels; they do not establish GPU,
multi-rank, Inductor performance or output-quality guarantees.
