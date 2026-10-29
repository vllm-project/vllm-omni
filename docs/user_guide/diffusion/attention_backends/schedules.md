# Attention Schedules

An attention schedule selects a named attention profile at zero-based denoising
step indices. Profiles are prepared at startup; requests select ranges, not new
backends. Steps outside a range use the base attention configuration.

Wan2.2 T2V, I2V, S2V and VACE publish progress in request mode.
MiniMax-H3 and HunyuanImage-3.0 publish it in request and step modes.
Configured profiles on any other model, including the Wan DMD2 pipelines
(`WanT2VDMD2Pipeline`, `WanI2VDMD2Pipeline`) and `CausVidWanPipeline`, fail at
startup with an `Unsupported diffusion_attention_schedule` error, before weights
are loaded.
Backend and model requirements still apply, and a backend's own kernel
fallbacks still run when a profile selects it.
A backend that a profile names is an explicit selection: if it cannot use a
dense 4D attention mask, the step fails instead of falling back to SDPA. A
role that the profile leaves out uses the platform default and falls back to
SDPA, as outside a schedule. This is checked on each scheduled step that sends
such a mask, not at startup. HunyuanImage-3.0 image attention, which sends the
mask with `full_attn_spans`, checks named backends at startup instead (see
[Compatibility limits](#compatibility-limits)).

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
- Other top-level keys are rejected at startup.

Each range has exactly `start`, `end` and `profile`:

| Field | Meaning |
| --- | --- |
| `start` | Integer >= 0; first included step. |
| `end` | Integer > `start`; first excluded step, or `null` for the sequence's end. Required. |
| `profile` | A profile declared at startup. |

Ranges must be ordered and non-overlapping. Adjacent ranges are allowed;
an open-ended range must be last. Gaps use the base configuration.
A bounded range in a request's own schedule must end within the actual
timestep sequence, which may differ from `num_inference_steps`. A bounded
range inherited from the server default may end later; it applies up to the
last step. An open-ended range that starts at or after the end of the sequence
selects nothing, so `{"start": 6, "end": null}` leaves a request with 6 or
fewer steps on the base configuration.
CFG evaluations share one step index.
Each S2V clip and each H3 seed/continuation window restarts at step 0.
Prompt encoding and decoding run with the base configuration.
HunyuanImage-3.0 computes the prompt K/V inside the step-0 DiT forward with the
step-0 profile and reuses it on later steps; only the AR KV-reuse prefill runs
with the base configuration. Deeper layers' prompt K/V also depend on earlier
layers' step-0 attention outputs, so on this model the step-0 profile affects
every step, not only step 0. A range that starts at step 0 with an approximate
profile therefore also changes the later steps; the server logs this at startup
when its default schedule does so.

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
`--diffusion-kv-cache-dtype` also applies to scheduled steps, so, as in the
base configuration, startup rejects it together with a profile `quant` spec
that sets `dtype_qk` or `dtype_vo`.

## Stage configuration

Set `diffusion_attention_schedule` on the denoising stage in a deploy YAML,
using the same object as the server flag. Set its base configuration separately
with `diffusion_attention_config`. In the bundled disaggregated H3,
HunyuanImage-3.0 and Wan2.2 (`wan2_2_eg`, `wan2_2_egd`) deployments, the
denoising stage is `stage_id: 1`. The Wan2.2 encode stage, the `wan2_2_egd`
decode stage and the separate decode stage of `minimax_h3_disaggregated_decode`
run no denoising and ignore request schedules. In `wan2_2_eg`, `wan2_2_egd`
and `minimax_h3_disaggregated_decode`, configure the schedule on stage 1 only,
in the deploy YAML or with `--stage-overrides`. The global flag also reaches
the stages that run no denoising, and a schedule on such a stage fails at
startup, before weights are loaded.

CLI overrides deep-merge profile mappings into the stage configuration.
A CLI `default` list replaces the stage list rather than appending ranges.

## Per-request configuration

| `attention_schedule` value | Effect |
| --- | --- |
| Omitted or `null` | Inherit defaults. |
| `[]` | Disable scheduling for this request. |
| Non-empty ranges | Replace defaults; do not merge. |

Video endpoints first apply a stage's `default_sampling_params`, if present;
an omitted or `null` request value preserves that stage default. A schedule
from `default_sampling_params` is checked as the request's own, so its bounded
ranges must end within the sequence; put ranges that may end later in the
schedule configuration's `default`.

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

Requests co-batch only with equal schedules after resolving stage defaults
and only when they all sent their schedule or all inherited it. An omitted
schedule does not co-batch with the same ranges sent explicitly, because an
explicit schedule is checked against the step count while an inherited
default is not. For the same reason `[]` never shares a batch with an
omitted schedule.
A scheduled H3 step batch keeps one packed transformer forward when every
request is at the same step and sigma and the selected profile's backends
isolate packed requests; otherwise it runs one forward per request so each
uses its own progress. A scheduled HunyuanImage-3.0 step batch always runs one
transformer forward per request.

Malformed schedules return HTTP 400 before dispatch. A request schedule that
names a profile the denoising stage does not configure returns HTTP 400 when
that stage admits the request, before denoising. In the disaggregated
deployments above, this happens after stage 0 has finished: the H3 encoder,
the HunyuanImage-3.0 AR stage or the Wan2.2 encode stage.
A bounded range in the request's own schedule that ends after the sequence
fails before denoising and retains the client error through
the runner and multiprocess executor. Asynchronous `/v1/videos` reports all of
these errors on the job record rather than the creation response.

## Compatibility limits

- Full compile scope and configured cache backends are rejected with profiles,
  even with empty default ranges. Under regional compilation each attention
  call is one opaque op inside the block graph, on every request including
  `[]`; the op picks the profile when it runs, so the block stays one graph.
  The op carries `attn_mask`, a `video_layout` without video spans and `extra`
  entries holding None, bools, ints or int tuples (a video grid). Other metadata
  (tensors, floats or strings in `extra`, joint tensors, video spans), HSDP
  and autograd keep an eager boundary that splits the block around attention.
  Different candidate output layouts may trigger recompilation.
- H3 rejects non-empty schedules with request-scoped Cache-DiT (`quality=high`)
  or `latent_refine`. Disable the schedule or the conflicting option.
  Step mode rejects these H3 options regardless of schedule. Non-empty
  default ranges with a `latent_refine` default in `--additional-config`
  fail at startup.
- Auto-padding sequence parallelism with `mask_sp_padding` enabled requires
  mask-capable profiles.
  Ring and scheduler-managed paged KV require profiles equivalent to the base
  backend and options. AllGather-KV rejects TRTLLM and sparse RainFusion.
  Configured KV-cache quantization requires dtype support in every candidate.
- Hunyuan image attention requires a backend that a profile names to take the
  attention mask and either honor `full_attn_spans` or consume a dense 4D mask,
  such as FLASH_ATTN, CUDNN_ATTN or TORCH_SDPA; a profile that names
  FLASHINFER_ATTN is rejected. A role that the profile leaves out uses the
  platform default, which falls back to SDPA where it cannot use the mask.
  Wan T2V, I2V and VACE VSA
  profiles require a VSA base that constructs its gate; Wan S2V passes no VSA
  metadata, so VSA falls back to SDPA there. H3 DiT profiles must exclude packed padding;
  VSA profiles require the VSA gate or a VSA base, and gated FastH3 layers accept only VSA.
  FastH3 checkpoints pin VSA sparsity, so their profiles cannot set
  `fastvideo_vsa_topk`. They also check each VSA profile's provider and
  precision at startup, because warm-up runs only the base.
- TRTLLM `target_sparsity` needs a per-layer calibration curve unless the
  layer is explicitly ignored.
- Regional `torch.compile` stays enabled: H3 selects the profile and runs
  attention inside its eager `torch.compiler.disable` attention region, so
  the compiled DiT blocks do not trace the selection.

Each attention layer prepares its own implementation for every distinct
profile. A profile identical to the base reuses the base implementation,
unless the profile carries its own calibration, including one propagated
from the checkpoint's `sparse_attention_config`. Each distinct FlashInfer
implementation adds a 128 MiB workspace per layer, unless FlashInfer runs
its cute-dsl backend (the automatic choice on Blackwell).

Startup warm-up does not exercise profiles. Warm up the base and each profile
before timing. CPU tests use stand-in kernels; they do not establish GPU,
multi-rank, Inductor performance or output-quality guarantees.
