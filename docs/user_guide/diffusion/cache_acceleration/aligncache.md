# AlignCache

AlignCache predicts selected diffusion model outputs while retaining the native
scheduler steps. Output quality and speed depend on the checkpoint and workload.

## Profiles and admission

Select a profile explicitly for each model; it is never inferred from the model
name. The default is Wan2.1 1.3B.

| Profile | Engineering adapter | Policy | Base threshold | Warmup / tail |
| --- | --- | --- | --- | --- |
| `wan21_reference_v1` | Wan2.1 T2V-1.3B on `WanPipeline` | `wan21_feedback_v1` | 0.065 | 12 / 1 |
| `wan21_14b_reference_v1` | Wan2.1 T2V-14B on `WanPipeline` | `wan21_feedback_v1` | 0.065 | 12 / 1 |
| `wan22_t2v_reference_v1` | Native Wan2.2 T2V-A14B | `wan22_feedback_v1` | 0.078 | 9 / 1 per component |
| `wan22_i2v_reference_v1` | Native Wan2.2 I2V-A14B on `WanImageToVideoPipeline` | `wan22_feedback_v1` | 0.065 | 9 / 1 per component |
| `wan22_ti2v_reference_v1` | Native Wan2.2 TI2V-5B on `WanPipeline` | `wan22_feedback_v1` | 0.065 | 9 / 1 |
| `qwen21_reference_v1` | Native Qwen-Image-2.1 T2I | `qwen_flow_v1` | 0.12 | 5 / 2 |
| `minimax_h3_reference_v1` | Native MiniMax-H3 T2VA, Euler | `minimax_h3_flow_v1` | Video 0.33, audio 0.165 | 5 / 1 |

Admission requires one CUDA GPU, BF16 native weights, eager execution, one
request and one output. Wan needs serial two-branch CFG; Wan2.1 checkpoints
load through the shared `WanPipeline` with one transformer and one guidance
scale. Wan2.2 A14B profiles keep one controller per high/low-noise expert;
TI2V-5B uses one controller and one guidance scale, with or without its
first-frame image. I2V-A14B predicts the noisy latent channels and keeps the
concatenated image condition native. Qwen also admits a single conditional
branch; H3 uses its native CFG-distilled joint AV forward.

Quantization is rejected unless a Qwen-Image-2.1 profile sets
`quantization` to `fp8`, which admits only the engine's online FP8
(`--quantization fp8`) and still requires BF16 activations; checkpoint-serialized
FP8 and other methods remain rejected. FP8 outputs differ from BF16, so compare
`none` and `force_exact` under the same quantization before using `accelerate`.

The following are rejected: batching, parallelism, other quantization, LoRA, offload,
reference-KV caching/offload (`kv_offload_config`), compile/decode graphs, step
execution, disaggregated stages, and paged KV. Also
rejected are Qwen image editing and extra transformer conditions, H3 reference
media, editing, locked audio, refine/upscale, continuation, FastH3 and VDNH3,
Wan2.1 I2V/FLF2V checkpoints with CLIP image embeddings, and Wan2.2 S2V,
Animate and VACE.

## Configuration

The backend and its options are engine/stage settings, not per-request HTTP
switches. Supply parameters inside the `align_cache` namespace:

```json
{
  "align_cache": {
    "profile": "qwen21_reference_v1",
    "mode": "force_exact"
  }
}
```

Use this object as Python `cache_config`, or its JSON form with `--cache-config`,
together with `--cache-backend align_cache --enforce-eager`. Before using
`accelerate` with a checkpoint, compare its native `none` and `force_exact`
outputs.

| Mode | Behavior |
| --- | --- |
| `passthrough` | Exact native outputs; no prediction histories |
| `force_exact` | Exact native outputs with history maintenance, no adaptive feedback |
| `trace` | Exact native outputs plus independent hypothetical cache decisions |
| `accelerate` | Execute exact or fast according to the selected policy |

Parsing applies schema defaults, then the selected profile, then explicit
overrides; nested fields keep unspecified profile values. Unknown keys, invalid
JSON, non-object JSON, nonfinite numbers, and bool-as-integer values fail
explicitly. Do not mix other backends' options into this namespace.

### Deploy configuration

For a single diffusion stage:

```yaml
stages:
  - stage_id: 0
    enforce_eager: true
    max_num_seqs: 1
    cache_backend: align_cache
    cache_config:
      align_cache:
        profile: qwen21_reference_v1
        mode: force_exact
        base_threshold: 0.12
        warmup_steps: 5
```

Pass the file with `--deploy-config`. The model must match the profile. CLI
fields override deploy fields; unspecified nested settings survive the merge.

### Common schema

| Field | Default / constraint |
| --- | --- |
| `schema_version` | `1`; other versions rejected |
| `profile` | `wan21_reference_v1`; one of the profile names above |
| `policy` | Inferred from profile; an explicit value must match it |
| `mode` | `accelerate`; one of the four modes above |
| `base_threshold` | Profile value; finite and nonnegative; zero disables fast |
| `warmup_steps`, `tail_steps` | Profile values; nonnegative integers |
| `max_cache_bytes` | `1073741824`; positive integer, controller memory only |
| `diagnostics` | `summary`; `off`, `summary`, or `trace` |
| `quantization` | `none`; `fp8` only with `qwen21_reference_v1` and online FP8 |
| `phase.early_end`, `phase.late_start` | Wan/H3: 0.15 / 0.86; Qwen: 0.2 / 0.8; ordered within [0, 1] |
| `phase.scales` | Wan/H3: 1.15 / 1 / 0.8; Qwen: 0.85 / 1 / 0.85; nonnegative |

Three-phase values are objects with `early`, `mid`, and `late` keys; partial
objects override only the supplied phases. A schedule with no fast window runs
entirely exact.

### Feedback schema

| Field under `feedback` | Wan default | Qwen / H3 differences |
| --- | --- | --- |
| `age_caps` | 5 / 4 / 3 | Qwen 2 / 4 / 2; H3 4 / 4 / 4 |
| `scale_init` | 1 / 1 / 1 | Flow policies reject explicit overrides |
| `scale_low` | 0.95 / 0.90 / 0.85 | Qwen 0.8 / 0.8 / 0.8 |
| `scale_high` | 1.03 / 1.06 / 1 | Qwen 1 / 1.15 / 1 |
| `ratio_ema` | 0.90 | Same defaults |
| `ratio_low`, `ratio_high` | 0.85 / 1.25 | Qwen 0.01 / 100 |
| `eps` | 1e-6 | Qwen 1e-12 |

Age caps are nonnegative integers. Scale and ratio bounds must be positive and
ordered, and `scale_init` must lie within the scale bounds. `ratio_ema` is in
[0, 1) and `eps` is positive. Wan/H3 age is the distance from a real evaluation;
Qwen also limits consecutive skips.

### Predictor and anchor schema

For Wan policies, `predictor` accepts:

- `enabled`, `second_order`, `online_scale`: booleans, default true.
- `stride`: fixed at 64.
- `ratio_ema`, `direction_ema`: default 0.8, in [0, 1).
- `beta`: 0 / 0.35 / 0.15, nonnegative.
- `ratio_tolerance`: 0 / 0.30 / 0.18, nonnegative.
- `direction_min`: 1 / 0.80 / 0.88, in [-1, 1].
- `scale_ema`: 0.85, in [0, 1).
- `scale_low`, `scale_high`: 0.85 / 1.18, positive and ordered.

Wan `anchor` fields are `enabled=true`, `early_exact=true`, `age_caps=0/1/2`,
`cond_cap=0/0.005/0.020`, `disagreement_cap=0/0.020/0.060`,
`alpha_init=1.279282/1.018750/0.972800`, `alpha_ema=0.85`, and
`blend=0/0.35/0.20`. Age caps are nonnegative integers; error caps are
nonnegative; alpha is in [-1.5, 1.5]; blend is in [0, 1].

Flow policies accept only `predictor.beta` as a predictor override (Qwen
0.2 / 0.4 / 0.2; H3 0 / 0.35 / 0.15). Qwen also accepts `anchor.blend`
(0 / 0.25 / 0), `anchor.cond_cap` (0.02 / 0.04 / 0.02) and
`anchor.disagreement_cap` (0.01 / 0.02 / 0.01). H3 has no anchor options because
video and audio are not conditional/unconditional branches.

Qwen alone accepts `flow.max_time_gap=0.2` (positive) and
`flow.max_consecutive_skips=4` (nonnegative integer). H3 alone accepts
`flow.audio_threshold=0.165` (nonnegative). Wan rejects the `flow` group.

## Diagnostics and failures

The logger `vllm_omni.diffusion.cache.aligncache` emits bounded request
summaries; Wan can also emit the most recent 256 scalar decisions. Flow summaries
report actual model calls and reason counts; `trace` mode never represents actual
acceleration. No prompt, embedding, or latent tensor is logged.

Zero fast evaluations can come from warmup/tail overlap, a zero threshold,
missing history, age limits, numerical safeguards, or insufficient capacity.
Capacity failure switches the request to exact computation before allocating
large histories. An invalid draft falls back before any native model call; real
model failures propagate without repeating the forward.

For Qwen, replacing the native prefix KV cache, losing its prefill state, or
changing branch/condition identity fails the request rather than counting as a
cache miss.

Disable the backend with `cache_backend="none"` and a fresh engine; an explicit
`none` is not overridden by environment fallback.
