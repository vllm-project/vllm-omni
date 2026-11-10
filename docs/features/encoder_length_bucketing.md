# Encoder Length Bucketing

Encoder length bucketing recovers the padding waste of variable-length audio
encoders. When a batch mixes short and long clips, the encoder normally
computes every item over the padded batch length; bucketing re-groups items by
their true lengths and computes one window per length rung instead, so short
items stop paying for long ones.

The feature is **off by default and opt-in per deployment**: it is enabled by a
stage configuration field, not by environment variables, and it engages only
for models whose per-model settings have been verified (see
[Supported models](#supported-models)). With the feature disabled, the encoder
runs exactly as it does without this code path.

When the feature is available for your model but disabled, the server logs one
line at startup telling you how to enable it — no need to discover the setting
by reading source.

## Supported models

| Model | `model_type` | Numerics with bucketing on |
| --- | --- | --- |
| [Step-Audio-2-mini](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/step_audio2) | `step_audio_2` | Encoder output numerically unchanged (mask-aware encoder) |
| [Covo-Audio-Chat](https://github.com/vllm-project/vllm-omni/tree/main/recipes/Tencent/Covo-Audio-Chat.md) | `covo_audio` | **Lossy** — the unmasked fixed-window Whisper path attends over the rung window instead of the full extractor padding, so outputs may differ from the un-bucketed forward |

Any other model runs the untouched forward: an unlisted `model_type` logs a
warning at install and the feature stays inert even when configured.

### The verified whitelist

Enabling is deliberately two-gated: the deployment opt-in above **and** an
entry in the verified whitelist. The whitelist (`_VERIFIED_MODELS` in
`vllm_omni/worker/bucketed_encoder.py`) carries each model's measured
per-model settings (`frames_per_second`, `frames_per_length`,
`context_margin`, `time_axis`, `output_kind`) and its numerical
classification. These are declared facts read from the model's own code and
per-model verification — not user-facing knobs: a wrong frame rate or time
axis would silently compute wrong windows, so configuration alone can never
enable an unlisted model. Setting `audio_bucket_seconds` for a model outside
the whitelist is safe: it takes no effect beyond the install warning.

### Supporting more models

Onboarding another model is a **code change, not a configuration change**:

1. The model's constructor declares its own encoder to
   `install_bucketed_encoders` (the wrapper's `lengths` call contract works
   with the feature off, so this is safe to ship unconditionally).
2. Measure the model's per-model settings and its numerical behavior, then
   add its whitelist entry and its lossless/lossy classification to the
   table above (mask-aware encoders keep the encoder output numerically
   unchanged; unmasked fixed-window encoders are lossy).

If you need bucketing for a model that is not listed, open an issue or PR
with the model's encoder `forward` signature and its mel frame rate — the
two supported models show the two integration shapes (positional lengths and
no lengths). Unlisted models are unaffected in every other respect.

## Enabling

Set `additional_config.audio_bucket_seconds` — a list of window lengths in
seconds — on the **stage that owns the encoder** (stage 0 for both supported
models). Both models use a 100 mel-frames-per-second rate, so
`[2, 4, 8, 16, 30]` selects 200/400/800/1600/3000-frame rungs.

### Deploy YAML

```yaml
stages:
  - stage_id: 0
    additional_config:
      audio_bucket_seconds: [2, 4, 8, 16, 30]
```

Pass the file with `--deploy-config` (or the offline `deploy_config=` argument).

### CLI (`--stage-overrides`)

No deploy YAML needed — the bundled `vllm_omni/deploy/<model_type>.yaml` is
loaded automatically and the override wins over it:

```bash
vllm serve Step-Audio-2-mini --omni --port 8091 \
    --stage-overrides '{"0": {"additional_config": {"audio_bucket_seconds": [2, 4, 8, 16, 30]}}}'
```

### Configuration rules

| Rule | Behavior |
| --- | --- |
| Unset or `[]` | Feature disabled (the default) |
| Ladder shorter than the longest clip | A batch containing an over-long item is served by the untouched forward (no truncation); keep the ladder at least as long as the longest clip to keep the benefit |
| Malformed value (not a list, or a non-numeric / non-positive entry) | `ValueError` at model construction |
| Wrong stage | No effect — set it on the stage that owns the encoder |

## What to expect

- The bucketed path runs only when it computes strictly fewer frames than the
  un-bucketed forward. Single-item and uniform-length batches have no padding
  waste to recover and run exactly as they do with the feature off.
- The rung window never truncates real content and keeps the producer's own
  padding values, so the copied frames are the ones the un-bucketed forward
  would read.
- Calls without per-item lengths, keyword-argument call forms, non-CUDA
  tensors, and unsupported output types fall back to the untouched forward;
  the fallback is logged instead of silent.
- These are mel-window and encoder-boundary guarantees only; they do not
  establish end-to-end transcript equivalence.

## See also

- [Stage and deploy configurations](../configuration/stage_configs.md#encoder-length-bucketing)
  — the configuration reference, including the lossiness classification and the
  window guarantees.
