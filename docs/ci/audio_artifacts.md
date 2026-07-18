# Audio Sample Artifacts in CI

vLLM-Omni's nightly and ready CI already saves benchmark results as JSON
artifacts so reviewers can diff metrics across runs. This document defines
the policy for also saving a curated subset of generated **audio clips** so
reviewers can *listen* to representative outputs without re-running the
bench.

## Why

- Catching audio regressions from JSON metrics alone is hard. A WER delta
  of +2% could mean "model is slightly worse" or "ASR misheard a Chinese
  homophone" — listening resolves it in seconds.
- TTS perf PRs touching the vocoder, talker, or code2wav stage need a
  listening pass before merge. Reviewers shouldn't need GPU access to do
  this.

## What gets saved

Per opt-in bench run, the collector writes to
`tests/dfx/perf/results/audio_samples/<label>/`:

- `NN_<utterance_id>.flac` — up to `--save-audio-samples N` FLAC clips
  (lossless, ~50% of raw WAV size). Falls back to `.wav` (PCM_16) if the
  installed libsndfile lacks FLAC support.
- `index.json` — manifest with `utterance_id`, `locale`, `ref_text`,
  `duration_sec`, `sample_rate`, `buffer_index`, `total_buffered`.

The sampler picks first, last, and evenly-spaced middle clips so reviewers
see both warmup and steady-state regimes.

## Size budget

| Configuration | Per bench | × jobs/day | Annualized |
|---|---|---|---|
| 8 clips × 5 s × FLAC | ~1 MB | 8 nightly → 8 MB/night | ~3 GB/year |
| 8 clips × 10 s × FLAC | ~2 MB | 8 nightly → 16 MB/night | ~6 GB/year |
| 8 clips × 10 s × Opus 48 kbps | ~0.5 MB | 8 nightly → 4 MB/night | ~1.5 GB/year |

The recommended default (8 FLAC clips) lands the entire nightly fleet at
**~10–20 MB per run**, well inside Buildkite's artifact tier.

## Policy: when to upload

| CI tier | Policy | Rationale |
|---|---|---|
| `test-nightly.yml` | **Always on** for TTS/Omni perf jobs (this PR enables `nightly-tts-performance` as the demonstrator). | Listening post for regressions; everyone hears the same clips across nights. |
| `test-ready.yml` | **Opt-in (Phase 2)**: failure-only, or behind a `ci/save-audio` PR label, or auto-on for paths touching `code2wav/`, `vocoder/`, `talker.py`. | Avoid wasting per-PR uploads on green runs that nobody listens to. |
| `test-merge.yml`, `test-weekly.yml` | Inherit nightly policy. | Same listening surface; no extra cost. |

The Phase 2 test-ready hook is intentionally **not** in this PR — proving
the framework end-to-end on one nightly job first, then expanding once the
maintainers agree on the policy.

## How to opt a bench in

In the bench JSON (`tests/dfx/perf/tests/test_*.json`):

```jsonc
{
  "task": "voice_clone",
  "dataset_name": "seed-tts",
  "seed_tts_wer_eval": true,
  "save_audio_samples": 8,
  "audio_samples_label": "qwen3_tts_base_voice_clone_quality_en"
}
```

The runner translates JSON keys to `--<kebab-case>` CLI flags. Equivalent
direct invocation:

```bash
vllm bench serve --omni \
  --dataset-name seed-tts \
  --save-audio-samples 8 \
  --audio-samples-label qwen3_tts_base_voice_clone_quality_en \
  ...
```

Env-var equivalents (`SAVE_AUDIO_SAMPLES`, `AUDIO_SAMPLES_DIR`,
`AUDIO_SAMPLES_LABEL`) work too, for ad-hoc shell runs.

## Caveats

- **Currently active path is `seed_tts_eval`.** Benches that decode audio
  outside `compute_seed_tts_wer_metrics` (e.g. ttsd, daily-omni audio mode)
  collect nothing today; expanding coverage means calling
  `get_collector().add(...)` from those data modules. Tracked as Phase 2.
- **Reference voice clones.** If a bench uses unreleased reference voices,
  uploading clones to Buildkite is a leak surface. Reviewers should confirm
  the dataset license before enabling for a new bench.
- **Buildkite UI does not auto-stream FLAC.** Clips are downloaded and
  played locally. If in-browser playback becomes a requirement, switch the
  helper to write Opus instead of FLAC (one-line change).

## Future work

- Add an HTML index that embeds `<audio>` tags per clip, uploaded as a
  single browsable artifact.
- Push clips to a small object store (S3/GCS) and embed signed URLs in the
  bench JSON for one-click listening from the summary `.xlsx`.
- Wire the collector into `ttsd`, `daily-omni`, and `random-mm` data modules.
- Add the test-ready failure-only / label-gated upload pattern.
