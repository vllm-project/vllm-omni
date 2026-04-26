# TTS Universal Benchmark

A model-agnostic serving benchmark for TTS models in vllm-omni. One CLI
(`bench_tts.py`) + one YAML registry (`model_configs.yaml`) drive perf and
quality runs for every registered checkpoint: **Qwen3-TTS** (Base / CustomVoice)
and **VoxCPM2** today, more to come.

The same three task types — `voice_clone`, `default_voice`, `voice_design` —
are wired into both the manual CLI and the DFX nightly CI matrix
(`tests/dfx/perf/tests/test_tts.json`).

## Quick start

### 1. Start the server

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base --omni --port 8000
```

The server auto-loads its Deploy YAML from `vllm_omni/deploy/qwen3_tts.yaml`
(Pipeline + Deploy schema introduced in #2383). No `--stage-configs-path` or
`--deploy-config` flag is needed for any registered model.

### 2. Run the benchmark (`vllm bench serve --omni`)

The primary, directly-controllable path. Copy-paste one of these and tweak
any bench flag (sampling params, endpoint, extra body, warmups, etc.):

#### voice_clone (Qwen3-TTS-Base, seed-tts dataset)

```bash
vllm bench serve --omni \
    --host 127.0.0.1 --port 8000 \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --backend openai-audio-speech \
    --endpoint /v1/audio/speech \
    --dataset-name seed-tts \
    --dataset-path /path/to/seed-tts-eval \
    --seed-tts-locale en \
    --num-prompts 20 --num-warmups 2 \
    --extra-body '{"task_type":"Base"}' \
    --max-concurrency 1 --request-rate inf \
    --percentile-metrics ttft,e2el,audio_rtf,audio_ttfp,audio_duration \
    --save-result --result-dir ./results
```

#### default_voice (Qwen3-TTS-CustomVoice, bundled seed_tts_smoke)

```bash
vllm bench serve --omni \
    --host 127.0.0.1 --port 8000 \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --backend openai-audio-speech \
    --endpoint /v1/audio/speech \
    --dataset-name seed-tts-text \
    --dataset-path benchmarks/build_dataset/seed_tts_smoke \
    --seed-tts-locale en \
    --num-prompts 20 --num-warmups 2 \
    --extra-body '{"voice":"Vivian","language":"English","task_type":"CustomVoice"}' \
    --max-concurrency 1 --request-rate inf \
    --percentile-metrics ttft,e2el,audio_rtf,audio_ttfp,audio_duration \
    --save-result --result-dir ./results
```

#### voice_design (Qwen3-TTS-CustomVoice, bundled seed_tts_design)

```bash
vllm bench serve --omni \
    --host 127.0.0.1 --port 8000 \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --backend openai-audio-speech \
    --endpoint /v1/audio/speech \
    --dataset-name seed-tts-design \
    --dataset-path benchmarks/build_dataset/seed_tts_design \
    --seed-tts-locale en \
    --num-prompts 20 --num-warmups 2 \
    --extra-body '{"task_type":"VoiceDesign","language":"English"}' \
    --max-concurrency 1 --request-rate inf \
    --percentile-metrics ttft,e2el,audio_rtf,audio_ttfp,audio_duration \
    --save-result --result-dir ./results
```

#### Add WER / SIM / UTMOS to any of the above

Append `--seed-tts-wer-eval` (and optionally `SEED_TTS_EVAL_DEVICE=cuda:0`
in the env, per PR #2558). This triggers the seed-tts-eval protocol:
Whisper-large-v3 ASR → WER, WavLM embeddings → SIM, balacoon/utmos → UTMOS.

### 3. Convenience wrapper (`bench_tts.py`)

If you're running the **canonical** configuration for a registered model,
`bench_tts.py` loads the right defaults from `model_configs.yaml` and
emits the exact `vllm bench serve --omni` command above — useful for
concurrency sweeps and multi-task runs:

```bash
# Smallest smoke — 5 prompts, concurrency=1
python benchmarks/tts/bench_tts.py \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --task voice_clone \
    --dataset-path /path/to/seed-tts-eval \
    --concurrency 1 --num-prompts 5 \
    --output-dir ./results

# Full concurrency sweep
python benchmarks/tts/bench_tts.py \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --task voice_clone \
    --dataset-path /path/to/seed-tts-eval \
    --concurrency 1 2 4 8 16 32 \
    --num-prompts 20 \
    --output-dir ./results

# With WER / SIM / UTMOS quality eval (adds ASR + embedding compute)
python benchmarks/tts/bench_tts.py \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --task voice_clone \
    --dataset-path /path/to/seed-tts-eval \
    --wer-eval \
    --concurrency 4 --num-prompts 200 \
    --output-dir ./results
```

### 4. Plot a sweep

```bash
python benchmarks/tts/plot_results.py \
    --results ./results/*.json \
    --output ./results/curve.png
```

Outputs TTFP / RTF / throughput curves (and a markdown table) for every
`(task, concurrency)` combination in the result set.

## Task types

| Task            | Dataset           | Request body                                        | Checkpoints that support it              |
|-----------------|-------------------|-----------------------------------------------------|------------------------------------------|
| `voice_clone`   | `seed-tts`        | `ref_audio` + `ref_text` + `task_type=Base`         | `Qwen3-TTS-*-Base`, `VoxCPM2`            |
| `default_voice` | `seed-tts-text`   | `voice=Vivian` + `task_type=CustomVoice`            | `Qwen3-TTS-*-CustomVoice`                |
| `voice_design`  | `seed-tts-design` | `instructions=<natural-language description>` + `task_type=VoiceDesign` | `Qwen3-TTS-*-CustomVoice` |

**`-CustomVoice` checkpoints do NOT ship `speaker_encoder` weights**, so
voice_clone requests raise `ValueError` at model runtime. Use `-Base` for
voice_clone.

## Adding a new TTS model

Drop an entry into `model_configs.yaml` — no Python changes required:

```yaml
models:
  <org>/<model-id>:
    supported_tasks: [voice_clone]          # or default_voice / voice_design
    backend: openai-audio-speech            # vllm bench serve backend
    endpoint: /v1/audio/speech              # OpenAI-compatible endpoint
    task_extra_body:                        # merged into every request's body
      voice_clone:
        task_type: Base
```

Then add the model's Deploy YAML under `vllm_omni/deploy/<model>.yaml`
(Pipeline + Deploy schema) and it's immediately benchable.

## Datasets

| Dataset            | Bundled? | Format            | Source                                                         |
|--------------------|----------|-------------------|----------------------------------------------------------------|
| `seed-tts-design`  | ✅       | 5-field meta.lst  | `benchmarks/build_dataset/seed_tts_design/en/meta.lst` (20 prompts) |
| `seed_tts_smoke`   | ✅       | 4-field meta.lst  | `benchmarks/build_dataset/seed_tts_smoke/en/meta.lst` (20 text-only) |
| `seed-tts`         | ❌       | 4-field meta.lst + WAVs | Google-Drive: [BytedanceSpeech/seed-tts-eval][seedtts] (~1.2 GB) |
| `seed-tts-text`    | ❌       | 4-field meta.lst  | Same archive as `seed-tts` (wav column unused)                 |

[seedtts]: https://github.com/BytedanceSpeech/seed-tts-eval

For manual voice_clone / default_voice runs against the full corpus, follow
`benchmarks/build_dataset/download_process_data_seedtts.md` and point
`--dataset-path` at the extracted `seedtts_testset` directory.

## DFX nightly CI

`tests/dfx/perf/tests/test_tts.json` wires three perf regimes plus quality:

| eval_phase    | concurrency | purpose                                                 | Baseline metrics                        |
|---------------|-------------|---------------------------------------------------------|-----------------------------------------|
| `latency`     | 1           | Single-request TTFP / RTF SLO                           | `median_audio_ttfp_ms`, `median_audio_rtf` |
| `throughput`  | 8           | Codec-batching cliff sentinel (PDF #272 concurrency≥8)  | `median_audio_ttfp_ms`, `median_audio_rtf` |
| `quality`     | 4           | WER / SIM / UTMOS regression (disabled in CI by default)| `mean_audio_rtf`                        |

Why `median_*` for latency/throughput and `mean_*` for quality: latency
distributions have cold-start tails that drag the mean; quality aggregates
over 200 prompts so single-request outliers don't matter.

Quality entries are `enabled: false` in CI because seed-tts-eval is not
staged in the Buildkite container (matches the precedent in
PR #2558 — quality runs are manual / release-validation, not nightly).

## Concurrency cliff regression sentinel

Observed on H20-3e, Qwen3-TTS-1.7B (measured pre-merge on this branch):

| Task          | Model         | c=1    | c=4    | **c=8**    | c=16   | c=32   |
|---------------|---------------|--------|--------|------------|--------|--------|
| voice_clone   | 1.7B-Base     | RTF 0.15 / TTFP 165ms | 0.28 / 412ms | **0.49 / 1701ms** | 0.72 / 3355ms | 0.77 / 3772ms |
| voice_design  | 1.7B-CustomVoice | RTF 0.08 / TTFP 53ms  | 0.11 / 154ms | **0.21 / 872ms**  | 0.33 / 1801ms | 0.38 / 1989ms |

Both models show a **4–6× TTFP jump from c=4 to c=8** while audio throughput
saturates around c=4–8 — the codec-bs=1 bottleneck documented in
vllm-project/vllm-omni#272. The `throughput` CI regime at c=8 is the
sentinel for regressions in this area.

## File layout

```
benchmarks/tts/
├── README.md                  (this file)
├── bench_tts.py               CLI — serve-mode benchmark driver
├── bench_voxcpm_offline.py    CLI — offline VoxCPM benchmark (sync + streaming)
├── plot_results.py            Generate per-task / per-concurrency curves
└── model_configs.yaml         Model registry (supported tasks + extra body)
```

## Related

- Upstream seed-tts-eval integration: vllm-project/vllm-omni#2558
- Pipeline + Deploy schema: vllm-project/vllm-omni#2383
- Concurrency cliff RFC: vllm-project/vllm-omni#272
