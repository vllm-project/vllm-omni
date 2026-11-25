# Benchmarking Multi-Stage Omni Models

Full flag reference: `docs/cli/bench/serve.md` (`vllm bench serve --omni --help=all`).
Implementation: `vllm_omni/entrypoints/cli/benchmark/serve.py` →
`vllm_omni/benchmarks/serve.py` (applies `vllm_omni/benchmarks/patch/patch.py`,
then runs upstream `vllm.benchmarks.serve.main_async`).

## Contents
- Entry point and core flags
- Metric definitions
- Accuracy evaluations
- Stage-level diagnostics (log-stats, profiler, orchestrator monitor)
- A/B comparison protocol

## Entry point and core flags

Always use `vllm bench serve --omni` against a running `vllm serve ... --omni`.

```bash
vllm bench serve --omni --port 8091 \
  --model /path/to/Qwen3-Omni-30B-A3B-Instruct \
  --endpoint /v1/chat/completions --backend openai-chat-omni \
  --dataset-name random --num-prompts 200 --max-concurrency 16 \
  --extra-body '{"modalities": ["text","audio"]}' \
  --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf \
  --metric-percentiles 50,90,99 --print-stage \
  --save-result --result-dir runs/<name>
```

| Flag | Notes |
|------|-------|
| `--backend` | Omni adds `openai-chat-omni`, `openai-audio-speech`, `openai-realtime-duplex`, `daily-omni`, `openai-image-edits-omni` |
| `--endpoint` | `/v1/chat/completions`, `/v1/audio/speech`, `/v1/realtime` |
| `--dataset-name` | Upstream (`random`, `random-mm`, `sharegpt`, `hf`) plus `daily-omni`, `videomme`, `omniinteract`, `seed-tts`, `seed-tts-text`, `seed-tts-design`, `ttsd`, `sound-effect` |
| `--extra-body` | Output modalities are set here, not via a dedicated flag |
| `--max-concurrency` / `--request-rate` | Fix one, sweep the other; `--request-rate inf` sends everything at t=0 |
| `--print-stage` | Per-stage metrics when the server returns stage snapshots |
| `--omni-request-timeout-s` | Per-request timeout, default 900 |
| `--ignore-eos` | Use for length-controlled `random` runs |
| `--save-result` / `--save-detailed` / `--result-dir` / `--result-filename` | Persist JSON |

Dataset-specific flag groups (`--daily-omni-*`, `--seed-tts-*`, `--videomme-*`,
`--omniinteract-*`) live in `vllm_omni/entrypoints/cli/benchmark/cli_args.py`.
Local media inputs need `--allowed-local-media-path` on the server.

## Metric definitions

Canonical names: `vllm_omni/metrics/definitions.py`; aggregation:
`vllm_omni/benchmarks/metrics/metrics.py`.

| Metric | Definition |
|--------|-----------|
| TTFT | Client wall time to first **text** token. Undefined for speech-only endpoints |
| TPOT / ITL | Mean / per-sample inter-chunk interval after the first token |
| E2EL | End-to-end request latency |
| AUDIO_TTFP (`audio_ttfp`) | Time to first non-empty **audio packet**; the key latency metric for spoken output |
| AUDIO_RTF (`audio_rtf`) | Generation latency / audio content duration; must stay < 1 for real-time streaming |
| AUDIO_DURATION | Seconds of audio content (PCM frames / sample rate) |
| audio_throughput | Total audio seconds / benchmark wall seconds |
| audio_underrun / `audio_continuity_ok_rate` | Simulated real-time playback gaps (default audible threshold 0.1 s, `VLLM_OMNI_BENCH_AUDIO_CONTINUITY_THRESHOLD_S`); see `vllm_omni/benchmarks/audio_continuity.py` |
| TTFC / TPOC / ICL / TPOP | Stage-level chunk metrics; definitions in the Multi-Stage section of `docs/cli/bench/serve.md` |

A good mean RTF does not guarantee smooth playback; check `audio_continuity_ok_rate`
for any streaming-quality claim.

## Accuracy evaluations

`benchmarks/accuracy/` is image-only. Omni accuracy runs through `vllm bench serve`:

| Eval | How | Result keys |
|------|-----|-------------|
| Seed-TTS WER | `--dataset-name seed-tts --seed-tts-wer-eval` (needs `vllm-omni[dev]`) | `seed_tts_content_error_*` |
| Daily-Omni | `--dataset-name daily-omni` with `--daily-omni-qa-json`, `--daily-omni-video-dir`; `--daily-omni-pack-mode qwen\|minicpm-interleave` | `daily_omni_accuracy*` |
| Video-MME | `--dataset-name videomme` | `videomme_accuracy*` |
| OmniInteract (duplex) | `--backend openai-realtime-duplex --dataset-name omniinteract --endpoint /v1/realtime --omniinteract-evaluate` | IA-QTF1 etc. |

Pair every performance change with an accuracy run on the same build when the
change touches sampling, numerics, graphs, or stage hand-off.

## Stage-level diagnostics

- **`--log-stats`** on `vllm serve`: per-request stage wall time, `stage_gen_time_ms`,
  and connector transfer timings (`tx_time_ms`, `rx_decode_time_ms`,
  `in_flight_time_ms`). See `docs/contributing/metrics.md`.
- **Torch profiler per stage**: pass `profiler_config` through `--stage-overrides`,
  warm up, then `POST /start_profile` and `POST /stop_profile` with
  `{"stages": [0, 1]}`. See `docs/contributing/profiling.md`.
- **Orchestrator monitor**: `--enable-orch-monitor` dumps poll/queue time series at
  shutdown (`VLLM_OMNI_ORCH_MONITOR_PATH`). Use it when latency is high but GPUs idle.
- **Prometheus**: `vllm_omni:*` families on `/metrics`.

## A/B comparison protocol

1. Record vLLM and vLLM-Omni commits, GPU model and count, deploy YAML, and the
   exact serve and bench commands.
2. Change one variable per comparison; keep dataset, seed, prompt count,
   concurrency, and `--extra-body` identical.
3. Warm up before measuring (CUDA graph capture and first-request compilation
   distort small runs).
4. Use enough requests for stable p99 (hundreds, not single digits) and repeat
   noisy runs.
5. Report mean and p99 for TTFT, AUDIO_TTFP, AUDIO_RTF, throughput, failures,
   plus `audio_continuity_ok_rate` for streaming output.
6. Attribute the delta with `--log-stats` / `--print-stage` before claiming which
   stage improved.
