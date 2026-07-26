# 20260726 Ascend 910C Single-Card Local Proxy Baseline

- Decision: keep as the pre-official local proxy baseline.
- Benchmark Git SHA: `652bb664bcbccc316505f932146f6416df1d0173`.
- Rules retrieval date: 2026-07-26.
- Official dataset, starter kit, metric definitions, thresholds, and score: `UNRESOLVED`.
- Hardware: one physical Ascend 910C card, NPU ID 0, two logical chips 0/1.
- HBM: 64 GiB per logical chip, 128 GiB aggregate.
- Deploy config: `vllm_omni/deploy/minicpmo_4_5_ascend_910c_1card.yaml`.
- Model: local `openbmb/MiniCPM-o-4_5` tree, bf16 model configuration.
- Model manifest SHA256: `bb23a7b8b90f583de88a36661ed0648e7d291aa505e23958e019e8cd3e844226`.
- Seed: 42; thinker max tokens: 256; talker max tokens: 256; warmups: 2.
- Raw artifact root: `artifacts/minicpmo_ascend/baseline-910c-1card-20260726/`.
- Artifact manifest SHA256: `f0ffcbb69b1b3baef96c9baa5d2571983dd8d9a0b0f03edc1e87c4503691e9c5`.
- Generated report SHA256: `69280b62904564de773445a021d7c75d05b96fe8683b097164e681d39e3f5967`.

## Gates

The machine-readable gate passed with no failures. Text-only output contained
no audio. Every audio response was non-empty 24 kHz PCM with ordered,
non-duplicate adjacent chunks.

| Request | Input | Output | Result | Audio chunks | PCM bytes |
| --- | --- | --- | --- | ---: | ---: |
| text_only | text | text | PASS | 0 | 0 |
| text_audio | text | text + audio | PASS | 8 | 679680 |
| image_audio | image | text + audio | PASS | 4 | 324480 |
| audio_audio | audio | text + audio | PASS | 3 | 224640 |
| video_audio | video | text + audio | PASS | 7 | 595200 |

## Performance

All latency values are seconds. Each configuration used 20 measured requests
after two warmups. Failed requests are excluded from metrics and would fail the
gate; this run had zero failures.

| Mode | C | OK/Fail | First text p50/p95/p99 | First audio p50/p95/p99 | E2E p50/p95/p99 | Req/s | Audio s/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| text | 1 | 20/0 | 0.060/0.062/0.064 | - | 0.862/0.875/0.876 | 1.159 | 0.000 |
| text | 2 | 20/0 | 0.071/0.096/0.118 | - | 0.917/1.272/1.608 | 2.011 | 0.000 |
| text | 4 | 20/0 | 0.082/0.120/0.120 | - | 0.934/0.988/0.989 | 4.231 | 0.000 |
| text_audio | 1 | 20/0 | 0.448/0.459/0.462 | 1.164/1.207/1.219 | 3.461/3.596/3.601 | 0.287 | 3.493 |
| text_audio | 2 | 20/0 | 0.474/0.495/0.504 | 1.724/1.851/1.860 | 4.355/4.517/4.523 | 0.457 | 6.026 |
| text_audio | 4 | 20/0 | 0.525/0.535/0.538 | 2.029/2.133/2.133 | 5.440/5.712/5.715 | 0.730 | 9.414 |

| Mode | C | Peak AI Core | Peak aggregate HBM MiB | HBM first/last delta MiB | Peak/first-last delta host GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| text | 1 | 66% | 89501 | 0 | 84.624/+0.608 |
| text | 2 | 66% | 89503 | 0 | 84.694/+0.054 |
| text | 4 | 64% | 89503 | 0 | 85.317/+0.074 |
| text_audio | 1 | 65% | 89504 | 0 | 85.436/+0.261 |
| text_audio | 2 | 64% | 89963 | +461 | 86.232/+0.701 |
| text_audio | 4 | 61% | 96833 | +4350 | 86.325/-0.007 |

The positive HBM deltas while increasing concurrency are persistent runtime
buffer/cache allocation. The stability run starts after those shapes have
already been exercised.

## Stability

The stability run used text-plus-audio at concurrency 4 for 100 measured
requests after two warmups.

| OK/Fail | Wall time | First text p50/p95/p99 | First audio p50/p95/p99 | E2E p50/p95/p99 | Req/s | Audio s/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100/0 | 134.413 s | 0.518/0.530/0.535 | 2.012/2.049/2.058 | 5.319/5.553/5.563 | 0.744 | 9.600 |

- Resource samples: 89.
- Peak AI Core: 62%.
- Peak aggregate HBM: 96836 MiB.
- Aggregate HBM first/last delta: 0 MiB.
- Peak host memory: 93.800 GiB.
- Host memory first/last delta: +5.566 GiB.
- Server errors, tracebacks, validation errors, or engine crashes: 0.
- NPU processes after clean shutdown: 0; HBM returned to 3096/2883 MiB idle usage.

The zero HBM delta proves no device-memory growth over this proxy stability
window. The host-memory delta uses system-wide `MemTotal - MemAvailable` and
includes page cache, so it is recorded as a follow-up signal rather than proof
of a process leak or leak-free host state.

## Reproduction

```bash
MODEL=/path/to/MiniCPM-o-4_5 \
ARTIFACT_DIR=artifacts/minicpmo_ascend/baseline/server \
bash benchmarks/competition/minicpmo_ascend/start_server.sh
```

```bash
VIDEO_INPUT=/path/to/Skiing.mp4 \
MODEL=/path/to/MiniCPM-o-4_5 \
MODEL_PATH=/path/to/MiniCPM-o-4_5 \
MODEL_MANIFEST=/path/to/model-sha256.txt \
OUTPUT_DIR=artifacts/minicpmo_ascend/baseline \
STABILITY_REQUESTS=100 \
bash benchmarks/competition/minicpmo_ascend/run_suite.sh \
  --concurrency 1 2 4 --num-requests 20 --warmups 2 \
  --seed 42 --thinker-max-tokens 256 --talker-max-tokens 256
```

When official material is published, record its version and checksums, replace
the proxy fixtures/requests with the official dataset and script, and rerun the
same gated flow. Do not compare a future official score directly with this
local proxy or infer unpublished weights from these measurements.
