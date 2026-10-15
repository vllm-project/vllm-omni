# Chatterbox Turbo and Original — H100

## Summary

- Vendor: Resemble AI
- Model: `ResembleAI/chatterbox-turbo` (Turbo) and `ResembleAI/chatterbox` (English Original)
- Task: English text-to-speech with the checkpoint's built-in voice or a voice cloned from one recording
- Mode: online `/v1/audio/speech` at up to 64 concurrent clients with the opt-in high-concurrency profiles; offline and default-profile serving as in the [A5000 recipe](Chatterbox-A5000.md)
- Hardware: 1x NVIDIA H100 SXM 80 GB
- Maintainer: @will-rice

## When to use this recipe

One 80 GB card serving many concurrent streams: the `*_high_concurrency`
profiles admit 64 requests, condition references on the GPU, and the
`*_bf16_mps` overlays add BF16 flow, native cuDNN attention and CUDA MPS. This
recipe also holds the comparison with the canonical upstream implementation.

## Supported model contract

The request contract is the one in the
[speech API](../../docs/serving/speech_api.md#chatterbox-turbo-and-original)
and the [A5000 recipe](Chatterbox-A5000.md#supported-model-contract).

| Profile on this hardware | Status |
| --- | --- |
| `chatterbox_turbo.yaml`, `chatterbox.yaml` (default: 8 AR slots, CPU conditioning) | runtime-qualified; nightly latency and throughput baselines below |
| `chatterbox_turbo_high_concurrency.yaml`, `chatterbox_high_concurrency.yaml` (64 requests, `conditioning_device: cuda:0`) | runtime-qualified at 64 HTTP clients |
| `chatterbox_turbo_high_concurrency_bf16_mps.yaml`, `chatterbox_high_concurrency_bf16_mps.yaml` (BF16 flow, cuDNN attention, MPS; Original starts with 30-token chunks) | runtime-qualified at 64 HTTP clients; Original is not real-time there |

Admission at 64 is a capacity contract, not a real-time guarantee: see the
evidence below for where each model keeps up with playback.

## References

- [Upstream implementation](https://github.com/resemble-ai/chatterbox) (`chatterbox-tts` 0.1.7, `5de7a54aa4e5e2baadb0182dde554908b48b85c2`)
- [A5000 recipe](Chatterbox-A5000.md): commands, offline use, limitations
- [Online examples](../../examples/online_serving/text_to_speech/README.md#chatterbox-turbo-and-original)
- [TTS benchmark README](../../benchmarks/tts/README.md): the shared `bench_tts.py` driver and `chatterbox_reference.py`
- [Nightly perf configuration](../../tests/dfx/perf/tests/test_tts.json)

## Hardware

- Accelerator model and per-device memory: NVIDIA H100 SXM, 80 GB
- Number of devices: 1; the API process and both stages share it (`conditioning_device: cuda:0`)
- Device interconnect: none used
- Host: Intel Xeon Platinum 8470 with a 22-CPU container quota and a 251 GB
  memory limit for the upstream comparison and nightly rows; not retained for
  the 64-client runs
- Qualification scope: CUDA only; the measurements below name their profile and revision

## Software environment

| Field | Upstream comparison and nightly baselines | 64-client qualification |
| --- | --- | --- |
| OS | Ubuntu 24.04.4 | Ubuntu 24.04 |
| Python | 3.12.3 | 3.12.3 |
| Driver / runtime | 580.126.09 (CUDA 13.0); PyTorch 2.13.0+cu130, torchaudio 2.11.0+cu130 | 595.91.07; PyTorch 2.13.0+cu130, torchaudio 2.11.0 |
| vLLM | 0.31.0 | 0.31.0 |
| Transformers / Diffusers | 5.14.1 / 0.40.0 | 5.14.1 / not recorded |
| vLLM-Omni | `c6fff58b8` (comparison and nightly) | `21bb93f10` |
| Checkpoints | Turbo `749d1c1a46eb10492095d68fbcf55691ccf137cd`; Original `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18` | same |

Upstream ran in its own environment: Python 3.11.15, `chatterbox-tts` 0.1.7,
PyTorch and torchaudio 2.6.0+cu124, NumPy 1.26.4, Transformers 5.2.0,
Diffusers 0.29.0, `resemble-perth` 1.0.1. Both sides used four CPU threads.

## Command

```bash
uv pip install -e '.[chatterbox]'
vllm serve ResembleAI/chatterbox-turbo --omni --port 8091 \
  --deploy-config vllm_omni/deploy/chatterbox_turbo_high_concurrency_bf16_mps.yaml
# Original: vllm serve ResembleAI/chatterbox --omni --port 8091 \
#   --deploy-config vllm_omni/deploy/chatterbox_high_concurrency_bf16_mps.yaml
```

The `_bf16_mps` overlays need `nvidia-cuda-mps-control` on `PATH`. Without MPS
or BF16 flow, use the `*_high_concurrency.yaml` profiles instead.

Load with the shared driver (`ResembleAI/chatterbox` for Original):

```bash
python benchmarks/tts/bench_tts.py \
  --model ResembleAI/chatterbox-turbo \
  --task voice_clone --locale en \
  --dataset-path benchmarks/build_dataset/seed_tts_smoke \
  --host 127.0.0.1 --port 8091 \
  --concurrency 64 --num-prompts 192 --num-warmups 64 \
  --output-dir results/turbo-c64
```

The comparison with upstream uses `benchmarks/tts/chatterbox_reference.py`;
its commands are in the [TTS benchmark README](../../benchmarks/tts/README.md#matched-chatterbox-reference-qualification).

## Verification

```bash
pytest tests/dfx/perf/scripts/run_benchmark.py --test-config-file tests/dfx/perf/tests/test_tts.json \
  -m "full_model and H100 and cards_1" -k chatterbox --run-level full_model
```

Expected: every case completes with zero failed requests, and the median
first-audio latency and real-time factor stay near the H100 baselines in
that file (the nightly rows below). The shared driver's report
must show `audio_continuity_ok` for every request of a warm Turbo run at 64
clients.

## Notes

- Memory usage: the 64-request profiles reserve 35 % (Turbo) or 40 %
  (Original, 128 CFG sequences at 2048 tokens) of the card for stage 0 and
  15 % for stage 1; whole-device usage during the single-request comparison
  was about 32 GB including workers and the resident encoders.
- Key flags: `conditioning_device: cuda:0` under stage 0 `additional_config`
  puts the reference encoders on the GPU; `flow_dtype: bfloat16` with
  `flow_attention_backend: cudnn` under stage 1 `hf_overrides` selects the
  faster flow at the cost of numerically different audio (HiFT stays FP32);
  `cuda_mps: true` shares the device between the API process and the stages.
- Known limitations: Original exceeds real-time at 64 clients and keeps up
  at 32 with the 30-token first chunk; Turbo's first cold wave carries the
  encoders' first-use cost (28 s p95 to first audio) that later waves do not.
  Everything in the A5000 recipe's limitations applies.

## Supported features

| Feature | Status |
| --- | --- |
| Built-in voice, voice cloning, async streaming | both variants, qualified at 64 HTTP clients |
| GPU reference conditioning | `conditioning_device: cuda:0`; every admitted reference is encoded in one batch, admission equal to stage 0's request capacity (64), a 128-entry cache |
| AR CUDA graphs | on by default; graph versus eager measured below |
| BF16 flow, cuDNN attention, CUDA MPS | opt-in overlays, measured below |
| Real-time at 64 clients | Turbo yes on warm and built-in workloads; Original no (32 clients yes) |
| Multilingual, VoiceDesign, instructions, prefix caching | unsupported |

## Qualification evidence

### Comparison with the canonical upstream implementation

Measured 2026-10-14 at `c6fff58b8` with `benchmarks/tts/chatterbox_reference.py`
on the host above: three fixed texts, seeds 0, 42 and 123, built-in and
cloned voice (the same `clone_2.wav`), nine timed requests per mode after two
warmup seeds per text, 108 timed generations in all. FP32 with TF32 off on
both sides, concurrency one, a 1000-token allowance; Omni in whole-utterance
mode on the default Model Runner V2, once with stage 0 eager and once with
stage 0 CUDA graphs (`--cuda-graphs`, still FP32); a fresh process per
backend, model and mode. Clone requests include fresh conditioning; model
loading, warmup and WAV writes are excluded. Upstream's `generate` includes
its CPU Perth watermark. Omni uses position-aligned flow noise, so equal
seeds do not give identical token draws.

| Model | Backend | Voice | E2E mean ± SD (s) | E2E p95 (s) | RTF mean ± SD | RTF p95 | GPU used p95 (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Turbo | upstream | builtin | 1.243 ± 0.897 | 2.571 | 0.163 ± 0.010 | 0.174 | 4315 |
| Turbo | upstream | clone | 1.436 ± 1.001 | 2.848 | 0.175 ± 0.020 | 0.202 | 4595 |
| Turbo | omni | builtin | 2.132 ± 1.557 | 4.329 | 0.285 ± 0.019 | 0.310 | 32336 |
| Turbo | omni | clone | 2.358 ± 1.535 | 4.592 | 0.299 ± 0.027 | 0.340 | 32526 |
| Turbo | omni, CUDA graphs | builtin | 0.585 ± 0.354 | 1.086 | 0.085 ± 0.016 | 0.106 | 32330 |
| Turbo | omni, CUDA graphs | clone | 0.689 ± 0.318 | 1.157 | 0.098 ± 0.025 | 0.128 | 32520 |
| Original | upstream | builtin | 2.002 ± 1.202 | 3.628 | 0.326 ± 0.035 | 0.360 | 4670 |
| Original | upstream | clone | 2.873 ± 1.786 | 5.376 | 0.313 ± 0.027 | 0.347 | 5277 |
| Original | omni | builtin | 2.300 ± 1.448 | 4.299 | 0.382 ± 0.041 | 0.428 | 31610 |
| Original | omni | clone | 3.400 ± 2.299 | 6.621 | 0.374 ± 0.024 | 0.403 | 32356 |
| Original | omni, CUDA graphs | builtin | 1.119 ± 0.579 | 1.945 | 0.199 ± 0.039 | 0.233 | 31594 |
| Original | omni, CUDA graphs | clone | 1.598 ± 0.932 | 2.931 | 0.189 ± 0.030 | 0.222 | 32340 |

Memory is whole-device usage sampled every 100 ms, including Omni's workers,
the resident encoders and the reserved KV cache. With stage 0 eager, Omni is
slower than upstream (built-in RTF: Turbo 0.285 against 0.163, Original
0.382 against 0.326). The CUDA-graph rows change only stage 0's
`enforce_eager`; with graphs on, Omni is faster than upstream (Turbo 0.085,
Original 0.199), so the eager gap is in stage 0 and CUDA graphs remove it.
The default profiles turn stage 0 graphs on. A profile on a different host places
the eager cost in vLLM's per-token host work, which depends on the host CPU;
it was not profiled here. The E2E SD mostly reflects the three text
lengths.

Quality, scored by the script's `quality` mode through the shared Seed-TTS
evaluator (`vllm_omni.benchmarks.data_modules.seed_tts_eval`) on the eager
pair: WER from `openai/whisper-large-v3` through `transformers` (revision
`06f233fe06e710322aca913c1bc4249a0d71fce1`), and SIM, the WavLM
(`microsoft/wavlm-base-plus`, `4c66d4806a428f2e922ccfa1a962776e232d487b`)
speaker-embedding cosine to the cloned reference clip. The built-in voice
ships as embeddings rather than a recording, so built-in rows have no SIM.
A final phrase is complete when the transcript's last three normalized words
match the text's.

| Model | Backend | Voice | Mean WER | Mean SIM | Final phrase complete |
| --- | --- | --- | --- | --- | --- |
| Turbo | upstream | builtin | 0.0000 | — | 9/9 |
| Turbo | upstream | clone | 0.0000 | 0.851 | 9/9 |
| Turbo | omni | builtin | 0.0000 | — | 9/9 |
| Turbo | omni | clone | 0.0000 | 0.849 | 9/9 |
| Original | upstream | builtin | 0.0000 | — | 9/9 |
| Original | upstream | clone | 0.0000 | 0.871 | 9/9 |
| Original | omni | builtin | 0.0022 | — | 9/9 |
| Original | omni | clone | 0.0000 | 0.870 | 9/9 |

All 108 outputs were finite, nonempty mono 24 kHz PCM. Original's one
built-in error is a single substituted article in the longest text. The
predeclared guards held: Omni's WER at most 0.05 above upstream, SIM at most
0.05 below, at most one extra missing final phrase per nine. Nine English
sentences per row show automated agreement, not statistical equivalence or
subjective parity, and these are not human listening scores.

### Nightly latency and throughput baselines, default profiles

Measured 2026-10-14 at `c6fff58b8` on the host above with the Verification
command: one pytest session, the eight Chatterbox cases of
`tests/dfx/perf/tests/test_tts.json`, all passed with zero failed requests.
The default profiles run stage 0 in BF16 with async chunk streaming and
stage 1 in FP32; requests go through `/v1/audio/speech` with the built-in
voice, the bundled `seed_tts_smoke` English texts and seed 42. Latency:
20 requests at concurrency one after two warmups; throughput: 80 requests at
concurrency eight after eight warmups; stress: 100 requests at two per second
after two warmups. Only stage 0's `enforce_eager` differs between a graph and
an eager row.

| Variant / AR mode | Workload | Mean RTF | Mean first audio (ms) | Mean audio duration (s) |
| --- | --- | --- | --- | --- |
| Turbo / graph | Latency | 0.110 | 223 | 3.984 |
| Turbo / eager | Latency | 0.283 | 419 | 3.984 |
| Original / graph | Latency | 0.302 | 505 | 3.436 |
| Original / eager | Latency | 0.398 | 685 | 3.482 |
| Turbo / graph | Throughput, 8 clients | 0.249 | 571 | 4.047 |
| Original / graph | Throughput, 8 clients | 0.839 | 1405 | 3.507 |
| Turbo / graph | Stress, 2 requests/s | 0.137 | 305 | 4.060 |
| Original / graph | Stress, 2 requests/s | 0.678 | 1138 | 3.517 |

The nightly JSON stores the median first-audio latency and median RTF of
every row, and the audio throughput of the throughput rows, under the H100
bucket from this run; B200 runs the same cases with no baseline. These rows
come from one bounded run per case and are qualification evidence, not a
benchmark: they cover English text of a few seconds and the built-in voice
only.

### 64 concurrent HTTP clients, CUDA overlays

Three synchronized waves of 64 clients per workload, 192 timed requests,
request-specific text and seed, streamed PCM, a 1000-token allowance. Cold
references are distinct gain envelopes of `clone_2.wav`, not different
speakers. Both models completed 576 of 576 requests with no error and no
repeated audio. The gap is the extra initial buffering needed when playback
starts at the first PCM byte, from PCM arrival times, with a 100 ms jitter
allowance per request.

| Model | Reference mode | p95 first audio (s) | p95 completion (s) | p95 RTF | p95 gap (s) |
| --- | --- | --- | --- | --- | --- |
| Turbo | Cold distinct | 28.378 | 36.039 | 1.800 | 0.000 |
| Turbo | Cached repeated | 1.513 | 7.384 | 0.362 | 0.000 |
| Turbo | Built-in | 1.640 | 8.126 | 0.459 | 0.000 |
| Original | Cold distinct | 9.799 | 28.569 | 1.326 | 0.799 |
| Original | Cached repeated | 3.533 | 25.598 | 1.201 | 2.217 |
| Original | Built-in | 3.543 | 18.078 | 1.243 | 0.478 |

Turbo's cold row includes the encoders' first use: its three cold waves had
p95 first audio of 28.4, 4.9 and 5.3 s. Stage 0 reached 64 active sequences
(128 for Original) in every warm and built-in wave; Turbo's cold waves peaked
at 64, 58 and 63. These numbers were produced by a per-model harness that has
since been replaced by the shared driver, whose `audio_underrun` and
`audio_continuity_ok` fields are the equivalent of the gap column.

Original at lower loads, same overlay, 128 warmups then 128 timed requests
of one ~65-word narration with one cached reference:

| First chunk | Clients | p95 RTF | p95 first audio (s) | p95 gap (s) |
| --- | --- | --- | --- | --- |
| 15 tokens | 8 | 0.346 | 1.376 | 0 |
| 15 tokens | 16 | 0.385 | 1.473 | 0 |
| 15 tokens | 32 | 0.589 | 2.043 | 0.327 |
| 15 tokens | 64 | 1.096 | 4.021 | 1.873 |
| 30 tokens | 32 | 0.560 | 2.130 | 0 |
| 30 tokens | 64 | 1.016 | 3.529 | 0.531 |

The Original overlay therefore starts with 30-token chunks and is real-time
at 32 clients on this workload, not at 64. Whisper large-v3 mean WER on 64
outputs of this narration was 0.22 % (Turbo) and 0.27 % (Original).

### WebSocket sentence frontend

Same overlays, three waves of 64 sessions per workload with the 1000-token
allowance; each session sends two text chunks, and the frontend yields five
sentence segments. 576 sessions per model completed with no error and no
repeated audio, 2880 validated segments each. p95 seconds from session
submission (later sentences wait behind earlier ones):

| Variant | Cold first audio / completion | Warm first audio / completion | Built-in first audio / completion |
| --- | --- | --- | --- |
| Turbo | 12.98 / 38.25 | 2.54 / 17.06 | 2.40 / 17.08 |
| Original | 11.99 / 44.30 | 4.00 / 37.07 | 4.04 / 29.01 |

These sessions were driven by the removed per-model harness; the shared
driver has no WebSocket mode, so they are not reproducible from this tree.
