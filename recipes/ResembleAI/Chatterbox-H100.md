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
`*_cuda` overlays add BF16 flow, native cuDNN attention and CUDA MPS. This
recipe also holds the comparison with the canonical upstream implementation.

## Supported model contract

The request contract is the one in the
[speech API](../../docs/serving/speech_api.md#chatterbox-turbo-and-original)
and the [A5000 recipe](Chatterbox-A5000.md#supported-model-contract).

| Profile on this hardware | Status |
| --- | --- |
| `chatterbox_turbo.yaml`, `chatterbox.yaml` (default: 8 AR slots, CPU conditioning) | runtime-qualified; nightly latency and throughput baselines below |
| `chatterbox_turbo_high_concurrency.yaml`, `chatterbox_high_concurrency.yaml` (64 requests, `conditioning_device: cuda:0`) | runtime-qualified at 64 HTTP clients |
| `chatterbox_turbo_high_concurrency_cuda.yaml`, `chatterbox_high_concurrency_cuda.yaml` (BF16 flow, cuDNN attention, MPS; Original starts with 30-token chunks) | runtime-qualified at 64 HTTP clients; Original is not real-time there |

Admission at 64 is a capacity contract, not a real-time guarantee: see the
evidence below for where each model keeps up with playback.

## References

- [Upstream implementation](https://github.com/resemble-ai/chatterbox) (`chatterbox-tts` 0.1.7, `5de7a54aa4e5e2baadb0182dde554908b48b85c2`)
- [A5000 recipe](Chatterbox-A5000.md): commands, offline use, limitations
- [Online examples](../../examples/online_serving/text_to_speech/README.md#chatterbox-turbo-and-original)
- [TTS benchmark README](../../benchmarks/tts/README.md): the shared `bench_tts.py` driver and `chatterbox_reference.py`
- [Nightly perf configuration](../../tests/dfx/perf/tests/test_chatterbox.json)

## Hardware

- Accelerator model and per-device memory: NVIDIA H100 SXM, 80 GB
- Number of devices: 1; the API process and both stages share it (`conditioning_device: cuda:0`)
- Device interconnect: none used
- Host memory: not retained for these runs
- Qualification scope: CUDA only; the measurements below name their profile and revision

## Software environment

| Field | Upstream comparison and nightly baselines | 64-client qualification |
| --- | --- | --- |
| OS | Ubuntu 24.04.3 | Ubuntu 24.04 |
| Python | 3.12.3 | 3.12.3 |
| Driver / runtime | 580.159.04; PyTorch 2.13.0+cu130, torchaudio 2.11.0+cu130 | 595.91.07; PyTorch 2.13.0+cu130, torchaudio 2.11.0 |
| vLLM | 0.31.0 | 0.31.0 |
| Transformers | 5.14.1 | 5.14.1 |
| vLLM-Omni | `cb4679038` (comparison), `0de536675` with `0a47e146a` (nightly) | `21bb93f10` |
| Checkpoints | Turbo `749d1c1a46eb10492095d68fbcf55691ccf137cd`; Original `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18` | same |

Upstream ran in its own environment: PyTorch and torchaudio 2.6.0+cu124,
NumPy 1.26.4, Transformers 5.2.0, Diffusers 0.29.0, Perth
`ff1c8ac55a976971245cdd53c18d6131ca00d993`. Both sides used four CPU threads.

## Command

```bash
uv pip install -e '.[chatterbox]'
vllm serve ResembleAI/chatterbox-turbo --omni --port 8091 \
  --deploy-config vllm_omni/deploy/chatterbox_turbo_high_concurrency_cuda.yaml
# Original: vllm serve ResembleAI/chatterbox --omni --port 8091 \
#   --deploy-config vllm_omni/deploy/chatterbox_high_concurrency_cuda.yaml
```

The `_cuda` overlays need `nvidia-cuda-mps-control` on `PATH`. Without MPS
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
pytest tests/dfx/perf/scripts/run_benchmark.py --test-config-file tests/dfx/perf/tests/test_chatterbox.json \
  -m "full_model and H100 and cards_1"
```

Expected: every case completes with zero failed requests, and the median
first-audio latency and real-time factor stay within the H100 baselines in
that file (the graph-mode latency rows below). The shared driver's report
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
| GPU reference conditioning | `conditioning_device: cuda:0`; batches of up to 8, a 64-request admission bound, a 128-entry cache |
| AR CUDA graphs | on by default; graph versus eager measured below |
| BF16 flow, cuDNN attention, CUDA MPS | opt-in overlays, measured below |
| Real-time at 64 clients | Turbo yes on warm and built-in workloads; Original no (32 clients yes) |
| Multilingual, VoiceDesign, instructions, prefix caching | unsupported |

## Qualification evidence

### Comparison with the canonical upstream implementation

Measured 2026-10-13 with `benchmarks/tts/chatterbox_reference.py`: three
fixed texts, seeds 0, 42 and 123, built-in and cloned voice (the same
`clone_2.wav`), nine timed requests per mode after two warmup seeds per text,
72 timed generations in all. FP32 with TF32 off on both sides, concurrency
one, a 1000-token allowance; Omni in whole-utterance mode with stage 0
eager; a fresh process per backend and model. Clone requests include fresh
conditioning; model loading, warmup and WAV writes are excluded. Upstream's
`generate` includes its CPU Perth watermark. Omni uses position-aligned flow
noise, so equal seeds do not give identical token draws.

| Model | Backend | Voice | E2E mean ± SD (s) | E2E p95 (s) | RTF mean ± SD | RTF p95 | GPU used p95 (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Turbo | upstream | builtin | 1.130 ± 0.803 | 2.276 | 0.149 ± 0.007 | 0.158 | 4315 |
| Turbo | upstream | clone | 1.340 ± 1.027 | 2.836 | 0.156 ± 0.008 | 0.166 | 4595 |
| Turbo | omni | builtin | 2.107 ± 1.572 | 4.327 | 0.278 ± 0.012 | 0.292 | 32334 |
| Turbo | omni | clone | 2.421 ± 1.430 | 4.511 | 0.320 ± 0.054 | 0.399 | 32524 |
| Original | upstream | builtin | 1.907 ± 1.236 | 3.688 | 0.302 ± 0.024 | 0.330 | 4952 |
| Original | upstream | clone | 2.718 ± 1.743 | 5.113 | 0.292 ± 0.019 | 0.315 | 5277 |
| Original | omni | builtin | 2.887 ± 1.678 | 5.302 | 0.494 ± 0.117 | 0.632 | 31588 |
| Original | omni | clone | 4.094 ± 2.955 | 8.780 | 0.443 ± 0.074 | 0.554 | 32332 |

Memory is whole-device usage sampled every 100 ms, including Omni's workers,
the resident encoders and the reserved KV cache. In this eager, FP32,
single-request configuration Omni is slower than upstream; with the default
profile (stage 0 CUDA graphs) the nightly latency row below reaches
upstream's Turbo RTF.

Quality, same Whisper large-v3 (FP32, temperature zero, English text
normalizer; checkpoint SHA256
`e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb`) and
the checkpoint's voice encoder for both backends. Speaker cosine is to the
reference embedding (clone) or the checkpoint speaker embedding (built-in):

| Model | Backend | Voice | Aggregate WER | Mean speaker cosine | Final phrase complete |
| --- | --- | --- | --- | --- | --- |
| Turbo | upstream | builtin | 0.0000 | 0.9389 | 9/9 |
| Turbo | upstream | clone | 0.0000 | 0.8417 | 9/9 |
| Turbo | omni | builtin | 0.0000 | 0.9362 | 9/9 |
| Turbo | omni | clone | 0.0000 | 0.8296 | 9/9 |
| Original | upstream | builtin | 0.0000 | 0.6241 | 9/9 |
| Original | upstream | clone | 0.0000 | 0.9123 | 9/9 |
| Original | omni | builtin | 0.0042 | 0.6411 | 9/9 |
| Original | omni | clone | 0.0000 | 0.9075 | 9/9 |

All 72 outputs were finite, nonempty mono 24 kHz PCM. The predeclared
guards held: Omni's WER at most 0.05 above upstream, speaker cosine at most
0.05 below, at most one extra missing final phrase per nine. Nine English
sentences per row show automated agreement, not statistical equivalence or
subjective parity.

### Nightly latency and throughput baselines, default profiles

Built-in voice, the bundled `seed_tts_smoke` English texts, seed 42. Latency:
20 requests at concurrency one after two warmups; throughput: 80 requests at
concurrency eight after eight warmups; stress: 100 requests at two per second
after two warmups. Only stage 0's `enforce_eager` differs between a graph and
an eager row.

| Variant / AR mode | Workload | Mean RTF | Mean first audio (ms) | Mean audio duration (s) |
| --- | --- | --- | --- | --- |
| Turbo / graph | Latency | 0.148 | 263 | 3.984 |
| Turbo / eager | Latency | 0.325 | 436 | 3.984 |
| Original / graph | Latency | 0.427 | 609 | 3.436 |
| Original / eager | Latency | 0.596 | 934 | 3.482 |
| Turbo / graph | Throughput, 8 clients | 0.286 | 592 | 4.026 |
| Original / graph | Throughput, 8 clients | 0.913 | 1603 | 3.494 |
| Turbo / graph | Stress, 2 requests/s | 0.187 | 371 | 4.063 |
| Original / graph | Stress, 2 requests/s | 0.976 | 1602 | 3.513 |

The nightly JSON stores the median first-audio latency and RTF of the graph
rows under the H100 bucket; B200 runs the same cases with no baseline.

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
