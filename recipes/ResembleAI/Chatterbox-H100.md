# Chatterbox Turbo and Original — H100

## Summary

- Vendor: Resemble AI
- Model: `ResembleAI/chatterbox-turbo` (Turbo) and `ResembleAI/chatterbox` (English Original)
- Task: English text-to-speech with the checkpoint's built-in voice or a voice cloned from one recording
- Mode: online `/v1/audio/speech` with the high-concurrency profiles, which admit 64 requests. Real-time playback holds up to 16 cloned-voice or 32 built-in clients for Turbo, and up to 8 built-in clients for Original (16 with the cuDNN/MPS overlay) but at no measured cloned-voice load. Offline and default-profile serving as in the [A5000 recipe](Chatterbox-A5000.md)
- Hardware: 1x NVIDIA H100 SXM 80 GB
- Maintainer: @will-rice

## When to use this recipe

One 80 GB card serving many concurrent streams: the `*_high_concurrency`
profiles admit 64 requests and condition references on the GPU, and the
`*_cudnn_mps` overlays add cuDNN flow attention and CUDA MPS to them. All
profiles run the flow decoder in BF16 and HiFT in FP32. Cloned voices cost
more than the built-in voice at every load, because each new reference is
encoded before its request starts; size for distinct references if the
traffic is cloning. This recipe also holds the comparison with the canonical
upstream implementation.

## Supported model contract

The request contract is the one in the
[speech API](../../docs/serving/speech_api.md#chatterbox-turbo-and-original)
and the [A5000 recipe](Chatterbox-A5000.md#supported-model-contract).

| Profile on this hardware | Status |
| --- | --- |
| `chatterbox_turbo.yaml`, `chatterbox.yaml` (default: 8 AR slots, CPU conditioning, BF16 flow, FP32 HiFT) | runtime-qualified; nightly latency and throughput baselines below |
| `chatterbox_turbo_high_concurrency.yaml`, `chatterbox_high_concurrency.yaml` (64 requests, `conditioning_device: cuda:0`) | runtime-qualified at 8–64 HTTP clients. Real-time: Turbo up to 16 cloned or 32 built-in clients; Original up to 8 built-in clients and at no measured cloned load |
| `chatterbox_turbo_high_concurrency_cudnn_mps.yaml`, `chatterbox_high_concurrency_cudnn_mps.yaml` (cuDNN attention + MPS over the high-concurrency profiles; Original starts with 30-token chunks) | runtime-qualified at 8–64 HTTP clients. Real-time: Turbo up to 8 cloned or 32 built-in clients; Original up to 16 built-in clients and at no measured cloned load |

Real-time here means p95 real-time factor below 1 and every stream
continuous (no playback underrun beyond 100 ms). Admission at 64 is a
capacity contract, not a real-time guarantee; the evidence below gives each
row.

## References

- [Upstream implementation](https://github.com/resemble-ai/chatterbox) (`chatterbox-tts` 0.1.7, `5de7a54aa4e5e2baadb0182dde554908b48b85c2`)
- [A5000 recipe](Chatterbox-A5000.md): commands, offline use, limitations
- [Online examples](../../examples/online_serving/text_to_speech/README.md#chatterbox-turbo-and-original)
- [TTS benchmark README](../../benchmarks/tts/README.md): the shared `bench_tts.py` driver, `chatterbox_reference.py` and `chatterbox_conditioning.py`
- [Nightly perf configuration](../../tests/dfx/perf/tests/test_tts.json)
- Seed-TTS evaluation set: [`zhaochenyang20/seed-tts-eval`](https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval) (`8f5e1aa2a35d42f42e940074c1983358b9491f89`)

## Hardware

- Accelerator model and per-device memory: NVIDIA H100 SXM, 80 GB
- Number of devices: 1; the API process and both stages share it (`conditioning_device: cuda:0`)
- Device interconnect: none used
- Host: Intel Xeon Platinum 8470 with a 22-CPU container quota and a 251 GB
  memory limit; the load driver ran on the same host
- Qualification scope: CUDA only; the measurements below name their profile and revision

## Software environment

| Field | Value |
| --- | --- |
| OS | Ubuntu 24.04.4 |
| Python | 3.12.3 |
| Driver / runtime | 580.126.09 (CUDA 13.0); PyTorch 2.13.0+cu130, torchaudio 2.11.0+cu130 |
| vLLM | 0.31.0 |
| Transformers / Diffusers | 5.14.1 / 0.40.0 |
| vLLM-Omni | `39d191c53` plus the BF16-flow default in `chatterbox_turbo.yaml` and `chatterbox.yaml` |
| Checkpoints | Turbo `749d1c1a46eb10492095d68fbcf55691ccf137cd`; Original `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18` |

Every measurement below comes from this host on 2026-10-15. Upstream ran in
its own environment: Python 3.11.15, `chatterbox-tts` 0.1.7, PyTorch and
torchaudio 2.6.0+cu124, NumPy 1.26.4, Transformers 5.2.0, Diffusers 0.29.0,
`resemble-perth` 1.0.1. Both sides of the upstream comparison used four CPU
threads.

## Command

```bash
uv pip install -e '.[chatterbox]'
vllm serve ResembleAI/chatterbox-turbo --omni --port 8091 \
  --deploy-config vllm_omni/deploy/chatterbox_turbo_high_concurrency.yaml
# Original: vllm serve ResembleAI/chatterbox --omni --port 8091 \
#   --deploy-config vllm_omni/deploy/chatterbox_high_concurrency.yaml
```

The `*_cudnn_mps` overlays need `nvidia-cuda-mps-control` on `PATH` and
native cuDNN SDPA. They are not faster for every workload; compare the two
tables below for yours.

Load with the shared driver (`ResembleAI/chatterbox` for Original). For
cloned voices, point `--dataset-path` at a Seed-TTS layout whose prompts
are distinct recordings longer than 5 s, with at least as many rows as
`--num-prompts`, and start a fresh server for each run, so that every
request encodes a new reference:

```bash
python benchmarks/tts/bench_tts.py \
  --model ResembleAI/chatterbox-turbo \
  --task voice_clone --locale en \
  --dataset-path /data/seed-tts-eval-ref5s \
  --host 127.0.0.1 --port 8091 \
  --concurrency 64 --num-prompts 192 --num-warmups 64 \
  --output-len 1000 --output-dir results/turbo-c64 --wer-eval \
  -- --metric-percentiles 50,95,99 --save-detailed
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
that file (the nightly rows below). With the high-concurrency Turbo
profile, the shared driver's report shows a 100.00 % streaming continuity
rate at 16 cloned-voice clients and at 32 built-in clients.

## Notes

- Memory usage: stage 0 reserves 35 % of the card (Turbo, and the default
  Original) or 40 % (high-concurrency Original). From the stage-0 startup
  lines: Turbo's 26.5 GiB of KV cache holds about 289,000 tokens, 141
  sequences of 2048 tokens; the high-concurrency Original's 30.1 GiB holds
  262,592 tokens, 128.2 sequences, just above its 128 guidance sequences.
  Stage 1 allocates no KV cache. Whole-device usage after startup was 30.2
  GB (Turbo) and 34.1 GB (Original) with the high-concurrency profiles, and
  about 32 GB during the single-request comparison.
- Key flags: `conditioning_device: cuda:0` under stage 0
  `additional_config` puts the reference encoders on the GPU;
  `flow_dtype: bfloat16` under stage 1 `hf_overrides` (the default in both
  profiles) runs the flow decoder under BF16 autocast with FP32 weights and
  FP32 HiFT, and `flow_dtype: float32` restores the FP32 flow;
  `flow_attention_backend: cudnn` selects cuDNN attention for the BF16
  flow; `cuda_mps: true` shares the device between the API process and the
  stages.
- Known limitations: cloned voices add the reference encoding to first
  audio (about 0.9 s at 8 Turbo clients, 5.7 s at 64); Original is not
  real-time for cloned voices at 8 clients on either overlay. Everything in
  the A5000 recipe's limitations applies.

## Supported features

| Feature | Status |
| --- | --- |
| Built-in voice, voice cloning, async streaming | both variants, measured at 8–64 HTTP clients with distinct references |
| GPU reference conditioning | `conditioning_device: cuda:0`; every admitted reference is encoded in one batch, admission equal to stage 0's request capacity (64); 64 references take 3.3 s (Turbo) or 2.2 s (Original) on their own |
| AR CUDA graphs | on by default; graph versus eager measured below |
| BF16 flow decoder, FP32 HiFT | default in every profile; quality and speed measured below |
| cuDNN flow attention, CUDA MPS | `*_cudnn_mps` overlays, measured below |
| Real-time with cloned voices | Turbo up to 16 clients (`*_high_concurrency`); Original at no measured load |
| Real-time with the built-in voice | Turbo up to 32 clients; Original up to 8 (`*_high_concurrency`) or 16 (`*_cudnn_mps`) |
| Multilingual, VoiceDesign, instructions, prefix caching | unsupported |

## Qualification evidence

### Comparison with the canonical upstream implementation

Measured with `benchmarks/tts/chatterbox_reference.py`: three fixed texts,
seeds 0, 42 and 123, built-in and cloned voice (the same `clone_2.wav`),
nine timed requests per mode after two warmup seeds per text, 108 timed
generations in all. Concurrency one, TF32 off, a 1000-token allowance.
Upstream runs FP32 throughout. Omni runs every weight in FP32 and the flow
decoder in BF16, as the deploy files set it, in whole-utterance mode on the
default Model Runner V2, once with stage 0 eager and once with stage 0
CUDA graphs (`--cuda-graphs`); a fresh process per backend, model and mode.
Clone requests include fresh conditioning; model loading, warmup and WAV
writes are excluded. Upstream's `generate` includes its CPU Perth
watermark. Omni uses position-aligned flow noise, so equal seeds do not
give identical token draws.

| Model | Backend | Voice | E2E mean ± SD (s) | E2E p95 (s) | RTF mean ± SD | RTF p95 | GPU used p95 (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Turbo | upstream | builtin | 1.230 ± 0.830 | 2.438 | 0.167 ± 0.016 | 0.190 | 4315 |
| Turbo | upstream | clone | 1.463 ± 1.045 | 2.912 | 0.176 ± 0.017 | 0.197 | 4595 |
| Turbo | omni | builtin | 2.268 ± 1.640 | 4.584 | 0.304 ± 0.020 | 0.326 | 31890 |
| Turbo | omni | clone | 2.504 ± 1.601 | 4.818 | 0.319 ± 0.028 | 0.348 | 32608 |
| Turbo | omni, CUDA graphs | builtin | 0.582 ± 0.326 | 1.044 | 0.087 ± 0.019 | 0.108 | 31904 |
| Turbo | omni, CUDA graphs | clone | 0.700 ± 0.317 | 1.164 | 0.101 ± 0.028 | 0.139 | 32094 |
| Original | upstream | builtin | 2.012 ± 1.226 | 3.675 | 0.326 ± 0.033 | 0.361 | 4670 |
| Original | upstream | clone | 2.933 ± 1.861 | 5.567 | 0.317 ± 0.025 | 0.349 | 5277 |
| Original | omni | builtin | 2.517 ± 1.529 | 4.603 | 0.424 ± 0.054 | 0.487 | 31572 |
| Original | omni | clone | 3.529 ± 2.213 | 6.613 | 0.401 ± 0.043 | 0.445 | 32100 |
| Original | omni, CUDA graphs | builtin | 1.207 ± 0.556 | 1.970 | 0.222 ± 0.057 | 0.285 | 31556 |
| Original | omni, CUDA graphs | clone | 1.640 ± 0.895 | 2.845 | 0.198 ± 0.038 | 0.240 | 32084 |

Memory is whole-device usage sampled every 100 ms, including Omni's workers,
the resident encoders and the reserved KV cache. With stage 0 eager, Omni is
slower than upstream (built-in RTF: Turbo 0.304 against 0.167, Original
0.424 against 0.326). The CUDA-graph rows change only stage 0's
`enforce_eager`; with graphs on, Omni is faster than upstream (Turbo 0.087,
Original 0.222), so the eager gap is in stage 0 and CUDA graphs remove it.
The default profiles turn stage 0 graphs on. The same script on this host
with an FP32 flow (copies of the two deploy files with `flow_dtype: float32`)
measured built-in RTF 0.290 eager and 0.088 with graphs for Turbo, and 0.415
and 0.203 for Original: at one request the BF16 flow costs up to 0.02 RTF,
and it pays off under load (see Flow precision below). The E2E SD mostly
reflects the three text lengths.

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
| Original | omni | clone | 0.0000 | 0.869 | 9/9 |

All 108 outputs were finite, nonempty mono 24 kHz PCM. Original's one
built-in error is a single substituted article in the longest text. The
predeclared guards held with the BF16 flow: Omni's WER at most 0.05 above
upstream, SIM at most 0.05 below, at most one extra missing final phrase
per nine. Nine English sentences per row show automated agreement, not
statistical equivalence or subjective parity, and these are not human
listening scores.

### Reference conditioning batch

`benchmarks/tts/chatterbox_conditioning.py` on `cuda:0`: 64 references cut
from `clone_2.wav` at 6.1, 8.3, 10.1 and 16 s with distinct amplitude
envelopes, encoded one at a time and then as one batch, three cycles.

| Model | 64 references, serial (s) | 64 references, one batch (s) | Peak device memory (MiB) | Serial/batch parity |
| --- | --- | --- | --- | --- |
| Turbo | 10.22–10.74 | 3.29–4.00 | 4493 | passed: tokens and features identical, speaker embedding within 1.2e-7 |
| Original | 8.23–8.90 | 2.18–2.29 | 3367 | passed: tokens and features identical, speaker embedding within 1.2e-7 |

Turbo's first batch took 4.00 s and the next two 3.30 s and 3.29 s. The
voice encoder ran once over all 784 partial windows, the speech tokenizer in two
calls of 64, and CAMPPlus in three calls grouped by equal waveform length.
Peak device memory is whole-device usage sampled every 100 ms.

### Nightly latency and throughput baselines, default profiles

Measured with the Verification command: one pytest session, the eight
Chatterbox cases of `tests/dfx/perf/tests/test_tts.json`, all passed with
zero failed requests. The default profiles run stage 0 in BF16 with async
chunk streaming and stage 1 with the BF16 flow and FP32 HiFT; requests go
through `/v1/audio/speech` with the built-in voice, the bundled
`seed_tts_smoke` English texts and seed 42. Latency: 20 requests at
concurrency one after two warmups; throughput: 80 requests at concurrency
eight after eight warmups; stress: 100 requests at two per second after two
warmups. Only stage 0's `enforce_eager` differs between a graph and an
eager row.

| Variant / AR mode | Workload | Mean RTF | Mean first audio (ms) | Mean audio duration (s) |
| --- | --- | --- | --- | --- |
| Turbo / graph | Latency | 0.121 | 240 | 3.984 |
| Turbo / eager | Latency | 0.288 | 429 | 3.984 |
| Original / graph | Latency | 0.370 | 587 | 3.436 |
| Original / eager | Latency | 0.437 | 761 | 3.482 |
| Turbo / graph | Throughput, 8 clients | 0.229 | 539 | 4.018 |
| Original / graph | Throughput, 8 clients | 0.735 | 1330 | 3.498 |
| Turbo / graph | Stress, 2 requests/s | 0.146 | 314 | 4.060 |
| Original / graph | Stress, 2 requests/s | 0.646 | 1064 | 3.518 |

The nightly JSON stores the median first-audio latency and median RTF of
every row, and the audio throughput of the throughput rows, under the H100
bucket from this run; B200 runs the same cases with no baseline. These rows
come from one bounded run per case and are qualification evidence, not a
benchmark: they cover English text of a few seconds and the built-in voice
only.

### Flow precision

Every profile now runs the flow decoder in BF16 (weights FP32, HiFT FP32).
The same-host ablation below changes only stage 1's `flow_dtype`: the FP32
rows serve copies of the deploy files with `flow_dtype: float32`, every
other key unchanged. Default profiles, stage 0 CUDA graphs, the nightly
workload (built-in voice, `seed_tts_smoke`, seed 42; 20 requests at one
client after two warmups, 80 at eight clients after eight), one server per
row pair:

| Model | Flow | Clients | Median first audio (ms) | Median RTF | p95 RTF | Audio throughput | Continuity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Turbo | FP32 | 1 | 229.5 | 0.1135 | 0.1346 | 8.919 | 100.00 % |
| Turbo | BF16 | 1 | 230.5 | 0.1148 | 0.1377 | 8.656 | 100.00 % |
| Turbo | FP32 | 8 | 568.8 | 0.2369 | 0.3048 | 32.713 | 100.00 % |
| Turbo | BF16 | 8 | 532.9 | 0.2143 | 0.2942 | 36.331 | 100.00 % |
| Original | FP32 | 1 | 531.5 | 0.2991 | 0.3678 | 3.313 | 100.00 % |
| Original | BF16 | 1 | 555.6 | 0.3510 | 0.4263 | 2.850 | 100.00 % |
| Original | FP32 | 8 | 1430.3 | 0.8400 | 1.1578 | 9.384 | 63.75 % |
| Original | BF16 | 8 | 1229.6 | 0.7140 | 0.9300 | 11.064 | 100.00 % |

High-concurrency profiles at 64 clients, 192 requests after a 64-request
warmup, the protocol of the concurrency section below; the FP32 rows each
ran on a fresh server, the BF16 rows are that section's:

| Model | Voice | Flow | p95 first audio (s) | p95 RTF | p95 underrun (s) | Continuity | Mean WER | Mean SIM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Turbo | cloned | FP32 | 7.844 | 3.979 | 1.536 | 26.56 % | 0.0262 | 0.887 |
| Turbo | cloned | BF16 | 7.672 | 3.587 | 0.848 | 42.71 % | 0.0297 | 0.887 |
| Turbo | built-in | FP32 | 2.086 | 1.579 | 1.472 | 0.52 % | 0.0609 | — |
| Turbo | built-in | BF16 | 1.544 | 0.956 | 0.231 | 77.60 % | 0.0620 | — |
| Original | cloned | FP32 | 13.292 | 8.534 | 11.397 | 3.12 % | 0.0229 | 0.884 |
| Original | cloned | BF16 | 9.383 | 5.395 | 4.073 | 5.73 % | 0.0195 | 0.884 |
| Original | built-in | FP32 | 5.625 | 4.864 | 7.332 | 0.00 % | 0.0628 | — |
| Original | built-in | BF16 | 3.294 | 2.519 | 2.039 | 0.52 % | 0.0639 | — |

The BF16 flow costs nothing measurable for Turbo at one client and 0.05
median RTF for Original; from eight clients on it raises throughput (Turbo
+11 %, Original +18 %) and, at 64 clients, cuts p95 RTF by 10–48 % and p95
underrun by 45–84 %. At 64 clients mean WER moves by at most 0.0035 in
either direction and mean SIM not at all; the single-request quality guards
above hold with the BF16 flow.

### Concurrent HTTP clients, high-concurrency profiles

The shared `bench_tts.py` driver at 8, 16, 32 and 64 clients: 96 requests at
8 and 16 clients, 128 at 32 and 192 at 64, each after a warmup of one
request per client at full concurrency; a 1000-token allowance, streamed
PCM, production sampling. Built-in rows use the bundled `seed_tts_smoke`
texts (about 4 s of audio each). Cloned rows use the English Seed-TTS
evaluation set, filtered to its 215 distinct prompt recordings longer than
5 s (Turbo's minimum), with each recording's own target text. Every cloned
request carries a different recording: 8 and 16 clients ran on one server
with disjoint sets of 96, after that server's built-in rows; 32 and 64
clients each ran on a fresh server. The only repeated reference in a run is
the first one, which the warmup also uses. Times in seconds; RTF is
end-to-end latency over audio duration, so it includes the wait for first
audio; underrun is the extra buffering a stream needs when playback starts
at its first byte; continuity is the share of requests whose underrun stays
within 100 ms. No request failed.

`chatterbox_turbo_high_concurrency.yaml` and `chatterbox_high_concurrency.yaml`:

| Model | Voice | Clients | p50 first audio | p95 first audio | p95 completion | p95 RTF | p95 underrun | Continuity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Turbo | cloned | 8 | 1.380 | 1.623 | 2.074 | 0.713 | 0.000 | 100.00 % |
| Turbo | cloned | 16 | 2.111 | 2.346 | 3.020 | 0.985 | 0.000 | 100.00 % |
| Turbo | cloned | 32 | 4.219 | 4.481 | 6.333 | 2.154 | 0.000 | 96.09 % |
| Turbo | cloned | 64 | 7.102 | 7.672 | 10.036 | 3.587 | 0.848 | 42.71 % |
| Turbo | built-in | 8 | 0.527 | 0.609 | 1.197 | 0.354 | 0.000 | 100.00 % |
| Turbo | built-in | 16 | 0.656 | 0.694 | 1.358 | 0.415 | 0.000 | 100.00 % |
| Turbo | built-in | 32 | 0.898 | 0.974 | 2.078 | 0.584 | 0.000 | 100.00 % |
| Turbo | built-in | 64 | 1.449 | 1.544 | 3.227 | 0.956 | 0.231 | 77.60 % |
| Original | cloned | 8 | 1.365 | 2.184 | 3.904 | 1.223 | 0.265 | 77.08 % |
| Original | cloned | 16 | 2.498 | 3.070 | 4.977 | 1.620 | 0.285 | 71.88 % |
| Original | cloned | 32 | 3.949 | 5.964 | 9.863 | 3.447 | 1.890 | 10.94 % |
| Original | cloned | 64 | 9.081 | 9.383 | 15.834 | 5.395 | 4.073 | 5.73 % |
| Original | built-in | 8 | 1.348 | 1.400 | 2.622 | 0.916 | 0.000 | 100.00 % |
| Original | built-in | 16 | 1.566 | 1.616 | 3.005 | 1.045 | 0.000 | 100.00 % |
| Original | built-in | 32 | 2.071 | 2.107 | 4.349 | 1.492 | 0.314 | 0.00 % |
| Original | built-in | 64 | 3.126 | 3.294 | 7.109 | 2.519 | 2.039 | 0.52 % |

`chatterbox_turbo_high_concurrency_cudnn_mps.yaml` and
`chatterbox_high_concurrency_cudnn_mps.yaml`:

| Model | Voice | Clients | p50 first audio | p95 first audio | p95 completion | p95 RTF | p95 underrun | Continuity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Turbo | cloned | 8 | 0.915 | 1.292 | 1.850 | 0.551 | 0.000 | 100.00 % |
| Turbo | cloned | 16 | 1.998 | 2.420 | 3.109 | 1.036 | 0.000 | 100.00 % |
| Turbo | cloned | 32 | 3.965 | 4.569 | 6.482 | 2.196 | 0.184 | 91.41 % |
| Turbo | cloned | 64 | 7.033 | 7.696 | 10.134 | 3.574 | 0.811 | 45.31 % |
| Turbo | built-in | 8 | 0.512 | 0.673 | 1.632 | 0.440 | 0.000 | 100.00 % |
| Turbo | built-in | 16 | 0.651 | 0.668 | 1.326 | 0.384 | 0.000 | 100.00 % |
| Turbo | built-in | 32 | 0.919 | 0.961 | 1.930 | 0.554 | 0.000 | 100.00 % |
| Turbo | built-in | 64 | 1.404 | 1.538 | 3.163 | 0.894 | 0.225 | 89.58 % |
| Original | cloned | 8 | 1.671 | 2.255 | 3.637 | 1.073 | 0.000 | 100.00 % |
| Original | cloned | 16 | 1.952 | 2.986 | 3.987 | 1.240 | 0.000 | 100.00 % |
| Original | cloned | 32 | 3.318 | 5.689 | 8.350 | 2.421 | 0.184 | 89.84 % |
| Original | cloned | 64 | 6.987 | 9.055 | 13.551 | 4.239 | 1.613 | 46.88 % |
| Original | built-in | 8 | 1.576 | 1.659 | 2.443 | 0.780 | 0.000 | 100.00 % |
| Original | built-in | 16 | 1.711 | 1.778 | 2.410 | 0.829 | 0.000 | 100.00 % |
| Original | built-in | 32 | 2.229 | 2.363 | 3.529 | 1.158 | 0.000 | 100.00 % |
| Original | built-in | 64 | 3.602 | 3.712 | 5.943 | 2.046 | 0.272 | 42.19 % |

Mean audio duration was 3.86–3.92 s (cloned) and 3.98–4.04 s (Turbo
built-in) or 3.59–3.65 s (Original built-in). Reading the tables:

- Conditioning cost: at the same client count, cloned first audio exceeds
  built-in first audio by the reference encoding. On the high-concurrency
  Turbo profile the p50 gap is 0.85 s at 8 clients, 1.46 s at 16, 3.32 s at
  32 and 5.65 s at 64; the Original's is 0.02, 0.93, 1.88 and 5.96 s.
- Turbo is real-time (p95 RTF below 1, every stream continuous) up to 16
  cloned clients on `*_high_concurrency` and 8 on `*_cudnn_mps`, and up to
  32 built-in clients on both. At 64 neither voice is: cloned streams are
  43–45 % continuous, built-in 78–90 %.
- Original is real-time up to 8 built-in clients on `*_high_concurrency`
  and 16 on `*_cudnn_mps`, where every stream also stays continuous at 32
  (p95 RTF 1.158). With cloned voices it is real-time at no measured load:
  `*_cudnn_mps` keeps every stream continuous through 16 clients, but p95
  RTF is 1.07 at 8 because of the encoding wait.

Quality at 64 clients, the shared Seed-TTS evaluator over all 192 outputs
of each run (no ASR or SIM failure; median WER 0 in every run):

| Model | Profile | Voice | Mean WER | Mean SIM (median) |
| --- | --- | --- | --- | --- |
| Turbo | `*_high_concurrency` | cloned | 0.0297 | 0.887 (0.891) |
| Turbo | `*_high_concurrency` | built-in | 0.0620 | — |
| Turbo | `*_cudnn_mps` | cloned | 0.0323 | 0.888 (0.893) |
| Turbo | `*_cudnn_mps` | built-in | 0.0658 | — |
| Original | `*_high_concurrency` | cloned | 0.0195 | 0.884 (0.890) |
| Original | `*_high_concurrency` | built-in | 0.0639 | — |
| Original | `*_cudnn_mps` | cloned | 0.0201 | 0.884 (0.889) |
| Original | `*_cudnn_mps` | built-in | 0.0637 | — |

The built-in rows score the bundled `seed_tts_smoke` texts and the cloned
rows the Seed-TTS targets, so WER is not comparable between voices.

The shared sentence-segmented WebSocket frontend is exercised by
`tests/e2e/online_serving/test_chatterbox.py`; its load is not measured on
this hardware.
