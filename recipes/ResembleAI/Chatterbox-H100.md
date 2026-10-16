# Chatterbox Turbo and Original — H100

## Summary

- Vendor: Resemble AI
- Model: `ResembleAI/chatterbox-turbo` (Turbo) and `ResembleAI/chatterbox` (English Original)
- Task: English text-to-speech with the checkpoint's built-in voice or a voice cloned from one recording
- Mode: online `/v1/audio/speech` with the one profile per model, each admitting 64 requests at once (Original's as 128 guided sequences), what one H100 fits. At 64 clients neither model plays in real time (the rows below); real-time playback held up to 16 cloned-voice or 32 built-in clients for Turbo, and up to 8 built-in clients for Original, in the earlier run at `39d191c53` on an overlay with the shipped profiles' keys. Offline use is covered below
- Hardware: 1x NVIDIA H100 SXM 80 GB
- Maintainer: @will-rice

## When to use this recipe

One card serving concurrent streams: each model has one profile, which
conditions references on the GPU, runs stage 0 with CUDA graphs and the flow
decoder in BF16 with HiFT in FP32, and admits the measured real-time
envelope. Cloned voices cost
more than the built-in voice at every load, because each new reference is
encoded before its request starts; size for distinct references if the
traffic is cloning. This recipe also holds the comparison with the canonical
upstream implementation.

## Supported model contract

| Task | Entrypoint |
| --- | --- |
| Built-in voice | `voice: "default"` on `/v1/audio/speech`; `VoiceConditioning.from_builtin` offline |
| Voice cloning | one `ref_audio` recording (URL or data URL), no transcript; `VoiceConditioner.prepare` offline |
| Long text | the shared WebSocket frontend with `split_granularity: "sentence"` ([speech API](../../docs/serving/speech_api.md#streaming-text-input-websocket)) |

| Input | Limit |
| --- | --- |
| Text | English; stage 0's 2048-token context holds the prompt and the output |
| Reference recording | mono; the shared 1 s minimum. Turbo repeats a clip of 5 s or less until it is longer (upstream refuses one). T3 conditions on the first 15 s (Turbo) or 6 s (Original), S3Gen on the first 10 s |
| Original controls | `extra_params.exaggeration` and `extra_params.cfg_weight`, finite and nonnegative; the deploy file's defaults are 0.5; Turbo rejects them |
| Speech tokens | `max_new_tokens` 1 to 1000, upstream's generation cap, default 1000 (40 s at 25 tokens/s) |

| Output | Value |
| --- | --- |
| Audio | mono, 24 kHz, float32 offline; WAV, PCM, MP3 or FLAC online; streamed PCM or WAV as delta chunks |
| Streaming | first chunk 15 tokens plus reference alignment, then 30 and 60 (`codec_chunk_frames`), a `codec_left_context_frames` of 50 tokens and position-aligned flow noise keep chunks consistent |

The request contract is the one in the
[speech API](../../docs/serving/speech_api.md#chatterbox-turbo-and-original).

| Profile on this hardware | Status |
| --- | --- |
| `chatterbox_turbo.yaml` (64 requests, `conditioning_device: cuda:0`, BF16 flow, FP32 HiFT) | measured at 64 clients at this head, built-in and cloned voices, with quality; 8–32-client rows from `39d191c53` on an overlay with the same keys |
| `chatterbox.yaml` (64 guided requests, `conditioning_device: cuda:0`, BF16 flow, FP32 HiFT) | as Turbo's row |

Real-time here means p95 real-time factor below 1 and every stream
continuous (no playback underrun beyond 100 ms). Admission at 64 is a
capacity contract, not a real-time guarantee; the evidence below gives each
row.

## References

- [Upstream implementation](https://github.com/resemble-ai/chatterbox) (`chatterbox-tts` 0.1.7, `5de7a54aa4e5e2baadb0182dde554908b48b85c2`)
- [Online examples](../../examples/online_serving/text_to_speech/README.md#chatterbox-turbo-and-original)
- [TTS benchmark README](../../benchmarks/tts/README.md): the shared `bench_tts.py` driver, `chatterbox_reference.py` and `chatterbox_conditioning.py`
- [Nightly perf configuration](../../tests/dfx/perf/tests/test_tts.json)
- Seed-TTS evaluation set: [`zhaochenyang20/seed-tts-eval`](https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval) (`8f5e1aa2a35d42f42e940074c1983358b9491f89`)

## Hardware

- Accelerator model and per-device memory: NVIDIA H100 SXM, 80 GB
- Number of devices: 1; the API process and both stages share it (`conditioning_device: cuda:0`)
- Device interconnect: none used
- Host: Intel Xeon Platinum 8470 with a 22-CPU container quota and a 251 GB
  memory limit; the load driver ran on the same host. The 64-client rows
  come from a second pod of the same type (GPU `GPU-0e708041`, same driver,
  CPU model, quota and memory limit); every other measurement from the first
- Qualification scope: CUDA only; the measurements below name their profile and revision

## Software environment

| Field | Value |
| --- | --- |
| OS | Ubuntu 24.04.4 |
| Python | 3.12.3 |
| Driver / runtime | 580.126.09 (CUDA 13.0); PyTorch 2.13.0+cu130, torchaudio 2.11.0+cu130 |
| vLLM | 0.31.0 |
| Transformers / Diffusers | 5.14.1 / 0.40.0 |
| vLLM-Omni | `feacefc03` for the 64-client rows and the conditioning batch; `39d191c53` plus the BF16-flow default in the deploy files for the upstream comparison, the nightly rows, the flow-precision rows and the 8–32-client rows |
| Checkpoints | Turbo `749d1c1a46eb10492095d68fbcf55691ccf137cd`; Original `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18` |

Every measurement below comes from the first host on 2026-10-15, except
the 64-client rows, from the second host on 2026-10-16. Upstream ran in
its own environment: Python 3.11.15, `chatterbox-tts` 0.1.7, PyTorch and
torchaudio 2.6.0+cu124, NumPy 1.26.4, Transformers 5.2.0, Diffusers 0.29.0,
`resemble-perth` 1.0.1. Both sides of the upstream comparison used four CPU
threads.

## Command

```bash
uv pip install -e '.[chatterbox]'
vllm serve ResembleAI/chatterbox-turbo --omni --port 8091 \
  --deploy-config vllm_omni/deploy/chatterbox_turbo.yaml
# Original: vllm serve ResembleAI/chatterbox --omni --port 8091 \
#   --deploy-config vllm_omni/deploy/chatterbox.yaml
```

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

Offline, with the same tokenizer, prompt and conditioning code serving uses:

```bash
uv run python - <<'PYTHON'
import soundfile as sf
import torch
from vllm_omni import Omni
from vllm_omni.model_executor.models.chatterbox.conditioning import (
    VoiceConditioning, build_prompt, load_text_tokenizer, tokenize_text,
)
from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig

model, variant, controls = "ResembleAI/chatterbox-turbo", "turbo", {}
# For Original, which takes its two controls (upstream's defaults shown):
# model, variant = "ResembleAI/chatterbox", "original"
# controls = {"exaggeration": 0.5, "cfg_weight": 0.5}
config = ChatterboxConfig(variant)
tokenizer = load_text_tokenizer(model, variant)
voice = VoiceConditioning.from_builtin(model)
ids = tokenize_text("Hello, this is Chatterbox speaking.", tokenizer, variant)
prompt = build_prompt(ids, voice, config, **controls)
engine = Omni(model=model)
try:
    output = list(engine.generate([prompt]))[0].outputs[0].multimodal_output
    audio = output["audio"]
    if isinstance(audio, list):
        audio = torch.cat([chunk.flatten() for chunk in audio])
    sf.write("output.wav", audio.flatten().numpy(force=True), config.sample_rate)
finally:
    engine.close()
PYTHON
```

To clone offline, read a recording with `soundfile.read` and replace `voice`
with `VoiceConditioner(model, config, torch.device("cpu")).prepare(wav, sample_rate)`.

## Verification

```bash
pytest tests/dfx/perf/scripts/run_benchmark.py --test-config-file tests/dfx/perf/tests/test_tts.json \
  -m "full_model and H100 and cards_1" -k chatterbox --run-level full_model
```

Expected: every case completes with zero failed requests, and the median
first-audio latency and real-time factor stay near the H100 baselines in
that file (the nightly rows below). With the 64-request Turbo overlay
measured below, the shared driver's report showed a 100.00 % streaming continuity
rate at 16 cloned-voice clients and at 32 built-in clients.

## Notes

- Memory usage: the fractions in the deploy files are derived for a 24 GB
  card (23.6 GiB) from this run's H100 stage-0 startup lines; no 24 GB card
  was run, and the API process's GPU conditioning (the reference
  conditioning batch below) is not in the fractions. Turbo: 1.12 GiB of
  weights with vLLM's own, 0.08 GiB of activations, 0.06 GiB of graphs and
  96 KiB of KV cache per token; 32 slots at 2048 tokens need 6.0 GiB, about
  7.3 GiB in all, under the 35 % (8.3 GiB). Original: 1.32 GiB, 0.16 GiB,
  0.08 GiB and 120 KiB per token; 16 sequences need 3.75 GiB, about 5.3 GiB
  in all, under the 30 % (7.1 GiB). Stage 1 allocates no KV cache; its 15 %
  bounds the flow's and HiFT's activations. The 64-client rows below ran
  with these fractions (159,248 KV-cache tokens for Turbo, 297,184 for
  Original at startup); the 8–32-client rows ran on the overlays' 0.35 and
  0.40.
- Key flags: `conditioning_device` under stage 0 `additional_config`
  places the reference encoders: on the stages' GPU (`cuda:0`) in every CUDA
  profile, and on the CPU on NPU, XPU and MUSA through each profile's
  `platforms:` block; the flow
  decoder runs under BF16 autocast with FP32 weights and FP32 HiFT, which
  is `ChatterboxConfig`'s default `flow_dtype`, and
  `hf_overrides: {flow_dtype: float32}` on stage 1 restores the FP32 flow.
- Known limitations: cloned voices add the reference encoding to first
  audio (5.0 s at the p50 for Turbo and 4.9 s for Original at 64 clients at
  this head; 0.9 s at 8 Turbo clients at `39d191c53`); neither model is
  real-time at 64 clients, and Original is not real-time for cloned voices
  at 8. English only
  (the multilingual checkpoint is not served); no VoiceDesign or free-form
  instructions; no prefix caching (prompts are placeholder ids whose
  embeddings differ per request). Text that needs more than 1000 speech
  tokens is cut at the cap, as upstream's `generate` is; both Omni and
  upstream produce repetition or unrelated speech on unsplit narration of
  several hundred words, so send long text through the sentence frontend.

## Supported features

| Feature | Status |
| --- | --- |
| Built-in voice, voice cloning, async streaming | both variants, measured at 64 HTTP clients with distinct references at this head and at 8–32 at `39d191c53` |
| GPU reference conditioning | `conditioning_device: cuda:0`; every admitted reference is encoded in one batch, admission equal to stage 0's request capacity; 64 references, the admission of both profiles, take 1.64–2.02 s (Turbo) or 0.51–0.68 s (Original) as one batch on H100 |
| AR CUDA graphs | on by default; graph versus eager measured below |
| BF16 flow decoder, FP32 HiFT | default in every profile; quality and speed measured below |
| Real-time with cloned voices | not at 64 clients; Turbo up to 16 clients at `39d191c53`; Original at no measured load |
| Real-time with the built-in voice | not at 64 clients; Turbo up to 32 clients and Original up to 8 at `39d191c53` |
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

`benchmarks/tts/chatterbox_conditioning.py` on `cuda:0` at this head: 64
references tiled from one 6.5 s Seed-TTS evaluation prompt
(`common_voice_en_103675.wav`) to 6.1, 8.3, 10.1 and 16 s with distinct
amplitude envelopes, encoded one at a time and then as one batch of 64 (the
admission of both profiles), three cycles. The same run at the commit before the batched device extraction
(same host, same clip) is the "before" column.

| Model | Serial before (s) | Serial (s) | One batch before (s) | One batch (s) | Peak allocated / reserved (MiB) | Serial/batch parity |
| --- | --- | --- | --- | --- | --- | --- |
| Turbo | 9.12–9.91 | 7.90–8.13 | 3.11–3.28 | 1.64–2.02 | 3237 / 3808 | conditioning tokens identical; one reference (the 8.3 s cut) gets one different decoder prompt token in the batch, before and after alike; prompt mel within 9.5e-7, speaker embedding within 1.4e-7, x-vector within 6.0e-6 |
| Original | 8.61–9.51 | 6.64–8.22 | 2.00–2.19 | 0.51–0.68 | 2193 / 2804 | tokens identical; prompt mel within 9.5e-7, speaker embedding within 1.3e-7, x-vector within 5.5e-6 |

In one batch the voice encoder ran once over all 752 partial windows, the
speech tokenizer in two calls of 64, and CAMPPlus in three calls grouped by
equal waveform length. The benchmark requires tokens to match exactly and
features within 1e-4, so the Turbo run reports the one-token difference as
a failure; the batched tokenizer's attention is not bit-identical to the
single clip's, and which clip flips a token depends on the recording.
Peak memory is torch's allocated and reserved device memory in the
benchmark process.

### Nightly latency and throughput baselines, the 8-slot profiles at `39d191c53`

Measured with the Verification command: one pytest session, the eight
Chatterbox cases of `tests/dfx/perf/tests/test_tts.json`, all passed with
zero failed requests, on the profiles of `39d191c53` (stage 0 admitting 8,
conditioning on the CPU), not yet rerun on the shipped 64-request profiles.
Those profiles ran stage 0 in BF16 with async chunk streaming and stage 1 with the BF16 flow and FP32 HiFT; requests go
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
other key unchanged. The 8-slot profiles at `39d191c53`, stage 0 CUDA graphs, the nightly
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

The 64-request overlays at 64 clients, 192 requests after a 64-request
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

### Concurrent HTTP clients

The shipped profiles at 64 clients, their admission, measured at this head
(`feacefc03`) on the second host. The 8–32-client rows are the earlier run
at `39d191c53` on an overlay whose keys equal the shipped profiles (stage 0
at 64 requests, Original's 128 sequences; stage 1 at 64 slots) except the
stage 0 memory fractions (0.35 and 0.40 there, 0.20 and 0.45 shipped),
which only change KV-cache headroom; that run's 64-client rows are kept
beneath for comparison.

The shared `bench_tts.py` driver: 192 requests at 64 clients (96 at 8 and
16, 128 at 32), each run after a warmup of one request per client at full
concurrency; a 1000-token allowance, streamed PCM, production sampling, one
server per model with its built-in rows before its cloned rows. Built-in
rows use the bundled `seed_tts_smoke` texts (about 4 s of audio each).
Cloned rows use the English Seed-TTS evaluation set, filtered to its 215
distinct prompt recordings longer than 5 s (upstream Turbo's bound, so no
reference is repeated), with each recording's own target text; every cloned
request carries a different recording, and the only repeated reference in
a run is the first one, which the warmup also uses. Times in seconds; RTF
is end-to-end latency over audio duration, so it includes the wait for
first audio; underrun is the extra buffering a stream needs when playback
starts at its first byte; continuity is the share of requests whose
underrun stays within 100 ms. No request failed in any run.

At this head, 64 clients:

| Model | Voice | p50 first audio | p95 first audio | p95 completion | p95 RTF | p50 underrun | p95 underrun | Continuity | Mean audio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Turbo | cloned | 6.418 | 6.927 | 9.201 | 3.207 | 0.000 | 0.137 | 91.67 % | 3.90 s |
| Turbo | built-in | 1.403 | 1.486 | 3.846 | 1.128 | 0.278 | 0.588 | 4.69 % | 4.01 s |
| Original | cloned | 7.868 | 9.371 | 15.096 | 5.290 | 2.930 | 3.619 | 4.17 % | 3.92 s |
| Original | built-in | 2.939 | 3.600 | 8.291 | 2.664 | 2.614 | 3.144 | 0.00 % | 3.62 s |

At `39d191c53` on the overlay, 8 to 64 clients (mean audio 3.86–3.92 s
cloned, 3.98–4.04 s Turbo built-in, 3.59–3.65 s Original built-in):

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

Not shipped: cuDNN flow attention with CUDA MPS over the same overlay at
`39d191c53` was mixed rather than faster (Turbo real-time to 8 cloned
clients instead of 16, p95 RTF 0.551 at 8; Original built-in to 16 clients
instead of 8, p95 RTF 0.829 at 16; at 64, Turbo built-in 89.58 % continuous
instead of 77.60 %).

Reading the tables:

- Conditioning cost: at the same client count, cloned first audio exceeds
  built-in first audio by the reference encoding. At 64 clients at this
  head the p50 gap is 5.0 s for Turbo and 4.9 s for Original; at
  `39d191c53` Turbo's was 0.85 s at 8 clients, 1.46 s at 16, 3.32 s at 32
  and 5.65 s at 64, and Original's 0.02, 0.93, 1.88 and 5.96 s.
- At 64 clients neither model is real-time (p95 RTF below 1 with every
  stream continuous) with either voice: Turbo's built-in p95 RTF is 1.13
  with 4.7 % of streams continuous and its cloned p95 RTF 3.21 with 91.7 %
  continuous; Original's are 2.66 with none and 5.29 with 4.2 %.
- Against the `39d191c53` run at 64, Turbo's cloned p50 first audio fell
  from 7.10 s to 6.42 s and continuity rose from 42.7 % to 91.7 %, which
  is the batched device conditioning; Turbo's built-in continuity fell from
  77.6 % to 4.7 % with p95 RTF 0.956 to 1.128, on a different host of the
  same type, and the two runs are not separated into code and host.
- At `39d191c53` Turbo was real-time up to 16 cloned clients and up to 32
  built-in clients, and Original up to 8 built-in clients; with cloned
  voices Original was real-time at no measured load (p95 RTF 1.22 at 8
  because of the encoding wait).

Quality at 64 clients at this head, the shared Seed-TTS evaluator over all
192 outputs of each run (no ASR or SIM failure; median WER 0 in every run):

| Model | Voice | Mean WER | Mean SIM (median) |
| --- | --- | --- | --- |
| Turbo | cloned | 0.0376 | 0.887 (0.893) |
| Turbo | built-in | 0.0643 | — |
| Original | cloned | 0.0261 | 0.883 (0.887) |
| Original | built-in | 0.0674 | — |

The built-in rows score the bundled `seed_tts_smoke` texts and the cloned
rows the Seed-TTS targets, so WER is not comparable between voices.

The shared sentence-segmented WebSocket frontend is exercised by
`tests/e2e/online_serving/test_chatterbox.py`; its load is not measured on
this hardware.
