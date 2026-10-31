# Chatterbox Turbo and Original — H100

## Summary

- Vendor: Resemble AI
- Model: `ResembleAI/chatterbox-turbo` (Turbo) and `ResembleAI/chatterbox` (English Original)
- Task: English text-to-speech with the checkpoint's built-in voice or a voice cloned from one recording
- Mode: online `/v1/audio/speech` with the one profile per model, each admitting 64 requests at once (Original's as 128 guided sequences), what one H100 fits. Real time (a median real-time factor below 1 in the nightly perf matrix, the repository's measure) holds for Turbo through 64 clients with either voice and for Original up to 16; the matrix below gives median first audio, real-time factor and throughput at 1, 8, 16 and 64 clients beside Qwen3-TTS on the same protocol. Offline use is covered below
- Hardware: 1x NVIDIA H100 SXM 80 GB
- Maintainer: @will-rice

## When to use this recipe

One card serving concurrent streams: each model has one profile, which
conditions references on the GPU, runs stage 0 with CUDA graphs and the flow
decoder in BF16 with HiFT in FP32, and admits 64 requests at once, what
one H100 fits. Cloned voices cost
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
| `chatterbox_turbo.yaml` (64 requests, `conditioning_device: cuda:0`, BF16 flow, FP32 HiFT) | measured through the nightly perf matrix at this head: 1, 8, 16 and 64 clients, built-in and cloned voices, and the 2-request/s stress case |
| `chatterbox.yaml` (64 guided requests, `conditioning_device: cuda:0`, BF16 flow, FP32 HiFT) | as Turbo's row |

Real time here is the repository's measure: a median real-time factor
below 1 in the nightly perf matrix. The matrix below gives the median
first audio, real-time factor and throughput at each load; the
flow-precision rows give p95 and playback continuity at 64 clients for an
earlier revision.

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
- Host: three pods of this type, all with driver 580.126.09 and a 251 GB
  memory limit, the load driver on the same host each time. The nightly
  matrix ran on an Intel Xeon Platinum 8480+ host with a 23.8-CPU container
  quota (GPU `GPU-afa1e129`); the reference conditioning batch on a Xeon
  Platinum 8470 host with a 22.1-CPU quota (`GPU-0e708041`); the upstream
  comparison and the flow-precision rows on another 8470 host with a 22-CPU
  quota
- Qualification scope: CUDA only; the measurements below name their profile and revision

## Software environment

| Field | Value |
| --- | --- |
| OS | Ubuntu 24.04.4 |
| Python | 3.12.3 |
| Driver / runtime | 580.126.09 (CUDA 13.0); PyTorch 2.13.0+cu130, torchaudio 2.11.0+cu130 |
| vLLM | 0.31.0 |
| Transformers / Diffusers | 5.14.1 / 0.40.0 |
| vLLM-Omni | the code of `feacefc03` for the nightly matrix and the conditioning batch; `39d191c53` plus the BF16-flow default in the deploy files for the upstream comparison and the flow-precision rows |
| Checkpoints | Turbo `749d1c1a46eb10492095d68fbcf55691ccf137cd`; Original `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18` |

The upstream comparison and the flow-precision rows are from 2026-10-15,
the conditioning batch and the nightly matrix from 2026-10-16. Upstream ran in
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
first-audio latency, real-time factor and audio throughput stay near the
H100 baselines in that file, which are the nightly matrix below.

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
  bounds the flow's and HiFT's activations. The nightly matrix below ran
  with these fractions: 159,248 KV-cache tokens for Turbo and 297,184 for
  Original at startup.
- Key flags: `conditioning_device` under stage 0 `additional_config`
  places the reference encoders: on the stages' GPU (`cuda:0`) in every CUDA
  profile, and on the CPU on NPU, XPU and MUSA through each profile's
  `platforms:` block; the flow
  decoder runs under BF16 autocast with FP32 weights and FP32 HiFT, which
  is `ChatterboxConfig`'s default `flow_dtype`, and
  `hf_overrides: {flow_dtype: float32}` on stage 1 restores the FP32 flow.
- Known limitations: cloned voices add the reference encoding to median
  first audio (Turbo 0.20 s at one client and 0.42 s at 64, Original 0.19 s
  and 1.18 s, on the nightly's Seed-TTS subset, whose references repeat
  and are cached after first use); Original's median real-time factor
  passes 1 between 16 and 64 clients with either voice. English only
  (the multilingual checkpoint is not served); no VoiceDesign or free-form
  instructions; no prefix caching (prompts are placeholder ids whose
  embeddings differ per request). Text that needs more than 1000 speech
  tokens is cut at the cap, as upstream's `generate` is; both Omni and
  upstream produce repetition or unrelated speech on unsplit narration of
  several hundred words, so send long text through the sentence frontend.

## Supported features

| Feature | Status |
| --- | --- |
| Built-in voice, voice cloning, async streaming | both variants, the nightly perf matrix at 1, 8, 16 and 64 clients |
| GPU reference conditioning | `conditioning_device: cuda:0`; every admitted reference is encoded in one batch, admission equal to stage 0's request capacity; 64 references, the admission of both profiles, take 1.64–2.02 s (Turbo) or 0.51–0.68 s (Original) as one batch on H100 |
| AR CUDA graphs | on by default; graph versus eager measured below |
| BF16 flow decoder, FP32 HiFT | default in every profile; quality and speed measured below |
| Real-time with cloned voices (median RTF below 1) | Turbo up to 64 clients (0.91 at 64); Original up to 16 (0.93), 2.10 at 64 |
| Real-time with the built-in voice (median RTF below 1) | Turbo up to 64 clients (0.74 at 64); Original up to 16 (0.84), 2.17 at 64 |
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

### Nightly perf matrix

Measured with the Verification command on 2026-10-16: one pytest session
over the Chatterbox cases of `tests/dfx/perf/tests/test_tts.json`, which
mirror the Qwen3-TTS cases. `latency` is 20 requests at one client after
two warmups; `throughput` is 80, 128 and 128 requests at 8, 16 and 64
clients, each after as many warmup requests as clients; `stress` is 100
requests at two per second; the `quality` cases are off in CI, as for every
model. Built-in rows (`default_voice`) use the bundled `seed_tts_smoke`
texts (about 4 s of audio each) with seed 42. Cloned rows (`voice_clone`)
use the English rows of `linyueqian/seed-tts-eval-subset`, as Qwen3-TTS's
do: one reference recording and its target text per row, 100 rows, so a
128-request run repeats references, which the speaker cache serves after
their first use, and Turbo repeats the subset's clips shorter than 5 s; the
requests carry the 1000-token cap the model enforces. Every case completed
every request. The graph rows are the shipped profiles; the eager rows set
only `enforce_eager: true` on stage 0. The values are the medians the
nightly stores, with audio throughput in seconds of audio per second.

| Model | Voice | Phase | Clients | Median first audio (ms) | Median RTF | Audio throughput |
| --- | --- | --- | --- | --- | --- | --- |
| Turbo | built-in | latency | 1 | 208 | 0.124 | — |
| Turbo | built-in | throughput | 8 | 504 | 0.233 | 33.8 |
| Turbo | built-in | throughput | 16 | 622 | 0.335 | 47.4 |
| Turbo | built-in | throughput | 64 | 1282 | 0.737 | 83.5 |
| Turbo | built-in | stress, 2/s | — | 322 | 0.160 | 8.0 |
| Turbo | cloned | latency | 1 | 412 | 0.166 | — |
| Turbo | cloned | throughput | 8 | 815 | 0.344 | 21.6 |
| Turbo | cloned | throughput | 16 | 729 | 0.376 | 38.6 |
| Turbo | cloned | throughput | 64 | 1707 | 0.907 | 65.5 |
| Turbo, eager stage 0 | built-in | latency | 1 | 396 | 0.323 | — |
| Original | built-in | latency | 1 | 520 | 0.382 | — |
| Original | built-in | throughput | 8 | 1361 | 0.765 | 10.7 |
| Original | built-in | throughput | 16 | 1481 | 0.838 | 19.5 |
| Original | built-in | throughput | 64 | 3020 | 2.170 | 30.5 |
| Original | built-in | stress, 2/s | — | 1339 | 0.796 | 6.7 |
| Original | cloned | latency | 1 | 715 | 0.395 | — |
| Original | cloned | throughput | 8 | 1675 | 0.864 | 9.0 |
| Original | cloned | throughput | 16 | 1733 | 0.926 | 16.5 |
| Original | cloned | throughput | 64 | 4201 | 2.098 | 29.5 |
| Original, eager stage 0 | built-in | latency | 1 | 768 | 0.462 | — |

Qwen3-TTS on the same protocol, its H100 baselines from the same file
(`Qwen3-TTS-12Hz-1.7B-CustomVoice` for the built-in voice, `-Base` for
cloning, two GPUs):

| Model | Voice | Phase | Clients | Median first audio (ms) | Median RTF | Audio throughput |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-TTS CustomVoice | built-in | latency | 1 | 47 | 0.137 | — |
| Qwen3-TTS CustomVoice | built-in | throughput | 8 | 75 | 0.186 | 38.4 |
| Qwen3-TTS CustomVoice | built-in | throughput | 16 | 714 | 0.315 | 55.4 |
| Qwen3-TTS CustomVoice | built-in | throughput | 64 | 5942 | 1.166 | 68.7 |
| Qwen3-TTS CustomVoice | built-in | stress, 2/s | — | 57 | 0.157 | 11.1 |
| Qwen3-TTS Base | cloned | latency | 1 | 141 | 0.159 | — |
| Qwen3-TTS Base | cloned | throughput | 8 | 227 | 0.331 | 24.2 |
| Qwen3-TTS Base | cloned | throughput | 16 | 647 | 0.364 | 40.4 |
| Qwen3-TTS Base | cloned | throughput | 64 | 4777 | 1.228 | 43.3 |

Turbo starts audio later than Qwen3-TTS at one and eight clients and
earlier from 16 clients on, and it is real-time (median real-time factor
below 1) through 64 clients with either voice where Qwen3-TTS's passes 1.
Original is real-time up to 16 clients and reaches 2.1–2.2 at 64. The earlier grids this recipe carried, from a per-model
harness and from the shared driver at other revisions and warmups, are
not reproduced here; the nightly matrix is the one protocol the
repository runs for every TTS model.

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

The 64-request overlays at `39d191c53` at 64 clients with the shared
`bench_tts.py` driver: 192 requests after a 64-request warmup, built-in
voice on the smoke texts and cloned voices with a distinct Seed-TTS
recording per request, 1000-token allowance, streamed PCM; the FP32 rows
each ran on a fresh server, the BF16 rows on the overlay's own servers:

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

The shared sentence-segmented WebSocket frontend is exercised by
`tests/e2e/online_serving/test_chatterbox.py`; its load is not measured on
this hardware.
