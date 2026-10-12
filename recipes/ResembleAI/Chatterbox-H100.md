# Chatterbox Turbo and Original — H100

These measurements used the combined model and runtime implementation. The
shared runtime changes are now reviewed separately in
[PR #8842](https://github.com/vllm-project/vllm-omni/pull/8842); the model branch
requires that prerequisite. Historical profiling and fresh split integration
validation are identified separately below.

Measurements on October 12, 2026 used one H100 SXM 80 GB, PyTorch
2.13.0/CUDA 13.0, vLLM 0.31.0 and Transformers 5.14.1. Serving runtime revision
`26e4d3910` removes GPU-to-CPU scalar synchronization from flow-mask repair.
The subsequent `39f251fb0` changes only decoder benchmark OOM reporting.

## Summary and model contract

| Field | Value |
| --- | --- |
| Vendor / checkpoints | ResembleAI: `chatterbox-turbo`, `chatterbox` |
| Runtime / task | vLLM-Omni two-stage English text-to-speech, built-in or cloned voice |
| Input / output | Text plus optional single reference; mono 24 kHz audio |
| Hardware | One NVIDIA H100 SXM, 80 GB; no inter-GPU communication |
| Recommended profiles | Default FP32 profiles for quality checks; opt-in `*_high_concurrency_cuda.yaml` for the BF16/cuDNN measurements below |
| Maintainer | PR author; see [#8700](https://github.com/vllm-project/vllm-omni/pull/8700) |

## Hardware and software environment

| Field | Recorded value |
| --- | --- |
| Accelerator / topology | 1× H100 SXM 80 GB; TP=1, one AR and one decoder worker |
| Host memory / OS / Python / driver | Not retained in this report |
| Runtime | PyTorch 2.13.0 / CUDA 13.0, vLLM 0.31.0, Transformers 5.14.1 |
| Precision | BF16 AR; FP32 HiFT; flow precision stated per result |
| Source revision | `26e4d3910`; benchmark reporting update `39f251fb0` |
| Qualification | Historical profiling environment; fresh integration environment and results below |

## Setup and commands

Install `vllm-omni[chatterbox]` and merge prerequisite [#8842](https://github.com/vllm-project/vllm-omni/pull/8842).
For built-in-voice and reference requirements, see the
[shared speech contract](../../docs/serving/speech_api.md#chatterbox-turbo-and-original)
and [upstream implementation](https://github.com/resemble-ai/chatterbox).

```bash
vllm-omni serve ResembleAI/chatterbox-turbo --omni --port 8091 \
  --deploy-config vllm_omni/deploy/chatterbox_turbo_high_concurrency_cuda.yaml
```

Use `ResembleAI/chatterbox` and `chatterbox_high_concurrency_cuda.yaml` for Original.
The acceleration overlays require `nvidia-cuda-mps-control` on PATH. They change
flow numerics; preserve FP32 HiFT and validate quality before deployment.
See the [offline guide](../../examples/offline_inference/text_to_speech/README.md#chatterbox-turbo-and-original)
and [online guide](../../examples/online_serving/text_to_speech/README.md#chatterbox-turbo-and-original)
for complete synthesis and playback commands.

## Supported features

| Feature | Status |
| --- | --- |
| Built-in / cloned voice, async streaming | Both variants; measured scopes below |
| Native AR CUDA graphs | Enabled by default; matched graph/eager measurements below |
| BF16 flow / cuDNN attention / MPS | Opt-in, H100 measurements below; numerical tradeoff |
| Multilingual / prefix caching | Unsupported |
| Real-time at 64 clients | Workload-dependent; no general guarantee |

## Verification

Expect nonempty mono 24 kHz PCM, completed requests, and independent request
cleanup. Run the commands in [benchmark documentation](../../benchmarks/tts/README.md#chatterbox-concurrency-and-conditioning-checks).
The nightly JSON records H100 baselines from the matched workload below;
historical workloads are not interchangeable baselines.

## Nightly workload and CUDA graph comparison

Fresh integration validation used model revision `0de536675` with prerequisite
runtime revision `0a47e146a`. Subsequent recipe, baseline and CI marker edits do
not change inference. The environment was Ubuntu 24.04.3, Python 3.12.3,
driver 580.159.04, PyTorch 2.13.0+cu130, torchaudio 2.11.0+cu130,
vLLM 0.31.0 and Transformers 5.14.1. The container had a 233.8 GiB memory
limit and a 23.8-core CPU quota; math threads were bounded to one. One H100
80 GB was visible. Package imports used local container storage. Checkpoint
snapshots were Turbo `749d1c1a46eb10492095d68fbcf55691ccf137cd` and Original
`5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18`.

Both variants passed real-weight MRv2 chunked offline synthesis through the
shared TTS helper (two batched prompts each), plus MP3 and FLAC speech API
checks with Whisper transcript validation. All 219 focused model, conditioning,
adapter, configuration and load-harness tests passed.

The benchmark used the bundled `benchmarks/build_dataset/seed_tts_smoke`
English text dataset, built-in voice, seed 42 and default deployment settings:
BF16 AR, FP32 eager decoder, eight stage-0 sequence slots. Original uses two
CFG slots per guided client. Only stage 0's `enforce_eager` differed in each
graph/eager pair. Latency used 20 requests at concurrency one and two warmups;
throughput used 80 requests at concurrency eight and eight warmups; stress
used 100 requests at two requests/second and two warmups. The shared runner
originally also emitted redundant 100-request concurrency-eight sweeps; those
reports are retained but excluded from the final nightly configuration and
baselines. All eight pytest cases passed, with zero failed benchmark requests.

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

These are end-to-end measurements, not isolated graph timings or 64-client
qualification. Original's generated duration differs slightly between modes.
The nightly JSON stores measured median first-audio latency and RTF, plus audio
throughput for the throughput cases, under the H100 hardware bucket. B200
mirror collection has no measured baseline or qualification claim.

## Bottlenecks and changes

Flow accounted for approximately 90–95% of isolated FP32 decoder time. Native
BF16 autocast applies only to flow; HiFT weights, compute, and returned audio
remain FP32. Selecting native cuDNN SDPA improves the BF16 flow further without
changing its masks. Unsupported cuDNN shapes fail explicitly.

At batch 64, isolated steady chunks contain 2.4 seconds of output audio:

| Model | FP32 decoder seconds | BF16 seconds | BF16 + cuDNN seconds |
| --- | --- | --- | --- |
| Turbo | 0.840 | 0.385 | 0.325 |
| Original | 4.502 | 1.623 | 1.331 |

These replay real checkpoint conditioning and codes, but exclude generation,
conditioning, transport and competing GPU stages. They cannot establish serving
latency. Original performs ten flow steps and internal classifier-free guidance;
Turbo performs two steps without decoder CFG.

Flow-mask repair also performed a host scalar read on every attention layer.
The repair now uses an equivalent Boolean tensor expression, and pad masks use
the already-known padded length. Real-GPU traces reduced decoder scalar reads
from 31 to 2 for Turbo and 143 to 2 for Original. Synchronization wait times
overlap GPU execution; their sum is not a predicted wall-clock speedup.

## Fully warmed Turbo batch-size sweep

The HTTP streaming workload uses 64 concurrent clients, 128 full-load warmup
requests, then 128 timed requests. Each request reads the same approximately
65-word English narration with a different request number and uses one cached
reference recording. Seed is zero and the output allowance is 1000 speech tokens.
This is a fixed-workload test, not a general latency guarantee.

| Decoder slots | MPS | p95 RTF | p95 first audio (s) | p95 playback deficit (s) | p99 deficit (s) |
| --- | --- | --- | --- | --- | --- |
| 8 | No | 0.552 | 2.443 | 0.612 | 0.615 |
| 16 | No | 0.435 | 2.077 | 0.195 | 0.773 |
| 32 | No | 0.373 | 1.713 | 0.076 | 0.077 |
| 64 | No | 0.383 | 1.782 | 0.062 | 0.124 |
| 32 | Yes | 0.332 | 1.444 | 0 | 0 |
| 64 | Yes | 0.332 | 1.395 | 0 | 0 |

Every row completed all 128 timed requests with zero failures. RTF is complete
request latency divided by generated audio duration. Playback begins at the
first chunk; the continuity metric allows 100 ms of scheduling jitter. Zero
reported deficit means no underrun beyond that tolerance. First-audio latency
still matters even when RTF is below one. Two small warmups do not warm kernels
for a 64-row flow batch and produce materially different tails.

The selected CUDA overlay retains 64 decoder slots: 32 and 64 have similar RTF,
and 64 has slightly lower first-audio latency in this run. Each scheduled flow
batch is decoded together; there is no internal four-request decoder limit.
HiFT groups incompatible stream lengths/cache states to preserve correctness.

## Matched Qwen3-TTS target

Qwen3-TTS 12Hz 1.7B Base used its existing
`qwen3_tts_high_concurrency_mrv2_single_gpu.yaml` profile on the same H100, with
the same references, target text, seed, allowance, 128 full-load warmups and
128 timed HTTP requests. Both rows completed all requests without failures:

| Model, 64 clients | p95 RTF | p95 first audio (s) | p95 playback deficit (s) | p99 deficit (s) |
| --- | --- | --- | --- | --- |
| Qwen3-TTS Base | 0.185 | 0.775 | 0.279 | 0.309 |
| Chatterbox Turbo CUDA | 0.332 | 1.395 | 0 | 0 |

Qwen generates this workload faster and starts audio sooner. Its smaller early
chunks still produce playback deficits under load; Turbo's larger initial
buffer avoids measured deficits in this warm workload. This does not establish
overall Qwen parity or a universal real-time service guarantee.

## Original concurrency

With the same BF16/cuDNN/MPS settings and 15-token initial chunks, Original
completed every request but did not sustain continuous playback at 64 clients.
Each row uses 128 warmups and 128 timed requests:

| Clients | p95 RTF | p95 first audio (s) | p95 playback deficit (s) | p99 deficit (s) |
| --- | --- | --- | --- | --- |
| 8 | 0.346 | 1.376 | 0 | 0 |
| 16 | 0.385 | 1.473 | 0 | 0.008 |
| 32 | 0.589 | 2.043 | 0.327 | 0.327 |
| 64 | 1.096 | 4.021 | 1.873 | 1.874 |

The Original CUDA profile preserves the 64-request admission contract. The
table distinguishes that capacity from a suitable interactive operating point.
It does not reduce the flow step count or disable either classifier-free
guidance path to make a performance number look better.

Increasing only the initial chunk setting to 30 tokens reduces repeated early
flow work. With the ten-second reference, alignment emits approximately
1.84 seconds in the first chunk instead of 0.64 seconds. The maximum remains
60 tokens. The same fully warmed protocol measured:

| Clients | p95 RTF | p95 first audio (s) | p95 playback deficit (s) | p99 deficit (s) |
| --- | --- | --- | --- | --- |
| 32 | 0.560 | 2.130 | 0 | 0 |
| 64 | 1.016 | 3.529 | 0.531 | 0.531 |

All 128 timed requests completed in each run. The opt-in Original CUDA profile
uses this 30-token initial setting; the ordinary profiles retain 15. At 32
clients it sustained continuous playback on this workload. At 64 it still
did not. A separate 64-output transcription check measured 0.19% mean WER
with zero ASR failures for the 30-token setting.

## Conditioning and quality boundaries

On an A5000, encoding 64 real speech references in batches of 8/16/32/64 took
3.189/3.074/3.055/3.031 seconds, with peak allocated memory of
0.83/1.15/1.78/3.05 GiB. All matched serial conditioning within the existing
tensor tolerances without changing encoder buffers. Retaining batches of eight
trades approximately 5% encoding time for substantially lower memory. The
64-request admission bound and duplicate-miss coalescing remain independent of
this encoder batch size.

Whisper-large-v3 measured mean WER of 0.22% on 64 Turbo BF16/cuDNN outputs and
0.27% on 64 Original outputs from this narration. A Qwen3-TTS Base run on the
same narration measured 0.11%. All three had zero ASR failures.
These checks establish narrow intelligibility evidence, not speaker similarity
or broad quality parity. BF16 changes numerical audio values. Cold-reference
variants derived from one recording stress cache misses, not distinct speakers.

The earlier fresh-server rerun was interrupted by insufficient Runpod balance.
Fresh split integration checks now have their own results below; the historical
warm performance and playback-deficit tables remain separate measurements.

## Fresh conditioning and protocol qualification

The October 12 integration environment and source revisions recorded above also
ran the opt-in CUDA profiles, including BF16/cuDNN flow, FP32 HiFT and MPS.
Each transport starts a fresh server. Each workload runs three synchronized
waves of 64 clients with the default 1000-token allowance: cold distinct
reference variants, a warm repeated reference and built-in voice. Request seeds
are their request indices. The current load harness uses the longer narration in the reproduction section. WebSocket
sessions send two input chunks and validate all five sentence segments, their
order, PCM and reconstructed text. Input chunk boundaries do not define sentence
boundaries. The initial harness incorrectly assumed they did; that failed
report was retained and the corrected harness rerun.

All four runs passed: 576 requests/sessions per run, zero errors and zero exact
audio replays. Each WebSocket run validated 2880 completed sentence segments.
The table reports p95 seconds from session submission; later WebSocket sentences
include time spent behind earlier sentences. These checks do not measure playback
deficits.

| Variant | Transport | Cold TTFA / completion | Warm TTFA / completion | Built-in TTFA / completion | Every-wave capacity: cold / warm / built-in |
| --- | --- | --- | --- | --- | --- |
| Turbo | HTTP | 16.33 / 24.84 | 2.20 / 9.11 | 1.97 / 9.00 | No / Yes / Yes |
| Turbo | WEBSOCKET | 12.98 / 38.25 | 2.54 / 17.06 | 2.40 / 17.08 | No / No / No |
| Original | HTTP | 10.84 / 31.51 | 7.15 / 27.56 | 3.82 / 17.03 | Yes / Yes / Yes |
| Original | WEBSOCKET | 11.99 / 44.30 | 4.00 / 37.07 | 4.04 / 29.01 | No / No / Yes |

Original HTTP reached 128 stage-0 sequences (two CFG branches per request) in
every wave, including cold conditioning. Turbo HTTP reached 64 in both warm
and built-in workloads in every wave; its cold wave peaks were 64/64/62.
WebSocket capacity confirmation was false except for Original built-in voice.
Sentence segmentation and conditioning stagger admissions; sampled peaks do not
establish sustained simultaneous occupancy or a universal real-time guarantee.

Browser checks generated built-in-voice speech through the Gradio streaming
demo for Turbo and Original. Both native players reached the end without a
media error. This verifies browser delivery and playback; it is not a concurrent
underrun measurement or a subjective speaker-quality assessment.

To reproduce HTTP qualification after starting the CUDA profile:

```bash
uv run python benchmarks/tts/chatterbox_load.py \
  ResembleAI/chatterbox-turbo tests/assets/qwen3_tts/clone_2.wav turbo-http.json \
  --url http://127.0.0.1:8091/v1/audio/speech \
  --concurrency 64 --cycles 3 --max-new-tokens 1000 --save-audio
```

For WebSocket, restart the server and add `--transport websocket`, changing the
URL to `ws://127.0.0.1:8091/v1/audio/speech/stream`. For Original, use
`ResembleAI/chatterbox`, its CUDA profile and `--stage0-sequences-per-request 2`.
The capacity field requires every wave to reach the active-sequence target;
client completion alone does not establish that target.

Historical profiling matrices, traces, quality logs, 64-output quality WAV sets,
and Original concurrency reports were saved locally before the earlier shutdown. The final hot
Qwen JSON was not copied before the shutdown; its table above uses the completed
report's metrics captured from the pod before it stopped. Do not represent that
capture as a complete raw report.

## Reproduction

Start Turbo with the CUDA overlay, and start Qwen3-TTS Base separately with
`qwen3_tts_high_concurrency_mrv2_single_gpu.yaml` on the same GPU. Use identical
Seed-TTS metadata and reference audio for both model-specific `voice_clone`
tasks. Restart between models and report cold conditioning separately.

The fixed target text was:

```text
Request number {index}. The river moves quietly past the old stone bridge today.
In the morning, people walk along the bank and listen to the birds in the trees.
A small boat carries fresh bread to the market across the water. By evening,
the streets become quiet again, and warm lights shine through the windows of the houses.
```

The reference is `tests/assets/qwen3_tts/clone_2.wav`. Warm metadata reuses the
same file for every row; cold metadata uses distinct gain variants of that
recording. The benchmark uses the same GPT-2 tokenizer only to count prompt
tokens for both models, not to tokenize model input inside the server.

```bash
vllm-omni serve ResembleAI/chatterbox-turbo --omni --port 8091 \
  --deploy-config vllm_omni/deploy/chatterbox_turbo_high_concurrency_cuda.yaml

uv run benchmarks/tts/bench_tts.py \
  --model ResembleAI/chatterbox-turbo --task voice_clone --locale en \
  --concurrency 64 --num-prompts 128 --num-warmups 128 \
  --dataset-path /path/to/workload --output-len 1000 --request-seed 0 \
  --output-dir /path/to/results --port 8091 -- \
  --tokenizer /path/to/common-gpt2-tokenizer \
  --save-detailed --metric-percentiles 50,95,99
```

The metadata reference transcript is supplied for Qwen's Base task; Chatterbox
does not require it. Add `--wer-eval` for a separate quality run. Inspect the
quality log and saved WAVs as well as the report: the current wrapper can omit
WER from its summary JSON. Use the recipe's separate public-API capacity harness
to measure active scheduler occupancy; client concurrency alone does not prove
64 simultaneously running model requests.

## Historical API qualification before full-batch decoding

On H100 SXM 80 GB with vLLM 0.31.0, PyTorch 2.13.0/CUDA 13.0,
Transformers 5.14.1, and Uvicorn 0.54.0, both variants completed 576 HTTP
requests and 576 WebSocket sessions (1152 sentence segments) each at 64 clients
with the default 1000-token allowance. Each transport used a fresh server and
three waves per workload. All requests passed, with no exact audio replay;
speech shared-memory entries returned to zero before each server was stopped.
Original HTTP/WebSocket results use final runtime revision `08352fddf` and
the 0.40 memory profile; Turbo uses `223f3c5f6`. These measurements precede
removal of the four-request decoder microbatches; the current full-batch decoder
has separate [H100 profiling results](Chatterbox-H100.md). The following
historical measurements are p95 seconds,
including cold conditioner initialization:

| Variant | Transport | Cold TTFA / completion | Warm TTFA / completion | Built-in TTFA / completion |
| --- | --- | --- | --- | --- |
| Turbo | HTTP | 27.74 / 34.14 | 3.17 / 7.44 | 3.52 / 7.10 |
| Turbo | WebSocket | 25.82 / 35.77 | 3.63 / 9.79 | 3.27 / 9.29 |
| Original | HTTP | 16.91 / 37.01 | 10.53 / 31.75 | 6.94 / 22.19 |
| Original | WebSocket | 17.63 / 46.48 | 8.24 / 43.27 | 6.41 / 30.99 |

These use the harness's short default text. The per-wave scheduler-capacity
check remains false for these runs because not every wave reached the target
stage-0 sequence count. They establish completed client concurrency with the
default allowance, rather than simultaneous full-context occupancy.

A separate 267-word unsplit HTTP workload completed 192 requests per variant
(64 per workload) at the 1000-token cap. Every workload reached 64 running Turbo
sequences or 128 running Original CFG sequences, and 64 decoder requests.
Original's initial 0.35 stage-0 memory allocation admitted only 57 pairs; the
final 0.40 profile allocated 264,624 KV tokens and reached all 64 pairs.
Live speech shared memory returned to zero. Long-workload p95 seconds were:

| Variant | Cold TTFA / completion | Warm TTFA / completion | Built-in TTFA / completion |
| --- | --- | --- | --- |
| Turbo | 29.09 / 58.05 | 4.47 / 28.78 | 3.84 / 11.70 |
| Original | 24.70 / 156.00 | 15.67 / 152.20 | 15.24 / 141.22 |

This establishes active capacity for the tested workload, not successful long
narration. Original outputs reached the 40-second ceiling; Turbo samples ended
after 10.2–33.32 seconds with severe repetition or unrelated speech. Separate
Whisper-small transcription of 18 short samples per variant gave text-cosine
ranges of 0.952–1.0 for Turbo and 0.848–1.0 for Original. The long samples were
substantially worse. Text cosine measures overlap, not ordered correctness;
PCM hashes do not establish semantic voice isolation. Prefer the sentence
frontend for long text; these results do not claim Qwen3-TTS quality or latency
parity. Long unsplit narration failed semantic checks in this configuration;
Both variants also failed the same long input at concurrency one. Pinned upstream
`chatterbox-tts` 0.1.7 built-in-voice controls likewise skipped/scrambled Original
content and produced unrelated Turbo speech (40.0 and 8.96 seconds respectively).
These stochastic controls characterize the limitation; they do not establish
backbone-logit or quality parity.

Real-weight conditioning separately matched serial execution for 64 reference
clips of four lengths over three cycles, for both variants on A5000 and H100.
On H100, eight-reference batches took 2.17–2.44 seconds per 64 Turbo references
and 1.67–1.92 seconds per 64 Original references, versus 3.21–3.37 and
2.71–3.04 seconds serially. Token and mel tensors matched exactly; speaker
embeddings remained within the benchmark's `1e-4` tolerance, with unchanged
encoder buffers. This measures encoding after initialization, separately from
the cold public-API timings above.
