# Chatterbox Turbo and Original

This model integration depends on the shared TTS runtime changes in
[PR #8842](https://github.com/vllm-project/vllm-omni/pull/8842). Merge that
prerequisite before using this branch; the model PR intentionally excludes
shared serving and runner fixes.

[Chatterbox Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) generates
mono 24 kHz English speech. The integration uses vLLM's GPT-2 backbone for
speech-token generation and S3Gen meanflow with HiFT for waveform decoding.
It supports the checkpoint's built-in voice and voice cloning from one recording.

## Installation and serving

Follow the source installation guide, then install the reference-conditioning extra:

```bash
uv pip install -e '.[chatterbox]'
vllm-omni serve ResembleAI/chatterbox-turbo --omni --host 127.0.0.1 --port 8091
```

The default deployment uses one CUDA GPU with 24 GB, Model Runner V2,
BF16 autoregressive generation with CUDA graphs, and eager FP32 waveform decoding.
Both stages also support Model Runner V1. Set `model_runner: v1` in a copy of
`vllm_omni/deploy/chatterbox_turbo.yaml` and pass it with `--deploy-config`.
Only the CUDA deployment has been validated.

Stage 1 can opt into BF16 flow autocast with
`hf_overrides: {flow_dtype: bfloat16}` while retaining FP32 HiFT and FP32
output audio. The default remains `float32`. This is a numerical tradeoff:
validate speech and speaker quality for the intended workload before enabling
it. Decoder-only measurements are not end-to-end performance guarantees.
On CUDA, add `flow_attention_backend: cudnn` to the same `hf_overrides` to
select native cuDNN SDPA with the existing flow masks. This requires BF16 flow;
unsupported devices or attention shapes fail. The default backend is `auto`.

For the measured H100 throughput settings, use the opt-in
`vllm_omni/deploy/chatterbox_turbo_high_concurrency_cuda.yaml` profile. It inherits
the 64-request profile and enables BF16 flow, native cuDNN attention, and CUDA
MPS across the API conditioner and model stages. It requires
`nvidia-cuda-mps-control` on PATH. HiFT remains FP32. The ordinary profiles retain
FP32 flow. See the [profiling report](Chatterbox-Profiling.md) for batch-size
measurements, warmup requirements, numerical checks, and latency limitations.

```bash
curl --fail http://127.0.0.1:8091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ResembleAI/chatterbox-turbo",
    "input": "Hello, welcome to the speech synthesis demonstration.",
    "voice": "default",
    "response_format": "wav",
    "max_new_tokens": 250,
    "seed": 42
  }' --output chatterbox.wav
```

## Reference voices

Supply `ref_audio` as an audio URL or base64 data URL, either a string or a
one-item list. The shared speech API's audio-access restrictions apply.
A transcript is unnecessary. Reference recordings must contain more than five
seconds of audio. Conditioning uses up to fifteen seconds for T3 and ten seconds
for S3Gen, with loudness normalization to -27 LUFS.

```json
{
  "model": "ResembleAI/chatterbox-turbo",
  "input": "Please read this new sentence in the reference voice.",
  "ref_audio": "data:audio/wav;base64,<base64-encoded-reference>",
  "response_format": "wav"
}
```

Each request carries its own voice tensors. Requests without `ref_audio` use
`conds.pt` from the selected model directory or Hub snapshot. Reference encoding
runs outside the API event loop; the encoder is initialized on its first
reference request. A bounded queue admits up to 64 conditioning requests, and
one encoder owner processes batches of up to eight. Identical cache misses share
one encoding operation, and a 128-entry cache stores completed conditioning.
The initialization lock does not serialize request generation.

## Chatterbox Original

Serve `ResembleAI/chatterbox` with the same command to select the Original
deployment (`vllm_omni/deploy/chatterbox.yaml`). It uses a Llama backbone,
classifier-free guidance, and ten-step S3Gen instead of Turbo's GPT-2 and
meanflow decoder. Both deployments expose the same speech and streaming API.
Original uses six seconds of reference audio for T3, ten seconds for S3Gen,
and does not apply Turbo's loudness normalization or five-second minimum.
Its opt-in acceleration profile is
`vllm_omni/deploy/chatterbox_high_concurrency_cuda.yaml`. It enables the same
BF16 flow, cuDNN attention and MPS settings as Turbo. Admitting 64 requests
does not guarantee real-time Original audio; consult the profiling report.
This overlay starts with 30-token chunks to reduce early flow work; the ordinary
profiles keep the 15-token initial setting.

Original accepts `extra_params` with `exaggeration` and `cfg_weight`, both
defaulting to 0.5. Values must be finite and nonnegative. Setting `cfg_weight`
to zero disables the unconditional companion. Otherwise each request occupies
two autoregressive sequence slots; the atomic scheduler keeps both branches
together, and only the conditional branch produces audio. Keep prefix caching,
chunked prefill, and asynchronous scheduling disabled for this deployment.

```json
{
  "model": "ResembleAI/chatterbox",
  "input": "Hello, welcome to the speech synthesis demonstration.",
  "voice": "default",
  "extra_params": {"exaggeration": 0.7, "cfg_weight": 0.5},
  "response_format": "wav"
}
```

Turbo supports English and checkpoint vocal-event tags such as `[laugh]`.
It does not support VoiceDesign, free-form `instructions`, speaker embeddings
from other models, or the Original model's exaggeration and CFG controls.

## Streaming and limits

For raw streamed audio, request `"stream": true`, `"stream_format": "audio"`,
and `"response_format": "pcm"` or `"wav"`. PCM is signed 16-bit little-endian
audio at 24 kHz. Complete responses consolidate the same internal chunks.
Set `--no-async-chunk` to decode each finished utterance once instead.

The deployment starts with 15-token chunks, ramps to 30 and 60, and waits for
the flow decoder's three-token lookahead. Reference-length alignment can
increase the first chunk. The decoder keeps a fixed 50-token left context
and stable noise positions for each request.

`max_new_tokens` caps speech tokens at 25 tokens per second, including EOS.
The public API accepts at most 4096 new speech tokens for either variant.
Original enforces the same 4096-token ceiling for offline sampling, including
unguided and stage-0-only requests; an explicit unlimited output budget is unsupported.
Original also accepts at most 2048 text tokens because its text positions are learned.
The default is 1000, subject to the remaining 2048-token stage-0 context.
The prompt occupies one speaker slot, up to 375 reference tokens, the tokenized
text, and one start token. Prefix caching must remain disabled because these
positions use placeholder IDs with request-specific embeddings.

When tuning the deployment, keep stage 1's `max_model_len` at least as large
as stage 0's and `max_num_batched_tokens` at least
`max_num_seqs * max_model_len`. These bounds cover the full utterance even
when a request raises the default output cap. Stage 1 requires FP32 for the
vocoder. Memory and latency depend on concurrency, text length, and reference
length; tune on the intended hardware.

## Conditioning load validation at 64 clients

Use the public streaming API harness for both deployments on the same hardware.
Unlike a configuration limit, this exercises reference loading, reference conditioning,
autoregressive generation, and waveform decoding together. The opt-in
`chatterbox_turbo_high_concurrency.yaml` and `chatterbox_high_concurrency.yaml`
profiles admit 64 requests (128 Original CFG branches), batch conditioning in
groups of eight and decode the scheduled waveform batch together. They explicitly
set stage 0 `hf_overrides.conditioning_device: cuda:0`: the API process and both
stages must share visible GPU zero on one host. The standard profiles retain CPU
conditioning. Include encoder weights and activations in the GPU memory budget.

Conditioning has one encoder owner, a 64-request admission bound, batches of up
to eight references, and a 128-entry content-keyed cache. Concurrent identical
references share encoding work; each caller receives separate conditioning
tensors. Cancellation retains admission until its encoder work finishes.

On an RTX A5000 (24 GB), revision `223f3c5f6` completed 576 Original WebSocket
sessions and 1152 sentence segments at 64 clients, with a 128-token cap, across
three waves each of cold distinct-reference, warm repeated-reference, and
built-in-voice workloads. All streams passed; speech shared-memory entries
returned to zero while the server remained running. Cold/warm/built-in session
p95 completion times were 111.64/106.40/66.27 seconds. These short runs do not
establish the default 1000-token capacity or latency parity with Qwen3-TTS.
The strict per-wave scheduler-capacity check was false: not every sampled wave
reached 128 simultaneous Original CFG branches.

On H100 SXM 80 GB with vLLM 0.31.0, PyTorch 2.13.0/CUDA 13.0,
Transformers 5.14.1, and Uvicorn 0.54.0, both variants completed 576 HTTP
requests and 576 WebSocket sessions (1152 sentence segments) each at 64 clients
with the default 1000-token allowance. Each transport used a fresh server and
three waves per workload. All requests passed, with no exact audio replay;
speech shared-memory entries returned to zero before each server was stopped.
Original HTTP/WebSocket results use final runtime revision `08352fddf` and
the 0.40 memory profile; Turbo uses `223f3c5f6`. These measurements precede
removal of the four-request decoder microbatches; the current full-batch decoder
has separate [H100 profiling results](Chatterbox-Profiling.md). The following
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

Start a fresh server before each model run so the first cold wave includes conditioner initialization.
Supply an actual speech recording of more than five seconds:

```bash
uv run benchmarks/tts/chatterbox_load.py \
  ResembleAI/chatterbox-turbo reference.wav turbo-c64-short.json \
  --concurrency 64 --cycles 3 --max-new-tokens 128
```

The harness sends three synchronized waves per workload: distinct cold references,
a repeated reference with a separately reported priming request, then the built-in voice.
A single recording produces deterministic, distinct amplitude-envelope variants
of real speech; these stress cache misses but are not 64 distinct speakers.
Alternatively, supply a directory containing at least `concurrency * cycles`
distinct WAV recordings. The report records the mode and reference hashes.
Cold cache claims require a fresh server and previously unseen recordings; cache
hits are inferred from repeated inputs, not measured by the client.

Repeat with `ResembleAI/chatterbox` and `--stage0-sequences-per-request 2`
after restarting the server for Original.
Run a separate longer workload with `--max-new-tokens 1000` and a fresh server;
the 128-token run is a short stress test and does not establish long-utterance
capacity. Keep hardware, reference recordings, concurrency, and caps identical
when comparing models or Qwen3-TTS using its model-specific benchmark.

For the measured long capacity probe, use a fresh server and this fixed input
(the paragraph is deliberately repeated). This probes occupancy, not narration quality:

```bash
uv run python - ResembleAI/chatterbox reference.wav original-c64-long.json \
  --concurrency 64 --stage0-sequences-per-request 2 --cycles 1 \
  --max-new-tokens 1000 --save-audio <<'PYTHON'
import benchmarks.tts.chatterbox_load as benchmark

paragraph = (
    "The morning train arrived beside the river just as the town began to "
    "wake. People carried baskets to the market, and a baker opened the "
    "windows of a small shop near the station. Across the square, children "
    "gathered around an old map and tried to trace the road that would take "
    "them into the hills. Their teacher explained that the stone bridge had "
    "connected the two valleys for generations. It had survived winter "
    "storms, spring floods, and the slow passage of countless travelers. "
    "Today they would cross that bridge together and follow the path "
    "through the orchard. They packed water, bread, and notebooks, then "
    "checked the weather one last time. A gentle breeze moved through the "
    "trees as they set off, listening to birds and watching the sunlight "
    "spread across the fields. "
)
benchmark.TEXT = "Request number {index}. " + paragraph + paragraph
benchmark.main()
PYTHON
```

For Turbo, change the model ID and use `--stage0-sequences-per-request 1`.

JSON includes every request, PCM hashes, reference hashes, errors, first audio
latency (TTFA), end-to-end p50/p95/p99/max, request throughput, audio throughput,
and peak outstanding client requests. Success requires all streams to finish
with nonzero mono 24 kHz PCM and no exact audio replay across distinct text/seed
requests. PCM integers are finite by construction; this cannot detect nonfinite
internal tensors before PCM conversion. Hashes detect exact replay, not semantic
speaker leakage. The harness polls `/metrics` every 250 ms and separately reports
per-stage peak running/waiting sequences and `capacity_confirmed`. Missing
metrics cannot confirm capacity. Original requires 128 running CFG branches for
64 requests. Cold conditioning may feed the GPU gradually; inspect each workload
separately. These gauges do not expose encoder cache counters, quality, or VRAM:
pair the report with server logs and GPU memory telemetry. A peak of 64 clients
alone does not establish active capacity or acceptable latency. HTTP 200 alone
never passes.

### Sentence-segmented WebSocket sessions

Validate the sentence frontend separately from HTTP streaming:

```bash
uv run benchmarks/tts/chatterbox_load.py \
  ResembleAI/chatterbox-turbo reference.wav turbo-c64-sentences.json \
  --transport websocket --url http://127.0.0.1:8091/v1/audio/speech/stream \
  --concurrency 64 --cycles 3 --max-new-tokens 128 --save-audio
```

This opens 64 overlapping sessions per wave with `split_granularity="sentence"`
and `stream_audio=true`. Each sends two text chunks, then `input.done`, and checks
ordered `audio.start`, binary PCM, `audio.done`, and `session.done` messages.
Each session has two independently validated sentence segments. The report adds
per-segment results and latency tails measured from session submission, so later
segments include time behind earlier sentences. Reference workloads, scheduler
metrics, capacity thresholds, and saved audio use the HTTP harness conventions.
For Original with CFG, include `--stage0-sequences-per-request 2`.

The WebSocket receiver continues draining input during generation so protocol
keepalive replies remain readable. Each connection permits up to 8 MiB of pending
messages and 4096 queued messages, in addition to the existing per-message limits.
Exceeding either backlog bound aborts active generation and closes with code 1008.
