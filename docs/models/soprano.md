# Soprano experimental native inference

This path supports English, the checkpoint's single fixed voice, TP1 and C1
eager offline inference. It uses native vLLM Qwen3 layers, weight loading and KV
cache, followed by the checkpoint's FP32 Soprano decoder. It does not invoke
Transformers generation or LMDeploy in a worker.

## Precision and deployment

Use an unmodified local Soprano-1.1-80M checkpoint with the explicit
`vllm_omni/deploy/soprano.yaml` pipeline. Do not change its HF architectures or
model type. The correctness baseline uses **FP32 AR and FP32 decoder**;
BF16 AR is not validated. Set `TRITON_F32_DEFAULT=ieee` before launching Python:
the selected native Triton attention backend otherwise uses reduced-precision
FP32 dot products. The model rejects a missing precision setting before loading
weights. This setting does not promise deterministic sampling across engines.

Install the `soprano` optional extra in the environment used for inference.
Its text dependencies are loaded when normalization is requested; model import
does not download weights, load the checkpoint or start a service.

Both stages share one GPU with `gpu_memory_utilization=0.20` each. AR KV memory
is explicitly 64 MiB; eager execution has no CUDA graph pool or prefix cache.
These settings do not establish concurrency or throughput claims.

## Complete utterances

The production offline frontend accepts the original English text, applies the
official cleaning and sentence splitting rules, validates all prompt lengths
without truncation, submits each sentence as a native request, and concatenates
audio in sentence order. Each sentence uses a fresh request/KV trajectory. The
sampling seed applies per sentence; the official batched HF implementation
consumes random numbers differently.

```python
from transformers import AutoTokenizer
from vllm_omni import Omni
from vllm_omni.model_executor.models.soprano.offline import synthesize

# checkpoint is an existing local directory; deploy is soprano.yaml above.
tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
engine = Omni(model=checkpoint, deploy_config=deploy)
try:
    result = synthesize(engine, text, tokenizer)
    # result.audio: mono FP32 CPU tensor; result.sample_rate: 32000.
    # result.request_ids: sentence order; result.truncated: any length stop.
finally:
    engine.close()
```

`Omni.generate` remains a single-synthesis-unit interface: raw text is cleaned
but multiple sentences are explicitly rejected in favor of `synthesize`.
Pretokenized units must contain 1..256 prompt tokens. The frontend validates
every sentence before admitting the first request; no token truncation or
hallucination retry is performed. The generation budget is 1..512 tokens per
sentence. Fixed-token predictor zero is the prompt's last hidden state; an EOS
step's predictor is excluded, while a length stop retains the final predictor.

The decoder needs at least two valid predictor vectors. An invalid handoff or
shorter generation completes with empty audio and `audio_synthesis_error=True`;
it must not kill the worker or be serialized as a successful empty WAV.
`synthesize` raises `SopranoSynthesisError` without returning partial utterance
audio, and the caller can reuse the same engine. Low-level `Omni.generate`
consumers must check the error flag themselves.

## Speech API: complete WAV responses

The registered `soprano_ar` adapter uses the same text frontend and native
pipeline, and the shared Speech API audio encoder and error mapping. The
checkpoint's ordinary Qwen3 architecture does not identify a speech deployment.

Launch with the original local checkpoint and explicit deployment:

```bash
TRITON_F32_DEFAULT=ieee HF_HUB_OFFLINE=1 \
  vllm serve "$SOPRANO_CHECKPOINT" --omni \
  --deploy-config vllm_omni/deploy/soprano.yaml \
  --host 127.0.0.1 --port 8000 --served-model-name soprano
```

`GET /health` and `GET /v1/models` expose readiness and the served model name.
The initial Speech API scope is one synthesis unit with 1..256 prompt tokens,
English, and a complete 32 kHz mono WAV response. Multiple units are explicitly
rejected; use the offline frontend for complete multi-sentence utterances.

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"soprano","input":"Hello, this is a speech test.","seed":44,"response_format":"wav"}' \
  --output soprano.wav
```

| Request option | Behavior |
| --- | --- |
| `voice` omitted/null or `default` | The checkpoint's single fixed voice; no borrowed speaker names |
| Any other voice, including uploaded voices | Rejected |
| `language` omitted, `en`, or `English` | English-only scope |
| `response_format` | `wav` only in this initial API |
| `sample_rate` | Native 32000; other requested rates rejected |
| `speed` | Default 1.0 only |
| `seed` | Optional nonnegative integer, passed through the common request path |
| `max_new_tokens` | 1..512, applied to the native AR stage |
| `extra_params` | Only native `temperature`, `top_p`, `top_k`, `repetition_penalty` |
| Reference audio/text/codes, embeddings, instructions, task controls | Unsupported and explicitly rejected |
| Streaming and word timestamps | Unsupported and explicitly rejected |

Voice and English tags are case-insensitive. Sampling defaults come from the
deploy YAML, and request overrides are validated with vLLM's `SamplingParams`.
A length stop is surfaced through `X-Audio-Truncated`. A zero/one-predictor
synthesis failure uses the common HTTP 500 JSON error response, without killing
the engine; it is never returned as a successful empty WAV. A normal request can
follow that failure on the same server.

Check HTTP status, media type and the WAV container before treating a downloaded
body as audio. The complete-response baseline does not establish streaming,
GPU batching, voice cloning, subjective quality or performance. Streaming has
its own deployment and checks below. This document does not add Soprano to the
supported-model table.

## Streaming PCM and SSE

The explicit `soprano_streaming.yaml` deployment enables native inter-stage
streaming. It retains eager FP32 execution, TP1 and the same two-stage memory
budget. `soprano.yaml` remains the complete-response deployment.

```bash
TRITON_F32_DEFAULT=ieee VLLM_USE_V2_MODEL_RUNNER=0 \
  vllm serve "$SOPRANO_MODEL_DIR" --omni \
  --deploy-config vllm_omni/deploy/soprano_streaming.yaml \
  --served-model-name soprano --host 127.0.0.1 --port 8000
```

Request raw mono signed 16-bit little-endian PCM at 32000 Hz:

```bash
curl --fail-with-body --no-buffer http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  --data-raw '{"model":"soprano","input":"Hello, this is a simple speech synthesis test.","voice":"default","response_format":"pcm","stream":true,"stream_format":"audio","seed":44}' \
  --output soprano-stream.pcm
```

This file is model-generated speech. Raw PCM has no WAV header; interpret it
with the response's `X-Audio-Sample-Rate` and `X-Audio-Channels` headers. Set
`stream_format` to `sse` for the existing `speech.audio.delta` / `speech.audio.done`
protocol, whose `audio` fields contain base64 audio deltas. Complete WAV responses
also work on the streaming deployment and concatenate the same emitted samples.

The connector's `soprano_chunk_size` controls hidden-vector chunks (default 4,
allowed range 1–64). Decoder state retains at most `8 + chunk_size` hidden rows
per request. The first scheduled decode needs `4 + chunk_size` valid predictors;
termination flushes exactly through sample `2048 * (T - 1)`. A short request may
finish before any regular chunk is available. Fewer than two valid predictors
produce an explicit synthesis error; no silent padding is applied.

Streaming uses the same predictor sequence as native offline inference, starting
with the prompt's last hidden state. It excludes the predictor of sampled EOS
and retains the last non-EOS predictor on a length stop. The official HF streaming
backend uses a different vector sequence; natural sampled waveforms from those
paths are not expected to match. Windowed and full decoding can also differ
numerically even for the same hidden states and must be compared separately.

Each synthesis unit owns its decoder state through finish, cancellation and
worker-slot reuse. The HTTP adapter still accepts one official synthesis unit per
request. For text split by the official cleaner/splitter, submit each unit as a
separate request and concatenate deltas in sentence order. This is not single
request multi-sentence fanout. CUDA graphs and speculative decoding are outside
this streaming baseline; overlapping HTTP requests do not establish C4 GPU
batching support. Human listening and browser playback remain separate acceptance
steps.
