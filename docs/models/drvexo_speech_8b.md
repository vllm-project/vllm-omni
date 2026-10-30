# DRVEXO Speech 8B

DRVEXO combines a dense Qwen3 language Thinker, a Qwen3-ASR audio encoder
and input projector, a text-conditioned Qwen3-TTS Talker/CodePredictor,
and a Codec. An optional FP32 General decision head scores explicit
caller-defined candidates on the resident Thinker. Intended integrations
include voice communications, outbound-call applications and voice control;
telephony gateways and actual tool/device execution belong to the application.

## Checkpoint and registration

Use a complete **local** DRVEXO unified safetensors checkpoint. The code PR does
not distribute model weights, evaluation targets or a hosted model ID. Each
BF16, FP8 or NVFP4 checkpoint is independently loadable and includes:

- `model.safetensors.index.json` and namespaced safetensors shards;
- `components.json` and stage-local `weights.json` tensor views;
- the root config/tokenizer, audio frontend, Talker and Codec configurations;
- projector, speaker and General-head components and their binding metadata.

The `drvexo_unified` loader selects component views without unpacking a second
set of weight files or reading pickle checkpoints. Config, model and pipeline
registrations use the ordinary vLLM-Omni plugin. All shard paths stay inside
the checkpoint. Missing components and changed head bindings fail loading.

The three-stage `drvexo_speech_8b` pipeline produces text at stage 0 and audio
at stage 2. Thinker output text conditions Talker; asynchronous chunks are
used on the Talker-to-Codec edge. A stored hidden-state bridge is retained for
checkpoint compatibility and is not the validated generated-text speech path.

## Development launch

Install the dependencies for the checked-out vLLM-Omni revision, then install
this checkout as described in the installation guide. Current upstream targets
vLLM 0.32.0. The original exported runtime was tested on vLLM 0.29.0; those
results do **not** validate the current upstream adapter. Check the PR's
matching-head validation report before relying on this development command:

```bash
vllm serve /absolute/path/to/complete-checkpoint --omni \
  --deploy-config vllm_omni/deploy/drvexo_speech_8b.yaml \
  --served-model-name drvexo --host 127.0.0.1 --port 8000 \
  --stage-init-timeout 1800 --init-timeout 2400
```

The opt-in deployment uses a single device, eager execution, one sequence,
no prefix caching and a 2048-token prefill batch. The General head requires
these settings. Chat has a 32768-token context budget; deployment settings
are not a claim of portability to every GPU, environment or context workload.

Do not reuse version-bound numerical caches from a different runtime. BF16,
FP8 and NVFP4 names describe supported stored/quantized components, not every
operator. The audio encoder remains BF16 in quantized bundles and the General
head runs on CPU FP32. Embeddings, norms, convolution and reference components
have precision exceptions. See the checkpoint's component manifests.

## Requests

Standard `/v1/chat/completions` requests support text and audio inputs. Supply
complete committed history, tool associations and authoritative product facts.
The adapter does not insert identity answers or use decision scores to
intercept ordinary chat. Audio recognition of the current input returns
`input_transcription` metadata; committed text history is preserved.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"drvexo","messages":[{"role":"user","content":"Hello"}],"modalities":["text"],"max_tokens":64}'
```

The following extensions are mounted only for the DRVEXO pipeline. They use
the `/v1` namespace so the existing `--api-key` authentication also covers
these routes. The original bundled runtime uses unversioned `/drvexo` paths;
use the versioned paths below with this upstream adapter:

| Route | Result |
| --- | --- |
| `POST /v1/drvexo/speech` | WAV from caller-supplied text |
| `POST /v1/drvexo/speech/stream` | NDJSON audio chunks and completion |
| `POST /v1/drvexo/chat/stream` | Text plus sentence audio in NDJSON |
| `POST /v1/drvexo/decision` | Explicit candidate scores and probabilities |

```bash
curl http://127.0.0.1:8000/v1/drvexo/decision \
  -H 'Content-Type: application/json' \
  -d '{"task":"choice","instruction":"Match the color in the state.","state":{"color":"blue"},"candidates":[{"id":"a","text":"blue"},{"id":"b","text":"red"}]}'
```

Decision tasks are `choice`, `noul` and `score`, with 2–151 unique candidate
IDs. The fixed protocol budgets are 1536 prefix tokens and 384 candidate
tokens, with shared instruction/state allocation and head truncation of each
candidate. They are independent of the chat context budget. Setting
`general_decision_head_enabled: false` skips the Thinker and head. Scores
are not calibrated confidence; `executable` is always false.

Speech/chat streams propagate cancellation to their uniquely owned engine
request. Disconnect-and-recovery checks are different from measuring GPU
abort reclamation latency. A model-selected candidate never establishes that
a call, transaction or device action succeeded.

## Validation scope

CPU tests cover actual tiny safetensors loading/relocation, path containment,
component completeness, input projection, quantization adapters, General
candidate validation, chunked-prefill ownership, disabled-head bypass,
cancellation and audio-history conversion. GPU execution and numerical output
must be validated against the exact PR commit and runtime. Earlier bundle
usability checks and benchmark scores are not matching-head evidence here.

This integration does not implement SIP/PSTN/WebRTC, full-duplex telephony,
production concurrency guarantees, human listening tests or speaker-similarity
acceptance. Base model, codec and data terms remain separate from this code's
license; code integration does not authorize publishing restricted data.
