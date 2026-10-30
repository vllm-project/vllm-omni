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
this checkout as described in the installation guide. The adapter was also
checked on vLLM 0.32.0 / DGX Spark GB10; see the validation scope below.
The original exported runtime uses vLLM 0.29.0 and different extension paths.

```bash
export VLLM_API_KEY="set-a-local-api-key"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
python -m vllm_omni.entrypoints.cli.main serve /absolute/path/to/complete-checkpoint --omni \
  --deploy-config vllm_omni/deploy/drvexo_speech_8b.yaml \
  --served-model-name drvexo --host 127.0.0.1 --port 8000 \
  --stage-init-timeout 1800 --init-timeout 1900
```

The opt-in deployment uses a single device, eager execution, one sequence,
no prefix caching and a 2048-token prefill batch. The General head requires
these settings. Chat has a 32768-token context budget; deployment settings
are not a claim of portability to every GPU, environment or context workload.

The tested local multiprocess stage path uses the serialization opt-in above;
checkpoint weights use safetensors. Quantized Codec loads BF16 state and promotes
it to FP32 for exact calibration-reference validation before packed replacement
and the manifest-selected surrounding compute conversion.

Do not reuse version-bound numerical caches from a different runtime. BF16,
FP8 and NVFP4 names describe supported stored/quantized components, not every
operator. The audio encoder remains BF16 in quantized bundles and the General
head runs on CPU FP32. Embeddings, norms, convolution and reference components
have precision exceptions. See the checkpoint's component manifests.

## Requests

Standard `/v1/chat/completions` requests support text and audio inputs.
For text plus WAV output, use `"modalities": ["text", "audio"]` and
`"audio": {"format": "wav"}`. The response combines text and audio in one choice.
Supply complete committed history, tool associations and authoritative product facts.
The adapter does not insert identity answers or use decision scores to
intercept ordinary chat. Audio recognition of the current input returns
`input_transcription` metadata; committed text history is preserved.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $VLLM_API_KEY" \
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
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -d '{"task":"choice","instruction":"Match the color in the state.","state":{"color":"blue"},"candidates":[{"id":"a","text":"blue"},{"id":"b","text":"red"}]}'
```

Decision tasks are `choice`, `noul` and `score`, with 2–151 unique candidate
IDs. The fixed protocol budgets are 1536 prefix tokens and 384 candidate
tokens, with shared instruction/state allocation and head truncation of each
candidate. They are independent of the chat context budget. Setting
`general_decision_head_enabled: false` skips the Thinker and head. Scores
are not calibrated confidence; `executable` is always false.

Enabled decisions score candidates sequentially, with one full shared-prefix
prefill per candidate and no prefix caching. At the 151-candidate cap this can
occupy the shared Thinker for substantially longer than a small decision;
there is no production latency guarantee. Each API process permits only one
enabled decision at a time and returns 429 with `Retry-After: 1` when busy.
The scoring deadline is 120 seconds; expiry aborts the owned candidate and
returns 504 after shielded cleanup. This does not reserve a separate chat lane;
use separate deployments for latency-sensitive traffic. The native head's
validated contract requires exactly 2048 batched prefill tokens, BF16 compute,
eager execution, one sequence and no prefix caching; other budgets are rejected.

At startup the General head verifies all `bound_artifacts`, including complete
weight shards, before accepting their binding. This deliberately adds a full
read of approximately 21.67 GB (BF16), 14.09 GB (FP8) or 10.96 GB (NVFP4) for
the supplied bundles, in addition to weight loading. It is an integrity check,
not a cached shortcut or a throughput measurement.

`/v1/drvexo/chat/stream` connects to the actual HTTP listening IPv4/IPv6 socket.
It forwards end-to-end authentication and tracing headers, excluding Host,
hop-by-hop and proxy-forwarding headers, and re-enters the same middleware and
logging stack. Direct TLS (`--ssl-keyfile`) and Unix socket (`--uds`) listeners
are explicitly unsupported by this NDJSON wrapper and return 503; use a local
HTTP listener for it. Standard chat/audio, sentence speech and decisions do
not use this internal HTTP hop. Both speech routes reject whitespace-only text
before starting synthesis. Sentence streaming accepts an omitted or null `n`
as one response, and rejects multiple choices.

The raw-audio frontend follows the vLLM 0.32 API documented by this checkout's
CUDA installation guide; vLLM 0.28 compatibility is not claimed.

Speech/chat streams propagate cancellation to their uniquely owned engine
request. Disconnect-and-recovery checks are different from measuring GPU
abort reclamation latency. A model-selected candidate never establishes that
a call, transaction or device action succeeded.

## Validation scope

CPU tests cover actual tiny safetensors loading/relocation, path containment,
component completeness, input projection, quantization adapters, General
candidate validation, chunked-prefill ownership, disabled-head bypass,
cancellation and audio-history conversion. The related local run passed 225
tests and 24 subtests. On GB10 with vLLM 0.32.0, each complete precision passed
17 existing usability checks plus 3 standard generated-token chat/audio checks;
5 unauthenticated route checks per precision returned 401. BF16's 17 ran at
`85722197e39eb08ae9952dc78d811303e22a5e06`; its additional 3 and all FP8/NVFP4
checks ran at `67072af55a6af930de24d2d7c8cd06f0e1504f87`. Later Codec changes
affect only quantized reference validation; explicit BF16 load dtype equals the previous resolved default.
Servers, suites and the final managed launcher retained actual zero exits.

The review follow-up passed 713 CPU tests and 24 subtests (one skipped).
A matching runtime-code BF16 run at `bd6c164d02d161100b7dc613e02031cc59272352`
passed 22/22 serving checks: the existing 20 above plus audio-input text-stream
disconnect and subsequent-chat recovery. The NDJSON chat case explicitly used
`n: null`; all five unauthenticated route probes returned 401. The server,
suite, managed main, `systemd-run` and outer launcher retained actual zero
exits, with an empty original cgroup and no experiment GPU residue. This run
used the complete BF16 deploy YAML, synchronous eager execution and the same
local component bundles. FP8/NVFP4 were not rerun after this review follow-up;
their earlier evidence remains separate. The two-chunk runner regression also
exercises cached scheduler offsets and batch reordering in both async-flag
modes on CPU; async GPU scheduling and speculative decoding remain unvalidated.

The environment used Torch 2.13.0+cu130, Transformers 5.16.1, FlashInfer
0.7.0.post1, Humming-kernels 0.1.17 and CUTLASS DSL 4.8.0. Installed Omni dev
metadata still produces a version warning; identify code by its source commit.
The environment includes system site-packages. First-use native kernel builds
can take several minutes. These checks establish functional local usability,
not fresh accuracy, cross-runtime numerical parity or throughput improvement.
Successful local shutdown logs also retain Python resource_tracker warnings
for three shared-memory objects. Zero exits and an empty cgroup/GPU do not
independently establish complete IPC-object lifecycle correctness.

This integration does not implement SIP/PSTN/WebRTC, full-duplex telephony,
production concurrency guarantees, human listening tests or speaker-similarity
acceptance. Base model, codec and data terms remain separate from this code's
license; code integration does not authorize publishing restricted data.
