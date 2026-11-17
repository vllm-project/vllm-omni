# DRVEXO 4B Unified

Experimental support for a **complete local merged BF16** `drvexo-4b-unified`
checkpoint. The 4B and [8B](drvexo_speech_8b.md) models share transport, APIs,
Talker and Codec code while preserving their different language architectures.
The 4B Thinker uses native Qwen3.5 hybrid attention and vision, an audio encoder
with its trained temporal connector, and a CPU FP32 candidate decision head.

## Package and launch

The complete 4B checkpoint is available on
[Hugging Face](https://huggingface.co/DRVEXO-LLM/drvexo-4b-unified) and
[ModelScope](https://modelscope.cn/models/DRVEXOLLM/drvexo-4b-unified).
Download the complete package before serving. The loader is local-only and does
not fetch separate component weights. It requires the root safetensors index,
all eight components, tokenizer, `assets/qwen35` vision processor, audio frontend,
Talker/Codec configs and component views. The v12 adapter must already be merged;
this loader rejects unmerged or quantized 4B packages. Retained source FP32
tensors are not rewritten merely because the package is named BF16.

Install vLLM 0.33 and this matching vLLM-Omni checkout, then run:

```bash
vllm serve /path/to/drvexo-4b-unified --omni \
  --deploy-config vllm_omni/deploy/drvexo_unified.yaml \
  --served-model-name drvexo-4b-unified --host 127.0.0.1 --port 8091 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml
```

The default profile is eager BF16, TP1, one sequence, an 8192-token context,
2048-token prefill chunks and disabled prefix caching. These are integration
defaults, not a throughput recommendation. The thinker-only deploy config
`drvexo_unified_thinker_only.yaml` isolates text, image, audio input and decisions;
speech output requires the complete three-stage pipeline.

The complete package always binds its decision head. Startup therefore rejects
`max_num_seqs != 1` or enabled prefix caching even for plain-chat deployments
that never call a decision endpoint. The eager/BF16, TP1 and 2048-token chunk
requirements are likewise enforced; incompatible settings are named in the
startup error. These constraints also apply to the thinker-only profile.

## Requests

Use `/v1/chat/completions` with `modalities: ["text"]` for text, one image or one
audio input. Image payloads use standard `image_url` content; audio uses
`input_audio`. Video is not supported. The default image area cap is 200704
pixels; supported image processor options can be supplied using
`mm_processor_kwargs`. Audio is resampled to 16 kHz and limited to 390 encoder
frames. Committed conversation history belongs in `messages`.

OpenAI `response_format` JSON Schema is propagated to Thinker only. Tool calls
require the configured parser shown above; named/required calls are returned as
parsed function calls with JSON arguments, not raw XML strings. These constraints
validate output format and do not establish argument semantics or tool execution.

The family shares `/v1/drvexo/decision`, `/v1/drvexo/speech`,
`/v1/drvexo/speech/stream` and `/v1/drvexo/chat/stream`; see the
[8B API guide](drvexo_speech_8b.md) for schemas. The 4B decision protocol uses
4096 prefix and 384 candidate tokens. It returns non-executable candidate scores.
The CPU head is structurally loaded, but its quality paired with the newly merged
backbone is not thereby accepted.

Explicit reference-voice requests use the existing `/v1/audio/speech` Base task
with `ref_audio` and `x_vector_only_mode: true`. The DRVEXO adapter marks this as
caller text, while chat-to-speech checks the exact generated-token provenance.
Both speech APIs intentionally require an exact caller-text round trip through
`talker/`: encode with `add_special_tokens=False`, reject any special/control
token IDs, then decode with `skip_special_tokens=False` and require equality to
the original input. The server does not trim, normalize or silently replace the
caller text. For example, if the deployed tokenizer collapses `"hello  world"`
to `"hello world"`, the original request returns HTTP 400 with
`Fixed speech tokenization changed the reply`; submit the intended canonical
text explicitly. The same rule applies to Unicode normalization and the
tokenizer's configured decoding cleanup. It preserves exact text for the frozen
Talker lexical interface; it does not promise that every ordinary text string is
accepted. Control tokens remain rejected even when they otherwise round-trip.

Standard speech runs this lexical check in a worker thread before resolving
reference audio or dispatching an engine request. It reuses the startup
Talker tokenizer when available; otherwise concurrent cold requests share one
worker-thread load, with a failed load remaining retryable. Prompt lengths and
precomputed text IDs use that same tokenizer, including when
`non_streaming_mode=true`; the root tokenizer belongs to Thinker.
Default-speaker sentence synthesis uses `/v1/drvexo/speech`. Native sentence
streaming returns incremental Code2Wav chunks; the chat stream splits generated
text at punctuation before synthesizing segments. Neither endpoint introduces
fixed identity replies or executes device commands.

## Validation limits

A local merged eight-component package was loaded through all three stages on
DGX Spark GB10 with vLLM 0.33. Bounded native checks exercised text, conversation,
JSON Schema, forced tools, schedule OCR, official DEV transcription, candidate
scores and complete/streamed 24 kHz audio. These are integration checks, not
benchmarks, human voice-quality approval or broad capability acceptance.
Reference speech also generated a finite complete WAV from an official DEV
recording, through Talker directly. A client disconnect after the first audio
chunk aborted its owned request, and a following default-speaker request
completed. Generated chat-to-audio exercised the separate token-provenance route.

One identity smoke answer still used the source model name with the product
developer. Photograph details, full head quality, custom-voice fidelity and
concurrent load need separate review. Source FP32 weights may execute as BF16
under this deployment; this changes runtime precision, not the checkpoint files.
The PR supplies no model weights, download endpoint or production deployment.
