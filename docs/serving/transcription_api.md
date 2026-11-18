# Transcription API

`POST /v1/audio/transcriptions` accepts an uploaded audio file and returns
recognized text. A deployed ASR model can optionally use
Qwen3-ForcedAligner-0.6B to add word timestamps.

ASR capability is determined by the resolved model's vLLM
`supports_transcription` contract, with a generation runner, rather than its
name or the number of pipeline stages. The transcription handler constructs
requests with `task_type="transcribe"` using the model's own prompt builder.

An injected ASR aligner consumes the immediately preceding stage's completed
transcript and the original prompt's decoded waveform. The audio must include
its sample rate as `(waveform, sample_rate)`; the HTTP handler supplies this
pair automatically. Multichannel waveforms use vLLM's standard mono
normalization before alignment. Multi-stage pipelines can use this bridge
when their terminal text model supports transcription.
A terminal LLM without that capability is rejected, even if an earlier stage
is ASR. The pipeline must preserve the corresponding audio and transcript;
this capability check does not verify their semantic accuracy.

The HTTP endpoint has a separate compatibility check: its entry model and
frontend configuration must describe the ASR model, with one final text output
from that entry stage. Plain transcription only requests text. For example,
a Ming Thinker followed by a Talker and a TTS aligner can still transcribe
audio with the Thinker. The TTS aligner's timestamps describe synthesized
speech and are never returned as timestamps for the uploaded audio.
Forced word alignment requires a pooling ASR aligner directly connected to
the entry stage, with no other timestamps output. ASR at a later stage
can use the aligner bridge through its pipeline, but does not automatically
enable this endpoint. Registering a model with vLLM alone does not create an
Omni pipeline or deployment profile.

For Qwen3-ASR-0.6B and Qwen3-ASR-1.7B, follow the
[CUDA deployment recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Qwen/Qwen3-ASR-CUDA.md).
The recipe records the validated hardware, checkpoints, and test scope.

## Request

Send `multipart/form-data`, with one audio file per request:

| Field | Meaning |
| ----- | ------- |
| `file` | Required audio file, decoded by the model's vLLM audio preprocessing path. |
| `model` | Served model name; when omitted, the server uses its configured name. |
| `language` | Optional language code, such as `en` or `zh`; omitted for automatic detection. |
| `prompt` | Optional transcription context, interpreted by the ASR model. |
| `response_format` | Qwen3-ASR supports `json` (default), `text`, and `verbose_json`. |
| `timestamp_granularities[]` | Use `word` with `verbose_json` to request forced word alignment. |
| `stream` | Set to `true` for text SSE with `json` or `text`. |
| `stream_include_usage` | Include a final token-usage event in a text stream. |
| `stream_continuous_usage_stats` | Include cumulative usage on text events when usage reporting is enabled. |

The upload-size limit comes from vLLM's `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB`.
Audio decoding, supported languages, and chunk duration follow the loaded
ASR model. The Qwen3-ASR profile uses non-overlapping audio chunks for long
recordings.

## Plain transcription

```bash
curl http://localhost:8000/v1/audio/transcriptions \
    -F file=@audio.wav \
    -F response_format=json
```

`json` returns the transcript in `text`; `text` returns a plain-text body.
Both skip forced alignment even if the aligner stage is loaded.

For text streaming, add `-N -F stream=true`. Events use
`object: "transcription.chunk"`, with text in `choices[].delta.content`, and
end with `data: [DONE]`. With `stream_include_usage=true`, the final usage
event has an empty `choices` list. Prompt tokens are counted once per audio
chunk, and completion tokens accumulate across deltas and chunks.

Text streaming here starts from a complete uploaded file. It is separate
from live audio input over a realtime WebSocket.

## Word timestamps

Start the server with `--forced-aligner Qwen/Qwen3-ForcedAligner-0.6B`, or
enable the stage with `--forced-aligner-config`. Then request:

```bash
curl http://localhost:8000/v1/audio/transcriptions \
    -F file=@audio.wav \
    -F response_format=verbose_json \
    -F 'timestamp_granularities[]=word'
```

The ASR stage recognizes the audio. The aligner then consumes the completed
transcript and original audio to produce `words`. For example, the response
shape is:

```json
{
  "duration": 2.4,
  "language": "English",
  "text": "Hello world.",
  "words": [
    {"word": "Hello", "start": 0.2, "end": 0.6},
    {"word": "world", "start": 0.7, "end": 1.2}
  ]
}
```

These values illustrate the format, not measured model output. Word times
are in **seconds**, relative to the start of the uploaded file. Chunk offsets
are included. Empty transcripts contribute no words or extra text separators,
while their audio duration still contributes to later timestamps.

## Native ASR responses and alignment limits

With an ASR aligner enabled, routing depends on the model and requested granularity:

| Request | Qwen3-ASR | Other compatible ASR models |
| ------- | --------- | --------------------------- |
| `json` or `text`, without timestamps | Text only | Native vLLM handling |
| `verbose_json`, granularity omitted | Forced word alignment | Native vLLM handling |
| `verbose_json` + `word` | Forced word alignment | Forced word alignment |
| `verbose_json` + `segment` | Rejected | Native vLLM handling |
| `verbose_json` + `word` + `segment` | Rejected | Rejected |

Other ASR models retain native response formats, segment timestamps, beam
search, and diarization where their vLLM implementation supports them and
forced word alignment is not requested. Their native handler also owns
validation of those requests. Without an ASR aligner, all requests for those
models use native handling; Qwen3-ASR `verbose_json` requires an ASR aligner.

Forced word alignment requires non-streaming `verbose_json` and does not
support beam search or combined word/segment output. The Qwen3-ASR Omni
transcription path also rejects beam search for plain responses. This
integration does not add word timestamps to chat, translation, or realtime
responses.

See the recipe's [word segmentation requirements](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Qwen/Qwen3-ASR-CUDA.md#word-segmentation)
for Japanese/Korean limitations and optional dependency compatibility.
Use the aligner's supported languages listed in its
[model card](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B); the ASR
model may recognize additional languages that the aligner does not support.

## Extending ASR language metadata

ASR models can inherit `TranscriptionLanguageMixin` from
`vllm_omni.model_executor.models.interfaces` and override the class method
`get_transcription_language(raw_text)` to expose language metadata from their
output. Return a language code or English language name, or `None` when it is
unknown or the streaming prefix is incomplete. The default returns `None`;
models without the method remain compatible. This method only parses existing
output and must not run additional inference or alter the transcript.

Qwen3-ASR adds this method in a thin subclass of the upstream vLLM model,
selected by the Omni model registry. Ming3 Audio implements it on its model
class. Transcript text still comes from each model's `post_process_output`.
Request languages take precedence over output metadata, including languages
already supplied by the native explicit detection flow, such as Whisper's.
Both the aligner bridge and streaming chunk joining use the language capability.
