# Speech API

vLLM-Omni provides an OpenAI-compatible API for text-to-speech (TTS) generation. Supported TTS models include:

- **Qwen3-TTS** (`Qwen/Qwen3-TTS-12Hz-*`) -- Qwen3-based TTS with CustomVoice, VoiceDesign, and Base (voice cloning) task types. Native output: 24 kHz; API output can be resampled to 8 kHz.
- **Fish Speech S2 Pro** (`fishaudio/s2-pro`) -- Dual-AR TTS with DAC codec. Supports text-to-speech and voice cloning via reference audio. Output: 44.1 kHz.
- **Voxtral TTS** (`mistralai/Voxtral-4B-TTS-2603`) -- AR + FlowMatching TTS with preset voices. Output: 24 kHz.
- **CosyVoice3** (`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`) -- 2-stage talker + flow-matching code2wav. Voice cloning via `ref_audio` + `ref_text` (no presets). Output: 24 kHz.

See the [Supported Models](#supported-models) section below for the full list, including OmniVoice, VoxCPM2, and MOSS-TTS-Nano.

!!! tip "Deployment recipes"
    TTS deployment recipes are published at
    [recipes.vllm.ai](https://recipes.vllm.ai) (e.g.
    [Qwen3-TTS](https://recipes.vllm.ai/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice),
    [Higgs-Audio v3](https://recipes.vllm.ai/bosonai/higgs-audio-v3-tts-4b)).
    The in-repo runbooks live under [`recipes/`](https://github.com/vllm-project/vllm-omni/tree/main/recipes).

Each server instance runs a single model (specified at startup via `vllm serve <model> --omni`).

## Quick Start

### Start the Server

```bash
# Qwen3-TTS: CustomVoice model (predefined speakers)
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --deploy-config vllm_omni/deploy/qwen3_tts.yaml \
    --omni \
    --port 8091 \
    --trust-remote-code \
    --enforce-eager

# Fish Speech S2 Pro
vllm serve fishaudio/s2-pro --omni --port 8091

# Voxtral TTS
vllm serve mistralai/Voxtral-4B-TTS-2603 --omni --port 8091

# CosyVoice3 (voice cloning only — supply ref_audio + ref_text per request)
vllm serve FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
    --omni --port 8091 --trust-remote-code
```

### Generate Speech

**Using curl:**

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, how are you?",
        "voice": "vivian",
        "language": "English"
    }' --output output.wav
```

**Using Python:**

```python
import httpx

response = httpx.post(
    "http://localhost:8091/v1/audio/speech",
    json={
        "input": "Hello, how are you?",
        "voice": "vivian",
        "language": "English",
    },
    timeout=300.0,
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

**Using OpenAI SDK:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8091/v1", api_key="none")

response = client.audio.speech.create(
    model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    voice="vivian",
    input="Hello, how are you?",
)

response.stream_to_file("output.wav")
```

## API Reference

### Endpoint

```text
POST /v1/audio/speech
Content-Type: application/json
```

### Request Parameters

#### OpenAI Standard Parameters

| Parameter | Type | Default | Description |
| ----------- | ------ | --------- | ------------- |
| `input` | string | **required** | The text to synthesize into speech |
| `model` | string | server's model | Model to use (optional, should match server if specified) |
| `voice` | string | "vivian" | Speaker name (e.g., vivian, ryan, aiden) |
| `response_format` | string | "wav" | Audio format: wav, mp3, flac, pcm, opus |
| `speed` | float | 1.0 | Playback speed (0.25-4.0) |

#### vLLM-Omni Extension Parameters

| Parameter | Type | Default | Description |
| ----------- | ------ | --------- | ------------- |
| `task_type` | string | null (inferred) | TTS task type: CustomVoice, VoiceDesign, or Base. For Qwen3-TTS, `ref_audio` or `ref_text` infers Base when this field is omitted, while an uploaded/precomputed voice always selects Base. Other omitted values select CustomVoice; VoiceDesign must be specified explicitly. |
| `sample_rate` | integer | model native | Target output sample rate. Qwen3-TTS supports 8000 or 24000 Hz; the model remains native 24 kHz internally and is resampled before encoding. |
| `language` | string | "Auto" | Language (see supported languages below) |
| `instructions` | string | "" | Voice style/emotion instructions |
| `max_new_tokens` | integer | 2048 | Maximum tokens to generate |
| `initial_codec_chunk_frames` | integer | null | Per-request initial chunk size override for TTFA tuning. When null, IC is computed dynamically based on server load. |
| `non_streaming_mode` | bool | null | Qwen3-TTS prompt construction mode override. Does not affect HTTP response streaming or async-chunk pipelining. When null, Qwen3-TTS uses model defaults: Base=false, CustomVoice/VoiceDesign=true. |
| `stream` | bool | false | When true, stream OpenAI `speech.audio.*` SSE events (requires `response_format="pcm"` or `"wav"`). For raw PCM/WAV byte streaming, set `stream_format="audio"`. |
| `stream_format` | string | null | Streaming output format. `"audio"` streams raw audio bytes as they are decoded; `"sse"` streams OpenAI `speech.audio.*` Server-Sent Events. If omitted, `stream=true` selects SSE and `stream=false` remains non-streaming. See [Response Format](#response-format). |

**Supported languages:** Only applicable to Qwen3-TTS. Derived from the model configuration (`talker_config.codec_language_id` in the checkpoint's `config.json`), plus `Auto`, which is always accepted. Official Qwen3-TTS checkpoints support: Auto, Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian.

#### Voice Clone Parameters (Base task)

| Parameter | Type | Default | Description |
| ----------- | ------ | --------- | ------------- |
| `ref_audio` | string | null | Reference audio (HTTP URL, base64 data URL, or `file://` URI with `--allowed-local-media-path`). Local files fold `mtime_ns` and `size` into cache keys to automatically reload on-disk edits; HTTP URLs and base64 URIs remain cached by string locator. |
| `ref_text` | string | null | Transcript of reference audio |
| `x_vector_only_mode` | bool | null | Use speaker embedding only (no ICL) |

### Response Format

The response shape depends on the streaming parameters:

**Non-streaming (default).** With `stream=false` and no `stream_format`, returns the
complete clip as binary audio data with an appropriate `Content-Type` header (e.g.
`audio/wav`). Because the raw-bytes body has no JSON carrier, successful
non-streaming responses from non-diffusion speech servers report usage through
response headers:

| Header | Description |
| --- | --- |
| `x-vllm-omni-input-tokens` | Total input tokens (`text_tokens` + `audio_tokens`). |
| `x-vllm-omni-output-tokens` | Generated codec/audio tokens. |
| `x-vllm-omni-total-tokens` | `input_tokens` + `output_tokens`. |
| `x-vllm-omni-input-text-tokens` | Tokens from the synthesized text (`input` plus `instructions`). |
| `x-vllm-omni-input-audio-tokens` | Reference-audio codec frames, non-zero only for in-context voice cloning. |

Diffusion-mode speech servers route through a separate response path and do not
emit these headers.

**Raw audio stream** (`stream_format="audio"`). Streams raw audio bytes (PCM or
WAV) as they are decoded.

Both streaming modes (`stream_format="audio"` and `"sse"`) require
`response_format="pcm"` or `"wav"`, and `speed` must be `1.0` (or omitted).

**SSE stream** (`stream=true` or `stream_format="sse"`). Streams [OpenAI
`speech.audio.*` Server-Sent Events](https://platform.openai.com/docs/api-reference/audio-streaming).
Each event has an `event:` line and a JSON `data:` line:

- `speech.audio.delta` — a base64 audio chunk:

    ```json
    { "type": "speech.audio.delta", "audio": "<base64>", "response_format": "pcm" }
    ```

- `speech.audio.done` — terminal event, carrying token `usage`:

    ```json
    {
        "type": "speech.audio.done",
        "usage": {
            "input_tokens": 119,
            "output_tokens": 77,
            "total_tokens": 196,
            "input_token_details": { "text_tokens": 18, "audio_tokens": 101 }
        }
    }
    ```

- `speech.audio.error` — emitted instead of `speech.audio.done` if generation fails:

    ```json
    { "type": "speech.audio.error", "error": { "message": "...", "type": "server_error", "param": null, "code": 500 } }
    ```

The `usage` object on `speech.audio.done` is the same shape returned per item by the
[batch endpoint](#batch-speech-generation):

- `input_tokens` = `text_tokens` + `audio_tokens`
    - `text_tokens`: tokens of the synthesized text (`input` plus `instructions`)
    - `audio_tokens`: reference-audio codec frames, non-zero only for in-context
      voice cloning (Base task); `0` for CustomVoice/VoiceDesign or x-vector-only
- `output_tokens`: generated codec tokens
- `total_tokens` = `input_tokens` + `output_tokens`

### Voices Endpoint

```text
GET /v1/audio/voices
```

Lists available voices for the loaded model.

```json
{
    "voices": ["aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian", "custom_voice_1"],
    "uploaded_voices": [
        {
            "name": "custom_voice_1",
            "consent": "user_consent_id",
            "created_at": 1738660000,
            "file_size": 1024000,
            "mime_type": "audio/wav",
            "ref_text": "The exact transcript of the audio sample.",
            "speaker_description": "warm narrator"
        }
    ]
}
```

`uploaded_voices` is always present (empty list when no custom voices have been uploaded). Fields `ref_text` and `speaker_description` are omitted per-entry when not provided at upload time.

```text
POST /v1/audio/voices
Content-Type: multipart/form-data
```

Upload a new voice sample for voice cloning in Base task TTS requests.

**Form Parameters:**

| Parameter | Type | Required | Description |
| ----------- | ------ | ---------- | ------------- |
| `audio_sample` | file | Yes | Audio file (max 10MB, supported formats: wav, mp3, flac, ogg, aac, webm, mp4) |
| `consent` | string | Yes | Consent recording ID |
| `name` | string | Yes | Name for the new voice |
| `ref_text` | string | No | Transcript of the audio. When provided, enables in-context voice cloning (higher quality). Without it, only the speaker embedding is extracted. |
| `speaker_description` | string | No | Free-form description of the voice (e.g. "warm narrator", "energetic presenter"). Stored as metadata and returned in `GET /v1/audio/voices`. |

**Response Example:**

```json
{
  "success": true,
  "voice": {
    "name": "custom_voice_1",
    "consent": "user_consent_id",
    "created_at": 1738660000,
    "mime_type": "audio/wav",
    "file_size": 1024000,
    "ref_text": "The exact transcript of the audio sample.",
    "speaker_description": "warm narrator"
  }
}
```

Fields `ref_text` and `speaker_description` are omitted when not provided at upload time.

**Usage Example:**

```bash
curl -X POST http://localhost:8091/v1/audio/voices \
  -F "audio_sample=@/path/to/voice_sample.wav" \
  -F "consent=user_consent_id" \
  -F "name=custom_voice_1" \
  -F "ref_text=The exact transcript of the audio sample." \
  -F "speaker_description=warm narrator"
```

## Streaming Text Input (WebSocket)

The `/v1/audio/speech/stream` WebSocket endpoint accepts text incrementally and generates audio per sentence as boundaries are detected.

> Note: text input is always streamed incrementally. Audio output remains sentence-scoped:
> use `stream_audio=false` for one binary frame per sentence, or `stream_audio=true` for one or more PCM chunks per sentence.

### WebSocket Protocol

Client -> Server:

| Message | Description |
| --------- | ------------- |
| `{"type": "session.config", ...}` | Session configuration (first message; may be resent between utterances to change it) |
| `{"type": "input.text", "text": "..."}` | Text chunk |
| `{"type": "input.done"}` | End of utterance: flushes the buffer and keeps the connection open |
| `{"type": "session.close"}` | End of connection |

Server -> Client:

| Message | Description |
| --------- | ------------- |
| `{"type": "audio.start", "utterance_index": 0, "sentence_index": 0, "sentence_text": "...", "format": "pcm", "sample_rate": 24000}` | Audio generation starting for the buffered input |
| Binary frame | Raw audio bytes (one or more PCM chunks when `stream_audio=true`) |
| `{"type": "audio.done", "utterance_index": 0, "sentence_index": 0, "total_bytes": 96000, "error": false}` | Audio complete for the buffered input |
| `{"type": "session.done", "utterance_index": 0, "total_sentences": N}` | Flushed utterance complete |
| `{"type": "error", "message": "..."}` | Non-fatal error |

### Flushing vs. Closing

`input.done` is a flush, not a disconnect. The server synthesizes the buffered
text, emits `session.done`, and then waits on the same connection for the next
utterance, so a client that speaks repeatedly (for example one driven by an
upstream LLM) pays the WebSocket handshake once instead of once per utterance.

- The session config is sticky. Send `input.text` again straight after
  `session.done` to reuse it, or send another `session.config` first to change
  voice, format, or reference audio. A `session.config` sent while text is
  still buffered is rejected so no pending input is silently dropped.
- An utterance is the flush unit, not a linguistic one: it is whatever text was
  buffered when `input.done` arrived, of any length, synthesized as one request.
  `utterance_index` counts those flushes across the connection, so it tells you
  which `input.done` a frame belongs to. `sentence_index` counts within one
  flush and so pairs with `total_sentences`, which means every utterance reports
  `sentence_index: 0` of `total_sentences: 1` (or `0` for an empty buffer).
- End the connection with `session.close`, or by closing the socket. An idle
  connection is still closed after the server's idle timeout, which now also
  applies to the gap between utterances.

### Session Config Parameters

All REST API parameters are supported, plus:

| Parameter | Type | Default | Description |
| ----------- | ------ | --------- | ------------- |
| `stream_audio` | bool | false | Stream one or more PCM chunks for the buffered input over WebSocket |

```bash
DELETE /v1/audio/voices/{name}
```

Delete an uploaded voice sample.

**Path Parameters:**

| Parameter | Type | Required | Description |
| ----------- | ------ | ---------- | ------------- |
| `name` | string | Yes | Name of the voice to delete |

**Response Example:**

```json
{
  "success": true,
  "message": "Voice 'custom_voice_1' deleted successfully"
}
```

**Error Response (404 Not Found):**

```json
{
  "success": false,
  "error": "Voice 'unknown_voice' not found"
}
```

**Usage Example:**

```bash
curl -X DELETE http://localhost:8091/v1/audio/voices/custom_voice_1
```

## Examples

### CustomVoice with Style Instruction

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "I am so excited!",
        "voice": "vivian",
        "instructions": "Speak with great enthusiasm"
    }' --output excited.wav
```

### VoiceDesign (Natural Language Voice Description)

```bash
# Start server with VoiceDesign model first
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
    --deploy-config vllm_omni/deploy/qwen3_tts.yaml \
    --omni \
    --port 8091 \
    --trust-remote-code \
    --enforce-eager
```

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello world",
        "task_type": "VoiceDesign",
        "instructions": "A warm, friendly female voice with a gentle tone"
    }' --output designed.wav
```

### Base (Voice Cloning)

```bash
# Start server with Base model first
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --deploy-config vllm_omni/deploy/qwen3_tts.yaml \
    --omni \
    --port 8091 \
    --trust-remote-code \
    --enforce-eager
```

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, this is a cloned voice",
        "task_type": "Base",
        "ref_audio": "https://example.com/reference.wav",
        "ref_text": "Original transcript of the reference audio",
        "non_streaming_mode": true
    }' --output cloned.wav
```

### Upload Voice

Upload voice (speaker embedding only):

```bash
curl -X POST http://localhost:8091/v1/audio/voices \
  -F "audio_sample=@/path/to/voice_sample.wav" \
  -F "consent=user_consent_id" \
  -F "name=custom_voice_1"
```

Upload voice with transcript (in-context cloning, higher quality):

```bash
curl -X POST http://localhost:8091/v1/audio/voices \
  -F "audio_sample=@/path/to/voice_sample.wav" \
  -F "consent=user_consent_id" \
  -F "name=custom_voice_2" \
  -F "ref_text=The exact transcript of the audio sample."
```

### Use Uploaded Voice

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, this is a cloned voice",
        "voice": "custom_voice_1"
    }' --output cloned.wav
```

For Qwen3-TTS, an uploaded voice is a Base voice-cloning input. The server
infers `task_type="Base"` when `voice` names an uploaded entry, so the request
must be sent to a Base checkpoint. Built-in presets such as `vivian` and
`ryan` remain CustomVoice speakers and require a CustomVoice checkpoint.

### Voice Storage & Caching

Uploaded voices are persisted to disk as a single `.safetensors` file per voice
(audio samples + metadata — name, consent, ref_text, sample_rate, created_at —
in the file header). On server restart the directory is scanned and all
previously uploaded voices are restored automatically, so uploads survive
process restarts.

Uploading an existing name overwrites the previous entry (a warning is logged).

Feature extraction artifacts (ref_code, speaker_embedding, DAC codes, etc.)
are cached in-process with a shared LRU so repeated requests with the same
`voice=...` skip the extraction pipeline. The cache is a true singleton across
all TTS model types; deleting a voice invalidates every model-type slot at
once.

### Precomputed Custom Voices

Qwen3-TTS Base and VoxCPM2 can load offline-precomputed voices at startup.
Qwen3-TTS precomputed voices follow the same Base task and checkpoint-matching
rules as uploaded voices.

Generate a directory containing `custom_voice_manifest.json` plus one
`.safetensors` file per voice, then set the pipeline-wide deploy config field:

```yaml
custom_voice_dir: /path/to/custom_voices
```

Qwen3-TTS profiles are created with:

```bash
python examples/online_serving/text_to_speech/qwen3_tts/precompute_custom_voice.py \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --voice-name alice \
  --ref-audio /path/to/reference.wav \
  --ref-text "Original transcript of the reference audio" \
  --mode icl \
  --output-dir /path/to/custom_voices
```

VoxCPM2 profiles are created with:

```bash
python examples/online_serving/text_to_speech/voxcpm2/precompute_custom_voice.py \
  --model openbmb/VoxCPM2 \
  --voice-name alice \
  --ref-audio /path/to/reference.wav \
  --mode ref_continuation \
  --prompt-text "Original transcript of the reference audio" \
  --output-dir /path/to/custom_voices
```

Only profiles whose safetensors payload can be loaded and validated are exposed
by `GET /v1/audio/voices`. Valid precomputed voices can be used in
`POST /v1/audio/speech` by passing `voice="alice"` without `ref_audio`.

**Configuration (environment variables):**

| Variable | Default | Description |
| ---------- | --------- | ------------- |
| `SPEAKER_SAMPLES_DIR` | `~/.cache/vllm-omni/speakers` | Directory for persisted uploaded speakers (`.safetensors` files). |
| `SPEAKER_MAX_UPLOADED` | `1000` | Maximum number of uploaded speakers kept on disk. Upload requests past the cap return 400. |

The in-memory LRU has a fixed 512 MiB byte budget.

## Batch Speech Generation

The batch endpoint synthesizes multiple texts in a single request, returning all results as JSON with base64-encoded audio.

### Endpoint

```text
POST /v1/audio/speech/batch
Content-Type: application/json
```

### Request Parameters

| Parameter | Type | Default | Description |
| ----------- | ------ | --------- | ------------- |
| `items` | array | **required** | List of items to synthesize (1–32) |
| `model` | string | server's model | Model to use |
| `voice` | string | null | Default voice for all items |
| `response_format` | string | "wav" | Default audio format for all items |
| `speed` | float | 1.0 | Default playback speed (0.25–4.0) |
| `task_type` | string | null | Default TTS task type |
| `language` | string | null | Default language |
| `instructions` | string | null | Default voice style instructions |
| `ref_audio` | string | null | Default reference audio (Base task) |
| `ref_text` | string | null | Default reference transcript (Base task) |
| `max_new_tokens` | integer | null | Default max tokens |
| `non_streaming_mode` | bool | null | Default Qwen3-TTS prompt construction mode override. Does not affect HTTP response streaming or async-chunk pipelining. When null, Qwen3-TTS uses model defaults: Base=false, CustomVoice/VoiceDesign=true. |

Each item in the `items` array requires only `input` (the text). All other fields are optional and override the batch-level defaults when set:

| Field | Type | Description |
| ------- | ------ | ------------- |
| `input` | string | **required** — text to synthesize |
| `voice` | string | Override voice for this item |
| `response_format` | string | Override format for this item |
| `speed` | float | Override speed for this item |
| `task_type` | string | Override task type |
| `language` | string | Override language |
| `instructions` | string | Override instructions |
| `ref_audio` | string | Override reference audio |
| `ref_text` | string | Override reference transcript |
| `max_new_tokens` | integer | Override max tokens |
| `non_streaming_mode` | bool | Override Qwen3-TTS prompt construction mode. Does not affect HTTP response streaming or async-chunk pipelining. When null, inherits the batch-level value (then the model default). |

### Response Format

```json
{
    "id": "speech-batch-abc123",
    "results": [
        {
            "index": 0,
            "status": "success",
            "audio_data": "<base64-encoded audio>",
            "media_type": "audio/wav",
            "usage": {
                "input_tokens": 119,
                "output_tokens": 77,
                "total_tokens": 196,
                "input_token_details": { "text_tokens": 18, "audio_tokens": 101 }
            }
        },
        {
            "index": 1,
            "status": "error",
            "error": "Input text cannot be empty"
        }
    ],
    "total": 2,
    "succeeded": 1,
    "failed": 1
}
```

Each successful item carries a `usage` object (errored items omit it):

- `input_tokens` = `text_tokens` + `audio_tokens`
    - `text_tokens`: tokens of the synthesized text (`input` plus `instructions`)
    - `audio_tokens`: reference-audio codec frames, non-zero only for in-context
      voice cloning (Base task); `0` for CustomVoice/VoiceDesign or x-vector-only
- `output_tokens`: generated codec tokens
- `total_tokens` = `input_tokens` + `output_tokens`

This is the same `usage` object emitted on the terminal `speech.audio.done` event
of the single endpoint's [SSE stream](#response-format) (`stream_format="sse"`).

### Examples

**Basic batch with shared defaults:**

```bash
curl -X POST http://localhost:8091/v1/audio/speech/batch \
    -H "Content-Type: application/json" \
    -d '{
        "items": [
            {"input": "Hello, how are you?"},
            {"input": "Goodbye, see you later!"}
        ],
        "voice": "vivian",
        "language": "English"
    }'
```

**Per-item overrides (different voices and formats):**

```bash
curl -X POST http://localhost:8091/v1/audio/speech/batch \
    -H "Content-Type: application/json" \
    -d '{
        "items": [
            {"input": "Hello!", "voice": "vivian", "response_format": "mp3"},
            {"input": "你好！", "voice": "ryan", "language": "Chinese"}
        ],
        "response_format": "wav"
    }'
```

**Voice cloning with shared reference audio (Base task):**

```bash
curl -X POST http://localhost:8091/v1/audio/speech/batch \
    -H "Content-Type: application/json" \
    -d '{
        "items": [
            {"input": "First sentence in the cloned voice."},
            {"input": "Second sentence in the cloned voice."}
        ],
        "task_type": "Base",
        "ref_audio": "https://example.com/reference.wav",
        "ref_text": "Transcript of the reference audio"
    }'
```

Setting `ref_audio` at the batch level applies it to all items, avoiding the need to repeat it per item.

**Decoding the response in Python:**

```python
import base64
import httpx

response = httpx.post(
    "http://localhost:8091/v1/audio/speech/batch",
    json={
        "items": [
            {"input": "First sentence."},
            {"input": "Second sentence."},
        ],
        "voice": "vivian",
    },
    timeout=300.0,
)

for result in response.json()["results"]:
    if result["status"] == "success":
        audio_bytes = base64.b64decode(result["audio_data"])
        with open(f"output_{result['index']}.wav", "wb") as f:
            f.write(audio_bytes)
```

### Configuration

| Parameter | Source | Default | Description |
| ----------- | -------- | --------- | ------------- |
| `tts_batch_max_items` | engine kwarg | 32 | Maximum number of items per batch request |

All items are fanned out to `generate()` concurrently. The engine's stage worker automatically batches them up to the configured `max_num_seqs` and queues the rest — no client-side throttling needed.

For best throughput, set both stages' `max_num_seqs` above 1 via `--stage-overrides`. On the current Qwen3-TTS CustomVoice benchmark, stage 1 performed best at `max_num_seqs: 10`:

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --omni --port 8091 --trust-remote-code --enforce-eager \
    --stage-overrides '{"0":{"max_num_seqs":10,"gpu_memory_utilization":0.2},
                        "1":{"max_num_seqs":10,"gpu_memory_utilization":0.2}}'
```

The bundled `qwen3_tts.yaml` uses a multi-request default and lets stage 1 batch chunks across in-flight requests. For latency-sensitive deployments, avoid forcing stage 1 back to `max_num_seqs: 1`; benchmark before reducing it below `10`.

The bundled config also sets `initial_codec_chunk_frames: 1`. This emits only the first audio chunk early for lower TTFA, then returns to the normal `codec_chunk_frames` window so Code2Wav does not repeatedly decode tiny overlapping chunks.

## Supported Models

### Qwen3-TTS

| Model | Task Type | Description |
| ------- | ----------- | ------------- |
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | CustomVoice | Predefined speaker voices with optional style control |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | VoiceDesign | Natural language voice style description |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Base | Voice cloning from reference audio or an uploaded/precomputed voice |
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | CustomVoice | Smaller/faster variant |
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Base | Smaller/faster Base variant for voice cloning, including uploaded/precomputed voices |

### Fish Speech S2 Pro

| Model | Description |
| ------- | ------------- |
| `fishaudio/s2-pro` | 4B dual-AR TTS with DAC codec (44.1 kHz). Supports text-to-speech and voice cloning. |

Fish Speech uses `ref_audio` and `ref_text` for voice cloning (no `task_type` needed). The `voice` field should be set to `"default"`. See the [Fish Speech section of the online TTS hub](../user_guide/examples/online_serving/text_to_speech.md#fish-speech-s2-pro) for details.

### Voxtral TTS

| Model | Description |
| ------- | ------------- |
| `mistralai/Voxtral-4B-TTS-2603` | 3B AR + FlowMatching TTS. Supports text-to-speech with preset voices. |

### CosyVoice3

| Model | Description |
| ------- | ------------- |
| `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | Voice cloning from `ref_audio` + `ref_text`. No built-in voice presets — upload a voice or pass `ref_audio`/`ref_text` per request. |

### OmniVoice

| Model | Description |
| ------- | ------------- |
| `k2-fsa/OmniVoice` | Pure-diffusion TTS. Supports voice cloning via `ref_audio` (with optional `ref_text`); no built-in voice presets. |

### VoxCPM2

| Model | Description |
| ------- | ------------- |
| `openbmb/VoxCPM2` | TTS + voice cloning with built-in speaker presets and uploaded-voice support. Accepts `voice` (preset or uploaded) or `ref_audio` + optional `ref_text`. |

### MOSS-TTS-Nano

| Model | Description |
| ------- | ------------- |
| `OpenMOSS-Team/MOSS-TTS-Nano` | Voice cloning only. Requires `ref_audio` (or an uploaded `voice`); no built-in voice presets. `ref_text` is accepted but ignored — upstream's `voice_clone` mode does not consume a transcript. |

## Error Responses

### 400 Bad Request

Invalid parameters:

```json
{
    "error": {
        "message": "Input text cannot be empty",
        "type": "BadRequestError",
        "param": null,
        "code": 400
    }
}
```

Qwen3-TTS task/checkpoint mismatch:

```json
{
    "error": {
        "message": "Qwen3-TTS CustomVoice checkpoint does not support task_type='Base'. Use task_type='CustomVoice' or load the matching Base checkpoint.",
        "type": "BadRequestError",
        "param": null,
        "code": 400
    }
}
```

### 404 Not Found

Model not found:

```json
{
    "error": {
        "message": "The model `xxx` does not exist.",
        "type": "NotFoundError",
        "param": "model",
        "code": 404
    }
}
```

## Troubleshooting

### "TTS model did not produce audio output"

Ensure you're using the correct model variant for your task type:

- CustomVoice task → CustomVoice model
- VoiceDesign task → VoiceDesign model
- Base task → Base model

Uploaded and precomputed Qwen3-TTS voices select the Base task, even when the
client supplies a different `task_type`; serve them with a Base checkpoint.
Built-in Qwen3-TTS presets remain CustomVoice voices.

### Server Not Running

```bash
# Check if server is responding
curl http://localhost:8091/v1/audio/voices
```

### Out of Memory

If you encounter OOM errors:

1. Use smaller model variant: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
2. Reduce `--gpu-memory-utilization`

### Unsupported Speaker

Use `/v1/audio/voices` to list available voices for the loaded model.

## Orchestration Loop (experimental)

Multi-stage omni deployments route stage outputs through a single orchestrator
loop. By default that loop polls every stage replica on a 1 ms cadence. An
opt-in event-driven mode replaces the poll with one reader task per live stage
replica awaiting its client directly, and switches the serving-side
final-output drain to a condition-variable wakeup at the same time.

**Configuration (environment variables):**

| Variable | Default | Description |
| --- | --- | --- |
| `VLLM_OMNI_EVENT_DRIVEN_ORCH` | `0` (off) | Switches the orchestration loop and the final-output drain from the legacy 1 ms poll to event-driven wakeups. Enabled by `1`, `true`, `yes`, or `on`, matched case-insensitively after surrounding whitespace is stripped; any other value leaves it off. |

Set it on the process that runs the orchestrator (stage 0 of an omni
deployment) before starting the server:

```bash
export VLLM_OMNI_EVENT_DRIVEN_ORCH=1
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --omni \
    --port 8091
```

The server logs the selected loop mode and its reader/poller counts once at
startup, so you can confirm which loop is live.

Routing, output ordering, and terminal-state behavior are identical on both
loops; only the poll cadence changes. Leaving the variable unset keeps the
legacy poll loop, which is the supported default.

**Known limitations:**

- The measured serving A/B (idle CPU 2.43% to 0.07%; TTFP p99 -32% at
  concurrency 8) predates the rebuild on the per-replica fault-isolation work
  in [#4285](https://github.com/vllm-project/vllm-omni/pull/4285). That work
  changed dead-replica handling and reader/poller lifecycle rather than the
  steady-state output path, and the parity suite covers it, but the serving
  A/B has not been re-run on the current head.
- The diffusion-poller branch is covered by unit tests only. Deployments whose
  stages all run as standard engine cores never exercise it, including GLM-TTS,
  which deploys its DiT without `stage_type: diffusion`.
- Concurrency 1 and 32 measured at parity with the legacy loop. At 32 the
  latency is admission-bound, which this mode does not address.

## Development

Enable debug logging:

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --deploy-config vllm_omni/deploy/qwen3_tts.yaml \
    --omni \
    --port 8091 \
    --trust-remote-code \
    --enforce-eager \
    --uvicorn-log-level debug
```
