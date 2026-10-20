# Ming-3.0-audio-flash

> Online multilingual ASR and speech QA with optional spoken answers

## Summary

- Vendor: inclusionAI
- Model: `inclusionAI/Ming-3.0-audio-flash`
- Tasks: 16 kHz multilingual ASR and speech QA with text or 44.1 kHz audio output
- Mode: Online serving with the OpenAI-compatible Chat Completions API
- Maintainer: Community

## References

- [Hugging Face model](https://huggingface.co/inclusionAI/Ming-3.0-audio-flash)
- [Hugging Face Tiny model](https://huggingface.co/inclusionAI/Ming-3.0-audio-tiny)
- [Upstream repository and examples](https://github.com/inclusionAI/Ming-3.0-audio-flash)

## Overview

Use this recipe to transcribe multilingual audio or answer a question spoken in an audio clip.
ASR and Speech QA share the Ming3 audio thinker; requests for audio output additionally run the bundled Ming talker and AudioVAE.

## CUDA

### Environment

- Linux, Python 3.11–3.13, and vLLM-Omni matching vLLM 0.32.0.
- Client examples use Bash, `curl`, `jq`, and GNU `base64`.
- Run commands from the repository root. Speech output also requires the [text normalization resources](#text-normalization-resources).

### Command

The default end-to-end configuration targets four 80 GB-class GPUs.
The thinker runs with TP=4 and the talker is colocated on GPU 3.

```bash
vllm serve inclusionAI/Ming-3.0-audio-flash \
    --omni \
    --deploy-config vllm_omni/deploy/ming3_audio_flash.yaml \
    --served-model-name ming3-audio-flash \
    --port 8091
```

`"modalities": ["text"]` stops after the thinker.
`"modalities": ["text", "audio"]` runs the talker and returns the same answer as text and speech.

### Text-only deployment

When the service only needs ASR or Speech QA text, avoid loading the talker:

```bash
vllm serve inclusionAI/Ming-3.0-audio-flash \
    --omni \
    --deploy-config vllm_omni/deploy/ming3_audio_flash_thinker_only.yaml \
    --served-model-name ming3-audio-flash \
    --port 8091
```

### Original audio in multi-turn history

The default profile `ming3_audio_flash.yaml` supports retaining original audio clips
across turns.

The profile permits up to 20 audio items per request, counted across all
messages. Other profiles keep their existing item limits. Each clip retains its own
feature length and placeholder; clips may have different durations or repeat an earlier
recording.

Send the full Chat Completions `messages` history, including earlier user
`audio_url` content and assistant text, followed by the new user audio. For example:

```json
{
  "model": "ming3-audio-flash",
  "messages": [
    {"role": "user", "content": [
      {"type": "audio_url", "audio_url": {"url": "https://example.com/turn-1.wav"}}
    ]},
    {"role": "assistant", "content": "Your previous answer."},
    {"role": "user", "content": [
      {"type": "text", "text": "Please answer the follow-up question."},
      {"type": "audio_url", "audio_url": {"url": "https://example.com/turn-2.wav"}}
    ]}
  ],
  "modalities": ["text"],
  "max_tokens": 256
}
```

Replace the example URLs with accessible recordings. All audio tokens and text
share the configured context limit. Increasing the item limit permits more clips and preprocessing
work; all audio and text still share the context and encoder budgets. Choose a
bound appropriate for the service. Retaining audio history does not itself enable cross-request
KV-cache reuse. This profile keeps the base profile's prefix-cache setting.

### Reusing an unchanged conversation prefix

The default profile `ming3_audio_flash.yaml` supports native Thinker prefix
caching while retaining up to 20 original audio clips. The profile enables
chunked prefill and aligned KDA-state checkpoints; Talker prefix caching remains
disabled. It requires the CUDA Bailing/KDA kernels supported by this recipe.

Resend the complete `messages` history with identical earlier text and audio.
The native cache reuses matching full-attention KV blocks and recurrent-state
checkpoints, then computes the unmatched suffix. Different audio content is part
of the native multimodal cache identity. Talker voice selection does not retain
or replace the Thinker conversation history.

Reuse is local to a worker and bounded by its cache capacity and block alignment.
Eviction, another replica, or changed earlier messages can produce a miss; an
unaligned tail may be recomputed. This is separate from audio-feature caching
and does not permanently pin a conversation in GPU memory. Use Chat Completions
usage details (`stream_options: {"include_usage": true}` for streaming) to inspect
cached prompt tokens; start the server with `--enable-prompt-tokens-details`
to expose them in `usage.prompt_tokens_details.cached_tokens`. A faster response
alone does not establish a cache hit.

### Talker decode graphs and low-latency collection

The full Flash and Tiny profiles enable the Talker's single-token Qwen2 decode
graph. Prefill remains eager, the outer Talker stage keeps `cudagraph_mode: NONE`,
and unsupported inputs or capture failures fall back to eager execution.
Graph buffers reset per request and do not provide Talker conversation reuse.

`low_latency_streaming` remains `false`. Enabling it collects the first available
audio latent promptly, then resumes normal VAE batching. BF16 VAE results can
vary with chunk boundaries; do not assume bitwise-identical waveforms across
chunk partitions. Cache checkpoints and graph buffers also consume GPU memory.

### Deployment profiles

Select a profile with `--deploy-config`. ASR profiles run only the Thinker for
text transcription; full profiles also include the Talker for speech output.

| Profile | GPUs / Thinker TP | Thinker context | Prefill token budget | Default output limit | no-repeat-ngram size |
| --- | --- | --- | --- | --- | --- |
| `ming3_audio_flash.yaml` | 4 / 4 | 131072 | 32768 | 65536 | - |
| `ming3_audio_flash_asr.yaml` | 4 / 4 | 131072 | 32768 | 32768 | 10 |
| `ming3_audio_flash_thinker_only.yaml` | 4 / 4 | 131072 | 32768 | 65536 | - |
| `ming3_audio_tiny.yaml` | 1 / 1 | 131072 | 32768 | 65536 | - |
| `ming3_audio_tiny_asr.yaml` | 1 / 1 | 131072 | 32768 | 32768 | 10 |
| `ming3_audio_tiny_thinker_only.yaml` | 1 / 1 | 131072 | 32768 | 65536 | - |

For example, select the Flash thinker-only profile:

```bash
vllm serve inclusionAI/Ming-3.0-audio-flash --omni \
    --deploy-config ming3_audio_flash_thinker_only.yaml \
    --served-model-name ming3-audio-flash --port 8091
```

The earlier Flash `*_128k`, `*_history`, `*_history_cache`, `*_llm_graph` and
`*_multiturn_graph` overlays have been consolidated into the base profiles.
Use `ming3_audio_flash.yaml` for speech output or the ASR/thinker-only profiles
for text output. Limit answer length with request `max_tokens` as needed.

Tiny profiles use the [Ming-3.0-audio-tiny checkpoint](https://huggingface.co/inclusionAI/Ming-3.0-audio-tiny), selected explicitly:

```bash
vllm serve inclusionAI/Ming-3.0-audio-tiny --omni \
    --deploy-config ming3_audio_tiny.yaml \
    --served-model-name ming3-audio-tiny --port 8091
```

The full Tiny profile places both stages on GPU 0. The ASR and thinker-only
variants run only the Thinker. Both Flash and Tiny ASR profiles enable 10-gram
repetition blocking for text-only transcription; use a full profile for speech output.
Tiny templates have configuration-resolution coverage; validate their checkpoint
compatibility, memory requirements and output quality on the target GPU.

## Verification

Use a full Thinker/Talker profile for the speech-output examples. Text-only profiles support transcription and text QA. Run client requests in a separate terminal after the server is ready.

### Transcribe with Chat Completions

```bash
curl http://localhost:8091/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "ming3-audio-flash",
      "messages": [{
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Please recognize the language of this speech and transcribe it with appropriate punctuation marks, placing a `\t` between the identified language and the transcribe."
          },
          {
            "type": "audio_url",
            "audio_url": {
              "url": "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/mary_had_lamb.ogg"
            }
          }
        ]
      }],
      "modalities": ["text"],
      "temperature": 0,
      "max_tokens": 256
    }' | jq -r '.choices[0].message.content'
```

### Ask a question in audio

The upstream [`examples/audio/`](https://github.com/inclusionAI/Ming-3.0-audio-flash/tree/main/examples/audio) directory includes `introduce_ant_group.wav`. Its [manifest](https://github.com/inclusionAI/Ming-3.0-audio-flash/blob/main/examples/audio/manifest.json) identifies the spoken question as “介绍一下蚂蚁集团。” (24 kHz mono, 2.48 seconds).

Download the clip once from the public source, then reuse it for this request and **Speak the answer** below. The download runs on the client before inference; no local model directory is needed:

```bash
BASE=https://raw.githubusercontent.com/inclusionAI/Ming-3.0-audio-flash/main/examples/audio
MING3_AUDIO_FILE=introduce_ant_group.wav
curl --fail --location "$BASE/introduce_ant_group.wav" --output "$MING3_AUDIO_FILE"

base64 -w0 "$MING3_AUDIO_FILE" | jq -Rs '{
  model: "ming3-audio-flash",
  messages: [
    {
      role: "system",
      content: "回答音频中的问题，回答不要使用markdown格式。"
    },
    {
      role: "user",
      content: [{
        type: "audio_url",
        audio_url: {url: ("data:audio/wav;base64," + .)}
      }]
    }
  ],
  modalities: ["text"],
  temperature: 0.6,
  max_tokens: 1024
}' | curl http://localhost:8091/v1/chat/completions \
    -H "Content-Type: application/json" \
    --data-binary @- | jq -r '.choices[0].message.content'
```

### Speak the answer

Select an explicit IP voice for a repeatable comparison. The two stages stay loaded in the service.

```bash
base64 -w0 "$MING3_AUDIO_FILE" | jq -Rs '{
  model: "ming3-audio-flash",
  messages: [
    {
      role: "system",
      content: "请用一句简短的话回答音频中的问题，不超过50个汉字，不要使用Markdown格式。"
    },
    {
      role: "user",
      content: [{
        type: "audio_url",
        audio_url: {url: ("data:audio/wav;base64," + .)}
      }]
    }
  ],
  modalities: ["text", "audio"],
  audio: {voice: {id: "灵小柔"}, format: "wav"},
  temperature: 0.6,
  max_tokens: 128
}' | curl --fail-with-body http://localhost:8091/v1/chat/completions \
    -H "Content-Type: application/json" \
    --data-binary @- -o response.json
jq -r '.choices[0].message.content' response.json
jq -r '.choices[0].message.audio.data' response.json \
    | base64 -d > ming3_speech_qa.wav
```

## Notes

- The ASR prompt requests `language<TAB>transcription`, but raw Chat Completions output can contain a literal `\t` separator instead. Clients that need only the transcript should accept either separator when removing the leading language field.
- Speech QA returns a direct answer rather than a transcript. It uses Chat Completions and is not passed through the ASR language-prefix postprocessor.
- Select Ming3 IP voices with Realtime `session.audio.output.voice: {"id": "灵小柔"}` (or Chat `audio.voice`). IDs in [the Ming3 IP voice list](../../vllm_omni/model_executor/models/ming_flash_omni/data/ip_voices.json) use IP captions without reference audio, before checking registered voices. Other IDs must be registered, including any model-provided `voice_presets`; unknown IDs fail.
- Ming3 reference mode accepts per-request `ref_audio` (a URL, data URL, or base64 audio) and optional `prompt_text`. Choose either reference audio or a voice ID. Omitting voice selection and reference inputs uses the IP voice `灵小柔`, without reference audio. Explicit `voice_mode: "reference"` requires reference audio, a registered voice, or a speaker embedding. Supply reference inputs in each request or select a registered voice.
- `normalize_text` controls linguistic text normalization. `scrub_markup` independently removes HTML, code fences and math markup before synthesis and defaults to `true`. Set both fields to `false` to disable markup scrubbing and linguistic normalization. Streaming segmentation and language-prefix handling still apply. Keep these controls fixed for the duration of a streaming request.
- The talker emits mono 44.1 kHz audio.
- Use Chat Completions for transcription; the `/v1/audio/transcriptions` endpoint is not currently available with `--omni`.
- The model handles audio chunking internally, so the deployment configuration does not set `max_audio_clip_s` or `min_energy_split_window_size`.
- Flash and Tiny profiles configure a 131072-token context. This includes input and output positions; a configured limit does not establish full-length quality or memory capacity. Validate the selected checkpoint and target GPUs before deployment.

### Text normalization resources

Full Flash/Tiny speech-output profiles enable `normalize_text: true` and `tn_mode: upstream`.
Install the `ming-tn` optional dependencies in the serving environment (`pynini==2.1.7` and `inflect`).
The native Talker reads `talker_re.json` and the four `{zh,en}_tn_{tagger,verbalizer}.fst` files from the selected model's `talker/talker_tn/` directory; vLLM-Omni does not bundle these resources.
The checkpoint resolver includes this directory in its `talker/**` download, and `--revision` selects the same snapshot for weights and normalization rules. A complete local export or cached snapshot works offline.

Upstream normalization reports missing dependencies, rules or graph files instead
of silently changing the spoken-text rules. Keep `talker/talker_tn/` when
copying an export for speech output. Explicit `tn_mode: legacy` or
`normalize_text: false` does not require these graphs and can change the
pronunciation of numbers, dates and amounts. Legacy mode segments text and
expands English numbers without the upstream Chinese RE/FST rules.
Text-only ASR does not load these resources.

## Standalone text to speech

Stop the full server before starting the standalone service on the same port. Run the request in a separate terminal after startup.

The official [Ming-3.0-audio-flash checkpoint](https://huggingface.co/inclusionAI/Ming-3.0-audio-flash) includes a `talker/` directory with the Talker config,
LLM tokenizer, AudioVAE and speaker extractor. The Ming3-specific profile loads
only the Talker assets, without loading Thinker weights. The resolver also accepts
a local export with the same layout; `--revision` selects the Hub snapshot.

The resolver stores generated config views and asset symlinks under `VLLM_CACHE_ROOT/ming_talker`, using vLLM's standard cache root.
Checkpoint files stay in the original local export or HF cache; they are not copied into the config view. The same checkpoint and config reuse the existing view.

```bash
vllm serve inclusionAI/Ming-3.0-audio-flash --omni \
    --deploy-config vllm_omni/deploy/ming3_audio_tts.yaml \
    --served-model-name ming3-audio-tts \
    --port 8091

curl http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"model":"ming3-audio-tts","input":"你好，很高兴认识你。","voice":"灵小柔","response_format":"wav"}' \
    --output speech.wav
```

For concurrent clients, `ming3_audio_tts_batched.yaml` serves up to 64
requests in one shared Talker batch: each step advances every active request by
one latent, with the Qwen2 decode, CFM, aggregator and stop head captured as one
CUDA graph per batch size. AudioVAE decoding stays per request. Batched decoding
uses FlashAttention over a per-request KV row, so latents differ from the
single-request path at bf16 rounding level, and seeded results depend on which
requests share a step. `max_num_seqs: 1` keeps the single-request path. In the
speech-output pipeline the same applies to the Talker stage: set stage 1
`max_num_seqs` above one to batch concurrent replies.

Omitting `voice` and reference inputs defaults to the IP voice `灵小柔`.
For voice cloning, pass `ref_audio` instead of `voice`, with optional `ref_text`.
Inline and registered reference audio accepts up to 16 clips per request,
with clip order and sample rates preserved. The clip count is checked before
loading inline references, and each decoded clip must be nonempty and finite
with a positive sample rate. There is no additional aggregate duration or sample-count
limit; the media loader's existing size/duration protections and the Speech API's
existing 1–30 second limit per clip still apply.
A registered `voice` name selects its stored reference audio and transcript.
An unregistered string is interpreted as an IP caption by this standalone endpoint.
The existing
`ming_flash_omni_tts.yaml` profile retains Ming2's nested `talker_config` layout
and its registered `voice` presets. Converted Ming3 deployments previously using
that Ming2 profile must select `ming3_audio_tts.yaml`.

## Registered voices

Checkpoints may provide `voice_presets/voice_prompts.json`, mapping each name
to `{"audio": "voice_audio/example.wav", "text": "reference transcript"}`.
Paths are relative to `voice_presets/`. These presets are imported into the
existing Speech API voice store at startup. A registered voice is available
only when included in the selected checkpoint or registered by the user. Query the running server before choosing a voice:

```bash
curl http://localhost:8091/v1/audio/voices

curl --fail-with-body http://localhost:8091/v1/audio/voices \
    -F 'audio_sample=@/path/to/reference.wav' \
    -F 'name=my_voice' \
    -F 'consent=your-consent-recording-id' \
    -F 'ref_text=The words spoken in the reference recording.'
```

Use the returned `id` in Chat Completions as
`"audio": {"format": "wav", "voice": {"id": "my_voice"}}`, or as `voice` in
`/v1/audio/speech`. Chat and Realtime match IDs against the Ming3 IP voice list first; all other
IDs must be registered. Realtime accepts `session.audio.output.voice: {"id": "my_voice"}`.
Do not combine a stored voice with per-request reference audio or a speaker
embedding. Runtime uploads take precedence over model presets with the same
name.

IP names are reserved (case insensitive): uploads and checkpoint presets must
use different names. `GET /v1/audio/voices` includes these built-in IDs.
After an upgrade, restored uploads with reserved names are excluded from the
voice registry with a warning; their files remain on disk. Re-register those
samples under different names to select them through any speech endpoint.
Checkpoint manifests containing reserved names fail validation before any
preset is imported. Standalone TTS, Chat and Realtime all prefer the built-in
IP and preserve its canonical caption, such as `US_Stella`.

Set `SPEAKER_SAMPLES_DIR` to a writable directory to persist uploaded voices
and imported presets across restarts. Voice registrations are loaded into each
server process. For multiple instances, use consistent presets and storage,
and ensure registrations reach the instance that serves subsequent requests.
Runtime voice uploads are rejected when using multiple API workers.

## Manual-turn Realtime

Connect to `ws://localhost:8091/v1/realtime?model=ming3-audio-flash` when
serving the full Thinker/Talker profile. After `session.created`, send:

```json
{"type":"session.update","session":{"audio":{"input":{"turn_detection":null},"output":{"voice":{"id":"my_voice"}}},"max_output_tokens":128}}
```

Wait for `session.updated`. To submit a text turn, send the following events
in order, waiting for `conversation.item.created` before `response.create`:

```json
{"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"Say hello briefly."}]}}
{"type":"response.create"}
```

For an audio turn, send `input_audio_buffer.append` events with base64-encoded
mono PCM16 little-endian audio at 24 kHz, then `input_audio_buffer.commit`.
Alternatively, `conversation.item.create` accepts an `input_audio` content part
with base64 `audio` and `format: "pcm16"` (the default) or `format: "wav"`.
Direct audio items must be nonempty and respect the configured maximum audio
file size. PCM is converted to WAV for Chat preprocessing; the WAV container
overhead counts toward this limit. Leave room for it when forming a PCM buffer.
Oversized items are rejected before entering conversation history.
Wait for `conversation.item.created` and send `response.create` to request an
answer. Committing input alone does not start generation.

`response.output_audio.delta` carries base64 PCM16 mono audio at 24 kHz;
`response.output_audio_transcript.delta` carries the answer text. Read through
`response.done`. This transport resamples the Talker's native 44.1 kHz output;
standalone Speech API WAV output retains 44.1 kHz.

`response.cancel` cancels generation. Voice changes are rejected after the
session has emitted audio. This adapter supports manual turns only: automatic
VAD, `conversation.item.truncate` and `conversation.item.delete` are not
implemented. It does not implement the full OpenAI Realtime protocol.
