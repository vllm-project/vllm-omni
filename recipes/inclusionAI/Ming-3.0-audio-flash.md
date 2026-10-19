# Ming-3.0-audio-flash

> Online multilingual ASR and speech QA with optional spoken answers

## Summary

- Vendor: inclusionAI
- Model: `inclusionAI/Ming-3.0-audio-flash`
- Tasks: 16 kHz multilingual ASR and speech QA with text or 44.1 kHz audio output
- Mode: Online serving with the OpenAI-compatible Chat Completions API
- Maintainer: Community

## When to use this recipe

Use this recipe to transcribe multilingual audio or answer a question spoken in an audio clip.
ASR and Speech QA share the Ming3 audio thinker; requests for audio output additionally run the bundled Ming talker and AudioVAE.

## Serve

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

### 2 x H200

For two H200 GPUs, the following example selects TP=2:

Example ming3_audio_flash_2gpu_h200.yaml

```yaml
base_config: ming3_audio_flash.yaml

stages:
  - stage_id: 0
    devices: "0,1"
    tensor_parallel_size: 2
    gpu_memory_utilization: 0.88

  - stage_id: 1
    devices: "1"
    gpu_memory_utilization: 0.10
```

Launch the server with

```bash
vllm serve /path/to/inclusionAI/Ming-3.0-audio-flash \
    --omni \
    --deploy-config vllm_omni/deploy/ming3_audio_flash_2gpu_h200.yaml \
    --init-timeout 1800 \
    --stage-init-timeout 1800 \
    --served-model-name ming3-audio-flash \
    --port 8091
```

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

To retain original audio clips across turns, explicitly select the history profile:

```bash
vllm serve inclusionAI/Ming-3.0-audio-flash --omni \
    --deploy-config ming3_audio_flash_history.yaml \
    --served-model-name ming3-audio-flash --port 8091
```

This overlay permits up to 16 audio items per request, counted across all
messages, while inheriting the normal Flash profile. Other profiles keep their
existing item limits. Each clip retains its own feature length and placeholder;
clips may have different durations or repeat an earlier recording.

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

Select `ming3_audio_flash_history_cache.yaml` to enable native Thinker prefix
caching while retaining up to 16 original audio clips. This profile enables
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

### Optional Talker decode graphs and low-latency collection

`ming3_audio_flash_llm_graph.yaml` enables a CUDA Graph for the Talker's
single-token Qwen2 decode. Prefill remains eager; the outer Talker stage keeps
`cudagraph_mode: NONE`. This is independent of the existing CFM graphs.
The decode graph is disabled by default and when `enforce_eager` is selected.
Unsupported inputs or capture failures fall back to eager execution.

To combine 128K context, up to 16 original audio clips, native Thinker KV/KDA
prefix reuse and Talker decode graphs, select the packaged profile:

```bash
vllm serve inclusionAI/Ming-3.0-audio-flash --omni \
    --deploy-config ming3_audio_flash_multiturn_graph.yaml \
    --served-model-name ming3-audio-flash --port 8091 \
    --enable-prompt-tokens-details
```

This profile explicitly sets the context limit to 131072 and the prefill batch
limit to 32768. It keeps `low_latency_streaming: false` and the outer Talker
`cudagraph_mode: NONE`. All audio and text share the context limit. Select the
history-cache profile instead when Talker decode graphs are not required.

Cache checkpoints and graph buffers consume GPU memory; choose cache capacity
and context limits for the available devices. CUDA Graphs reset the Talker's
static KV state for each request and do not provide Talker conversation reuse.

`low_latency_streaming` defaults to `false` in all shipped profiles. Setting it
to `true` in the Talker's `hf_overrides` collects the first available audio
latent promptly, then resumes normal VAE batching. It is independent of prefix
caching and decode graphs. BF16 VAE results can vary with chunk boundaries;
even faster execution with this option disabled can change queue grouping.
Do not assume bitwise-identical waveforms across different chunk partitions.

### Optional 128K and Tiny profiles

Select these profiles explicitly with `--deploy-config`; adding the files does
not change automatic profile selection. The Flash overlays inherit the matching
base profile's execution mode, sampling controls and Talker configuration.

| Profile | GPUs / Thinker TP | Thinker context | Prefill token budget | Default output limit |
| --- | --- | --- | --- | --- |
| `ming3_audio_flash_128k.yaml` | 4 / 4 | 131072 | 32768 | 65536 |
| `ming3_audio_flash_asr_128k.yaml` | 4 / 4 | 131072 | 32768 | 32768 |
| `ming3_audio_flash_thinker_only_128k.yaml` | 4 / 4 | 131072 | 32768 | 65536 |
| `ming3_audio_tiny.yaml` | 1 / 1 | 131072 | 32768 | 65536 |
| `ming3_audio_tiny_asr.yaml` | 1 / 1 | 131072 | 32768 | 65536 |
| `ming3_audio_tiny_thinker_only.yaml` | 1 / 1 | 131072 | 32768 | 65536 |

For example, select the full Flash 128K profile:

```bash
vllm serve inclusionAI/Ming-3.0-audio-flash --omni \
    --deploy-config ming3_audio_flash_128k.yaml \
    --served-model-name ming3-audio-flash --port 8091
```

The context limit includes input and generated positions. The output limit is a
ceiling, not a target; set request `max_tokens` to the intended answer length.
Long inputs and a larger prefill budget can increase compute and working memory.
Validate memory capacity, warmup time and latency on the target GPUs; configuring
128K does not establish full-length performance or quality.

Tiny profiles require a compatible Tiny checkpoint, selected explicitly:

```bash
vllm serve /path/to/Ming-3.0-audio-tiny --omni \
    --deploy-config ming3_audio_tiny.yaml \
    --served-model-name ming3-audio-tiny --port 8091
```

They use the existing Ming3 pipeline and place both stages on GPU 0; the
thinker-only variant omits the Talker. The Tiny
ASR profile enables 10-gram repetition blocking and retains its own asynchronous
execution settings, rather than copying the Flash ASR execution mode.
These templates have configuration-resolution coverage, but have not been GPU
validated with a Tiny checkpoint. Check the checkpoint's context support and
single-GPU memory requirements before deployment.

## Transcribe with Chat Completions

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

## Ask a question in audio

Example inputs:

```bash
MING3_AUDIO_FILE=/path/to/inclusionAI/Ming-3.0-audio-flash/examples/audio_files/introduce_ant_group.wav
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
  temperature: 0,
  max_tokens: 1024
}' | curl http://localhost:8091/v1/chat/completions \
    -H "Content-Type: application/json" \
    --data-binary @- | jq -r '.choices[0].message.content'
```

## Speak the answer

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
  additional_information: {voice_mode: "ip", ip_name: "灵小柔"},
  temperature: 0,
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
- Ming3 IP voices use `additional_information: {"voice_mode": "ip", "ip_name": "灵小柔"}`. IP captions and Ming2 registered presets such as `DB30` are different controls.
- Ming3 reference mode accepts per-request `ref_audio` (a URL, data URL, or base64 audio) and optional `prompt_text`. Choose either reference audio or IP controls. Omitting the controls retains the configured reference fallback; `voice_name: null` does not disable it.
- `normalize_text` controls linguistic text normalization. `scrub_markup` independently removes HTML, code fences and math markup before synthesis and defaults to `true`. Set both fields to `false` to disable markup scrubbing and linguistic normalization. Streaming segmentation and language-prefix handling still apply. Keep these controls fixed for the duration of a streaming request.
- The talker emits mono 44.1 kHz audio.
- Use Chat Completions for transcription; the `/v1/audio/transcriptions` endpoint is not currently available with `--omni`.
- The model handles audio chunking internally, so the deployment configuration does not set `max_audio_clip_s` or `min_energy_split_window_size`.
- The default Flash profiles retain an 8192-token context limit. Select an explicit `*_128k.yaml` overlay for the larger limits documented above.

## Standalone text to speech

Converted Ming3 exports contain a `talker/` directory with the Talker config,
LLM tokenizer, AudioVAE and speaker extractor. Serve that export with the
Ming3-specific profile. The resolver accepts a local export or a Hub repository
with that layout; `--revision` selects the Hub snapshot. Download remote storage
exports into a local directory before serving.

```bash
vllm serve /path/to/converted-Ming3-export --omni \
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

For voice cloning, pass `ref_audio` instead of `voice`, with optional `ref_text`.
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
existing Speech API voice store at startup. Names such as `DB30` and
`lingguang` are available only when included in the selected checkpoint or
registered by the user. Query the running server before choosing a voice:

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
`/v1/audio/speech`. Chat and Realtime require a registered name for this form;
use the explicit IP controls documented above for an IP caption.
Do not combine a stored voice with per-request reference audio or a speaker
embedding. Runtime uploads take precedence over model presets with the same
name.

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
