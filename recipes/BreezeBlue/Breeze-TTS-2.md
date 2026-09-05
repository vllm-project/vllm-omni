# Breeze-TTS-2

> Text-to-speech serving (plain / voice design / voice clone / voice direction)

## Summary

- Vendor: BreezeBlue
- Model: `BreezeBlue/Breeze-TTS-2` (`BreezeForConditionalGeneration`)
- Task: Text-to-speech with speaker tags, natural-language style instructions, single-reference voice cloning, and reference+instruction "voice direction"
- Mode: Online serving with the OpenAI-compatible `/v1/audio/speech` API (streaming and non-streaming)
- Maintainer: Community

## When to use this recipe

Use this recipe when you want a known-good starting point for serving
Breeze-TTS-2 with vLLM-Omni. Breeze-TTS-2 is a two-stage AR TTS pipeline:

| Stage | Components | Output |
| ----- | ---------- | ------ |
| 0 (talker, `LLM_AR`) | T5Gemma2 text encoder + Qwen3 backbone + 2052-way codebook-0 head + 12-layer depth decoder | 16-codebook codec frames |
| 1 (codec, `LLM_GENERATION`) | Bundled Qwen3-TTS audio tokenizer decoder (Mimi fallback) | 24 kHz mono waveform |

Four prompt modes are selected automatically from the request fields:

| Mode | Trigger | Fields |
| ---- | ------- | ------ |
| `tts_plain` | only `input` (default `voice=S0`) | `input`, `voice` |
| `tts_instruction` (voice design) | `input` + `instructions` | style/emotion/pace directives, not spoken |
| `ref_clone_tata` (clone) | `ref_audio` + `ref_text` + `input` | reference clip and its exact transcript |
| `ref_edit_tata` (voice direction) | reference trio + `instructions` | keep the reference timbre, direct the delivery |

Current scope: greedy sampling with `cfg_scale=1.0`; CFG ≠ 1.0 and
`negative_prompt` are rejected with explicit client errors until companion
request support lands.

## References

- Issue: [#6656 [New Model]: Breeze TTS 2](https://github.com/vllm-project/vllm-omni/issues/6656)
- Related examples under `examples/`:
  [`examples/online_serving/text_to_speech/breeze_tts_2/`](../../examples/online_serving/text_to_speech/breeze_tts_2/),
  [`examples/offline_inference/text_to_speech/breeze_tts_2/`](../../examples/offline_inference/text_to_speech/breeze_tts_2/)

## Environment

- OS: Linux
- Python: 3.10+
- vLLM / vLLM-Omni: use versions from your current checkout
- GPU: verified on 1x NVIDIA L20; VRAM sizing follows the deploy YAML
  (stage 0 `gpu_memory_utilization` dominant, stage 1 codec is small)

## Command

Start the server from the repository root. The deploy config
(`vllm_omni/deploy/breeze_tts_2.yaml`, async chunk streaming by default)
auto-loads from the checkpoint's `model_type`; pass it explicitly if you
customized it:

```bash
vllm serve BreezeBlue/Breeze-TTS-2 \
    --deploy-config vllm_omni/deploy/breeze_tts_2.yaml \
    --omni --port 8091
```

## Verification

Plain synthesis (speaker tag `S0`..`S9`, default `S0`):

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "model": "BreezeBlue/Breeze-TTS-2",
        "input": "Hello, this is Breeze TTS 2 running on vLLM-Omni.",
        "voice": "S0",
        "response_format": "wav",
        "sample_rate": 24000
    }' --output breeze_plain.wav
```

Voice design (instruction only, no reference audio):

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "[laughs] Welcome to tonight's story time.",
        "instructions": "A warm young woman, clear voice, lively delivery.",
        "response_format": "wav"
    }' --output breeze_design.wav
```

Voice cloning (`ref_audio` and `ref_text` must be provided together):

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "This is new text spoken in the cloned voice.",
        "ref_audio": "file:///path/to/reference.wav",
        "ref_text": "The exact transcript of the reference audio.",
        "response_format": "wav"
    }' --output breeze_clone.wav
```

Voice direction (reference + instruction):

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "We need to discuss what happened last night.",
        "ref_audio": "file:///path/to/reference.wav",
        "ref_text": "The exact transcript of the reference audio.",
        "instructions": "Speak slowly with a restrained, serious tone.",
        "response_format": "wav"
    }' --output breeze_direction.wav
```

Streaming PCM (SSE `speech.audio.delta` events):

```bash
curl -N -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Streaming output from Breeze TTS 2.",
        "stream": true,
        "stream_format": "audio",
        "response_format": "pcm",
        "sample_rate": 24000
    }' --output breeze_stream.pcm
```

Expected results: finite, non-silent 24 kHz mono WAV files; streaming requests
return incremental PCM deltas. On 1x L20 (greedy, single request after
warm-up) a few seconds of speech completes in roughly 3.4–3.8 s wall time.

## Notes

- **Sample rate**: only `24000` is accepted; other values fail validation
  before inference.
- **Speed / seeds / multi-reference**: `speed` must be `1.0`;
  `task_type=VoiceDesign`, `ref_audio_2`, and speaker embeddings are not
  supported yet.
- **CFG**: `guidance_scale`/`cfg_scale` must be `1.0`. Non-1.0 values and
  `negative_prompt` return a client error.
- **Length budget**: prompt length is bounded by stage 0 `max_model_len=4096`;
  synthesis length by `max_new_tokens` (default 2048 frames ≈ 164 s at the
  12.5 Hz codec frame rate) — set `max_new_tokens` for shorter caps.
- **Checkpoint layout**: the checkpoint must contain the `audio_tokenizer/`
  subdirectory (bundled Qwen3-TTS codec) for the default streaming path; the
  Mimi fallback is used only when that directory is absent and does not
  support async-chunk streaming.
- **License**: the reference inference code is Apache 2.0; the checkpoints are
  distributed under the separate BreezeBlue Research and Non-Commercial
  License. Commercial use requires written authorization from BreezeBlue.
