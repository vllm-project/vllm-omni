# Breeze-TTS-2

> Text-to-speech serving (plain / voice design / voice clone / voice direction)

## Summary

- Vendor: BreezeBlue
- Model: `BreezeBlue/Breeze-TTS-2` (`BreezeForConditionalGeneration`)
- Task: Text-to-speech with speaker tags, natural-language style instructions, single-reference voice cloning, and reference+instruction "voice direction"
- Mode: Online serving with the OpenAI-compatible `/v1/audio/speech` API (streaming and non-streaming); offline `Omni` example
- Hardware: 1x NVIDIA L20 48GB (CUDA)
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

## Supported model contract

| Item | Value |
| ---- | ----- |
| Tasks | Text-to-speech through `/v1/audio/speech` (online) and the offline `Omni` example |
| Modes | plain, voice design (`instructions`), voice clone (`ref_audio` + `ref_text`), voice direction (reference + `instructions`) |
| Languages | English and Chinese (model card) |
| Reference audio | exactly one clip per request with its exact transcript; encoded by the bundled Qwen3-TTS tokenizer to 16 codebooks at 12.5 Hz |
| Output | 24 kHz mono; `wav` or `pcm`; streaming SSE `speech.audio.delta` |
| Length | prompt bounded by stage-0 `max_model_len=4096`; synthesis bounded by `max_new_tokens` codec frames (default 2048, about 164 s) |
| Sampling | greedy (`temperature=0`) with `cfg_scale=1.0`; other guidance values and `negative_prompt` are rejected |
| Deployment profile | `vllm_omni/deploy/breeze_tts_2.yaml`: two stages on one GPU, async-chunk streaming with 8-frame codec chunks, stage 0 `gpu_memory_utilization=0.80`, stage 1 `0.15` |

## References

- Issue: [#6656 [New Model]: Breeze TTS 2](https://github.com/vllm-project/vllm-omni/issues/6656)
- Related examples under `examples/`:
  [`examples/online_serving/text_to_speech/breeze_tts_2/`](../../examples/online_serving/text_to_speech/breeze_tts_2/),
  [`examples/offline_inference/text_to_speech/breeze_tts_2/`](../../examples/offline_inference/text_to_speech/breeze_tts_2/)

## Hardware

- Accelerator model and per-device memory: 1x NVIDIA L20 48GB
- Number of devices: 1 (both stages are placed on device 0)
- Device interconnect: not applicable (single device)
- Host memory: no special requirement; per-request codec state is small
- Qualification scope: functional serving of all four prompt modes, streaming PCM, and 4 concurrent mixed-mode requests (greedy, `cfg_scale=1.0`). No other accelerator is qualified by this recipe.

## Software environment

- OS: Linux
- Python: 3.12.12
- Driver / runtime: NVIDIA driver 580.82.07, CUDA 13.0 (PyTorch 2.13.0+cu130)
- vLLM version: 0.28.0
- vLLM-Omni version or commit: the commit that introduced this recipe (branch based on `main` @ `67ef0b64`)
- Transformers: 5.14.1

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

## Supported features

| Feature | Status | Notes |
| ------- | ------ | ----- |
| Streaming output | ✓ | `async_chunk: true`, 8-frame inter-stage chunks, SSE `speech.audio.delta` PCM |
| Voice cloning | ✓ | one reference clip plus transcript per request |
| Voice design / voice direction | ✓ | `instructions` field, with or without a reference |
| Classifier-free guidance | ✗ | `cfg_scale` must be `1.0`; `negative_prompt` rejected (follow-up) |
| Non-greedy sampling | ✗ | greedy decoding only |
| Prefix caching | ✗ | disabled in the deploy config |
| Tensor / pipeline parallelism | untested | single-GPU profile only |
| Quantization | ✗ | not supported in this release |
| CUDA graphs | not tuned | depth decoder and stage-1 codec run eagerly |
| Platforms | CUDA | qualified on 1x NVIDIA L20 |
