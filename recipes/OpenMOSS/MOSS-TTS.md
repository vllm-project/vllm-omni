# MOSS-TTS

## Summary

- Vendor: OpenMOSS
- Models: `OpenMOSS-Team/MOSS-TTS` (8B), `OpenMOSS-Team/MOSS-TTS-v1.5` (8B),
  `OpenMOSS-Team/MOSS-TTS-Realtime` (1.7B), `OpenMOSS-Team/MOSS-TTSD-v1.0` (8B),
  `OpenMOSS-Team/MOSS-SoundEffect` (8B), `OpenMOSS-Team/MOSS-VoiceGenerator` (1.7B)
- Task: Text-to-speech synthesis, sound effect generation, zero-shot voice design
- Mode: Online serving via the OpenAI-compatible `/v1/audio/speech` API; offline inference
- Maintainer: Community

## When to use this recipe

Use this recipe for 24 kHz multilingual TTS with voice cloning (20 languages including
Chinese and English). Choose a variant based on your latency and quality requirements:

| Model | Params | Use case |
|---|---|---|
| MOSS-TTS | 8B | General TTS, highest quality |
| MOSS-TTS-v1.5 | 8B | General TTS upgrade of 1.0: 31 languages, steadier cloning, `[pause Xs]` markers (set `language` for best results); same `MossTTSDelay` API |
| MOSS-TTS-Realtime | 1.7B | Lowest latency (TTFB ~180 ms), streaming-first |
| MOSS-TTSD-v1.0 | 8B | Multi-turn dialogue TTS |
| MOSS-SoundEffect | 8B | Sound effect synthesis from text description |
| MOSS-VoiceGenerator | 1.7B | Zero-shot voice design |

All variants share the same codec (`OpenMOSS-Team/MOSS-Audio-Tokenizer`, ~7 GB) and
output 24 kHz mono audio.

## References

- Offline inference example: [`examples/offline_inference/text_to_speech/moss_tts/`](../../examples/offline_inference/text_to_speech/moss_tts/)
- Deploy configs: [`vllm_omni/deploy/moss_tts.yaml`](../../vllm_omni/deploy/moss_tts.yaml) and variants
- HuggingFace org: <https://huggingface.co/OpenMOSS-Team>

## Hardware Support

### GPU

#### 1x H100 80GB — MOSS-TTS (8B)

##### Environment

- OS: Linux
- Python: 3.11+
- CUDA 12.8
- vLLM-Omni version: see `vllm_omni/__version__.py`

##### Command

```bash
# The codec is loaded automatically from OpenMOSS-Team/MOSS-Audio-Tokenizer.
# Override the path with MOSS_TTS_CODEC_PATH if you have a local copy.
vllm serve OpenMOSS-Team/MOSS-TTS --omni --port 8091
```

##### Verification

Voice cloning (provide a reference audio clip):

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "model": "OpenMOSS-Team/MOSS-TTS",
        "input": "Hello, this is a voice cloning test.",
        "voice": "default",
        "ref_audio": "https://raw.githubusercontent.com/OpenMOSS/MOSS-TTS/main/assets/audio/zh_1.wav",
        "response_format": "wav"
    }' --output output.wav
```

##### Notes

- Peak GPU memory: ~18 GB for the talker (8B) + ~8 GB for the codec decoder on the same device.
  Use `gpu_memory_utilization: 0.85` in `moss_tts.yaml` (default).
- Output: 24 kHz mono WAV.
- The `MOSS_TTS_CODEC_PATH` environment variable overrides the codec checkpoint location.

---

#### 1x A10G 24GB — MOSS-TTS-Realtime (1.7B)

##### Environment

- OS: Linux
- Python: 3.11+
- CUDA 12.8

##### Command

```bash
vllm serve OpenMOSS-Team/MOSS-TTS-Realtime --omni --port 8091
```

##### Verification

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "model": "OpenMOSS-Team/MOSS-TTS-Realtime",
        "input": "This is a low-latency streaming TTS test.",
        "voice": "default",
        "ref_audio": "https://raw.githubusercontent.com/OpenMOSS/MOSS-TTS/main/assets/audio/zh_1.wav",
        "response_format": "wav",
        "stream": true
    }' --output output.wav
```

##### Notes

- Peak GPU memory: ~6 GB for the talker (1.7B) + ~8 GB for the codec decoder.
- First-audio latency (TTFB): ~180 ms on A10G.
- `codec_chunk_frames: 15` in `moss_tts_realtime.yaml` for lower TTFA than the 8B variant.

---

#### 1x A10G 24GB — MOSS-SoundEffect (8B, sound effect synthesis)

##### Command

```bash
vllm serve OpenMOSS-Team/MOSS-SoundEffect --omni --port 8091
```

##### Verification

Sound effect synthesis takes a text description instead of reference audio:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "model": "OpenMOSS-Team/MOSS-SoundEffect",
        "input": "Thunder rumbling, rain pattering on a tin roof.",
        "response_format": "wav"
    }' --output thunder.wav
```

##### Notes

- No `ref_audio` required or accepted for MOSS-SoundEffect.
- Input field maps to the `ambient_sound` parameter in the upstream processor.
- Rate: ~12.5 tokens per second; longer descriptions produce longer audio.
