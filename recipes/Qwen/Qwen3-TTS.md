# Qwen3-TTS — Text-to-Speech Recipe

**Model:** Qwen3-TTS series (Qwen/Qwen3-TTS-0.6B, Qwen/Qwen3-TTS-8B)
**Task:** Text-to-Speech (offline inference + online serving)
**Docs:** [Qwen3-TTS docs](https://vllm-omni.readthedocs.io)
**Examples:** [examples/offline_inference/qwen3_tts](../../examples/offline_inference/qwen3_tts/)

---

## Model Variants

| Model | Size | Task Types |
|---|---|---|
| Qwen/Qwen3-TTS-0.6B | 0.6B | CustomVoice, VoiceDesign, Base |
| Qwen/Qwen3-TTS-8B | 8B | CustomVoice, VoiceDesign, Base |

**Task types:**
- **CustomVoice** — generate speech with a known speaker identity
- **VoiceDesign** — generate speech from a descriptive voice instruction
- **Base** — voice cloning using a reference audio + transcript

---

## Environment

```bash
pip install vllm-omni
pip install qwen-tts
```

> **Note:** Requires `transformers >= 4.57.0` for auto voice.
> Voice cloning (Base task) requires `transformers >= 5.3.0`.

---

## Hardware: NVIDIA CUDA

### Tested configurations
- 1× A100 80GB (CUDA 12.4)
- 1× H100 80GB (CUDA 12.4)

Qwen3-TTS-0.6B fits comfortably on a single 16GB GPU.
Qwen3-TTS-8B requires at least 24GB VRAM.

---

## Offline Inference

```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni/examples/offline_inference/qwen3_tts

# CustomVoice — single sample
python end2end.py --query-type CustomVoice

# CustomVoice — batch
python end2end.py --query-type CustomVoice --use-batch-sample

# VoiceDesign — single sample
python end2end.py --query-type VoiceDesign

# VoiceDesign — batch
python end2end.py --query-type VoiceDesign --use-batch-sample

# Base (voice cloning) — ICL mode
python end2end.py --query-type Base --mode-tag icl
```

**Expected output:** WAV files written to the current directory, e.g.
`output_0_9cf29896-3ca9-4f93-a5c5-de7dac752561.wav`

---

## Online Serving

### Launch the server

```bash
# 0.6B variant
vllm serve Qwen/Qwen3-TTS-0.6B --omni --port 8091

# 8B variant
vllm serve Qwen/Qwen3-TTS-8B --omni --port 8091
```

### Make a request

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-0.6B",
    "input": "Hello, welcome to vLLM-Omni text-to-speech.",
    "voice": "default"
  }' \
  --output output.wav
```

**Verify:** `output.wav` should be a valid WAV file:

```bash
file output.wav
# output.wav: RIFF (little-endian) data, WAVE audio
```

---

## Important Flags

| Flag | Description |
|---|---|
| `--omni` | Required to enable omni-modality serving |
| `--port` | Server port (default 8000) |
| `--query-type` | Task type: `CustomVoice`, `VoiceDesign`, `Base` |
| `--use-batch-sample` | Run batch inference with multiple prompts |
| `--mode-tag icl` | Enable in-context learning mode for Base task |

---

## Known Limitations

- Online serving currently supports auto voice only; voice cloning (Base task) is offline only
- Multi-worker serving is not recommended; use single worker for stability
- Offline demos require `VLLM_USE_V1=0` and `VLLM_WORKER_MULTIPROC_METHOD=spawn`

---

## Links

- [Offline inference example](../../examples/offline_inference/qwen3_tts/)
- [Online serving TTS docs](../../docs/user_guide/examples/online_serving/text_to_speech.md)
- [Qwen3-TTS model card](https://huggingface.co/Qwen/Qwen3-TTS-8B)
- [QwenLM/Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
