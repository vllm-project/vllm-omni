# Miso TTS Installation

## Prerequisites

- Linux system with CUDA GPU
- Python 3.10+
- `uv` package manager

## Installation Steps

1. **Install system dependencies** (required for moshi):
   ```bash
   sudo apt-get install python3-dev
   ```

2. **Install vLLM**:
   ```bash
   uv pip install vllm==0.23.0 --torch-backend=auto
   ```

3. **Install vLLM-Omni with Miso TTS dependencies** (from the repo root):
   ```bash
   uv pip install -e ".[miso_tts]"
   ```

4. **Login to Hugging Face** (required for Miso TTS model):
   ```bash
   huggingface-cli login
   ```

## Running the Server

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve MisoLabs/MisoTTS \
  --omni \
  --port 8091
```

`vllm_omni/deploy/miso_tts.yaml` is loaded by default; pass `--deploy-config <path>` to override it.

## Making a Request

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MisoLabs/MisoTTS",
    "input": "Hello from Miso TTS",
    "voice": "0"
  }' \
  --output output.wav
```

## Notes

- Miso TTS requires accepting the model license on Hugging Face before use
- The model uses the `moshi` library for codec operations
- `voice` is the speaker id as a string (`"0"` to `"9"`, default `"0"`); `max_new_tokens` caps generated frames (80 ms each)
