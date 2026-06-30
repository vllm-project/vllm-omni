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

3. **Clone vLLM-Omni** (miso branch):
   ```bash
   git clone -b miso --single-branch https://github.com/Nightwing-77/vllm-omni.git
   cd vllm-omni
   ```

4. **Install vLLM-Omni with Miso TTS dependencies**:
   ```bash
   uv pip install -e ".[miso_tts]"
   ```

5. **Login to Hugging Face** (required for Miso TTS model):
   ```bash
   huggingface-cli login
   ```

## Running the Server

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve MisoLabs/MisoTTS \
  --omni \
  --port 8091 \
  --stage-config vllm_omni/deploy/miso_tts.yaml \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1
```

## Making a Request

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "miso_tts",
    "input": "Hello from Miso TTS",
    "voice": "0"
  }' \
  --output output.wav
```

## Notes

- Miso TTS requires accepting the model license on Hugging Face before use
- The model uses the `moshi` library for codec operations
- Default speaker ID is 0, valid range is 0-9
