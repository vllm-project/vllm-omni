# Fun-Audio-Chat-8B

## Overview

Fun-Audio-Chat-8B is a multimodal speech dialogue model developed by Alibaba's FunAudioLLM team, supporting:

- **Speech-to-Text (S2T)**: Speech understanding and transcription
- **Speech-to-Speech (S2S)**: End-to-end voice conversation (audio in, audio out)

## Model Architecture

Fun-Audio-Chat uses a 3-stage pipeline for S2S functionality:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Fun-Audio-Chat S2S Pipeline                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Stage 0: Main Model (Fun-Audio-Chat-8B)                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Input: Audio (16kHz WAV) + Text Prompt                              │   │
│  │ Process: Audio Encoder → Qwen3-8B LLM → Text Generation             │   │
│  │ Output: Text Response + Hidden States (for S2S)                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                               ↓                                             │
│  Stage 1: CRQ Decoder (Residual Quantization Decoder)                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Input: Hidden States from Stage 0                                   │   │
│  │ Process: CRQ Transformer (28 layers, 1024 hidden) → Speech Tokens   │   │
│  │ Output: Speech Tokens @ 25Hz (group_size=5, codebook=6565)          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                               ↓                                             │
│  Stage 2: CosyVoice (Token-to-Waveform Synthesis)                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Input: Speech Tokens from Stage 1                                   │   │
│  │ Process: CosyVoice3-0.5B Flow Matching → Audio Waveform             │   │
│  │ Output: Audio Waveform @ 24kHz                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Hardware Requirements

| Mode                     | GPU Configuration | VRAM Required            |
| ------------------------ | ----------------- | ------------------------ |
| S2T (understanding only) | 1x GPU            | ~20GB                    |
| S2S (full pipeline)      | 1x GPU            | ~35GB                    |
| S2S (recommended)        | 2x GPU            | GPU0: ~20GB, GPU1: ~15GB |

**Tested on:**

- 2x NVIDIA H100 80GB
- 1x NVIDIA A100 80GB (single-card S2S)

## Installation

### 1. Install vLLM-Omni

```bash
cd /path/to/vllm-omni
pip install -e .
```

### 2. Install CosyVoice (S2S mode only)

```bash
# Clone CosyVoice repository
git clone https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice

# Install dependencies
pip install -e .

# Install Matcha-TTS
cd third_party/Matcha-TTS
pip install -e .
```

Or set environment variable pointing to existing CosyVoice installation:

```bash
export COSYVOICE_PATH=/path/to/CosyVoice
```

### 3. Download Models

Models will be downloaded automatically on first run. You can also pre-download:

```bash
# Use China mirror for faster downloads (optional)
export HF_ENDPOINT=https://hf-mirror.com

# Download Fun-Audio-Chat-8B
huggingface-cli download FunAudioLLM/Fun-Audio-Chat-8B

# Download CosyVoice3 (S2S mode only)
huggingface-cli download FunAudioLLM/Fun-CosyVoice3-0.5B-2512
```

## Quick Start

### Speech-to-Text (S2T) Mode

Speech understanding only, outputs text:

```bash
cd examples/offline_inference/fun_audio_chat

# Use default audio
bash run_s2t.sh

# Use custom audio
python end2end.py --mode s2t --audio-path /path/to/audio.wav
```

### Speech-to-Speech (S2S) Mode

Full voice conversation, audio in → audio out:

```bash
cd examples/offline_inference/fun_audio_chat

# Single GPU mode (requires ~35GB VRAM)
bash run_s2s.sh

# Dual GPU mode (recommended)
bash run_s2s_2gpu.sh

# Use custom audio
python end2end.py --mode s2s --audio-path /path/to/audio.wav --output-dir output
```

## Command Line Arguments

```bash
python end2end.py [OPTIONS]

Options:
  --mode              Run mode: s2t (speech-to-text) or s2s (speech-to-speech)
                      Default: s2t

  --audio-path, -a    Input audio file path (supports WAV, MP3, FLAC)
                      Default: uses built-in test audio

  --output-dir        Output directory
                      Default: output

  --model-path        Fun-Audio-Chat model path
                      Default: FunAudioLLM/Fun-Audio-Chat-8B

  --cosyvoice-path    CosyVoice3 model path (S2S mode only)
                      Default: auto-download

  --stage-configs     Custom stage config file path
                      Default: auto-select s2t or s2s config

  --speaker           Speaker name or embedding path
                      Default: 中文女

  --sampling-rate     Input audio sampling rate
                      Default: 16000

  --temperature       Generation temperature
                      Default: 0.7

  --max-tokens        Maximum tokens to generate
                      Default: 2048

  --seed              Random seed
                      Default: 42

  --enable-stats      Enable detailed statistics logging
                      Default: False
```

## Configuration Files

### S2T Config (1 stage)

```yaml
# vllm_omni/model_executor/stage_configs/fun_audio_chat.yaml
stage_args:
  - stage_id: 0
    runtime:
      devices: "0"
    engine_args:
      model_arch: FunAudioChatForConditionalGeneration
      gpu_memory_utilization: 0.8
      engine_output_type: token
```

### S2S Config (3 stages)

```yaml
# vllm_omni/model_executor/stage_configs/fun_audio_chat_s2s.yaml
stage_args:
  - stage_id: 0 # Main: audio understanding
    runtime:
      devices: "0"
    engine_args:
      model_arch: FunAudioChatForConditionalGeneration
      gpu_memory_utilization: 0.7
      engine_output_type: latent # output hidden states

  - stage_id: 1 # CRQ: speech token generation
    runtime:
      devices: "1" # can be on different GPU
    engine_args:
      model_arch: FunAudioChatCRQDecoder

  - stage_id: 2 # CosyVoice: speech synthesis
    runtime:
      devices: "1"
    engine_args:
      model_arch: FunAudioChatCosyVoice
```

## Output Format

### S2T Mode

```
output/
├── request_0.txt     # Transcribed text
└── request_0.json    # Full metadata
```

### S2S Mode

```
output/
├── request_0.txt     # Transcribed text
├── request_0.wav     # Generated audio (24kHz)
└── request_0.json    # Full metadata
```

## FAQ

### Q: librosa backend error?

Install ffmpeg:

```bash
sudo apt update
sudo apt install ffmpeg
```

### Q: Out of memory?

1. Use S2T mode (only needs ~20GB)
2. Use dual-GPU configuration
3. Lower `gpu_memory_utilization`

### Q: CosyVoice import failed?

Ensure paths are set correctly:

```bash
export COSYVOICE_PATH=/path/to/CosyVoice
export PYTHONPATH=$COSYVOICE_PATH:$COSYVOICE_PATH/third_party/Matcha-TTS:$PYTHONPATH
```

### Q: Model download timeout?

Use China mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## References

- [Fun-Audio-Chat Official Repo](https://github.com/FunAudioLLM/Fun-Audio-Chat)
- [Fun-Audio-Chat-8B Model](https://huggingface.co/FunAudioLLM/Fun-Audio-Chat-8B)
- [CosyVoice3 Model](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- [CosyVoice Repo](https://github.com/FunAudioLLM/CosyVoice)

## License

Apache 2.0
