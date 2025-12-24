# Fun-Audio-Chat-8B Online Serving

## Overview

This example demonstrates how to deploy Fun-Audio-Chat-8B as an online service with vLLM-Omni.

Supports:

- **S2T (Speech-to-Text)**: Audio understanding and transcription via OpenAI-compatible API
- **S2S (Speech-to-Speech)**: Full voice conversation with audio response
- **Gradio Demo**: Interactive web interface for voice chat

## Quick Start

### 1. Launch the Server

```bash
# S2T mode (default)
vllm serve FunAudioLLM/Fun-Audio-Chat-8B --omni --port 8091

# S2S mode (with speech synthesis)
vllm serve FunAudioLLM/Fun-Audio-Chat-8B --omni --port 8091 \
    --stage-configs-path /path/to/fun_audio_chat_s2s.yaml
```

Or use the provided scripts:

```bash
cd examples/online_serving/fun_audio_chat

# S2T mode
bash run_server_s2t.sh

# S2S mode
bash run_server_s2s.sh
```

### 2. Send Requests

#### Via Python Client

```bash
# Text query
python openai_chat_completion_client.py --query-type text

# Audio query
python openai_chat_completion_client.py --query-type use_audio

# With custom audio file
python openai_chat_completion_client.py --query-type use_audio --audio-path /path/to/audio.wav
```

#### Via curl

```bash
bash run_curl_request.sh use_audio
```

#### Via Gradio Demo

```bash
# Launch Gradio web interface
bash run_gradio_demo.sh
```

Then open http://localhost:7861 in your browser.

## API Endpoints

### Chat Completions (OpenAI-compatible)

```
POST /v1/chat/completions
```

#### Text Request

```json
{
  "model": "FunAudioLLM/Fun-Audio-Chat-8B",
  "messages": [
    { "role": "system", "content": "You are a helpful voice assistant." },
    { "role": "user", "content": "Hello!" }
  ],
  "max_tokens": 2048,
  "temperature": 0.7
}
```

#### Audio Request

```json
{
  "model": "FunAudioLLM/Fun-Audio-Chat-8B",
  "messages": [
    { "role": "system", "content": "You are a helpful voice assistant." },
    {
      "role": "user",
      "content": [
        { "type": "audio", "audio_url": "data:audio/wav;base64,..." },
        { "type": "text", "text": "What did you hear?" }
      ]
    }
  ],
  "max_tokens": 2048
}
```

#### S2S Response (with audio output)

```json
{
  "id": "chatcmpl-xxx",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "I heard..."
      },
      "multimodal_output": {
        "audio": "base64-encoded-wav"
      }
    }
  ]
}
```

## Command Line Arguments

### Python Client

```bash
python openai_chat_completion_client.py [OPTIONS]

Options:
  --query-type, -q    Query type: text, use_audio
                      Default: use_audio

  --audio-path, -a    Path to local audio file or URL
                      Default: uses built-in test audio

  --prompt, -p        Custom text prompt
                      Default: auto-generated

  --model, -m         Model name
                      Default: FunAudioLLM/Fun-Audio-Chat-8B

  --api-base          API base URL
                      Default: http://localhost:8091/v1

  --output-dir        Output directory for audio files
                      Default: output
```

### Gradio Demo

```bash
python gradio_demo.py [OPTIONS]

Options:
  --model             Model name
                      Default: FunAudioLLM/Fun-Audio-Chat-8B

  --api-base          API base URL
                      Default: http://localhost:8091/v1

  --ip                Host/IP for Gradio
                      Default: 127.0.0.1

  --port              Port for Gradio
                      Default: 7861

  --share             Share publicly via Gradio link
                      Default: False
```

## Server Configuration

### S2T Mode (1 stage)

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

### S2S Mode (3 stages)

```yaml
# vllm_omni/model_executor/stage_configs/fun_audio_chat_s2s.yaml
stage_args:
  - stage_id: 0 # Main: audio understanding
    runtime:
      devices: "0"
    engine_args:
      model_arch: FunAudioChatForConditionalGeneration
      engine_output_type: latent

  - stage_id: 1 # CRQ: speech token generation
    runtime:
      devices: "1"
    engine_args:
      model_arch: FunAudioChatCRQDecoder

  - stage_id: 2 # CosyVoice: speech synthesis
    runtime:
      devices: "1"
    engine_args:
      model_arch: FunAudioChatCosyVoice
```

## Hardware Requirements

| Mode              | GPU Configuration | VRAM Required            |
| ----------------- | ----------------- | ------------------------ |
| S2T               | 1x GPU            | ~20GB                    |
| S2S               | 1x GPU            | ~35GB                    |
| S2S (recommended) | 2x GPU            | GPU0: ~20GB, GPU1: ~15GB |

## FAQ

### Q: Server fails to start?

Check if the port is already in use:

```bash
lsof -i :8091
```

### Q: Audio response is empty in S2S mode?

Ensure CosyVoice is properly installed:

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
- [vLLM-Omni Documentation](../../../README.md)

## License

Apache 2.0
