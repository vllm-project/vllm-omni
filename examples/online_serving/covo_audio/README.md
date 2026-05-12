# Covo-Audio-Chat

## Installation

Please refer to [README.md](../../../README.md)

> **Note**
> Covo-Audio code2wav requires `torchdiffeq`. Install it with: `pip install torchdiffeq`

## Run examples (Covo-Audio-Chat)

### Launch the Server

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve tencent/Covo-Audio-Chat --omni \
    --trust-remote-code --port 18091
```

> The default deploy config is located at `vllm_omni/deploy/covo_audio.yaml` and is loaded automatically by the model registry — no `--deploy-config` flag needed for default use.

### Send Request

Get into the example folder:
```bash
cd examples/online_serving/covo_audio
```

#### Audio input chat

Using the default audio asset:
```bash
python openai_chat_completion_client.py
```

Using a custom audio file:
```bash
python openai_chat_completion_client.py --audio-path /path/to/audio.wav
```

#### Streaming mode

```bash
python openai_chat_completion_client.py --audio-path /path/to/audio.wav --stream
```

### Command-line Arguments

The Python client supports the following arguments:

- `--prompt` (or `-p`): Text prompt (default: `"请回答这段音频里的问题。"`)
- `--audio-path`: Path to audio file. Supports local file paths or HTTP/HTTPS URLs. Common audio formats: WAV, MP3, OGG, FLAC, M4A. Uses the default audio asset if omitted
- `--output-audio-path` (or `-o`): Output path for generated audio (default: `./audio_0.wav`)
- `--model`: Model name for the API. Auto-detected from server if omitted
- `--stream`: Enable streaming mode

## Pipeline

Covo-Audio-Chat uses a 2-stage pipeline:

- **Stage 0 (fused_thinker_talker):** The 7B LLM generates interleaved text and audio tokens in a single autoregressive pass.
- **Stage 1 (code2wav):** A BigVGAN-based vocoder converts the extracted audio codes into a 24kHz WAV waveform.

## Environment

- GPU: 1x H100 (80 GiB)
- Stage 0 (7B LLM): ~16 GiB VRAM
- Stage 1 (BigVGAN vocoder): ~2 GiB VRAM

## Important: System Prompt

Covo-Audio-Chat requires a specific system prompt to enable interleaved text+audio generation. Without it, Stage 0 produces text-only tokens, and the output audio will be **silent**.

The system prompt must include the instruction:

```
请用文本和音频进行对话，交替生成5个文本token和15个音频token，音频部分使用发音人：default_female
```

The full system prompt is defined in `vllm_omni/model_executor/models/covo_audio/prompt_utils.py` (`COVO_AUDIO_SYSTEM_PROMPT`).

## FAQ

If you encounter `ModuleNotFoundError: No module named 'librosa'`, install it with:
```bash
pip install librosa
```

### Audio output is silent (0.1s, all zeros)

This means Stage 0 did not generate any audio tokens. Most likely causes:

1. **Missing system prompt** -- see the "Important: System Prompt" section above. The model needs the interleaved generation instruction in the system message.
2. **Missing `ignore_eos: true` / `stop_token_ids: [151645]`** in Stage 0 sampling params -- without these, the model stops at `<|endoftext|>` before generating audio tokens.
