# MiMo-Audio

> Online serving for multimodal audio chat, speech understanding, and audio generation

## Summary

- Vendor: XiaomiMiMo
- Model: `XiaomiMiMo/MiMo-Audio-7B-Instruct`
- Task: Multimodal chat with text and audio input; text and/or audio output; TTS-style audio generation
- Mode: Online serving with the OpenAI-compatible `/v1/chat/completions` API
- Maintainer: Community

## When to use this recipe

Use this recipe when you want to serve MiMo-Audio for audio understanding,
text-only chat, or generated speech responses. The bundled deployment runs
MiMo-Audio as a two-stage pipeline: Stage 0 is the fused thinker/talker that
produces text and RVQ audio codes, and Stage 1 is the Code2Wav decoder that
turns those codes into 24 kHz waveform audio.

## References

- Upstream model:
  [`XiaomiMiMo/MiMo-Audio-7B-Instruct`](https://huggingface.co/XiaomiMiMo/MiMo-Audio-7B-Instruct)
- Audio tokenizer:
  [`XiaomiMiMo/MiMo-Audio-Tokenizer`](https://huggingface.co/XiaomiMiMo/MiMo-Audio-Tokenizer)
- Online serving example:
  [`examples/online_serving/mimo_audio/README.md`](../../examples/online_serving/mimo_audio/README.md)
- Offline inference example:
  [`examples/offline_inference/mimo_audio/README.md`](../../examples/offline_inference/mimo_audio/README.md)
- Deploy config:
  [`vllm_omni/deploy/mimo_audio.yaml`](../../vllm_omni/deploy/mimo_audio.yaml)
- Chat template:
  [`examples/online_serving/mimo_audio/chat_template.jinja`](../../examples/online_serving/mimo_audio/chat_template.jinja)
- Related issue or discussion:
  [vllm-project/vllm-omni#2645](https://github.com/vllm-project/vllm-omni/issues/2645)

## Hardware Support

This recipe records MiMo-Audio configurations that were personally validated.
Additional GPU, ROCm, NPU, and XPU sections can be added as community
validation lands.

## GPU

### 1x A100 80GB PCIe

#### Environment

- OS: Linux 6.8.0-124-generic with glibc 2.39
- Python: 3.12.13
- GPU: NVIDIA A100 80GB PCIe
- Driver / runtime: NVIDIA driver 580.126.20, CUDA runtime 13.0.88
- PyTorch: 2.11.0+cu130
- vLLM version: 0.24.0
- vLLM-Omni version or commit: 0.24.0 source checkout
- Transformers: 5.12.1

#### Command

Start the server from the repository root:

```bash
hf download XiaomiMiMo/MiMo-Audio-Tokenizer \
  --local-dir /workspace/models/MiMo-Audio-Tokenizer

export MIMO_AUDIO_TOKENIZER_PATH=/workspace/models/MiMo-Audio-Tokenizer

vllm serve XiaomiMiMo/MiMo-Audio-7B-Instruct \
  --omni \
  --served-model-name MiMo-Audio-7B-Instruct \
  --deploy-config vllm_omni/deploy/mimo_audio.yaml \
  --chat-template examples/online_serving/mimo_audio/chat_template.jinja \
  --port 8091 \
  --log-stats
```

The bundled `mimo_audio.yaml` places both stages on GPU 0 with async-chunk
streaming through the shared-memory connector.

#### Verification

Text-only chat smoke test:

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiMo-Audio-7B-Instruct",
    "messages": [
      {"role": "user", "content": "Reply with the phrase: MiMo-Audio text smoke test passed."}
    ],
    "modalities": ["text"],
    "stream": false
  }'
```

Observed output:

```text
choices[0].message.content: MiMo-Audio text smoke test passed.
finish_reason: stop
usage: prompt_tokens=34, completion_tokens=29, total_tokens=63
```

Audio-generation smoke test using the bundled client:

```bash
cd examples/online_serving/mimo_audio

OPENAI_BASE_URL=http://localhost:8091/v1 \
python openai_chat_completion_client_for_multimodal_generation.py \
  --query-type text \
  --prompt "Read this sentence in a calm and friendly voice."
```

Observed output:

```text
[req 0_chatcmpl-9141929dee9dd90a] Chat completion output from text: Read this sentence in a calm and friendly voice.
[req 0_chatcmpl-9141929dee9dd90a] Audio saved to .//text/audio_0.wav
WAV: 24 kHz, mono, 16-bit PCM, 7.36 s
```

Multi-round audio input smoke test using the bundled base64 message fixture:

```bash
cd examples/online_serving/mimo_audio

OPENAI_BASE_URL=http://localhost:8091/v1 \
python openai_chat_completion_client_for_multimodal_generation.py \
  --query-type multi_audios \
  --message-json ../../offline_inference/mimo_audio/message_base64_wav.json
```

Observed output:

```text
[req 0_chatcmpl-a6297fc07f0f497d] Chat completion output from text: <non-empty Chinese response>
[req 0_chatcmpl-a6297fc07f0f497d] Audio saved to .//multi_audios/audio_0.wav
WAV: 24 kHz, mono, 16-bit PCM, 57.12 s
```

#### Notes
- Install `flash-attn` for your CUDA and PyTorch stack before validating audio
generation. On CUDA GPUs, missing or incompatible FlashAttention can make
generated audio noise-only even when the server appears healthy.
- `MIMO_AUDIO_TOKENIZER_PATH` is required and must point to a local snapshot of the tokenizer weights. In vLLM-Omni 0.24.0, MiMo checks this value with
  `os.path.exists()`, so a Hugging Face repo id is not sufficient.
- MiMo-Audio is not compatible with the default chat template. Always pass
  `--chat-template examples/online_serving/mimo_audio/chat_template.jinja`.
