# GLM-TTS for Chinese/English TTS on 1x GPU

## Summary

- Vendor: zai-org
- Model: `zai-org/GLM-TTS`
- Task: Zero-shot voice-cloned text-to-speech synthesis
- Mode: Online serving with the OpenAI-compatible `/v1/audio/speech` API
- Maintainer: Community

## When to use this recipe

Use this recipe to serve GLM-TTS as a two-stage TTS system (AR + DiT
flow-matching) for Chinese and English speech synthesis. Every request is
conditioned on reference audio and its transcript.

## References

- Upstream or canonical docs:
  [zai-org/GLM-TTS on HuggingFace](https://huggingface.co/zai-org/GLM-TTS)
- GitHub repository:
  [zai-org/GLM-TTS](https://github.com/zai-org/GLM-TTS)
- Related example under `examples/`:
  [`examples/online_serving/text_to_speech/README.md#glm-tts`](../../examples/online_serving/text_to_speech/README.md#glm-tts)
- Offline inference example:
  [`examples/offline_inference/text_to_speech/README.md#glm-tts`](../../examples/offline_inference/text_to_speech/README.md#glm-tts)

## Hardware Support

### GPU

### 1x A40 48GB

#### Environment

- OS: Linux
- Python: 3.10+
- Driver / runtime: NVIDIA CUDA environment with A40 48GB GPU
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Command

Start the server from the repository root:

```bash
vllm-omni serve zai-org/GLM-TTS --omni --trust-remote-code --port 8091
```

Async chunking is enabled by default in the bundled deployment config. For
the sync (non-streaming) path:

```bash
vllm-omni serve zai-org/GLM-TTS --omni --trust-remote-code --port 8091 --no-async-chunk
```

Use a custom deploy config for advanced cases:

```bash
vllm-omni serve zai-org/GLM-TTS --omni --trust-remote-code --port 8091 \
  --deploy-config /path/to/your_glm_tts_overrides.yaml
```

#### Verification

Run the bundled OpenAI-compatible client with reference audio:

```bash
python examples/online_serving/text_to_speech/glm_tts/openai_speech_client.py \
  --text "你好，这是一个语音合成测试。" \
  --ref-audio file:///path/to/ref.wav \
  --ref-text "这是参考音频的文本内容。"
```

For a quick API smoke test:

```bash
curl http://localhost:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zai-org/GLM-TTS",
    "input": "你好，这是一个语音合成测试。",
    "response_format": "wav",
    "ref_audio": "file:///path/to/ref.wav",
    "ref_text": "这是参考音频的文本内容。"
  }' --output test.wav
```

Voice cloning with reference audio:

```bash
python examples/online_serving/text_to_speech/glm_tts/openai_speech_client.py \
  --text "你好，这是语音克隆测试。" \
  --ref-audio file:///path/to/ref.wav \
  --ref-text "这是参考音频的文本内容。"
```

#### Notes

- Hardware scope: the default bundled config is CUDA-only and verified on 1x A40 48GB (~16.6 GiB peak); fits 24GB cards. Split stages across GPUs for higher concurrency.
- Memory usage: ~18-20GB total (AR ~10GB, DiT ~8GB); both stages share GPU 0 by default.
- Audio output: 24kHz mono WAV via HiFT vocoder (Vocos2D 32kHz fallback with resampling).
- Key flags: `--omni` is required; `--trust-remote-code` is needed for the GLM-TTS phoneme tokenizer; the DiT stage enables bucketed model-internal CUDA graphs with eager fallback.
- Voice cloning: requires `ref_audio` + `ref_text` together. Reference audio should be 3-10 seconds. Feature extraction (WhisperVQ tokenizer, CampPlus ONNX, mel) runs on the model side.
- Known limitations: First request may be slow due to lazy model loading (WhisperVQ, CampPlus ONNX). Warm-cache RTF is approximately 0.6-0.7x on A40.
