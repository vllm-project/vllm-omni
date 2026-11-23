# GLM-TTS for Chinese/English TTS

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
vllm serve zai-org/GLM-TTS --omni --trust-remote-code --port 8091
```

Async chunking is enabled by default in the bundled deployment config. For
the sync (non-streaming) path:

```bash
vllm serve zai-org/GLM-TTS --omni --trust-remote-code --port 8091 --no-async-chunk
```

Use a custom deploy config for advanced cases:

```bash
vllm serve zai-org/GLM-TTS --omni --trust-remote-code --port 8091 \
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

### 2x non-standard RTX 4090 48GB

This configuration was personally validated on two non-standard RTX 4090 cards
reporting 49,140 MiB each in `nvidia-smi`. Stock 24GB RTX 4090 cards were not
validated.

#### Environment

- OS: Ubuntu 22.04.1 LTS
- NVIDIA driver: 580.76.05
- Python: 3.12.13
- PyTorch: 2.11.0+cu130
- CUDA runtime and JIT toolkit: 13.0
- vLLM: 0.26.0
- vLLM-Omni: commit `66a9b0c8`, version
  `0.26.0rc2.dev5+g66a9b0c84`
- FlashInfer: 0.6.14

The host's default `nvcc` was 11.8. The working CUDA toolkit came from the
Python environment, so `CUDA_HOME` was set to
`$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13`.

#### Command

From the repository root, start the async-chunk server with:

```bash
export CUDA_HOME="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

CUDA_VISIBLE_DEVICES=0,1 vllm-omni serve zai-org/GLM-TTS \
  --omni \
  --trust-remote-code \
  --port 8091 \
  --deploy-config vllm_omni/deploy/glm_tts.yaml \
  --stage-overrides '{"1":{"devices":"1"}}' \
  --allowed-local-media-path "${PWD}/tests/assets"
```

Stage 0 runs on GPU 0, while the override places stage 1 on GPU 1. For the sync
variant, add `--no-async-chunk` to the same command.

#### Verification

Run the Chinese voice-cloning check:

```bash
python examples/online_serving/text_to_speech/glm_tts/openai_speech_client.py \
  --model zai-org/GLM-TTS \
  --text "今天天气很好，我们一起学习语音模型推理。" \
  --ref-audio "file://${PWD}/tests/assets/glm_tts/jiayan_zh.wav" \
  --ref-text "他当时还跟线下其他的站姐吵架，然后，打架进局子了。" \
  --response-format wav \
  --output glm_tts_zh.wav
```

Run the English voice-cloning check with the same Chinese reference:

```bash
python examples/online_serving/text_to_speech/glm_tts/openai_speech_client.py \
  --model zai-org/GLM-TTS \
  --text "Hello, this is a voice cloning test with GLM-TTS." \
  --ref-audio "file://${PWD}/tests/assets/glm_tts/jiayan_zh.wav" \
  --ref-text "他当时还跟线下其他的站姐吵架，然后，打架进局子了。" \
  --response-format wav \
  --output glm_tts_en.wav
```

Inspect either output:

```bash
ffprobe -v error -select_streams a:0 \
  -show_entries stream=codec_name,sample_rate,channels \
  -of default=noprint_wrappers=1 glm_tts_zh.wav
```

Expected output:

```text
codec_name=pcm_s16le
sample_rate=24000
channels=1
```

The English smoke test produced a valid WAV with an actual duration of 5.020
seconds.

#### Observed warm-cache results

These observations are from one controlled run, not a hardware or performance
guarantee. Cold-start and JIT costs were excluded. Values are mean +/- standard
deviation.

##### Measurement procedure

- For the single-GPU server, use the documented command with
  `CUDA_VISIBLE_DEVICES=0` and omit `--stage-overrides`; both bundled stages
  remain on GPU 0. For the dual-GPU server, use the command as shown.
- Async chunking is the default. Add `--no-async-chunk` for sync mode.
- For non-streaming measurements, use the exact Chinese client command above.
  For streaming, add `--stream` to that command and change the output filename
  to `glm_tts_zh.pcm`; streaming output is raw PCM, not WAV.
- The fixed Chinese prompt and reference were exactly those shown in
  Verification and yielded 3.900 seconds of audio. Each case used one warm-up
  followed by three measured requests at concurrency 1.
- Wall time starts immediately before the HTTP POST and stops after the complete
  response body. TTFA runs from immediately before the POST to the first
  non-empty audio bytes. RTF is wall time divided by audio duration.
- Peak total device memory used was sampled every 0.2 seconds and reported by
  NVML via `nvmlDeviceGetMemoryInfo.used`. The readings include background and
  baseline use and are not baseline-subtracted.

| GPUs | Mode | Response | Wall time (s) | TTFA (s) | RTF |
|---:|---|---|---:|---:|---:|
| 1 | Async | Non-streaming | 2.144 +/- 0.076 | -- | 0.550 +/- 0.019 |
| 2 | Async | Non-streaming | 1.969 +/- 0.130 | -- | 0.505 +/- 0.033 |
| 1 | Async | Streaming | 2.135 +/- 0.093 | 1.245 +/- 0.178 | 0.548 +/- 0.024 |
| 2 | Async | Streaming | 1.891 +/- 0.006 | 1.170 +/- 0.042 | 0.485 +/- 0.002 |
| 1 | Sync | Non-streaming | 1.822 +/- 0.016 | -- | 0.467 +/- 0.004 |
| 2 | Sync | Non-streaming | 1.792 +/- 0.023 | -- | 0.459 +/- 0.006 |

Moving stage 1 to GPU 1 reduced async non-streaming wall time by 8.2%. It
reduced async streaming wall time by 11.4% and TTFA by 6.0%, and reduced sync
wall time by 1.6%. Single- and dual-GPU outputs were bitwise identical only
within matching execution and response modes.

| Configuration | GPU 0 | GPU 1 |
|---|---:|---:|
| Single-GPU async | 31.52 GiB | -- |
| Dual-GPU async | 29.23 GiB | 2.91 GiB |
| Single-GPU sync | 31.47 GiB | -- |
| Dual-GPU sync | 29.23 GiB | 2.91 GiB |
