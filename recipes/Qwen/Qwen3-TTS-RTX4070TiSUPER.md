# Qwen3-TTS 0.6B CustomVoice on RTX 4070 Ti SUPER 16GB

## Summary

- Vendor: Qwen
- Model: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
- Task: Text-to-speech
- Mode: Online serving
- Hardware: 1x NVIDIA GeForce RTX 4070 Ti SUPER, 16376 MiB
- Maintainer: Community

## When to use this recipe

Serving Qwen3-TTS on a 16 GB consumer card. The 0.6B CustomVoice checkpoint is
the smallest Qwen3-TTS deployment and runs with the bundled deploy config
unmodified, so no `gpu_memory_utilization` override is required at this size.

## Supported model contract

See [`Qwen3-TTS.md`](./Qwen3-TTS.md) for the task, voice, and language contract
shared by all Qwen3-TTS profiles. Only the 0.6B CustomVoice checkpoint was
qualified on this card; the 1.7B checkpoints were not measured here.

## References

- Shared recipe: [`Qwen3-TTS.md`](./Qwen3-TTS.md)
- Deploy config: [`vllm_omni/deploy/qwen3_tts.yaml`](../../vllm_omni/deploy/qwen3_tts.yaml)

## Hardware

- Accelerator model and per-device memory: NVIDIA GeForce RTX 4070 Ti SUPER, 16376 MiB
- Number of devices: 1
- Device interconnect: not applicable, single device
- Host memory: not relevant, no CPU offload in this profile
- Qualification scope: online serving of the 0.6B CustomVoice checkpoint, short
  prompts, up to 8 concurrent requests

## Software environment

- OS: Ubuntu 24.04.4 LTS (WSL2), kernel 6.6.114.1-microsoft-standard-WSL2
- Python: 3.12.3
- Driver / runtime: NVIDIA 610.60 / CUDA 13.2
- PyTorch: 2.13.0+cu132
- vLLM version: 0.29.0
- vLLM-Omni version or commit: `01a2f93256975c7ff9565c5414b0fe0225bc4765`

## Command

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
    --deploy-config vllm_omni/deploy/qwen3_tts.yaml \
    --omni --port 8091
```

The default deploy config runs unmodified on this card. Both stages (talker and
code2wav) share GPU 0 at `gpu_memory_utilization: 0.3` each, so the `0.15`
profile documented for the RTX 4090 is not needed here.

## Verification

**English synthesis:**

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, this is Qwen3-TTS running on RTX 4070 Ti SUPER.",
        "voice": "vivian",
        "language": "English"
    }' --output test_english.wav
```

**Chinese synthesis:**

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "你好，这是在RTX 4070 Ti SUPER上运行的语音合成测试。",
        "voice": "vivian",
        "language": "Chinese"
    }' --output test_chinese.wav
```

Both requests return 24 kHz mono 16-bit PCM WAV. Clip length varies between runs
because stage 0 samples with `temperature: 0.9` and `top_k: 50`; across eight
runs the English prompt produced between 6.3 s and 11.7 s of audio.

## Notes

- Memory usage: **9753 MiB / 16376 MiB** at idle and **9768 MiB / 16376 MiB**
  peak under a burst of 8 concurrent requests, with the default deploy config.
  These are whole-device totals; the desktop compositor and background
  applications on this host hold about 1.0 GiB, measured at 1045 MiB immediately
  before startup and 1015 MiB immediately after shutdown. The server therefore
  accounts for roughly 8.5 GiB, leaving about 6.4 GiB free.
- Engine-reported breakdown: stage 0 (talker) weights 1.91 GiB, KV cache
  2.01 GiB (18,816 tokens, 4.59x maximum concurrency at 4,096 tokens per
  request), CUDA graphs 0.17 GiB, peak activation 0.81 GiB; stage 1 (code2wav)
  weights 0.48 GiB, CUDA graphs 0.18 GiB.
- Startup reaches `Application startup complete` in about 80 s once the
  FlashInfer sampling-kernel cache is warm. The first run on a cold cache took
  roughly 2.5 minutes.
- Known limitations: the memory figures cover short prompts at a concurrency of
  8 over roughly 5 s of sampling; other text lengths and concurrency levels were
  not measured. Whisper transcription reproduces the sentence frame every time,
  but the model name and the digits in `4070` transcribe inconsistently, and
  phrase repetition appeared in 2 of 7 transcribed runs. These are observations
  only; no root cause was established and audio was not assessed by listening.
