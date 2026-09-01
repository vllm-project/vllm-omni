# Qwen3-TTS

> Text-to-speech serving (CustomVoice / VoiceDesign / Base)

## Summary

- Vendor: Qwen
- Model: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` (and VoiceDesign / Base variants)
- Task: Text-to-speech with predefined voices, voice design, or voice cloning
- Mode: Online serving with the OpenAI-compatible `/v1/audio/speech` API
- Maintainer: Community

## When to use this recipe

Use this recipe when you want a known-good starting point for serving Qwen3-TTS
models with vLLM-Omni and validate the deployment with the existing TTS client
examples in this repository.

Qwen3-TTS supports three task types, each backed by a dedicated model checkpoint:

| Task Type     | Model                                    | Description                                                   |
| ------------- | ---------------------------------------- | ------------------------------------------------------------- |
| `CustomVoice` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`   | Predefined speaker voices with optional style/emotion control |
| `VoiceDesign` | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`   | Generate speech from a natural language voice description     |
| `Base`        | `Qwen/Qwen3-TTS-12Hz-1.7B-Base`          | Voice cloning from reference audio + transcript               |

Smaller 0.6B variants are also available for `CustomVoice` and `Base`.

## References

- Related examples under `examples/`:
  [`examples/online_serving/text_to_speech/qwen3_tts/`](../../examples/online_serving/text_to_speech/qwen3_tts/),
  [`examples/offline_inference/text_to_speech/qwen3_tts/`](../../examples/offline_inference/text_to_speech/qwen3_tts/)
- Related issue or discussion:
  [RFC: add recipes folder](https://github.com/vllm-project/vllm-omni/issues/2645)

## Environment

- OS: Linux
- Python: 3.10+
- vLLM / vLLM-Omni: use versions from your current checkout, >=0.20.0

## Command

Start the server from the repository root. Pick the model that matches your
task type:

```bash
# CustomVoice (predefined speakers with optional style control)
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --deploy-config vllm_omni/deploy/qwen3_tts.yaml \
    --omni --port 8091

# VoiceDesign (natural language voice description)
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
    --deploy-config vllm_omni/deploy/qwen3_tts.yaml \
    --omni --port 8091

# Base (voice cloning from reference audio)
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --deploy-config vllm_omni/deploy/qwen3_tts.yaml \
    --omni --port 8091
```

Alternatively, use the convenience script:

```bash
./examples/online_serving/text_to_speech/qwen3_tts/run_server.sh                  # Default: CustomVoice
./examples/online_serving/text_to_speech/qwen3_tts/run_server.sh VoiceDesign      # VoiceDesign
./examples/online_serving/text_to_speech/qwen3_tts/run_server.sh Base             # Base (voice clone)
```

The bundled deploy config (`vllm_omni/deploy/qwen3_tts.yaml`) enables async
chunking for low first-audio latency. For advanced deployment tuning, pass a
custom deploy config:

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --deploy-config /path/to/your_qwen3_tts_overrides.yaml \
    --omni --port 8091 --trust-remote-code
```

## Verification

**Quick smoke test with curl (CustomVoice):**

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, how are you?",
        "voice": "vivian",
        "language": "English"
    }' --output output.wav
```

**CustomVoice with emotion instruction:**

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "I am so excited!",
        "voice": "vivian",
        "instructions": "Speak with great enthusiasm"
    }' --output excited.wav
```

**List available voices:**

```bash
curl http://localhost:8091/v1/audio/voices
```

**Using the Python client:**

```bash
cd examples/online_serving/text_to_speech/qwen3_tts

# CustomVoice
python openai_speech_client.py \
    --text "Hello, how are you?" \
    --speaker vivian --language English

# VoiceDesign (requires VoiceDesign model)
python openai_speech_client.py \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
    --task-type VoiceDesign \
    --text "Hello world" \
    --instructions "A warm, friendly female voice"

# Base / voice clone (requires Base model)
python openai_speech_client.py \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --task-type Base \
    --text "Hello, this is a cloned voice" \
    --ref-audio /path/to/reference.wav \
    --ref-text "Transcript of the reference audio"
```

**Streaming audio (low latency):**

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, how are you?",
        "voice": "vivian",
        "language": "English",
        "stream": true,
        "stream_format": "audio",
        "response_format": "pcm"
    }' --no-buffer | play -t raw -r 24000 -e signed -b 16 -c 1 -
```

**Offline inference (no server needed):**

```bash
python examples/offline_inference/text_to_speech/qwen3_tts/end2end.py --query-type CustomVoice
python examples/offline_inference/text_to_speech/qwen3_tts/end2end.py --query-type CustomVoice --streaming
```

## Notes

- Memory usage: The deploy config allocates `gpu_memory_utilization: 0.3` per stage (talker + code2wav share a single GPU). For the 0.6B variants or constrained GPUs, adjust via `--gpu-memory-utilization`.
- Key flags: `--omni` is required. `--deploy-config` points to the bundled two-stage pipeline config.
- Async chunking: Enabled by default in `qwen3_tts.yaml` for streaming-friendly first-audio latency. Raw audio streaming requires `stream=true`, `stream_format="audio"`, and `response_format="pcm"`.
- Task/model matching: Each task type requires its matching model checkpoint. Using a CustomVoice model for a Base (voice clone) request will fail.
- Base codec termination: Base requests without an explicit `max_new_tokens`
  use a text-scaled safety ceiling (at least 192 codec frames and no more than
  the configured model limit). If the Talker reaches that ceiling without
  codec EOS, non-streaming serving discards the incomplete audio and retries
  once with a fresh seed; an explicit `seed` or `max_new_tokens` disables the
  retry. SSE and WebSocket clients receive structured errors and must discard
  previously emitted audio when the error contains `"action":"discard"`;
  raw PCM streams terminate with a connection error after any partial bytes
  already sent.
- Known limitations: The server serves one model variant at a time. To switch task types (e.g., CustomVoice to Base), restart the server with the corresponding model.

## Hardware Support

## GPU

### 1x RTX 4090 24GB (0.6B CustomVoice)

#### Environment

- OS: Ubuntu 24.04 LTS
- Python: 3.12.3
- PyTorch: 2.11.0+cu130
- Driver / runtime: NVIDIA 580.126.09 / CUDA 13.0
- GPU: NVIDIA GeForce RTX 4090, 24 GB
- vLLM version: 0.21.0
- vLLM-Omni version or commit: 0.21.0rc3.dev91+gd4c13950

#### Command

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
    --deploy-config vllm_omni/deploy/qwen3_tts.yaml \
    --omni --port 8091
```

The default deploy config (`qwen3_tts.yaml`) works without modification on the
RTX 4090. Both stages (talker + code2wav) share GPU 0 with
`gpu_memory_utilization: 0.3` each.

#### Verification

**English synthesis:**

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, this is Qwen3-TTS running on RTX 4090.",
        "voice": "vivian",
        "language": "English"
    }' --output test_english.wav
```

**Chinese synthesis:**

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "你好，这是在RTX 4090上运行的语音合成测试。",
        "voice": "vivian",
        "language": "Chinese"
    }' --output test_chinese.wav
```

**With emotion instruction:**

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "I am so excited about this!",
        "voice": "vivian",
        "language": "English",
        "instructions": "Speak with great enthusiasm"
    }' --output test_emotion.wav
```

#### Notes

- Memory usage: **~13.5 GiB / 24 GiB** at idle with the default deploy config
  (`gpu_memory_utilization: 0.3` per stage). The 0.6B model weights occupy only
  ~2.4 GiB (Stage 0: 1.91 GiB, Stage 1: 0.45 GiB); the remainder is KV cache
  pre-allocated at startup. To reduce idle footprint to ~5 GiB, use a custom
  deploy config with lower utilization (inference peak ~10 GiB):

  ```yaml
  # Custom deploy config for Qwen3-TTS-12Hz-0.6B on RTX 4090
  # Copy vllm_omni/deploy/qwen3_tts.yaml and override gpu_memory_utilization:
  stages:
    - stage_id: 0
      gpu_memory_utilization: 0.15
    - stage_id: 1
      gpu_memory_utilization: 0.15
  ```

### 1x AMD MI300X, 1.7B checkpoints

#### Environment

- OS: Linux 6.8.0-134-generic, x86_64
- Container: official ROCm image built from `docker/Dockerfile.rocm`
- Python: 3.12.13
- PyTorch: 2.11.0+gitd0c8b1f
- Driver / runtime: AMD 6.19.14.31400000 / ROCm 7.2.53211
- GPU: one AMD Instinct MI300X, `gfx942:sramecc+:xnack-`, 191.69 GiB visible HBM
- vLLM version: 0.27.0+rocm723
- vLLM Omni version or commit: `3fecb6953ca8dc51210cc0421ef24552267a41ef`
- Installed vLLM Omni package metadata: `0.27.0rc2.dev44+g55abdade9.rocm`
- ONNX Runtime: onnxruntime-rocm 1.22.2.post3 with `ROCMExecutionProvider`
- transformers: 5.15.0

#### Command

The MI300X tests also worked without `enforce_eager`. To use that setup, remove `enforce_eager: true` from stage 1 under `platforms.rocm` in `vllm_omni/deploy/qwen3_tts.yaml`.

Set this environment variable before you start Qwen3-TTS:

```bash
export MIOPEN_FIND_MODE=FAST
```

#### Verification

Each row records one offline request with one prompt and batch size one. Request time is the logged `e2e_wall_time_ms`, and stage 1 time is the logged `e2e_stage_1_wall_time_ms`.

| Checkpoint | Code2Wav configuration | `MIOPEN_FIND_MODE` | Request time | Stage 1 time | Output duration | Real time factor | Maximum sampled device memory |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| CustomVoice | Eager | Unset | 1.683 s | 1.613 s | 5.68 s | 0.30 | 60.91 GiB |
| CustomVoice | Graphs allowed | `FAST` | 1.326 s | 1.259 s | 5.68 s | 0.23 | 62.19 GiB |
| VoiceDesign | Eager | Unset | 1.195 s | 1.180 s | 4.40 s | 0.27 | 60.91 GiB |
| VoiceDesign | Graphs allowed | `FAST` | 1.127 s | 1.054 s | 4.40 s | 0.26 | 62.21 GiB |
| Base | Eager | Unset | 13.330 s | 13.263 s | 4.32 s | 3.09 | 61.74 GiB |
| Base | Graphs allowed | `FAST` | 14.314 s | 14.245 s | 4.80 s | 2.98 | 62.84 GiB |

All six processes finished successfully, and each output passed the 24 kHz mono WAV checks.

#### Notes

- The current ROCm config keeps Code2Wav in eager mode.
- The tested alternative allows graphs and sets [`MIOPEN_FIND_MODE=FAST`](https://rocm.docs.amd.com/projects/MIOpen/en/develop/reference/env_variables.html). MIOpen uses a saved kernel choice when available and uses its immediate fallback otherwise.
- CustomVoice used a graph for one of four logged Code2Wav batches. VoiceDesign used a graph for two of four batches. Base used no graphs for its four logged batches.
- The recorded request time with graphs allowed was lower for CustomVoice and VoiceDesign and higher for Base. The Base runs generated different amounts of audio, so their request times are not a direct comparison.
- Each setup was tested with one request, so the timing is not a performance benchmark.
- GPU memory was sampled once per second with `rocm-smi`.
