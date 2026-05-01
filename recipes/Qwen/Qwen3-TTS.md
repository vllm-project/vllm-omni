# Qwen3-TTS for text-to-speech

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

| Task Type     | Model                                    | Description                                                  |
| ------------- | ---------------------------------------- | ------------------------------------------------------------ |
| `CustomVoice` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`  | Predefined speaker voices with optional style/emotion control |
| `VoiceDesign` | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`  | Generate speech from a natural language voice description     |
| `Base`        | `Qwen/Qwen3-TTS-12Hz-1.7B-Base`         | Voice cloning from reference audio + transcript              |

Smaller 0.6B variants are also available for `CustomVoice` and `Base`.

## References

- Upstream or canonical docs:
  [`docs/user_guide/examples/online_serving/qwen3_tts.md`](../../docs/user_guide/examples/online_serving/qwen3_tts.md)
- Related examples under `examples/`:
  [`examples/online_serving/qwen3_tts/`](../../examples/online_serving/qwen3_tts/),
  [`examples/offline_inference/qwen3_tts/`](../../examples/offline_inference/qwen3_tts/)
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
./examples/online_serving/qwen3_tts/run_server.sh                  # Default: CustomVoice
./examples/online_serving/qwen3_tts/run_server.sh VoiceDesign      # VoiceDesign
./examples/online_serving/qwen3_tts/run_server.sh Base             # Base (voice clone)
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
cd examples/online_serving/qwen3_tts

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
        "response_format": "pcm"
    }' --no-buffer | play -t raw -r 24000 -e signed -b 16 -c 1 -
```

**Offline inference (no server needed):**

```bash
python examples/offline_inference/qwen3_tts/end2end.py --query-type CustomVoice
python examples/offline_inference/qwen3_tts/end2end.py --query-type CustomVoice --streaming
```

## Notes

- Memory usage: The deploy config allocates `gpu_memory_utilization: 0.3` per stage (talker + code2wav share a single GPU). For the 0.6B variants or constrained GPUs, adjust via `--gpu-memory-utilization`.
- Key flags: `--omni` is required. `--deploy-config` points to the bundled two-stage pipeline config.
- Async chunking: Enabled by default in `qwen3_tts.yaml` for streaming-friendly first-audio latency. Streaming requires `stream=true` with `response_format="pcm"`.
- Task/model matching: Each task type requires its matching model checkpoint. Using a CustomVoice model for a Base (voice clone) request will fail.
- Known limitations: The server serves one model variant at a time. To switch task types (e.g., CustomVoice to Base), restart the server with the corresponding model.
