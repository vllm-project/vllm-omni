# Fish Speech S2 Pro

## Summary

- Vendor: FishAudio
- Model: `fishaudio/s2-pro`
- Task: Text-to-speech synthesis with optional voice cloning
- Mode: Online serving with the OpenAI-compatible `/v1/audio/speech` API
- Maintainer: Community

## When to use this recipe

Use this recipe as a practical baseline for running `fishaudio/s2-pro` for
high-quality text-to-speech synthesis. Fish Speech S2 Pro outputs 44.1 kHz
audio and supports voice cloning from reference audio.

## References

- User guide: [`docs/user_guide/examples/online_serving/fish_speech.md`](../../docs/user_guide/examples/online_serving/fish_speech.md)
- Example guide: [`examples/online_serving/fish_speech/README.md`](../../examples/online_serving/fish_speech/README.md)
- Related issue or discussion:
  [RFC: add recipes folder](https://github.com/vllm-project/vllm-omni/issues/2645)

## Hardware Support

This recipe documents reference GPU configuration for Fish Speech S2 Pro.
Other hardware and configurations are welcome as community validation lands.

## GPU

### 1x A800 80GB

#### Environment

- OS: Linux
- Python: 3.10+
- Required package: `fish-speech` (for DAC codec)
- CUDA 12.8
- vLLM version: 0.19.0
- vLLM-Omni version or commit: c93359bb354a6aa5c14d062430cb85b2c4db251e

```bash
# Install PortAudio dependency if missing (required by pyaudio which is a dependency of fish-speech)
# For Ubuntu/Debian:
if ! dpkg -l | grep -q libportaudio2; then
    sudo apt-get update && sudo apt-get install -y libportaudio2 portaudio19-dev
fi
pip install fish-speech
```

#### Command

```bash
vllm serve fishaudio/s2-pro --omni --port 8091
```

Notes:

- `--omni` is required.
- The default deploy config `vllm_omni/deploy/fish_qwen3_omni.yaml` is loaded
  automatically by model registry (HF `model_type=fish_qwen3_omni`).


#### Verification

Basic TTS:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, how are you?",
        "voice": "default",
        "response_format": "wav"
    }' --output output.wav
```
[output.wav](https://github.com/user-attachments/files/27134970/output.wav)

Voice cloning:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, this is a cloned voice.",
        "voice": "default",
        "ref_audio": "https://example.com/reference.wav",
        "ref_text": "Transcript of the reference audio."
    }' --output cloned.wav
```
[reference.wav](https://github.com/user-attachments/files/27134971/reference.wav) <br>
[cloned.wav](https://github.com/user-attachments/files/27134969/cloned.wav)



#### Notes

- Key flags: `--omni` is required.
- Known limitations: Output audio is 44.1 kHz mono WAV. Voice cloning requires
  both `ref_audio` and `ref_text` parameters.
- Memory usage: Model loads at ~48.3 GiB, peaks at ~48.9 GiB during inference
  headroom for video frames and audio caches.
