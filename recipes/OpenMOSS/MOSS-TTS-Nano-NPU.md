# MOSS-TTS-Nano on Ascend NPU

## Summary

- Model: `OpenMOSS-Team/MOSS-TTS-Nano`
- Task: multilingual text-to-speech with a reference voice
- Hardware: one Ascend 910B3 64GB NPU
- Output: 48 kHz mono audio
- Serving API: OpenAI-compatible `/v1/audio/speech`

MOSS-TTS-Nano combines a small autoregressive language model with
`OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano`. Both components run on the same
NPU. The codec is a separate checkpoint and must be available even when the
main model has already been downloaded.

## Environment

Use an aligned vLLM, vLLM-Ascend, and vLLM-Omni environment. The following
configuration was validated:

- Image: `quay.nju.edu.cn/ascend/vllm-omni:v0.28.0`
- Ascend device: 910B3 64GB
- CANN: 9.1.0
- vLLM: 0.28.0
- Execution mode: eager

See the [NPU installation guide](../../docs/getting_started/installation/npu.md)
for the complete container device and driver mounts.

## Download checkpoints

```bash
export HF_HOME=/models/huggingface-cache

hf download OpenMOSS-Team/MOSS-TTS-Nano

# This dependency is loaded from its model ID by the MOSS-TTS-Nano config.
hf download OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano
```

For an air-gapped server, populate `HF_HOME` with the audio-tokenizer snapshot
before startup and set `HF_HUB_OFFLINE=1`. Do not omit the audio tokenizer:
the main model directory does not contain its weights.

## Start the server

Expose one physical NPU and keep the stage's logical device as `0`:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export HF_HOME=/models/huggingface-cache

vllm serve OpenMOSS-Team/MOSS-TTS-Nano \
    --omni \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8091
```

`--trust-remote-code` is required because both checkpoints provide custom
Transformers model implementations. The bundled `moss_tts_nano.yaml` deploy
configuration enables eager execution; ACL graph capture is not used for this
model path.

## Voice-cloning request

`ref_audio` accepts an HTTP URL or a base64 data URL. A non-streaming request
returns a WAV file:

```bash
curl http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "model": "OpenMOSS-Team/MOSS-TTS-Nano",
        "input": "Hello, this is a voice cloning test on Ascend NPU.",
        "stream": false,
        "response_format": "wav",
        "ref_audio": "https://example.com/reference.wav"
    }' \
    --output output.wav
```

Streaming raw PCM uses the same endpoint:

```bash
curl http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "model": "OpenMOSS-Team/MOSS-TTS-Nano",
        "input": "你好，这是一段流式语音合成测试。",
        "stream": true,
        "stream_format": "audio",
        "response_format": "pcm",
        "ref_audio": "https://example.com/reference.wav"
    }' \
    --output output.pcm
```

Raw PCM output is signed 16-bit little-endian mono at 48 kHz.

## Validation

The Ascend 910B3 validation covered:

- English non-streaming WAV output;
- Chinese streaming PCM output;
- repeat generation with the same seed producing byte-identical WAV files;
- two concurrent non-streaming requests;
- waveform checks for a non-empty, finite, non-silent signal at 48 kHz.

The first request includes one-time NPU kernel warmup. In the validated setup,
the first non-streaming request took about 22.7 seconds, a repeated request
took about 7.5 seconds, and streaming produced its first audio chunk in about
0.38 seconds. These figures are functional reference measurements, not a
performance guarantee.

## Current limitations

- The model runs in eager mode; ACL graph mode has not been enabled.
- Concurrent HTTP requests are accepted but generated serially. Do not raise
  the stage's `max_num_seqs` above `1`: the remote model and audio tokenizer
  share RNG and streaming decode state, and interleaved generation can corrupt
  audio.
- The remote model's CUDA-only `flash_attention_2` path is replaced with
  PyTorch SDPA on Ascend.
- The current non-CUDA model wrapper loads the language model and audio codec
  in float32. Quantized checkpoints have not been validated.
- A reference audio clip is required for `voice_clone` mode.
