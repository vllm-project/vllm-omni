# Voxtral TTS for text-to-speech

## Summary

- Vendor: Mistral AI
- Model: `mistralai/Voxtral-4B-TTS-2603`
- Task: Text-to-speech with model-provided voice presets or voice cloning
- Mode: Online serving with the OpenAI-compatible `/v1/audio/speech` API
- Maintainer: Community

## When to use this recipe

Use this recipe when you want a known-good starting point for serving Voxtral TTS
models with vLLM-Omni and validate the deployment with the existing TTS client
examples in this repository.

## References

- Upstream model card: <https://huggingface.co/mistralai/Voxtral-4B-TTS-2603>
- Related examples under `examples/`:
  [`examples/offline_inference/text_to_speech/voxtral_tts/end2end.py`](../../examples/offline_inference/text_to_speech/voxtral_tts/end2end.py)
  [`examples/online_serving/text_to_speech/voxtral_tts/gradio_demo.py`](../../examples/online_serving/text_to_speech/voxtral_tts/gradio_demo.py)
- Related issue or discussion:
  [RFC: add recipes folder](https://github.com/vllm-project/vllm-omni/issues/2645)

## Hardware Support

## GPU

### 1x NVIDIA GeForce RTX 5090 32GB

#### Environment

- OS: Ubuntu 24.04.4 LTS
- Python: 3.12.3
- PyTorch: 2.11.0+cu130
- Driver / runtime: NVIDIA driver 580.159.03, CUDA 13.0
- GPU: NVIDIA GeForce RTX 5090, 32 GB (`32607 MiB`)
- vLLM version: 0.23.0
- vLLM-Omni version or commit: 0.23.0rc2.dev140+gbe60d7c7c (`be60d7c7`)
- Tested with: `mistral-common==1.11.5`

#### Command

```bash
vllm serve mistralai/Voxtral-4B-TTS-2603 \
    --omni \
    --port 8091 \
    --trust-remote-code
```

The deploy config is auto-loaded from `vllm_omni/deploy/voxtral_tts.yaml`.

#### Verification

Health check and voices:

```bash
curl -s -o /tmp/voxtral_health.txt -w '%{http_code}' http://127.0.0.1:8091/health
curl -s http://127.0.0.1:8091/v1/audio/voices
```

Non-streaming WAV:

```bash
curl -X POST http://127.0.0.1:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Voxtral-4B-TTS-2603",
    "input": "Hello, this is Voxtral TTS running on an NVIDIA GeForce RTX 5090 with vLLM-Omni.",
    "voice": "casual_female",
    "language": "English",
    "response_format": "wav"
  }' \
  --output /tmp/voxtral_5090.wav
```

Streaming PCM:

```bash
curl -X POST http://127.0.0.1:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Voxtral-4B-TTS-2603",
    "input": "Hello, this is Voxtral TTS streaming PCM on an NVIDIA GeForce RTX 5090.",
    "voice": "casual_female",
    "language": "English",
    "stream": true,
    "response_format": "pcm"
  }' \
  --output /tmp/voxtral_5090_stream.pcm

ffmpeg -f s16le -ar 24000 -ac 1 -i /tmp/voxtral_5090_stream.pcm /tmp/voxtral_5090_stream.wav -y
ffprobe -hide_banner /tmp/voxtral_5090_stream.wav
```

#### Notes

- Output: 24 kHz mono WAV/PCM; both non-streaming and streaming requests passed.
- Memory usage: ~27.3 GiB after warmup (Stage 0 ~25.5 GiB, Stage 1 ~1.7 GiB).
- Key flags: `--omni` is required; `--trust-remote-code` is recommended.

### 1x RTX 4090 24GB

#### Environment

- OS: Ubuntu 22.04.5 LTS
- Python: 3.12.11
- PyTorch: 2.11.0+cu130
- Driver / runtime: CUDA 13.0
- GPU: NVIDIA GeForce RTX 4090, 24 GB
- vLLM version: 0.20.0
- vLLM-Omni version or commit: 0.20.0rc2.dev99+g857356d5b
- Required package: mistral_common >= 1.10.0

#### Command

Online:

```bash
vllm serve mistralai/Voxtral-4B-TTS-2603 \
    --omni \
    --port 8091
```

The deploy config is auto-loaded from `vllm_omni/deploy/voxtral_tts.yaml`

#### Verification

**Quick smoke test with curl:**

```bash
curl -X POST http://127.0.0.1:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Voxtral-4B-TTS-2603",
    "input": "Hello, this is Voxtral TTS running with vLLM-Omni.",
    "voice": "casual_female",
    "language": "English",
    "response_format": "wav"
  }' \
  --output voxtral.wav
```

**Streaming audio:**

```bash
curl -X POST http://127.0.0.1:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Voxtral-4B-TTS-2603",
    "input": "Hello, this is Voxtral TTS streaming PCM.",
    "voice": "casual_female",
    "language": "English",
    "stream": true,
    "stream_format": "audio",
    "response_format": "pcm"
  }' \
  --output voxtral_stream.pcm
```
Convert the raw PCM output to WAV
```bash
ffmpeg -f s16le -ar 24000 -ac 1 -i voxtral_stream.pcm voxtral_stream.wav -y
ffprobe -hide_banner voxtral_stream.wav
```

#### Notes

- Memory usage: Stage 0 (`audio_generation`) ~18.95 GiB, Stage 1
  (`audio_tokenizer`) ~1.55 GiB. The observed server-startup peak was
  `21006 MiB / 24564 MiB` (`20.51 GiB / 23.99 GiB`, about 85.5%).
  Both stages share GPU 0 via the deploy config (`gpu_memory_utilization: 0.8`
  for Stage 0, `0.1` for Stage 1).
- Key flags: `--omni` is required. `--port 8091` is optional and only selects
  the HTTP port.
- Known limitation: Voxtral TTS outputs mono audio at 24 kHz. And with the public
mistralai/Voxtral-4B-TTS-2603 checkpoint. running the
voice-cloning e2e test ([`Voice cloning (capability gated upstream)`](../../examples/offline_inference/text_to_speech/README.md#voice-cloning-capability-gated-upstream)
) or adding a `ref_audio` request parameter will fail
because the public checkpoint does not provide the encoder weights needed to
convert reference audio into conditioning features. The expected failure is: `RuntimeError: encode_waveforms requires encoder weights which are not available in the open-source checkpoint.`

### 1x NVIDIA RTX A6000 48GB

#### Environment

- OS: Ubuntu 22.04
- Python: 3.12
- Driver / runtime: NVIDIA driver 580.126.09, CUDA 13.0
- GPU: NVIDIA RTX A6000, 48 GB
- vLLM version: 0.22.0
- vLLM-Omni version or commit: 0.22.0rc2.dev8+g41a795b5d
- Required package: `pip install mistral-common`

#### Command

```bash
vllm serve mistralai/Voxtral-4B-TTS-2603 --omni --port 8091
```

#### Verification

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Voxtral-4B-TTS-2603",
    "input": "Artificial intelligence is transforming the way we interact with technology.",
    "voice": "neutral_female",
    "response_format": "wav"
  }' -o voxtral_test.wav
```

Supported voice presets: `ar_male`, `casual_female`, `casual_male`, `cheerful_female`,
`de_female`, `de_male`, `es_female`, `es_male`, `fr_female`, `fr_male`, `hi_female`,
`hi_male`, `it_female`, `it_male`, `neutral_female`, `neutral_male`, `nl_female`,
`nl_male`, `pt_female`, `pt_male`.

#### Notes

- Memory usage: Peak VRAM ~40.5 GiB during inference.
- Generation time: ~8 seconds for ~200 words of English text.
- Audio quality is high; mono 24 kHz output.
