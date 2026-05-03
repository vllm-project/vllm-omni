# OmniVoice

## Summary

- Vendor: K2-FSA
- Model: `k2-fsa/OmniVoice`
- Task: Multilingual text-to-speech synthesis
- Mode: Online serving with the OpenAI-compatible `/v1/audio/speech` API
- Maintainer: Community / @fray1024

## When to use this recipe

Use this recipe to serve `k2-fsa/OmniVoice` with vLLM-Omni on a single NVIDIA
L20 48GB GPU for multilingual text-to-speech synthesis.

This recipe validates the OpenAI-compatible `/v1/audio/speech` online serving
path, including English TTS, Chinese TTS, and a raw `curl` smoke test.

## References

- Online serving example:
  [`examples/online_serving/omnivoice`](../../examples/online_serving/omnivoice)
- Related issue:
  [vllm-project/vllm-omni#2645](https://github.com/vllm-project/vllm-omni/issues/2645)

## Hardware Support

This recipe documents the tested GPU configuration for OmniVoice online
serving on 1x NVIDIA L20 48GB.
Other hardware and configurations are welcome as community validation lands.

## GPU

### 1x NVIDIA L20 48GB

#### Environment

- OS: Linux / Ubuntu 22.04
- GPU: 1x NVIDIA L20 48GB
- Driver: 580.126.09
- CUDA: 13.0
- Python: 3.12.13
- PyTorch: 2.10.0+cu130
- vLLM version: 0.19.0
- vLLM-Omni commit: `dd0fa02547aae9f57e1cb1c80b7db50a4161b8d2`
- Transformers: 4.57.6
- Hugging Face cache: `HF_HOME=$HOME/hf_cache`

If Hugging Face access is slow or unavailable from the runtime environment,
configure a reachable mirror before downloading the model:

```bash
export HF_HOME=$HOME/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
```

Use a `huggingface-hub` version compatible with `transformers`:

```bash
uv pip install "huggingface-hub>=0.34.0,<1.0"
```

#### Download

```bash
hf download k2-fsa/OmniVoice --cache-dir $HF_HOME/hub --max-workers 8
```

After download, the model cache should contain:

```text
$HF_HOME/hub/models--k2-fsa--OmniVoice
```

#### Command

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve k2-fsa/OmniVoice \
  --host 0.0.0.0 \
  --port 8091 \
  --trust-remote-code \
  --omni
```

Notes:

- `--omni` is required for vLLM-Omni serving.
- `--trust-remote-code` is required for this model.
- The service exposes the OpenAI-compatible `/v1/audio/speech` endpoint.

#### Verification

Health check:

```bash
curl http://127.0.0.1:8091/health
```

Observed result:

```text
Server is healthy after 35 seconds.
```

English TTS:

```bash
python examples/online_serving/omnivoice/speech_client.py \
  --api-base http://127.0.0.1:8091 \
  --model k2-fsa/OmniVoice \
  --text "Hello, this is an OmniVoice smoke test on one NVIDIA L20 GPU." \
  --output omnivoice_l20_en.wav
```

Observed result:

```text
Model: k2-fsa/OmniVoice
Text: Hello, this is an OmniVoice smoke test on one NVIDIA L20 GPU.
Generating audio...
Audio saved to: omnivoice_l20_en.wav
```

Chinese TTS:

```bash
python examples/online_serving/omnivoice/speech_client.py \
  --api-base http://127.0.0.1:8091 \
  --model k2-fsa/OmniVoice \
  --text "你好，这是一个 OmniVoice 的中文语音合成测试。" \
  --language Chinese \
  --output omnivoice_l20_zh.wav
```

Observed result:

```text
Model: k2-fsa/OmniVoice
Text: 你好，这是一个 OmniVoice 的中文语音合成测试。
Language: Chinese
Generating audio...
Audio saved to: omnivoice_l20_zh.wav
```

Raw `/v1/audio/speech` request:

```bash
curl -sS \
  -w "\nhttp_code=%{http_code}\nsize_download=%{size_download}\ntime_total=%{time_total}\n" \
  -X POST "http://127.0.0.1:8091/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "k2-fsa/OmniVoice",
    "input": "This is a raw curl smoke test for OmniVoice on one NVIDIA L20 GPU.",
    "voice": "default",
    "response_format": "wav"
  }' \
  --output omnivoice_l20_curl.wav
```

Observed result:

```text
http_code=200
size_download=203564
time_total=0.659745
```

Audio validation:

```text
omnivoice_l20_en.wav:
  size_bytes: 190124
  channels: 1
  sample_width: 2
  sample_rate: 24000
  frames: 95040
  duration_sec: 3.96

omnivoice_l20_zh.wav:
  size_bytes: 188204
  channels: 1
  sample_width: 2
  sample_rate: 24000
  frames: 94080
  duration_sec: 3.92

omnivoice_l20_curl.wav:
  size_bytes: 203564
  channels: 1
  sample_width: 2
  sample_rate: 24000
  frames: 101760
  duration_sec: 4.24
```

During the final check after the smoke tests, `nvidia-smi` showed
3660 MiB / 46068 MiB GPU memory in use on the NVIDIA L20 GPU.

#### Notes

- This recipe validates OmniVoice online serving with the default voice setting.
- English and Chinese TTS requests were tested.
- A raw `/v1/audio/speech` HTTP request was also tested with `curl`.
- The generated WAV files were non-empty and readable.
- In the tested environment, output audio was mono WAV at 24 kHz.
- If Hugging Face is not reachable from the runtime environment, configure
  `HF_ENDPOINT=https://hf-mirror.com` or another reachable mirror before model
  download.
- Voice cloning and voice design are not covered by this smoke test.
