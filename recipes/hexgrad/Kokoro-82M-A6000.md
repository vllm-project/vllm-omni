# Kokoro-82M text-to-speech — RTX A6000

> Kokoro-82M with its 54 built-in voices, served through `/v1/audio/speech`
> (non-streaming and streaming) or offline `Omni` as one single-stage pipeline
> on one RTX A6000.

## Summary

- Vendor: hexgrad
- Model: `hexgrad/Kokoro-82M` (`kokoro-v1_0.pth` plus `voices/*.pt`, loaded as published)
- Task: text-to-speech with built-in voices
- Mode: online `/v1/audio/speech` (non-streaming, raw PCM/WAV stream, SSE) and offline `Omni`
- Hardware: 1x RTX A6000 48GB
- Maintainer: Community

## When to use this recipe

Use this recipe to serve a small, Apache-2.0 TTS model behind an
OpenAI-compatible speech endpoint: low first-audio latency through streaming,
many concurrent requests on one GPU, and a fixed set of voices rather than
voice cloning.

## Supported model contract

| Property | Value |
| --- | --- |
| Voices | the checkpoint's 54 style packs under `voices/` (default `af_heart`); `GET /v1/audio/voices` lists them |
| Languages | a voice's first letter selects the G2P language: `a` American English, `b` British English, `e` Spanish, `f` French, `h` Hindi, `i` Italian, `p` Brazilian Portuguese, `j` Japanese, `z` Mandarin. `language` (`en-us`, `en-gb`, `es`, `fr-fr`, `hi`, `it`, `pt-br`, `ja`, `zh`) overrides it |
| G2P extras | `vllm-omni[kokoro]` covers the first seven languages; Japanese and Mandarin also need `misaki[ja]` / `misaki[zh]`, and without them the server answers HTTP 400 for those voices |
| Text | plain text; long input is split into chunks of at most 510 phonemes, one network pass and one engine step each, so streams deliver one chunk at a time |
| Request fields | `speed` (native: scales predicted durations, also with streaming) and `seed`. Voice cloning and instruction fields (`ref_audio`, `ref_text`, `instructions`, `task_type`, ...) are rejected with HTTP 400 |
| Output | mono 24 kHz; offline `multimodal_output["audio"]` |
| Failures | a non-streaming request whose synthesis fails gets HTTP 500; a failing stream ends early |
| Deploy profile | `vllm_omni/deploy/kokoro.yaml`: fp32, eager, `max_num_seqs: 4`, `max_tokens: 2048` chunks per request, no KV cache |

## References

- Model card: <https://huggingface.co/hexgrad/Kokoro-82M>
- Reference implementation: <https://github.com/hexgrad/kokoro> (`kokoro` 0.9.4)
- Examples: [online](../../examples/online_serving/text_to_speech/README.md#kokoro-82m) and [offline](../../examples/offline_inference/text_to_speech/README.md#kokoro-82m) TTS READMEs
- Supported models: [`docs/models/supported_models.md`](../../docs/models/supported_models.md)
- Tests: `tests/e2e/online_serving/test_kokoro.py`, `tests/e2e/offline_inference/test_kokoro.py`

## Hardware

- Accelerator model and per-device memory: NVIDIA RTX A6000 48GB
- Number of devices: 1
- Device interconnect: not applicable
- Host memory: not material (no CPU offload)
- Qualification scope: online and offline e2e tests at `advanced_model`, and the measurements below

## Software environment

- OS: Linux
- Python: not recorded for the validation run
- Driver / runtime: CUDA 13.0 through torch 2.13.0+cu130; the host had no `nvcc`
- vLLM version: 0.32.0 (`5df63a72` wheel)
- vLLM-Omni version or commit: measured at `9c532193` (base `92595ee4`). Later
  commits change failure handling and the splitting of long non-English chunks,
  not the network or the deploy profile.

## Command

```bash
uv pip install "vllm-omni[kokoro]"
# misaki otherwise pip-installs this spaCy model on first use:
python -m spacy download en_core_web_sm
# Without nvcc/ninja on the host, also: export VLLM_USE_FLASHINFER_SAMPLER=0

vllm-omni serve hexgrad/Kokoro-82M --omni --port 8091
```

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"model": "hexgrad/Kokoro-82M", "input": "Hello from Kokoro.", "voice": "af_heart"}' \
    --output kokoro.wav
```

Offline `Omni` usage is in the
[offline README](../../examples/offline_inference/text_to_speech/README.md#kokoro-82m).

## Verification

```bash
pytest -s -v tests/e2e/online_serving/test_kokoro.py tests/e2e/offline_inference/test_kokoro.py \
    -m advanced_model --run-level advanced_model
```

Expected: online 8/8 and offline 2/2 pass, with Whisper transcripts matching
the input text and the streaming HNR check at `min_hnr_db=-3.0` (clean Kokoro
speech measures about -1.6 to +1.7 dB in the reference package).

With the same phonemes, voice and seed in fp32, the network output was
bit-identical to the reference `kokoro` 0.9.4 package for 5 voices (en-US,
en-GB, es, fr) at speeds 0.9, 1.0 and 1.2.

## Notes

- Memory usage: weights take 0.31 GiB; peak activation memory is about 1 GiB
  per chunk, allocated outside `gpu_memory_utilization`.
- Performance (fp32, eager, `max_num_seqs=4`):

  | Workload | Result |
  | --- | --- |
  | Network forward, 287 phonemes → 19.05 s audio | 213-270 ms (reference `KModel`, same env: 417 ms) |
  | `/v1/audio/speech`, single warm request, same text | median 294 ms |
  | Long input, 3 chunks → 79.5 s audio | 1.7 s non-streaming; streaming first audio 0.82 s |
  | 16 concurrent requests, 140 s audio total | 4.4 s wall (32x realtime), p50 latency 2.8 s |

- Startup downloads all 54 `voices/*.pt` packs next to the weights and loads
  them on the GPU. On a host without network access, pre-fetch the repository
  (`hf download hexgrad/Kokoro-82M`) and set `HF_HUB_OFFLINE=1`.
- Key flags: `enforce_eager: true`; `gpu_memory_utilization` is a startup
  headroom check, not a KV budget.
- Known limitations: each engine step synthesizes its requests one after
  another (no batched forward yet). Japanese and Mandarin voices, and Hindi,
  Italian and Portuguese audio quality, were not qualified on this hardware.

## Supported features

| Feature | Status | Guide |
| --- | --- | --- |
| `/v1/audio/speech` | supported, non-streaming and streaming (PCM/WAV, SSE) | [`docs/serving/speech_api.md`](../../docs/serving/speech_api.md) |
| Offline `Omni.generate` | supported | [offline README](../../examples/offline_inference/text_to_speech/README.md#kokoro-82m) |
| Native `speed` | supported, including streaming | — |
| Voice cloning, instructions | not supported (HTTP 400) | — |
| CUDA graphs, prefix caching, TP/PP | not supported (eager, single GPU) | — |
