# Consumer devices (8 GiB and below)

vLLM-Omni is usually documented on data-centre GPUs. This page collects what has actually been
measured to fit on a single small consumer GPU, and how to run it.

All numbers below come from one machine and one run: **1× RTX 3060 Laptop, 6 GiB (6144 MiB) of
device memory, driver 617.42, vLLM 0.34.0**. Treat them as an upper bound on what 8 GiB cards can do,
not as a guarantee on different hardware.

## Check what fits before installing anything

```bash
vllm-omni doctor
```

`doctor` reads the minimum-memory field of every structured recipe under `recipes/models/` and prints
the variants whose documented minimum fits each attached device, then the smallest documented tiers
that nothing reaches yet:

```text
Devices:
- NVIDIA GeForce RTX 3060 Laptop GPU (cuda, 6.0 GiB dedicated)
  1 documented variant across 1 model fit: `OpenMOSS-Team/MOSS-TTS-Nano`
```

`vllm-omni doctor --json` produces the same data for scripts and CI. Recipes that have never been
measured on a small device are absent by design; see [Minimum memory tiers](../models/memory_tiers.md)
for the full ladder: 3 variants fit 8 GiB, none documents a minimum in the 12 GiB band, and 11 fit
16 GiB at the time of writing.

## A verified 6 GiB speech stack

Speech is the only modality that fits comfortably here. Two localhost services cover it: text to
speech from vLLM-Omni, and speech to text from vLLM itself, which already ships the audio encoder
path. Both numbers were sampled with one `nvidia-smi` poller while both servers were under load.

| Role | Model | Weights | Latency measured |
| --- | --- | --- | --- |
| Text to speech | `OpenMOSS-Team/MOSS-TTS-Nano-100M` | 0.31 GiB | 6.0 s for 5.5 s of audio (RTF ≈ 1.1) |
| Speech to text | `Qwen/Qwen3-ASR-0.6B` | 1.53 GiB + 0.58 GiB KV | 0.76 s for 7.9 s of audio (RTF ≈ 0.10) |

Both services up at once use **3.16 GiB** and peak at **3.21 GiB** while both are serving, so a 6 GiB
card keeps ~2.8 GiB of headroom for the desktop. The TTS figure is a short sentence; longer prompts
land closer to RTF 3 on this card. Sending both requests at the same time only moves the
peak by 50 MiB; it costs TTS latency (6.0 s → 9.2 s) and leaves speech to text untouched
(0.76 s → 0.89 s).

### 1. Text to speech

```bash
vllm-omni serve OpenMOSS-Team/MOSS-TTS-Nano-100M --omni --port 8123
```

The 100M talker plus the `MOSS-Audio-Tokenizer-Nano` codec load in about two minutes on a cold cache,
most of it downloading the codec. It is voice-cloned: `ref_audio` is required and must be an HTTP(S)
URL, a `file://` URI under `--allowed-local-media-path`, or a base64 data URL.

```bash
# data URL, so no extra flags are needed; supply your own 3-10 s reference clip
REF=$(base64 -w0 reference.wav)
curl -X POST http://localhost:8123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"OpenMOSS-Team/MOSS-TTS-Nano-100M\",
       \"input\": \"Hello from a 6 GiB laptop.\",
       \"ref_audio\": \"data:audio/wav;base64,${REF}\",
       \"response_format\": \"wav\"}" --output reply.wav
```

The recipe ([`recipes/models/OpenMOSS-Team/MOSS-TTS-Nano.yaml`](../../recipes/models/OpenMOSS-Team/MOSS-TTS-Nano.yaml))
carries the same numbers and a longer usage guide.

### 2. Speech to text

`/v1/audio/transcriptions` is not wired into the vLLM-Omni API server yet, so use the vLLM server for
the transcription model and point clients at a second port in the meantime:

```bash
vllm serve Qwen/Qwen3-ASR-0.6B --served-model-name qwen3-asr --port 8100 \
  --gpu-memory-utilization 0.55 --max-model-len 2048 --max-num-seqs 1 \
  --enforce-eager --trust-remote-code
```

```bash
curl -X POST http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-asr",
       "messages": [{"role": "user", "content": [
         {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,'"$(base64 -w0 clip.wav)"'"}},
         {"type": "text", "text": "Transcribe the audio."}]}],
       "max_tokens": 256, "temperature": 0}'
```

The reply wraps the transcript in `<asr_text>…</asr_text>` and may prefix a detected language, so
strip both before handing the text to a user or to a downstream pipeline:

```text
language Chinese<asr_text>...</asr_text>
```

`--gpu-memory-utilization 0.55` leaves room for the TTS service. Lower it further when both run
together, and prefer `--max-num-seqs 1` on cards this small.

## What does not fit

- **Image and video generation** is documented from 13 GiB up (the lowest entry is
  `flux-3-image-klein` in nvfp4), which no 6 GiB card can hold. Nothing below that has been measured,
  so `doctor` reports the smaller bands as unmeasured rather than guessing.
- **Multi-stage audio models** need per-stage budgets. Qwen3-TTS-0.6B, for example, only starts on
  6 GiB when both stages are pinned explicitly, because the default single value cannot be shared
  across stages:

  ```bash
  vllm-omni serve Qwen/Qwen3-TTS-12Hz-0.6B-Base --omni --port 8124 \
    --stage-overrides '{"0": {"gpu_memory_utilization": 0.62, "max_model_len": 2048},
                        "1": {"gpu_memory_utilization": 0.25, "max_model_len": 2048}}'
  ```

  Its 1.7B sibling does not fit at any setting tried on 6 GiB: weights alone take 4.11 GiB and the
  remaining memory is below the KV cache floor.
- **Apple Silicon and CPU** are not serving targets yet: MPS has no platform plugin, and the CPU
  platform only exists as a fallback for argument parsing. No tier in the ladder is reachable from
  either.

## Reporting your own numbers

`vllm-omni doctor` and the memory ladder only know what recipes declare. If you measure a model on a
small card, add `vram_minimum_gb` to its variant in `recipes/models/<vendor>/<model>.yaml` and
regenerate the artifacts with `python tools/gen_memory_tiers.py`. The TTS numbers on this page are the
ones recorded in the MOSS-TTS-Nano recipe; the speech-to-text numbers were measured for this page only
and are not yet a recipe.
