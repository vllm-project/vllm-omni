# MOSS-TTS-Nano Offline Inference

## Overview

Single-stage offline TTS pipeline using the 0.1B MOSS-TTS-Nano AR LM and MOSS-Audio-Tokenizer-Nano codec. Outputs 48 kHz stereo WAV.

## Quick Start

```bash
python end2end.py --text "Hello, this is MOSS-TTS-Nano."
```

The first run downloads `OpenMOSS-Team/MOSS-TTS-Nano` and `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano` from Hugging Face.

## Usage

```
python end2end.py [OPTIONS]

Options:
  --text TEXT               Text to synthesize (default: "Hello, this is MOSS-TTS-Nano speaking.")
  --voice VOICE             Built-in voice preset (default: Junhao). See voice table below.
  --mode MODE               voice_clone (default) or continuation
  --prompt-audio PATH       Reference WAV/MP3 for custom voice cloning
  --prompt-text TEXT        Reference transcript (continuation mode)
  --max-new-frames N        Max AR frames, default 375 (~14 s audio)
  --seed INT                Random seed for reproducibility
  --audio-temperature F     Audio sampling temperature (default: 0.8)
  --audio-top-k N           Audio top-k sampling (default: 25)
  --audio-top-p F           Audio top-p sampling (default: 0.95)
  --text-temperature F      Text layer temperature (default: 1.0)
  --batch                   Run a built-in batch of diverse samples (ZH/EN/FR)
  --output-dir DIR          Directory for WAV outputs (default: /tmp/moss_tts_nano_output)
  --deploy-config PATH      Override deploy YAML (defaults to vllm_omni/deploy/moss_tts_nano.yaml)
  --stage-init-timeout INT  Timeout in seconds for stage init (default: 120)
```

## Examples

```bash
# Built-in Chinese voice
python end2end.py --text "你好，这是MOSS-TTS-Nano的语音合成演示。" --voice Junhao

# Built-in English voice
python end2end.py --text "Hello from MOSS-TTS-Nano." --voice Ava

# Custom voice clone
python end2end.py \
    --text "Hello, this is a cloned voice." \
    --prompt-audio /path/to/reference.wav \
    --prompt-text "Exact transcript of the reference audio."

# Batch synthesis (ZH/EN/FR)
python end2end.py --batch --output-dir /tmp/batch_output

# Reproducible output
python end2end.py --text "Deterministic test." --seed 42
```

## Built-in Voice Presets

| Voice | Language |
|-------|----------|
| `Junhao` | ZH |
| `Zhiming` | ZH |
| `Weiguo` | ZH |
| `Xiaoyu` | ZH |
| `Yuewen` | ZH |
| `Lingyu` | ZH |
| `Ava` | EN |
| `Bella` | EN |
| `Adam` | EN |
| `Nathan` | EN |
| `Sakura` | JA |
| `Yui` | JA |
| `Aoi` | JA |
| `Hina` | JA |
| `Mei` | JA |

## Deploy Config

Runtime knobs live in `vllm_omni/deploy/moss_tts_nano.yaml` (auto-loaded;
override with `--deploy-config PATH`). Key stage-level settings:

```yaml
stages:
  - stage_id: 0
    gpu_memory_utilization: 0.3   # ~2 GB VRAM; increase for faster init
    max_num_seqs: 4               # concurrent requests
    max_model_len: 4096
```

## Output Format

WAV files, 48 kHz, stereo (2-channel). The codec interleaves stereo as `[L, R, L, R, ...]` in the flat tensor returned by the model.

## Troubleshooting

- **`libnvrtc.so.13: cannot open shared object file`**: torchaudio 2.10+ torchcodec backend requires NVRTC. The model patches `torchaudio.load/save` automatically at load time to fall back to soundfile.
- **`flash_attn not installed`**: The model falls back to `sdpa` attention automatically.
- **Empty audio**: Check that `--text` is non-empty and the model loaded successfully (look for "MOSS-TTS-Nano LM loaded" in logs).
