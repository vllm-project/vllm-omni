# Covo-Audio-Chat (Offline Inference)

## Setup

Please refer to the [stage configuration documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/) to configure memory allocation appropriately for your hardware setup.

> **Note**
> Covo-Audio code2wav requires `torchdiffeq`. Install it with: `pip install torchdiffeq`

## Run examples

Get into the example folder:
```bash
cd examples/offline_inference/covo_audio
```

### Audio input chat

Using the default audio asset:
```bash
python end2end.py
```

Using a custom audio file:
```bash
python end2end.py --audio-path /path/to/audio.wav
```

Using a local model:
```bash
python end2end.py -m /path/to/Covo-Audio-Chat --output-dir ./my_output
```

### Command-line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--model-name` | `-m` | `tencent/Covo-Audio-Chat` | Model path or HuggingFace model ID |
| `--text` | `-t` | `请回答这段音频里的问题。` | Text prompt / question for the audio |
| `--audio-path` | `-a` | default audio asset | Path to local audio file |
| `--sampling-rate` | | `16000` | Sampling rate for audio loading (Hz) |
| `--output-dir` | | `./output_audio` | Output directory for generated files |
| `--num-prompts` | | `1` | Number of prompts to generate |
| `--stage-configs-path` | | (auto) | Path to stage configs YAML file |
| `--log-stats` | | `false` | Enable detailed statistics logging |
| `--stage-init-timeout` | | `300` | Stage initialization timeout (seconds) |
| `--batch-timeout` | | `5` | Batching timeout (seconds) |
| `--init-timeout` | | `300` | Overall initialization timeout (seconds) |
| `--shm-threshold-bytes` | | `65536` | Shared memory threshold (bytes) |

## Pipeline

Covo-Audio-Chat uses a 2-stage pipeline:

- **Stage 0 (fused_thinker_talker):** The 7B LLM generates interleaved text and audio tokens in a single autoregressive pass.
- **Stage 1 (code2wav):** A BigVGAN-based vocoder converts the extracted audio codes into a 24kHz WAV waveform.

## Output

The script generates two files per request in the output directory:

- `{request_id}.txt` -- prompt and generated text
- `{request_id}.wav` -- generated audio (24kHz WAV)

## FAQ

If you encounter `ModuleNotFoundError: No module named 'librosa'`, install it with:
```bash
pip install librosa
```

## Environment

- GPU: 1x A100 (80 GiB)
- Stage 0 (7B LLM): ~16 GiB VRAM
- Stage 1 (BigVGAN vocoder): ~2 GiB VRAM
