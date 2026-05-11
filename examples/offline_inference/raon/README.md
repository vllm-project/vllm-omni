# Raon-Speech

This directory contains an offline demo for running the Raon-Speech model with vLLM Omni. It builds task-specific inputs and generates WAV files locally.

## Model Overview

**Raon** (`KRAFTON/Raon-Speech-9B`) is a 9B speech language model developed by KRAFTON. It uses a Thinker+Talker (AR + audio codec decoder) architecture for high-quality speech synthesis and voice cloning.

Supported tasks:

- **TTS**: Basic text-to-speech synthesis.
- **TTS_ICL**: Voice cloning using a reference audio file.

## Setup

Please refer to the [stage configuration documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/) to configure memory allocation for your hardware.

When running the offline demo from Docker, allocate enough shared memory for
vLLM-Omni stage communication and PyTorch multiprocessing:

```bash
docker run --gpus all --shm-size=16g ...
```

`--ipc=host` is also acceptable in environments where host IPC sharing is
allowed.

The default stage config for Raon is at:
```
vllm_omni/model_executor/stage_configs/raon.yaml
```

## Quick Start

```bash
# Basic TTS
python end2end.py --query-type tts

# Voice cloning with a reference audio file
python end2end.py --query-type tts_icl --ref-audio /path/to/reference.wav
```

Generated audio files are saved to `output_audio/` by default.

## Task Usage

### TTS (Basic)

Single sample:

```bash
python end2end.py --query-type tts
```

Batch sample (multiple built-in prompts):

```bash
python end2end.py --query-type tts --use-batch-sample
```

Custom prompts from a file (one per line):

```bash
python end2end.py --query-type tts --txt-prompts prompts.txt --batch-size 4
```

### TTS (Voice Clone)

Single sample using a default reference audio URL:

```bash
python end2end.py --query-type tts_icl
```

With a local reference audio file:

```bash
python end2end.py --query-type tts_icl --ref-audio /path/to/reference.wav
```

Batch sample:

```bash
python end2end.py --query-type tts_icl --use-batch-sample --ref-audio /path/to/reference.wav
```

## Streaming Mode

Add `--streaming` to stream audio chunks progressively via `AsyncOmni` (requires `async_chunk: true` in the stage config, which is the default in `raon.yaml`):

```bash
python end2end.py --query-type tts --streaming --output-dir /tmp/out_stream
```

Each audio codec chunk is logged as it arrives. The TTFA (time to first audio) is reported for the first chunk. The final WAV file is written once generation completes.

## Batched Decoding

Pass multiple prompts via `--txt-prompts` with a matching `--batch-size`:

```bash
python end2end.py --query-type tts \
    --txt-prompts prompts.txt \
    --batch-size 4
```

`--batch-size` must be a power of two (1, 2, 4, 8, ...).

## Notes

- Use `--output-dir` to change the output folder.
- The default stage config path is `vllm_omni/model_executor/stage_configs/raon.yaml`. Override with `--stage-configs-path`.
- For voice cloning, provide a clear, clean reference audio (3-8 seconds recommended).
- Long-form TTS defaults to a final-chunk EOS minimum gate with best-of-k and
  final retry disabled. Override with:
  - `RAON_TTS_LONG_ENABLE_FINAL_BEST_OF_K`
  - `RAON_TTS_LONG_ENABLE_FINAL_EOS_MIN_GATE`
  - `RAON_TTS_LONG_FINAL_EOS_MIN_DURATION_RATIO`
  - `RAON_TTS_LONG_ENABLE_FINAL_EOS_RETRY`
- When running in Docker, use `--shm-size=16g` or `--ipc=host` to avoid shared
  memory exhaustion during concurrent audio generation.
