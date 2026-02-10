# Qwen3-TTS

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/qwen3_tts>.


This directory contains an offline demo for running Qwen3 TTS models with vLLM Omni. It builds task-specific inputs and generates WAV files locally.

## Model Overview

Qwen3 TTS provides multiple task variants for speech generation:

- **CustomVoice**: Generate speech with a known speaker identity (speaker ID) and optional instruction.
- **VoiceDesign**: Generate speech from text plus a descriptive instruction that designs a new voice.
- **Base**: Voice cloning using a reference audio + reference transcript, with optional mode selection.

## Setup
Please refer to the [stage configuration documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/) to configure memory allocation appropriately for your hardware setup.

### ROCm Dependencies

You will need to install these two dependencies `onnxruntime-rocm` and `sox`.

```
pip uninstall onnxruntime # should be removed before we can install onnxruntime-rocm
pip install onnxruntime-rocm sox
```

## Quick Start

Run a single sample for a task:

```
python end2end.py --query-type CustomVoice
```

Generated audio files are saved to `output_audio/` by default.

## Task Usage

### CustomVoice

Single sample:

```
python end2end.py --query-type CustomVoice
```

Batch sample (multiple prompts in one run):

```
python end2end.py --query-type CustomVoice --use-batch-sample
```

### VoiceDesign

Single sample:

```
python end2end.py --query-type VoiceDesign
```

Batch sample:

```
python end2end.py --query-type VoiceDesign --use-batch-sample
```

### Base (Voice Clone)

Single sample:

```
python end2end.py --query-type Base
```

Batch sample:

```
python end2end.py --query-type Base --use-batch-sample
```

Mode selection for Base:

- `--mode-tag icl` (default): standard mode
- `--mode-tag xvec_only`: enable `x_vector_only_mode` in the request

Examples:

```
python end2end.py --query-type Base --mode-tag icl
```

## Notes

- The script uses the model paths embedded in `end2end.py`. Update them if your local cache path differs.
- Use `--output-dir` (preferred) or `--output-wav` to change the output folder.

## Example materials

??? abstract "end2end.py"
    ``````py
    --8<-- "examples/offline_inference/qwen3_tts/end2end.py"
    ``````
