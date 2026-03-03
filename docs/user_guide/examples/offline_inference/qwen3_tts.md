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

## Streaming Mode

Add `--streaming` to stream audio chunks progressively via `AsyncOmni` (requires `async_chunk: true` in the stage config):

```bash
python end2end.py --query-type CustomVoice --streaming --output-dir /tmp/out_stream
```

Each 25-frame Code2Wav chunk is logged as it arrives. The final WAV file is written once generation
completes. This demonstrates that audio data is available progressively rather than only at the end.

> **Note:** Streaming uses `AsyncOmni` internally. The non-streaming path (`Omni`) is unchanged.

## Batched Decoding

The Code2Wav stage (stage 1) supports batched decoding, where multiple requests are decoded in a single forward pass through the SpeechTokenizer. To use it, provide a stage config with `max_batch_size > 1` and pass multiple prompts via `--txt-prompts` with a matching `--batch-size`.

```
python end2end.py --query-type CustomVoice \
    --txt-prompts benchmark_prompts.txt \
    --batch-size 4 \
    --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_tts_batch.yaml
```

**Important:** `--batch-size` must match a CUDA graph capture size (1, 2, 4, 8, 16...) because the Talker's code predictor KV cache is sized to `max_num_seqs`, and CUDA graphs pad the batch to the next capture size. Both stages need `max_batch_size >= batch_size` in the stage config for batching to take effect. If only stage 1 has a higher `max_batch_size`, it won't help — stage 1 can only batch chunks from requests that are in-flight simultaneously, which requires stage 0 to also process multiple requests concurrently.

## Notes

- The script uses the model paths embedded in `end2end.py`. Update them if your local cache path differs.
- Use `--output-dir` to change the output folder.

## Example materials

??? abstract "benchmark_prompts.txt"
    ``````txt
    --8<-- "examples/offline_inference/qwen3_tts/benchmark_prompts.txt"
    ``````
??? abstract "end2end.py"
    ``````py
    --8<-- "examples/offline_inference/qwen3_tts/end2end.py"
    ``````
