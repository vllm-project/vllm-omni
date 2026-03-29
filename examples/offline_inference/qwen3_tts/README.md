# Qwen3-TTS

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

Each Code2Wav chunk is logged as it arrives (default 25 frames; configurable via `codec_chunk_frames`
in the stage config). The initial chunk size is dynamically selected based on server load for reduced
TTFA, and can be overridden per-request via the `initial_codec_chunk_frames` API field. The final WAV file is written once generation
completes. This demonstrates that audio data is available progressively rather than only at the end.

> **Note:** Streaming uses `AsyncOmni` internally. The non-streaming path (`Omni`) is unchanged.

## Batched Decoding

The Code2Wav stage (stage 1) supports batched decoding, where multiple requests are decoded in a single forward pass through the SpeechTokenizer. To use it, provide a stage config with `max_num_seqs > 1` and pass multiple prompts via `--txt-prompts` with a matching `--batch-size`.

```
python end2end.py --query-type CustomVoice \
    --txt-prompts benchmark_prompts.txt \
    --batch-size 4 \
    --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_tts_batch.yaml
```

**Important:** `--batch-size` must match a CUDA graph capture size (1, 2, 4, 8, 16...) because the Talker's code predictor KV cache is sized to `max_num_seqs`, and CUDA graphs pad the batch to the next capture size. Both stages need `max_num_seqs >= batch_size` in the stage config for batching to take effect. If only stage 1 has a higher `max_num_seqs`, it won't help — stage 1 can only batch chunks from requests that are in-flight simultaneously, which requires stage 0 to also process multiple requests concurrently.

## Notes

- The script uses the model paths embedded in `end2end.py`. Update them if your local cache path differs.
- Use `--output-dir` to change the output folder.

## End-to-End Benchmark Script

`end2end_benchmark_script.py` is an offline benchmark tool that runs warmup + test rounds with a single Omni instance to avoid repeated model loading. It prints per-round wall time and optional detailed pipeline summary at the end.

### Quick Example

```bash
VLLM_OMNI_USE_V2_RUNNER=1 python examples/offline_inference/qwen3_tts/end2end_benchmark_script.py \
    --query-type Base \
    --streaming \
    --log-stats \
    --enable-diffusion-pipeline-profiler \
    --test-rounds 7 \
    --warmup-rounds 3
```

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--query-type`, `-q` | str | `CustomVoice` | Query type. Choices: `CustomVoice`, `VoiceDesign`, `Base` |
| `--warmup-rounds` | int | `3` | Number of warmup rounds (results discarded) |
| `--test-rounds` | int | `7` | Number of test rounds (results saved and timed) |
| `--log-stats` | flag | disabled | Enable writing detailed statistics |
| `--streaming` | flag | disabled | Stream audio chunks via AsyncOmni |
| `--batch-size` | int | `1` | Number of prompts per batch (must be a power of two) |
| `--output-dir` | str | `output_audio` | Output directory for generated wav files |
| `--txt-prompts` | str | None | Path to a `.txt` file with one prompt per line |
| `--stage-configs-path` | str | None | Path to a stage configs file |
| `--stage-init-timeout` | int | `300` | Timeout for initializing a single stage in seconds |
| `--use-batch-sample` | flag | disabled | Use batch input sample for the selected query type |
| `--mode-tag` | str | `icl` | Mode tag for Base query. Choices: `icl`, `xvec_only` |
| `--enable-diffusion-pipeline-profiler` | flag | disabled | Enable diffusion pipeline profiler to display stage durations |

### Environment Variables

| Variable | Description |
|---|---|
| `VLLM_OMNI_USE_V2_RUNNER` | Set to `1` to use the V2 runner |

### Execution Modes

- **Sync mode** (default): Uses `Omni` for synchronous generation.
- **Streaming mode** (`--streaming`): Uses `AsyncOmni` to stream audio chunks progressively, logging TTFA and inter-chunk timing.

### Output

- Generated wav files are saved to the `--output-dir` directory, named `output_<round>_<request_id>.wav`.
- A benchmark summary is printed at the end, including per-round wall time, average, min, and max.
- When `--log-stats` is enabled, a detailed pipeline summary is also printed with per-stage timing and throughput metrics.
