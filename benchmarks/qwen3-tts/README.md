# Qwen3-TTS Benchmark

Benchmarks for Qwen3-TTS text-to-speech models, comparing vLLM-Omni streaming serving against HuggingFace Transformers offline inference.

## Prerequisites

```bash
pip install matplotlib aiohttp soundfile numpy tqdm
pip install qwen_tts  # for HF baseline
```

## Quick Start

Run the full benchmark (vllm-omni + HF baseline) with a single command:

```bash
cd benchmarks/qwen3-tts
bash run_benchmark.sh
```

Results (JSON + PNG plots) are saved to `results/`.

### Common options

```bash
# Only vllm-omni (skip HF baseline)
bash run_benchmark.sh --async-only

# Only HF baseline
bash run_benchmark.sh --hf-only

# Use a different model (e.g. 1.7B)
MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice bash run_benchmark.sh --async-only

# Use batch_size=4 config for higher throughput
STAGE_CONFIG=vllm_omni/configs/qwen3_tts_bs4.yaml bash run_benchmark.sh --async-only

# Custom GPU, prompt count, concurrency levels
GPU_DEVICE=1 NUM_PROMPTS=20 CONCURRENCY="1 4" bash run_benchmark.sh
```

## Manual Steps

### 1) Start the vLLM-Omni server

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm_omni.entrypoints.cli.main serve \
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice" \
    --omni --host 127.0.0.1 --port 8000 \
    --stage-configs-path benchmarks/qwen3-tts/vllm_omni/configs/qwen3_tts_bs1.yaml \
    --trust-remote-code
```

### 2) Run online serving benchmark

```bash
python vllm_omni/bench_tts_serve.py \
    --port 8000 \
    --num-prompts 50 \
    --max-concurrency 1 4 10 \
    --config-name "async_chunk" \
    --result-dir results/
```

### 3) Run HuggingFace baseline

```bash
python transformers/bench_tts_hf.py \
    --model "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice" \
    --num-prompts 50 \
    --gpu-device 0 \
    --result-dir results/
```

### 4) Generate comparison plots

```bash
python plot_results.py \
    --results results/bench_async_chunk_*.json results/bench_hf_transformers_*.json \
    --labels "vllm-omni" "hf_transformers" \
    --output results/comparison.png
```

## Stage Configs

| Config | Batch Size | Description |
|--------|:----------:|-------------|
| `vllm_omni/configs/qwen3_tts_bs1.yaml` | 1 | Single-request processing (lower latency) |
| `vllm_omni/configs/qwen3_tts_bs4.yaml` | 4 | Concurrent request processing (higher throughput) |

Both configs use a 2-stage pipeline (Talker -> Code2Wav) with `async_chunk` streaming enabled. The `SharedMemoryConnector` streams codec frames (25-frame chunks with 25-frame context overlap) between stages.

The model is specified via the CLI `--model` flag (or `MODEL` env var in `run_benchmark.sh`), so the same configs work for both the 0.6B and 1.7B model variants.

## Metrics

- **TTFP (Time to First Audio Packet)**: Time from request to first audio chunk (streaming latency)
- **E2E (End-to-End Latency)**: Total time from request to complete audio response
- **RTF (Real-Time Factor)**: E2E latency / audio duration. RTF < 1.0 means faster-than-real-time synthesis
- **Throughput**: Total audio seconds generated per wall-clock second
