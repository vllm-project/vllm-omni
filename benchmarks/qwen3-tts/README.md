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

# Use a Voice Clone model
MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base TASK_TYPE=Base bash run_benchmark.sh --async-only

# Use batch size 16 for higher throughput
BATCH_SIZE=16 bash run_benchmark.sh --async-only

# Custom GPU, prompt count, concurrency levels
GPU_DEVICE=1 NUM_PROMPTS=20 CONCURRENCY="1 4" bash run_benchmark.sh
```

## Manual Steps

### 1) Start the vLLM-Omni server

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm_omni.entrypoints.cli.main serve \
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice" \
    --omni --host 127.0.0.1 --port 8000 \
    --deploy-config vllm_omni/deploy/qwen3_tts.yaml \
    --stage-overrides '{"0":{"max_num_seqs":1,"gpu_memory_utilization":0.3,"max_num_batched_tokens":512},"1":{"max_num_seqs":1,"gpu_memory_utilization":0.3,"max_num_batched_tokens":8192}}' \
    --trust-remote-code
```

### 2) Run online serving benchmark

```bash
python benchmarks/qwen3-tts/vllm_omni/bench_tts_serve.py \
    --port 8000 \
    --num-prompts 50 \
    --max-concurrency 1 4 10 \
    --config-name "async_chunk" \
    --result-dir results/
```

### 3) Run HuggingFace baseline

```bash
python benchmarks/qwen3-tts/transformers/bench_tts_hf.py \
    --model "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice" \
    --num-prompts 50 \
    --gpu-device 0 \
    --result-dir results/
```

### 4) Generate comparison plots

```bash
python benchmarks/qwen3-tts/plot_results.py \
    --results results/bench_async_chunk_*.json results/bench_hf_transformers_*.json \
    --labels "vllm-omni" "hf_transformers" \
    --output results/comparison.png
```

## Batch-size presets

The bench script loads the bundled production deploy (`vllm_omni/deploy/qwen3_tts.yaml`) and layers per-stage budgets on top via `--stage-overrides`, driven by the `BATCH_SIZE` env var. Each batch size picks compatible per-stage `max_num_seqs`, `max_num_batched_tokens`, and `gpu_memory_utilization` defaults:

| `BATCH_SIZE` | Description |
|:--:|-------------|
| `1` (default) | Single-request processing (lowest latency) |
| `4`  | Moderate-throughput concurrent processing |
| `16` | High-throughput concurrent processing |

The 2-stage pipeline (Talker -> Code2Wav) runs with `async_chunk` streaming enabled via the prod deploy; the `SharedMemoryConnector` streams codec frames (25-frame chunks with 25-frame context overlap) between stages.

The model is specified via the CLI `--model` flag (or `MODEL` env var), so the same bench script works for both the 0.6B and 1.7B model variants.

## Metrics

- **TTFP (Time to First Audio Packet)**: Time from request to first audio chunk (streaming latency)
- **E2E (End-to-End Latency)**: Total time from request to complete audio response
- **RTF (Real-Time Factor)**: E2E latency / audio duration. RTF < 1.0 means faster-than-real-time synthesis
- **Throughput**: Total audio seconds generated per wall-clock second
