# Fish Speech S2 Pro Benchmark

Benchmarks for Fish Speech S2 Pro text-to-speech model, comparing vLLM-Omni streaming serving against sglang-omni as reference baseline.

Related issue: [#2432](https://github.com/vllm-project/vllm-omni/issues/2432)

## Prerequisites

```bash
# Fish Speech DAC codec dependency (required for vllm-omni to serve the model)
pip install fish-speech

# Benchmark client dependencies
pip install aiohttp numpy tqdm matplotlib
```

For sglang-omni comparison:
```bash
git clone https://github.com/sgl-project/sglang-omni.git ~/Dev/sglang-omni
cd ~/Dev/sglang-omni
uv venv .venv -p 3.12 && source .venv/bin/activate
uv pip install -e ".[s2pro]"
```

## Quick Start

Run the vllm-omni benchmark with a single command:

```bash
cd benchmarks/fish-speech
bash run_benchmark.sh
```

Results (JSON + PNG plots) are saved to `results/`.

### Common options

```bash
# Compare vllm-omni against sglang-omni
bash run_benchmark.sh --compare

# Only sglang-omni
bash run_benchmark.sh --sglang-only

# Custom stage config
STAGE_CONFIG=/path/to/custom.yaml bash run_benchmark.sh

# Custom GPU, prompt count, concurrency levels
GPU_DEVICE=1 NUM_PROMPTS=20 CONCURRENCY="1 4" bash run_benchmark.sh
```

## Manual Steps

### 1) Start the vLLM-Omni server

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm_omni.entrypoints.cli.main serve \
    "fishaudio/s2-pro" \
    --omni --host 127.0.0.1 --port 8091 \
    --stage-configs-path vllm_omni/model_executor/stage_configs/fish_speech_s2_pro.yaml \
    --trust-remote-code --enforce-eager
```

### 2) Run online serving benchmark

```bash
python benchmarks/fish-speech/vllm_omni/bench_tts_serve.py \
    --port 8091 \
    --num-prompts 50 \
    --max-concurrency 1 4 10 \
    --config-name "vllm_omni" \
    --result-dir results/
```

### 3) (Optional) Run sglang-omni benchmark

```bash
# Start sglang-omni server (in a separate terminal)
cd ~/Dev/sglang-omni
sgl-omni serve --model-path fishaudio/s2-pro \
    --config examples/configs/s2pro_tts.yaml --port 8000

# Run benchmark
python benchmarks/fish-speech/sglang_omni/bench_tts_serve.py \
    --port 8000 \
    --num-prompts 50 \
    --max-concurrency 1 4 10 \
    --config-name "sglang_omni" \
    --result-dir results/
```

### 4) Generate comparison plots

```bash
python benchmarks/qwen3-tts/plot_results.py \
    --results results/bench_vllm_omni_*.json results/bench_sglang_omni_*.json \
    --labels "vllm-omni" "sglang-omni" \
    --title "Fish Speech S2 Pro" \
    --output results/comparison.png
```

## Stage Config

By default, the benchmark uses the upstream stage config at `vllm_omni/model_executor/stage_configs/fish_speech_s2_pro.yaml`. This is a 2-stage pipeline (Slow AR -> DAC Decoder) with `async_chunk` streaming enabled, `max_num_seqs: 4` for the AR stage and `max_num_seqs: 1` for the DAC decoder. The `SharedMemoryConnector` streams codec frames (25-frame chunks with 25-frame context overlap, ~21.5 Hz codec rate).

To use a custom config, set `STAGE_CONFIG`:
```bash
STAGE_CONFIG=/path/to/custom.yaml bash run_benchmark.sh
```

## Metrics

- **TTFP (Time to First Audio Packet)**: Time from request to first audio chunk (streaming latency)
- **E2E (End-to-End Latency)**: Total time from request to complete audio response
- **RTF (Real-Time Factor)**: E2E latency / audio duration. RTF < 1.0 means faster-than-real-time synthesis
- **Throughput**: Total audio seconds generated per wall-clock second

## Architecture Notes

The benchmark scripts import shared infrastructure from `tts_bench_utils.py` (dataclasses, HTTP streaming client, metrics computation, result formatting). Only the model-specific payload construction and audio parameters live in the per-model wrappers (`vllm_omni/bench_tts_serve.py` and `sglang_omni/bench_tts_serve.py`).
