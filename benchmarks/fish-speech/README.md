# Fish Speech S2 Pro Benchmark

Benchmarks for Fish Speech S2 Pro text-to-speech model, comparing vLLM-Omni streaming serving against sglang-omni as reference baseline.

Related issue: [#2432](https://github.com/vllm-project/vllm-omni/issues/2432)

Hardware note: the benchmark results collected in this directory were measured on NVIDIA H100 PCIe GPUs. Fish Audio's S2 Pro technical report reports its official numbers on H200, so absolute latency/RTF numbers should not be compared directly without accounting for the hardware difference.

## Prerequisites

```bash
# Fish Speech DAC codec dependency (required for vllm-omni to serve the model)
pip install fish-speech

# Benchmark client dependencies
pip install aiohttp numpy tqdm matplotlib
```

For sglang-omni comparison:
```bash
# Install sglang-omni with Fish Speech S2 Pro support and make sure
# `sgl-omni` is available on your PATH.
#
# One option is a source install:
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni
uv venv .venv -p 3.12 && source .venv/bin/activate
uv pip install -e ".[s2pro]"
```

## Quick Start

### 1) Start the vLLM-Omni server

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm_omni.entrypoints.cli.main serve \
    "fishaudio/s2-pro" \
    --omni --host 127.0.0.1 --port 8091 \
    --stage-configs-path benchmarks/fish-speech/config/vllm_omni/fish_speech_s2_pro.yaml \
    --trust-remote-code \
    --enforce-eager
```

### 2) (Optional) Start the sglang-omni server

```bash
sgl-omni serve --model-path fishaudio/s2-pro \
    --config benchmarks/fish-speech/config/sglang_omni/s2pro_tts_upstream.yaml \
    --port 8000
```

For high-VRAM GPUs, an additional benchmark variant is provided:

```bash
sgl-omni serve --model-path fishaudio/s2-pro \
    --config benchmarks/fish-speech/config/sglang_omni/s2pro_tts_gpu_vocoder.yaml \
    --port 8000
```

### 3) Run the benchmark script

```bash
cd benchmarks/fish-speech

# vllm-omni only
bash run_benchmark.sh

# compare both running servers
bash run_benchmark.sh --compare

# only benchmark an already-running sglang-omni server
bash run_benchmark.sh --sglang-only
```

Results (JSON + PNG plots) are saved to `results/`.

### Common options

```bash
# Compare vllm-omni against sglang-omni
bash run_benchmark.sh --compare

# Only sglang-omni
bash run_benchmark.sh --sglang-only

# Custom prompt count, concurrency levels, or ports
NUM_PROMPTS=20 CONCURRENCY="1 4" bash run_benchmark.sh
PORT=8092 SGLANG_PORT=8001 bash run_benchmark.sh --compare
```

## Manual Steps

### Run online serving benchmark against the running vLLM-Omni server
```bash
python benchmarks/fish-speech/vllm_omni/bench_fish_server.py \
    --port 8091 \
    --num-prompts 50 \
    --max-concurrency 1 4 10 \
    --config-name "vllm_omni" \
    --result-dir results/
```

### (Optional) Run the sglang-omni benchmark client directly

```bash
python benchmarks/fish-speech/sglang_omni/bench_fish_server.py \
    --port 8000 \
    --num-prompts 50 \
    --max-concurrency 1 4 10 \
    --config-name "sglang_omni" \
    --result-dir results/
```

### Generate comparison plots

```bash
python benchmarks/fish-speech/plot_results.py \
    --results benchmarks/fish-speech/results/bench_vllm_omni_*.json benchmarks/fish-speech/results/bench_sglang_omni_*.json \
    --labels "vllm-omni" "sglang-omni" \
    --title "Fish Speech S2 Pro" \
    --output benchmarks/fish-speech/results/comparison.png
```

## Stage Config

This benchmark vendors both serving configs under `benchmarks/fish-speech/config/`:

- `benchmarks/fish-speech/config/vllm_omni/fish_speech_s2_pro.yaml`
- `benchmarks/fish-speech/config/sglang_omni/s2pro_tts_upstream.yaml`
- `benchmarks/fish-speech/config/sglang_omni/s2pro_tts_gpu_vocoder.yaml`

The default vllm-omni config is a 2-stage pipeline (Slow AR -> DAC Decoder) with `async_chunk` streaming enabled, `max_num_seqs: 4` for the AR stage and `max_num_seqs: 1` for the DAC decoder. The `SharedMemoryConnector` streams codec frames (25-frame chunks with 25-frame context overlap, ~21.5 Hz codec rate).

For sglang-omni, two config variants are provided:

- `s2pro_tts_upstream.yaml` mirrors the upstream minimal config at `examples/configs/s2pro_tts.yaml` and therefore uses the upstream default behavior. In upstream `sglang-omni`, `stream_vocoder_device` is not set in the minimal YAML, but `create_sglang_tts_engine_executor(...)` defaults it to `cpu` when omitted, so this config represents the memory-conservative default path. In benchmark runs, this default path may still show partial request failures under load even without moving the streaming vocoder onto GPU.
- `s2pro_tts_gpu_vocoder.yaml` expands the pipeline explicitly and changes the streaming vocoder path to `stream_vocoder_device: cuda:0`. This variant is intended for high-VRAM GPUs to show the performance of a full-GPU streaming path.

In practice, the two sglang-omni configs illustrate a memory/performance tradeoff: the upstream default is more conservative on GPU memory, while the GPU-vocoder variant can deliver much better latency/RTF when it fits, but may OOM on smaller-memory GPUs. This memory sensitivity is itself an important comparison point in this benchmark. On H20-3e hardware, serving-time VRAM usage was approximately 127.5 GB for sglang-omni versus 87.3 GB for vllm-omni on the same Fish Speech workload. That lower memory footprint is a practical vllm-omni advantage because it enables a full-GPU pipeline on a wider range of hardware.

To use a custom vllm-omni config, start the server with:
```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm_omni.entrypoints.cli.main serve \
    "fishaudio/s2-pro" \
    --omni --host 127.0.0.1 --port 8091 \
    --stage-configs-path /path/to/custom.yaml \
    --trust-remote-code
```

## Metrics

- **TTFP (Time to First Audio Packet)**: Time from request to first audio chunk (streaming latency)
- **E2E (End-to-End Latency)**: Total time from request to complete audio response
- **RTF (Real-Time Factor)**: E2E latency / audio duration. RTF < 1.0 means faster-than-real-time synthesis
- **Throughput**: Total audio seconds generated per wall-clock second

## Architecture Notes

The benchmark scripts import shared infrastructure from `fish_bench_utils.py` (dataclasses, HTTP streaming client, metrics computation, result formatting). Only the model-specific payload construction and audio parameters live in the per-model wrappers (`vllm_omni/bench_fish_server.py` and `sglang_omni/bench_fish_server.py`).
