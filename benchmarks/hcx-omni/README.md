# HyperCLOVAX-SEED-Omni-8B Benchmarks

Measures end-to-end latency and throughput for:

| Mode | Input → Output |
|------|----------------|
| T2T  | Text → Text (thinker only) |
| T2V  | Text → Text + Image |
| S2S  | Audio + Text → Text + Audio |

## Prerequisites

Start the server first:

```bash
cd examples/online_serving/hcx_omni
./run_server.sh --model naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B
```

## Run Benchmarks

```bash
# All modes (10 requests each, sequential)
bash benchmarks/hcx-omni/run_benchmark.sh

# Custom settings via env vars
NUM_PROMPTS=50 CONCURRENCY=4 MODE=t2v bash benchmarks/hcx-omni/run_benchmark.sh

# S2S with a real audio file
python benchmarks/hcx-omni/benchmark_hcx_omni.py \
    --mode s2s --num-prompts 20 --audio-file /path/to/speech.wav

# Save results to JSON
python benchmarks/hcx-omni/benchmark_hcx_omni.py \
    --mode all --num-prompts 10 --output-json results.json
```

## Metrics

```
t2v Results:
  mode                : t2v
  total               : 10
  success             : 10
  success_rate        : 100.0%
  latency_mean        : 18.43s
  latency_p50         : 17.91s
  latency_p90         : 21.34s
  latency_p99         : 22.10s
  latency_min         : 15.20s
  latency_max         : 22.10s
```

Expected latency ranges (A100 80GB × 6):

| Mode | p50 latency | Notes |
|------|------------|-------|
| T2T  | ~2–4 s    | Thinker only |
| T2V  | ~15–25 s  | Thinker + 50-step diffusion |
| S2S  | ~5–12 s   | Thinker + BigVGAN vocoder |
