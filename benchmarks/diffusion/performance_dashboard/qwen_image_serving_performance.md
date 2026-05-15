# Qwen-Image Serving Performance Dashboard

This document describes how to deploy and benchmark **Qwen-Image** using vLLM-Omni. It includes service startup configuration, acceleration-related options, benchmark methodology, dataset settings, and performance results.

---

# 1. Overview

Qwen-Image is a multimodal text-to-image generation model served through the vLLM-Omni infrastructure.

This document covers:

* Service launch configuration (including acceleration options)
* Benchmark scripts and usage
* Dataset and workload settings
* Performance measurement results
* Reproducibility guidelines

---

# 2. Test Environment
| Component | Specification |
|------------|----------------|
| GPU | NVIDIA A100-SXM4-80GB |
| Diffusion Attention Backend | FlashAttention |

# 3. Service Launch Configuration

## 3.1 Basic Serving Command

```bash
vllm serve Qwen/Qwen-Image --omni \
    --port 8091
```

To replay step-wise continuous batching, compare
`vllm serve Qwen/Qwen-Image --omni --port 8089 --step-execution --max-num-seqs 1`
against the same command with `--max-num-seqs 8`. `--step-execution` is the
feature gate, and increasing `--max-num-seqs` above `1` lets the scheduler
keep more compatible requests active at once.

## 3.2 Key Parameters

| Parameter             | Description              |
| --------------------- | ------------------------ |
| `--cfg-parallel-size` | CFG parallelism degree   |
| `--ulysses-degree`    | Ulysses parallel degree  |
| `--vae-patch-parallel-size`    | VAE parallel degree  |
| `--tensor-parallel-size` | Tensor parallelism degree |

Record these parameters when reporting performance results.

---

# 4. Benchmark Script

## 4.1 Benchmark Entry

```bash
python benchmarks/diffusion/diffusion_benchmark_serving.py \
    --backend vllm-omni \
    --dataset <DATASET_NAME> \
    --task t2i \
    --num-prompts <N> \
    --max-concurrency <C> \
    --enable-negative-prompt \
    --random-request-config <CFG>
```

## 4.2 Key Benchmark Arguments

| Parameter              | Description                       |
| ---------------------- | --------------------------------- |
| `--backend`            | Serving backend (use `vllm-omni`) |
| `--dataset`            | Dataset name (`random` or custom) |
| `--task`               | Task type (e.g., `t2i`)           |
| `--num-prompts`        | Total number of requests          |
| `--max-concurrency`    | Client-side concurrency           |
| `--warmup-concurrency` | Warmup concurrency for batch-shape prewarming |
| `--random-request-config`| JSON string defining random request |

---

# 5. Dataset & Workload Settings

## 5.1 Recommended Evaluation Configurations

### Dataset A ( 512 Resolution)

* Dataset: `random`
* Task: t2i
* Concurrency: 1
* Mix Resolution
```
[
 {"width":512,"height":512,"num_inference_steps":20,"weight":1}
]
```

### Dataset B (1536 Resolution)

* Dataset: `random`
* Task: t2i
* Concurrency: 1
* Mix Resolution
```
[
 {"width":1536,"height":1536,"num_inference_steps":35,"weight":1}
]
```

### Dataset C (Mix Resolution)

* Dataset: `random`
* Task: t2i
* Concurrency: 1
* Mix Resolution
```
[
 {"width":512,"height":512,"num_inference_steps":20,"weight":0.15},
 {"width":768,"height":768,"num_inference_steps":20,"weight":0.25},
 {"width":1024,"height":1024,"num_inference_steps":25,"weight":0.45},
 {"width":1536,"height":1536,"num_inference_steps":35,"weight":0.15}
]
```
---

## 5.2 Example Benchmark Command

```bash
python benchmarks/diffusion/diffusion_benchmark_serving.py \
    --backend vllm-omni \
    --dataset random \
    --task t2i \
    --num-prompts 1 \
    --max-concurrency 1 \
    --enable-negative-prompt \
    --random-request-config '[
        {"width":512,"height":512,"num_inference_steps":20,"weight":1}
    ]'
```

For a continuous-batching A/B replay, run the same benchmark command twice
against the two serving configs above and keep traffic plus warmup settings
identical across both runs. If the measured run uses `--max-concurrency 8`,
warm with `--warmup-requests 8 --warmup-concurrency 8` so the first measured
batch does not include compile or CUDA-graph capture overhead.

---

# 6. Performance Metrics

The following metrics are collected during benchmarking:

| Metric             | Description                   | Unit    |
| ------------------ | ----------------------------- | ------- |
| Mean Latency        | Mean of latency       | seconds |
| P99 Latency        | P99 of latency             | seconds |

---

# 7. Performance Results

| Dataset Configuration | Max Concur. | CFG | Usp | Tp | VAE Parallel | Mean Latency (s) | P99 Latency (s) |
|-----------------------|-----|-----|-----|----|--------------|------------------|------------------|
| Dataset A | 1 | 2 | 2 | Off | Off          | 2.2087          | 2.2087          |
| Dataset B | 1 | 2 | 2 | Off | Off          | 19.6739          | 19.6739          |
| Dataset C | 1 | 2 | 2 | Off | Off          | 5.67259        | 18.6234 |
---

# 8. Reproducibility Checklist

To ensure consistent and comparable benchmark results:

* Record GPU type
* Record parallel configuration
* Record benchmark parameters (resolution, concurrency, number of prompts)
* Ensure no background workload on GPUs during testing
* When comparing continuous batching, keep warmup settings the same across runs

---

This document serves as the official Qwen-Image serving performance reference under vLLM-Omni.
