# Wan2.2 Serving Performance Dashboard

This document describes how to deploy and benchmark **Wan-AI/Wan2.2-T2V-A14B-Diffusers** using vLLM-Omni. It includes service startup configuration, acceleration-related options, benchmark methodology, dataset settings, and performance results.

---

# 1. Overview

Wan-AI/Wan2.2-T2V-A14B-Diffusers is a multimodal text-to-video generation model served through the vLLM-Omni infrastructure.

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
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --omni \
    --port 8091
```

## 3.2 Key Parameters

| Parameter             | Description              |
| --------------------- | ------------------------ |
| `--cfg-parallel-size` | CFG parallelism degree   |
| `--ulysses-degree`    | Ulysses parallel degree  |
| `--vae-patch-parallel-size`    | VAE parallel degree  |
| `--tensor-parallel-size` | Tensor parallelism degree |
| `--use-hsdp` | Enable Hybrid Sharded Data Parallel to shard model weights across GPUs |

Record these parameters when reporting performance results.

---

# 4. Benchmark Script

## 4.1 Benchmark Entry

```bash
python benchmarks/diffusion/diffusion_benchmark_serving.py \
    --backend v1/videos \
    --dataset <DATASET_NAME> \
    --task t2v \
    --num-prompts <N> \
    --max-concurrency <C> \
    --enable-negative-prompt \
    --random-request-config <CFG>
```

## 4.2 Key Benchmark Arguments

| Parameter              | Description                       |
| ---------------------- | --------------------------------- |
| `--backend`            | Serving backend (use `v1/videos`) |
| `--dataset`            | Dataset name (`random` or custom) |
| `--task`               | Task type (e.g., `t2v`)           |
| `--num-prompts`        | Total number of requests          |
| `--max-concurrency`    | Client-side concurrency           |
| `--random-request-config`| JSON string defining random request |

---

# 5. Dataset & Workload Settings

## 5.1 Recommended Evaluation Configurations

### Dataset A (480p)

* Dataset: `random`
* Task: t2v
* Concurrency: 1
* Mix Resolution
```
[
    {"width":854,"height":480,"num_inference_steps":3,"num_frames":80,"fps":16,"weight":1}
]
```
### Dataset B (720p)

* Dataset: `random`
* Task: t2v
* Concurrency: 1
* Mix Resolution
```
[
    {"width":1280,"height":720,"num_inference_steps":6,"num_frames":80,"fps":16,"weight":1}
]
```
### Dataset C (Mix Resolution)

* Dataset: `random`
* Task: t2v
* Concurrency: 1
* Mix Resolution
```
[
 {"width":854,"height":480,"num_inference_steps":3,"num_frames":80,"fps":16,"weight":0.15},
 {"width":854,"height":480,"num_inference_steps":4,"num_frames":120,"fps":24,"weight":0.25},
 {"width":1280,"height":720,"num_inference_steps":6,"num_frames":80,"fps":16,"weight":0.6}
]
```
---

## 5.2 Example Benchmark Command

```bash
python benchmarks/diffusion/diffusion_benchmark_serving.py \
    --backend v1/videos \
    --dataset random \
    --task t2v \
    --num-prompts 1 \
    --max-concurrency 1 \
    --enable-negative-prompt \
    --random-request-config '[
        {"width":854,"height":480,"num_inference_steps":18,"num_frames": 33,"fps":16",weight":1}
    ]'
```

---

# 6. Performance Metrics

The following metrics are collected during benchmarking:

| Metric             | Description                   | Unit    |
| ------------------ | ----------------------------- | ------- |
| Mean Latency        | Mean of latency       | seconds |
| P99 Latency        | P99 of latency             | seconds |

---

# 7. Performance Results

| Dataset Configuration | Max Concur. | CFG | Usp | Tp | Hsdp | VAE Parallel | Mean Latency (s) | P99 Latency (s) |
|-----------------------|-----|-----|-----|-----|----|--------------|------------------|------------------|
| Dataset A | 1 | 2 | 2 | 1 | On | 1          | 24.6766          | 24.6766          |
| Dataset A | 1 | 2 | 2 | 1 | On | 4          | 21.6810          | 21.6810          |
| Dataset B | 1 | 2 | 2 | 1 | On | 1          | 124.6639         | 124.6639          |
| Dataset B | 1 | 2 | 2 | 1 | On | 4          | 117.44          | 117.44          |
| Dataset C | 1 | 2 | 2 | 1 | On | 1          | 79.2175        | 124.2565 |
| Dataset C | 1 | 2 | 2 | 1 | On | 4          | 74.4977        | 117.710 |
---

# 8. Reproducibility Checklist

To ensure consistent and comparable benchmark results:

* Record GPU type
* Record parallel configuration
* Record benchmark parameters (resolution, concurrency, number of prompts)
* Ensure no background workload on GPUs during testing

---

This document serves as the official Wan2.2 serving performance reference under vLLM-Omni.
