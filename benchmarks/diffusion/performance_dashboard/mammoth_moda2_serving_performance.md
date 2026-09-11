# MammothModa2-Preview Serving Performance Dashboard

This document describes how to deploy and benchmark **bytedance-research/MammothModa2-Preview** (T2I) using vLLM-Omni. It includes service startup configuration, benchmark methodology, dataset settings, and performance results.

MammothModa2 is a two-stage AR→DiT text-to-image pipeline: stage 0 is an autoregressive LLM that encodes the prompt into a visual-token grid, and stage 1 is a generation-LLM DiT that renders the final image (`final_output_type: image`). Unlike single-stage diffusion pipelines, `/v1/images/generations` requests traverse both stages, and the DiT stage runs with `max_num_seqs: 1` in the default deploy config, so concurrent requests serialize on the DiT stage while the AR stage batches.

> **Legacy-topology baseline**: all numbers in this dashboard were measured on the legacy topology, where the MammothModa2 DiT stage runs as an LLM-generation stage (`stage_type: llm`) built from `VllmConfig`, served through the LLM-typed image-stage compatibility path. #7134 migrates this pipeline to the shared diffusion runtime (DiT registered as a standard `DIFFUSION` stage built from `OmniDiffusionConfig`). These results should be treated as the pre-migration baseline and are expected to be rerun after #7134 lands; they are not directly comparable with post-migration numbers.

---

# 1. Overview

This document covers:

* Service launch configuration
* Benchmark scripts and usage
* Dataset and workload settings
* Performance measurement results
* Reproducibility guidelines

---

# 2. Test Environment

| Component | Specification |
|------------|----------------|
| GPU | NVIDIA A800-SXM4-80GB (1x) |
| CPU | 18 cores |
| Model weights | `bytedance-research/MammothModa2-Preview` |
| Deploy config | `vllm_omni/deploy/mammoth_moda2.yaml` (bundled) |
| Stage 0 (AR) | `gpu_memory_utilization: 0.5`, `enforce_eager: true`, `max_num_seqs: 100` |
| Stage 1 (DiT) | `gpu_memory_utilization: 0.3`, `enforce_eager: true`, `max_num_seqs: 1` |
| Default sampling | `text_guidance_scale: 9.0`, `cfg_range: [0.0, 1.0]`, `num_inference_steps: 50` |

---

# 3. Service Launch Configuration

## 3.1 Basic Serving Command

```bash
vllm-omni serve bytedance-research/MammothModa2-Preview --omni \
    --port 8091 \
    --trust-remote-code
```

When serving from a local weights directory (no `model_index.json`), pass the bundled deploy config explicitly so the `mammoth_moda2` pipeline is selected:

```bash
vllm-omni serve /path/to/MammothModa2-Preview --omni \
    --port 8091 \
    --trust-remote-code \
    --deploy-config vllm_omni/deploy/mammoth_moda2.yaml
```

## 3.2 Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--deploy-config` | Path to the deploy YAML; defaults to the bundled `mammoth_moda2.yaml` for the HF checkpoint |
| `size` (request) | `WxH`, e.g. `1024x1024`; routed to the AR grid and the DiT `target_w/target_h` |
| `num_inference_steps` (request) | DiT sampling steps (default 50) |
| `guidance_scale` (request) | Text guidance scale (default 9.0) |

Note: the benchmark script's `--warmup-num-inference-steps` shortcut is not usable for this pipeline — a reduced step count truncates the AR stage's visual-token output and crashes the DiT stage. Use `--warmup-requests 0` and warm up with a real request instead.

---

# 4. Benchmark Script

## 4.1 Benchmark Entry

```bash
python benchmarks/diffusion/diffusion_benchmark_serving.py \
    --base-url http://localhost:8091 \
    --model bytedance-research/MammothModa2-Preview \
    --endpoint /v1/images/generations \
    --task t2i \
    --dataset random \
    --num-prompts 8 \
    --width <W> --height <H> \
    --num-inference-steps <S> \
    --max-concurrency <C> \
    --warmup-requests 0 \
    --seed 142 \
    --output-file <OUT>.json
```

## 4.2 Dataset

`random` synthetic T2I prompts, fixed seed 142, 8 prompts per configuration (4 for 2048x2048 due to per-request latency).

---

# 5. Performance Results

All requests succeeded (0 failures across the sweep). Latency is end-to-end per request (AR + DiT + VAE decode + base64 response).

> Note: these are legacy-topology baseline results (see the callout in the overview); rerun after the #7134 diffusion-runtime migration before comparing across topologies.

## 5.1 Latency

| Resolution | Steps | Concurrency | Mean (s) | P50 (s) | P95 (s) | P99 (s) |
|------------|-------|-------------|----------|---------|---------|---------|
| 512x512    | 20    | 1           | 21.8     | 21.7    | 22.1    | 22.2    |
| 512x512    | 20    | 2           | 24.8     | 25.2    | 25.6    | 25.8    |
| 512x512    | 20    | 4           | 28.5     | 29.2    | 31.1    | 31.8    |
| 1024x1024  | 50    | 1           | 96.3     | 95.8    | 99.8    | 100.8   |
| 1024x1024  | 50    | 2           | 121.5    | 125.4   | 127.9   | 128.5   |
| 1024x1024  | 50    | 4           | 150.9    | 157.6   | 178.6   | 186.3   |
| 2048x2048  | 50    | 1           | 271.9    | 271.6   | 273.6   | 273.8   |

## 5.2 Throughput

| Resolution | Steps | Concurrency | Throughput (img/s) |
|------------|-------|-------------|--------------------|
| 512x512    | 20    | 1           | 0.0459             |
| 512x512    | 20    | 2           | 0.0798             |
| 512x512    | 20    | 4           | 0.1329             |
| 1024x1024  | 50    | 1           | 0.0104             |
| 1024x1024  | 50    | 2           | 0.0162             |
| 1024x1024  | 50    | 4           | 0.0238             |
| 2048x2048  | 50    | 1           | 0.0037             |

## 5.3 Stage Breakdown (1024x1024, 50 steps, concurrency 1, from server logs)

| Stage | Role | Gen time (s) | Notes |
|-------|------|--------------|-------|
| 0     | AR prompt encoder | 77.4 | 4,161 output tokens (64x64 grid + 1), TTFT 82 ms, 18.6 ms/token |
| 1     | DiT renderer      | 17.2 | 1,048,576 pixels (1024x1024), image TTFT 17.2 s |

The AR stage dominates end-to-end latency (~80%) at the default configuration; the DiT stage is the serialization bottleneck under concurrency (`max_num_seqs: 1`).

## 5.4 Endpoint Fix Verification

Before the serving fix, `/v1/images/generations` returned HTTP 503 ("No diffusion stage found in multi-stage pipeline") because the pipeline's image-output stage is a generation-LLM DiT rather than a `diffusion`-typed stage. After the fix all requests return HTTP 200 with a valid base64 image (verified across every sweep configuration above).

---

# 6. Reproducibility

1. Start the server as in §3.1 and wait for `Application startup complete`.
2. Warm up with one real request (any size) — do not use reduced-step warmups.
3. Run the sweep commands from §4.1 for each row in §5.
4. Server-side per-stage timings are printed in the engine stats table (`stage_gen_time_ms`, `output_unit_count`) at `--log-stats` level info.

Raw per-configuration JSON outputs are stored under `benchmarks/diffusion/performance_dashboard/mammoth_moda2_serving_results/`.
