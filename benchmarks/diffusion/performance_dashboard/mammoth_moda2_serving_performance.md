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
| Sampling params (per request) | `text_guidance_scale: 9.0`, `cfg_range: [0.0, 1.0]`, `num_inference_steps: 50` sent via `--extra-body` / `--num-inference-steps` |

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
    --extra-body '{"text_guidance_scale": 9.0, "cfg_range": [0.0, 1.0]}' \
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

## 5.3 Stage Breakdown (concurrency 1, from server logs)

Per-stage wall time from the server's `[OmniTiming]` records across the §5.1/§5.4/§5.5 sweeps (n = 18/9/5 per row; at concurrency 1 stage wall time equals `stage_gen_time_ms` in the engine stats table):

| Resolution | Steps | AR stage (s) | DiT stage (s) | E2E (s) | AR share | AR output tokens | AR ms/token |
|------------|-------|--------------|---------------|---------|----------|------------------|-------------|
| 512x512    | 20    | 20.3         | 1.7           | 22.0    | 92%      | 1,057            | 19.2        |
| 1024x1024  | 50    | 78.9         | 17.2          | 96.1    | 82%      | 4,161            | 19.0        |
| 2048x2048  | 50    | 156.3        | 114.8         | 271.2   | 58%      | 8,154            | 19.2        |

Stage 0 is the AR prompt encoder (TTFT 82 ms at 1024x1024); stage 1 is the DiT renderer. Three observations:

1. **AR per-token cost is constant (~19 ms/token) across resolutions**, so AR time is fully predictable: the AR stage decodes a fixed visual grid whose length scales with resolution (1,057 → 4,161 → 8,154 tokens), giving `AR ≈ tokens × 19 ms`. This is also why §5.4's step sweep is sub-linear — the AR token count is step-invariant.
2. **DiT time scales super-linearly with pixels**: a 16x pixel increase (512 → 2048) multiplies DiT time by 67 (1.7 → 114.8 s).
3. **The dominant stage flips with resolution**: AR is 92% of E2E at 512x512 (DiT optimization is nearly pointless there) but 58% at 2048x2048, where both stages matter.

At concurrency > 1, per-stage wall times additionally include intra-stage queueing (each stage serializes at `max_num_seqs: 1`), so the c2/c4 rows in §5.1 cannot be decomposed additively from this table.

## 5.4 Inference-Step Sensitivity (1024x1024 / 512x512, concurrency 1)

Steps 25 at 1024x1024 (half the default 50) and steps 25 at 512x512, same protocol as §5.1:

| Resolution | Steps | Concurrency | Mean (s) | P50 (s) | P95 (s) | P99 (s) |
|------------|-------|-------------|----------|---------|---------|---------|
| 1024x1024  | 25    | 1           | 85.2     | 85.3    | 87.8    | 88.7    |
| 1024x1024  | 25    | 4           | 113.7    | 115.8   | 126.1   | 129.4   |
| 512x512    | 25    | 1           | 21.9     | 21.8    | 22.2    | 22.3    |

Halving DiT steps does not halve end-to-end latency (96.3 → 85.2 s at 1024): the AR stage emits a fixed count of visual tokens (4,161 at 1024x1024) regardless of step count, so its ~79 s contribution (§5.3) is step-invariant; only the DiT sampling time scales with steps. At 512x512, steps 20 → 25 adds ~0.1 s per step (21.8 → 21.9 s).

## 5.5 Prompt-Length Sensitivity (long vs short prompts)

A ~430-word long prompt (repeated via `--random-request-config '[{"weight":1,"prompt":"..."}]'`) against the default short synthetic prompt, same seed and settings:

| Resolution | Steps | Concurrency | Prompt | Mean (s) | P99 (s) | Throughput (img/s) |
|------------|-------|-------------|--------|----------|---------|---------------------|
| 1024x1024  | 50    | 1           | short  | 96.3     | 100.8   | 0.0104              |
| 1024x1024  | 50    | 1           | long   | 93.6     | 94.4    | 0.0107              |
| 1024x1024  | 50    | 4           | short  | 150.9    | 186.3   | 0.0238              |
| 1024x1024  | 50    | 4           | long   | 141.7    | 174.2   | 0.0255              |
| 512x512    | 20    | 1           | short  | 21.8     | 22.2    | 0.0459              |
| 512x512    | 20    | 1           | long   | 22.2     | 22.5    | 0.0450              |

Prompt length has no material effect on latency (the long-prompt rows are within run-to-run variance and even slightly faster with tighter P99 tails). This confirms the AR stage's cost is dominated by decoding the fixed 4,161-token visual grid, not by prompt encoding: the KV prefill of a few hundred extra text tokens is negligible against 4,161 decode steps.

## 5.6 Peak GPU Memory (per-stage process, nvidia-smi polling at 1 Hz)

Sampled during the §5.4/§5.5 sweeps (stage processes identified by PID; both stages share the single A800):

| Configuration (worst case) | stage0_ar (MB) | stage1_dit (MB) |
|----------------------------|----------------|-----------------|
| 1024x1024, 50 steps, c4    | 39,290         | 11,104          |
| all other configs          | 39,194         | 11,102–11,104   |

Memory is essentially flat across the sweep: the AR stage grows by only ~96 MB at concurrency 4 and the DiT stage by 2 MB, because both stages pre-allocate from `gpu_memory_utilization` budgets (0.5 / 0.3) at startup. Combined steady-state usage ≈ 50.4 GiB of the 80 GiB device.

## 5.7 DiT-Stage Component Attribution (pipeline profiler)

With `enable_diffusion_pipeline_profiler: true` in the stage-1 deploy config (`MammothModa2DiTPipeline` targets: `gen_transformer.forward`, `gen_vae.decode`, `gen_image_condition_refiner.forward`), three warm 1024x1024 / 50-step requests (profiler adds a `synchronize` per call, so timings are indicative, not throughput-grade):

| Component | Per request (s) | Share of DiT pipeline |
|-----------|-----------------|-----------------------|
| `gen_transformer.forward` (100 calls: 50 steps x 2 passes) | 14.01–14.07 | 98.6% |
| `gen_vae.decode` (1 call) | 0.148 | 1.0% |
| `gen_image_condition_refiner.forward` (1 call) | 0.005 | 0.03% |
| whole `MammothModa2DiTPipeline.forward` | 14.18–14.27 | — |

The DiT latency is ~99% transformer sampling; VAE decode costs only ~0.15 s per image at 1024x1024 and is not a bottleneck. Per-step transformer time is ~140 ms (two forward passes per step: ~2.8 s nominal x CFG), consistent with §5.3's 17.2 s image-TTFT figure once base64 encoding and stage handoff are included.

## 5.8 Endpoint Fix Verification

Before the serving fix, `/v1/images/generations` returned HTTP 503 ("No diffusion stage found in multi-stage pipeline") because the pipeline's image-output stage is a generation-LLM DiT rather than a `diffusion`-typed stage. After the fix all requests return HTTP 200 with a valid base64 image (verified across every sweep configuration above).

---

# 6. Reproducibility

1. Start the server as in §3.1 and wait for `Application startup complete`.
2. Warm up with one real request (any size) — do not use reduced-step warmups.
3. Run the sweep commands from §4.1 for each row in §5.
4. Server-side per-stage timings are printed in the engine stats table (`stage_gen_time_ms`, `output_unit_count`) at `--log-stats` level info.
5. New rows use the same seed (`--seed 142`), `--num-prompts 8`, and `--warmup-requests 0` protocol as §4.1; when saving results with `--output-file`, name them `<res>_s<steps>_c<conc>[_long].json` to keep runs comparable (steps-25, long-prompt, peak-memory and profiler runs).
6. To reproduce §5.6 peak-memory numbers, poll `nvidia-smi --query-gpu=memory.used --format=csv -l 1` per stage PID while a sweep is in flight (e.g. `nvidia-smi pmon` / `--query-compute-apps`).
7. To reproduce §5.7, copy the bundled deploy config, set `enable_diffusion_pipeline_profiler: true` on the stage-1 entry, restart, and grep `[DiffusionPipelineProfiler]` lines from the server log; timings include a `torch.cuda.synchronize` per call.
