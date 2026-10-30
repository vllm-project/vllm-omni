# MiniMax H3 on Ascend 950PR / 950DT

> Joint video and audio generation — Ascend NPU deployment guide for the
> four-card Ascend 950PR / 950DT route

## Summary

- Vendor: MiniMaxAI
- Model: [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- Tasks: T2VA, FL2VA, and Ref2VA
- Mode: OpenAI-compatible `/v1/videos` HTTP serving
- Hardware: Ascend 950PR / Ascend 950DT, 4x NPU (128 GB HBM per device)
- Maintainer: Community

This recipe covers the four-card 950PR / 950DT configuration at 1344x768.
For the eight-card Atlas 800I A2 / A3 route, see
[MiniMax-H3-NPU.md](MiniMax-H3-NPU.md); checkpoint layout, container
preparation, MindIE-SD installation, and the ffmpeg/decord dependencies are
described there and apply unchanged.

## Environment

Use the A5 variant of the official vLLM-Omni NPU images (see
[NPU installation](../../docs/getting_started/installation/npu.md)):

```bash
export IMAGE=quay.io/ascend/vllm-omni:v0.29.0-a5
```

Check the [tag list](https://quay.io/repository/ascend/vllm-omni?tab=tags)
for newer releases. Inside the container, install MindIE-SD and the optional
Ref2VA media dependencies as described in
[MiniMax-H3-NPU.md § Environment](MiniMax-H3-NPU.md#environment).

## Start a server

Task selection follows the eight-card recipe: pass `--task-type fl2va` or
`--task-type ref2va` to pick the task partition. The configurations below use
4-card USP, 4-card text-encoder TP, and VAE `tile` parallelism.

The two recommended configurations differ in how they fit memory: the
lossless configuration enables distributed layerwise offload (DLO) to avoid
activation OOM at longer durations, while the lossy configuration fits the
maximum supported MiniMax-H3 workload shape without OOM, so DLO is omitted
for performance.

### Recommended lossless configuration (Ascend 950PR / 950DT)

```bash
export PORT=9098
export MODEL=/path/to/MiniMax-H3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=4000
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_NPU_SOCKET_PORT_RANGE="auto"

vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --task-type fl2va \
  --num-gpus 4 \
  --usp 4 \
  --ring 1 \
  --text-encoder-tp-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --vae-patch-parallel-size 4 \
  --enable-diffusion-pipeline-profiler \
  --enable-distributed-layerwise-offload \
  --diffusion-attention-backend FLASH_ATTN
```

H3 is CFG-distilled, so `--cfg-parallel-size` must remain 1. The first
request includes regional compilation; warm the server once before measuring
steady-state latency.

### Recommended lossy configuration (Ascend 950PR / 950DT)

Adds EQBSA sparse attention (block-sparse with Q/K INT8 + V FP8 mixed
precision) and MXFP8 online quantization, and drops DLO:

```bash
export PORT=9098
export MODEL=/path/to/MiniMax-H3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=4000
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_NPU_SOCKET_PORT_RANGE="auto"

vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --task-type fl2va \
  --num-gpus 4 \
  --usp 4 \
  --ring 1 \
  --text-encoder-tp-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --vae-patch-parallel-size 4 \
  --enable-diffusion-pipeline-profiler \
  --diffusion-attention-config '{"default":{"backend":"RAINFUSION_ATTN","block_sparse":{"sparsity":0.8,"precision":"mix","start_step":8,"end_step":12}}}' \
  --diffusion-quantization-config '{"transformer":{"method":"mxfp8"}}'
```

The recommended lossy configuration uses a **start=8 / end=12** dense-fallback
window; see below for more aggressive settings.

### Optional optimizations

The following sections list only the **increment or replacement** relative to
the recommended lossless configuration; environment variables and the
remaining flags stay unchanged.

#### Tuning the EQBSA dense-fallback window

`start_step` and `end_step` are the numbers of leading/trailing steps that
fall back to dense attention — not step indices. `precision` defaults to
`bf16`; EQBSA requires setting it to `mix` explicitly. For 1344x768, 50-step
T2VA:

| Configuration | `start_step` | `end_step` | Notes |
| --- | ---: | ---: | --- |
| Quality first (recommended) | 8 | 13 | General and complex-motion scenes |
| Balanced | 0 | 13 | Balanced |
| Speed first | 0 | 0 | Simple, low-motion scenes |

When frames look discontinuous or details unstable, increase `end_step`
first, then `start_step`; do not compensate by lowering `sparsity` alone.
These pairings were validated at 1344x768 / 50 steps only; re-evaluate for
other resolutions or step counts.

#### Cache-DiT

Append to enable DiT block caching with TaylorSeer extrapolation — see the
[Cache-DiT guide](../../docs/user_guide/diffusion/cache_acceleration/cache_dit.md):

```bash
  --cache-backend cache_dit \
  --enable-cache-dit-summary \
  --cache-config '{"Fn_compute_blocks":2,"Bn_compute_blocks":1,"max_warmup_steps":4,"residual_diff_threshold":0.4,"max_continuous_cached_steps":4,"enable_taylorseer":true,"taylorseer_order":2}'
```

#### MXFP4 online quantization

Append to switch the DiT to MXFP4 online quantization:

```bash
  --diffusion-quantization-config '{"transformer":{"method":"mxfp4"}}'
```

and remove `--enable-distributed-layerwise-offload`.

## Request examples

Identical to the eight-card recipe; see
[MiniMax-H3-NPU.md § HTTP API examples](MiniMax-H3-NPU.md#http-api-examples)
and the GPU recipe's full parameter table.

## Benchmarks (Ascend 950PR)

Measured with `--enable-diffusion-pipeline-profiler` on vLLM-Omni 0.28.0:
lossless rows use the recommended lossless configuration (with DLO), lossy
rows the recommended lossy configuration (EQBSA + MXFP8, without DLO). Memory
figures are per device.

| Task | Resolution | Frames | Duration (s) | Config | E2E (s) | DiT total (s) | DiT per-step (s) | DiT steps | Text encode (s) | Ref video encode (s) | Ref audio encode (s) | VAE decode (s) | Ref preprocess (s) | Post-process (s) | CPU MP4 (s) | Resident weights (GB) | Peak memory (GB) |
| ---- | ---- | ---- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| t2va | 1344x768 | 124 | 5 | lossless | 203.95 | 198.10 | 4.04 | 49 | 0.57 | NA | NA | 5.05 | NA | 0.001 | 0.23 | 3.97 | 22.2 |
| t2va | 1344x768 | 362 | 15 | lossless | 1489.17 | 1474.35 | 30.08 | 49 | 0.69 | NA | NA | 13.45 | NA | 0.001 | 0.82 | 3.97 | 32.14 |
| t2va | 1344x768 | 124 | 5 | lossy | 113.86 | 109.81 | 2.24 | 49 | 0.03 | NA | NA | 3.82 | NA | 0.001 | 0.21 | 56.99 | 64.38 |
| t2va | 1344x768 | 362 | 15 | lossy | 762.17 | 749.40 | 15.29 | 49 | 0.03 | NA | NA | 12.19 | NA | 0.001 | 0.61 | 56.99 | 74.26 |
| ref2va | 1344x768 | 124 | 5 | lossless | 745.43 | 730.22 | 14.9 | 49 | 1.16 | 8.15 | 0.13 | 5.12 | 0.41 | 0.001 | 0.28 | 3.97 | 23.82 |
| ref2va | 1344x768 | 362 | 15 | lossless | 5887.24 | 5847.57 | 119.33 | 49 | 2.54 | 22.07 | 0.23 | 13.34 | 0.81 | 0.001 | 0.70 | 3.97 | 32.53 |
| ref2va | 1344x768 | 124 | 5 | lossy | 400.40 | 387.82 | 7.91 | 49 | 0.95 | 7.05 | 0.09 | 3.91 | 0.41 | 0.001 | 0.21 | 56.6 | 66.03 |
| ref2va | 1344x768 | 362 | 15 | lossy | 3037.84 | 3001.31 | 61.24 | 49 | 2.28 | 20.64 | 0.15 | 12.1 | 0.81 | 0.001 | 0.70 | 56.6 | 74.76 |

The resident-weight figures reflect the DLO strategy difference: with DLO
(lossless rows) DiT weights live in host memory and only the active layer
stays on the device, while without DLO (lossy rows) the quantized weights are
resident in HBM. Peak memory stays well inside the 128 GB per-device capacity
in both cases. These numbers describe the validated shapes rather than a
general throughput guarantee.

## Known limitations

- Task serving is partitioned by `--task-type`: T2VA/FL2VA and Ref2VA load
  different DiTs, so switching between them requires a restart with the other
  partition.
- H3 currently executes one generation request per diffusion batch.
- The first regional-compile request is a warmup and should not be included
  in steady-state performance measurements.
- VAE patch parallelism requires size 1 or the full DiT group size and
  supports the H3 native `tile` mode only.
- The lossy configuration (EQBSA + MXFP8) is validated for T2VA and Ref2VA;
  use the lossless configuration for FL2VA or re-validate first.
- The lossless configuration trades throughput for activation headroom via
  DLO; if your workload never approaches the maximum duration, dropping
  `--enable-distributed-layerwise-offload` may improve latency at the risk of
  OOM on longer durations.

## Additional resources

- [MiniMax-H3.md](MiniMax-H3.md) — full GPU guide
- [MiniMax-H3-NPU.md](MiniMax-H3-NPU.md) — eight-card Atlas 800I A2 / A3
  guide (checkpoint, container, and dependency details)
- [NPU installation](../../docs/getting_started/installation/npu.md)
- [RainFusion attention](../../docs/user_guide/diffusion/attention_backends/rainfusion.md)
  — block-sparse knobs and tuning
- [Online quantization](../../docs/user_guide/quantization/online.md)
  — INT8 / MXFP8 / MXFP4
- [Cache-DiT guide](../../docs/user_guide/diffusion/cache_acceleration/cache_dit.md)
- [Supported models](../../docs/models/supported_models.md)
- [Video API](../../docs/serving/videos_api.md)
