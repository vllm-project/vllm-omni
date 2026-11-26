# MiniMax H3 on Ascend 950PR / 950DT

> Joint video and audio generation — Ascend NPU deployment guide for
> four-card and eight-card Ascend 950PR / 950DT routes

## Summary

- Vendor: MiniMaxAI
- Model: [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- Tasks: T2VA, FL2VA, and Ref2VA
- Mode: OpenAI-compatible `/v1/videos` HTTP serving
- Hardware: Ascend 950PR / Ascend 950DT, 4x or 8x NPU
  (128 GB HBM per device)
- Maintainer: Community

This recipe covers four-card and eight-card 950PR / 950DT configurations at
1344x768. The eight-card profile was qualified on a host with two four-NPU
UB domains.
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

### Eight-card configuration on a split-domain 950PR host

This profile was qualified on an x86_64 Atlas 150 host with eight 128 GB
Ascend 950PR NPUs. NPU 0-3 form one UB domain and NPU 4-7 form the other;
communication between the domains traverses SYS/NUMA.

On this tested topology, do not set `HCCL_NPU_SOCKET_PORT_RANGE=auto`. With
automatic NPU-side socket-port allocation enabled, startup and `/health`
succeeded, but the first video request failed during `HcclBroadcast` on the
rank 0 to rank 4 path. Leaving the variable unset restored successful
generation. See
[#10645](https://github.com/vllm-project/vllm-omni/issues/10645).

The following T2VA BF16 command was qualified with the software environment
listed in the eight-card evidence section below:

```bash
export PORT=9098
export MODEL=/path/to/MiniMax-H3
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=14400
export VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT=14400
export VLLM_OMNI_INPUT_WAIT_TIMEOUT_S=14400
export VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=14400
export HCCL_CONNECT_TIMEOUT=120
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
unset HCCL_NPU_SOCKET_PORT_RANGE

vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --task-type fl2va \
  --num-gpus 8 \
  --usp 8 \
  --ring 1 \
  --text-encoder-tp-size 8 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --vae-patch-parallel-size 8 \
  --enable-diffusion-pipeline-profiler
```

The `--task-type fl2va` partition serves both T2VA and FL2VA. Restart with
`--task-type ref2va` and the Ref2VA checkpoint for Ref2VA requests. The
qualification matrix records each task's tested DiT parallel layout; do not
assume that one layout is interchangeable across all three tasks.

### Ref2VA on a 950PR that reports ~86 GB HBM

The sections above match a 128 GB part. On a 950PR whose `npu-smi` reports
86016 MB, `--usp 4` does not shard the DiT: Ulysses shards the sequence, and
each card holds a full copy of the weights. The Ref2VA DiT is about 62 GB in
bf16, and `vllm serve` with `--usp 4` and no quantization raises
`torch.OutOfMemoryError` after about 79 GiB is already allocated. Tensor
parallel 4 in bf16 uses about 15.5 GB per card. USP4 plus MXFP8 uses about
31 GB per card and fits.

For Ref2VA on that device, keep the environment variables and server flags
of the recommended lossy configuration, switch `--task-type` to `ref2va`,
and narrow the dense window with the flags below. `skip_layers` keeps those
DiT blocks dense on every step. The first request is a warmup; exclude it
from steady-state measurements. Measured at 1344x768, 24 fps, 50 steps,
vLLM-Omni 0.32.0, MindIE-SD 3.2.0.

| Point | `start_step` | `end_step` | `skip_layers` | 5 s diffuse |
| --- | ---: | ---: | --- | ---: |
| Speed | 4 | 4 | `0-1,48-49` | 312.5 s |
| Sharper | 6 | 6 | `0-2,47-49` | 354.2 s |

```bash
  --task-type ref2va \
  --diffusion-attention-config '{"default":{"backend":"RAINFUSION_ATTN","block_sparse":{"sparsity":0.8,"precision":"mix","start_step":4,"end_step":4,"skip_layers":"0-1,48-49"}}}' \
  --diffusion-quantization-config '{"transformer":{"method":"mxfp8"}}'
```

A 15 s run of the speed point took 1836 s; that run has no quality comparison.

Quality against the dense USP4+MXFP8 run of the same prompt
(Laplacian-variance ratio): 0.81 for the speed point, 0.91 for the sharper
point. When frames look soft, move toward the sharper row before lowering
`sparsity`.

#### RainFusion plus Cache-DiT

Same USP4 + MXFP8 stack, with a shorter dense tail and Cache-DiT. Across
five repeats on vLLM-Omni 0.32.0, the 5 s diffuse median is 166.17 s
(wall-clock median 181.5 s), with 31 of 50 denoise steps cached. The same
configuration passed at seeds 2202 (165.1 s) and 2303 (165.9 s). Quality
versus the dense run: 0.924 sharpness / 0.950 flicker / 0.984 luma.

```bash
  --task-type ref2va \
  --diffusion-attention-config '{"default":{"backend":"RAINFUSION_ATTN","block_sparse":{"sparsity":0.8,"precision":"mix","start_step":4,"end_step":2,"skip_layers":"0-2,47-49"}}}' \
  --diffusion-quantization-config '{"transformer":{"method":"mxfp8"}}' \
  --cache-backend cache_dit \
  --enable-cache-dit-summary \
  --cache-config '{"Fn_compute_blocks":2,"Bn_compute_blocks":1,"max_warmup_steps":4,"residual_diff_threshold":0.07,"max_continuous_cached_steps":3,"enable_taylorseer":true,"taylorseer_order":1}'
```

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
[MiniMax-H3-NPU.md § FastH3 HTTP API examples](MiniMax-H3-NPU.md#fasth3-http-api-examples)
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

### Eight-card qualification evidence

The following results used `quay.io/ascend/vllm-omni:v0.33.0rc1-a5`,
vLLM-Omni `0.1.dev1+g08e9267ed.npu`, vLLM-Ascend
`0.19.1rc2.dev3349+g221fb096c.d20261104`, torch-npu `2.10.0.post7`, and
MindIE-SD `3.2.0`. Output resolution was 1344x768. The 124-frame and
362-frame workloads are referred to as 5 s and 15 s respectively.

Each row is one representative successful run. Run-to-run variance and a
quality comparison were not measured. Peak memory is reported per NPU. The
T2VA FlashGen4 rows use four denoising steps, so they are not an equal-step
comparison against the 50-step BF16 rows.

| Task | Frames | Configuration | DiT layout | Steps | E2E (s) | DiT (s) | DiT/step (s) | Peak memory (MB) |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| T2VA | 124 | BF16 dense | USP8 | 50 | 137.625 | 133.701 | 2.674 | 33,445 |
| T2VA | 124 | FlashGen4 + RainFusion + MXFP8 (mix) | USP8 | 4 | 8.218 | 4.385 | 1.096 | 57,958 |
| T2VA | 362 | BF16 dense | USP8 | 50 | 761.321 | 750.742 | 15.015 | 34,562 |
| T2VA | 362 | FlashGen4 + RainFusion + MXFP8 (mix) | USP8 | 4 | 23.732 | 13.313 | 3.328 | 58,825 |
| FL2VA | 124 | BF16 dense | TP8 | 50 | 148.112 | 144.015 | 2.880 | 33,491 |
| FL2VA | 124 | RainFusion + MXFP8 (mix) | TP8 | 50 | 120.154 | 116.008 | 2.320 | 29,726 |
| FL2VA | 362 | BF16 dense | TP8 | 50 | 788.380 | 777.414 | 15.548 | 34,742 |
| FL2VA | 362 | RainFusion + MXFP8 (mix) | TP8 | 50 | 492.401 | 481.516 | 9.630 | 31,274 |
| Ref2VA | 124 | BF16 dense | TP8 | 50 | 410.165 | 399.129 | 7.983 | 33,513 |
| Ref2VA | 124 | RainFusion + MXFP8 (mix) | USP8 | 50 | 213.919 | 202.815 | 4.056 | 58,290 |
| Ref2VA | 362 | BF16 dense | TP8 | 50 | 2,802.427 | 2,771.573 | 55.431 | 41,215 |
| Ref2VA | 362 | RainFusion + MXFP8 (mix) | USP8 | 50 | 1,389.345 | 1,358.997 | 27.180 | 59,978 |

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
- On a 950PR with ~86 GB HBM, USP4 bf16 does not fit; see the
  "Ref2VA on a 950PR that reports ~86 GB HBM" section above.

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
