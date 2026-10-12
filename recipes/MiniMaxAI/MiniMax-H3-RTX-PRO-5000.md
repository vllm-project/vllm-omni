# MiniMax-H3 on RTX PRO 5000 Blackwell GPUs

This recipe runs MiniMax-H3 in BF16 on 72 GiB RTX PRO 5000 Blackwell GPUs. It
contains the validated two-GPU DLO configuration and the recommended resident
configurations: TP1 x Ulysses2 with 20 resident layers on two GPUs, TP2 x
Ulysses2 on four GPUs, and TP4 x Ulysses2 on eight GPUs. The four- and
eight-GPU routes do not require offload.

## Capacity requirements

| Resource | Requirement |
| --- | ---: |
| GPUs | 2, 4, or 8 x RTX PRO 5000 Blackwell |
| GPU HBM | 72 GiB per GPU |
| Checkpoint storage | 135 GiB per partition |
| Available system RAM | 200 GiB minimum |
| Recommended system RAM | 384 GiB |

`FL2VA` and `Ref2VA` are separate checkpoint partitions. Start one server at a
time on a host sized for the minimum system-memory requirement.

## PCIe topology and GPU order

RTX PRO 5000 does not provide NVLink. Before starting the server, identify the
closest PCIe pairs and their NUMA affinity:

```bash
nvidia-smi topo -m
nvidia-smi nvlink -s
```

For two GPUs, select the closest PCIe pair on one NUMA node. For four GPUs,
select all cards from one NUMA node. On the validated host,
physical GPUs `(0,1)` and `(2,3)` are the two `PXB` pairs on NUMA node 0. The
order `CUDA_VISIBLE_DEVICES=0,2,1,3` maps the Ulysses groups to those local
pairs. For eight GPUs, the validated order is `0,4,1,5,2,6,3,7`, with host
memory interleaved across NUMA nodes 0 and 1. Do not copy these IDs blindly:
reproduce the same PCIe and NUMA relationships on the target host.

## Recommended serving configurations

### Two GPUs

Two 72 GiB cards require distributed layerwise offload. The validated route
uses TP1 x Ulysses2, keeps 20 leading DiT layers resident, and streams
rank-local weights without AllGather. Eager execution avoids regional-compile
instability on this offload path.

```bash
export MODEL_ROOT=/path/to/MiniMax-H3
export MODEL="${MODEL_ROOT}/FL2VA"
export PORT=8091

CUDA_VISIBLE_DEVICES=0,1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
numactl --cpunodebind=0 --membind=0 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 2 \
  --tensor-parallel-size 1 \
  --usp 2 \
  --ring 1 \
  --text-encoder-tp-size 2 \
  --vae-patch-parallel-size 2 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend CUDNN_ATTN \
  --enable-distributed-layerwise-offload \
  --dlo-no-use-allgather \
  --dlo-resident-layers 20 \
  --enforce-eager
```

### Four GPUs

The validated baseline uses TP2 x Ulysses2, text-encoder TP4, VAE patch
parallelism 4, and explicit cuDNN BF16 attention. Selecting the backend
explicitly keeps the recipe independent of platform-default backend changes.

```bash
export MODEL_ROOT=/path/to/MiniMax-H3
export MODEL="${MODEL_ROOT}/FL2VA"
export PORT=8091

CUDA_VISIBLE_DEVICES=0,2,1,3 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
numactl --cpunodebind=0 --membind=0 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 4 \
  --tensor-parallel-size 2 \
  --usp 2 \
  --ring 1 \
  --text-encoder-tp-size 4 \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend CUDNN_ATTN
```

### Eight GPUs

The recommended eight-GPU route uses TP4 x Ulysses2, text-encoder TP8, VAE
patch parallelism 8, and host-memory interleaving across both NUMA nodes.

```bash
export MODEL_ROOT=/path/to/MiniMax-H3
export MODEL="${MODEL_ROOT}/FL2VA"
export PORT=8091

CUDA_VISIBLE_DEVICES=0,4,1,5,2,6,3,7 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
numactl --interleave=0,1 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 8 \
  --tensor-parallel-size 4 \
  --usp 2 \
  --ring 1 \
  --text-encoder-tp-size 8 \
  --vae-patch-parallel-size 8 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend CUDNN_ATTN
```

For the four- and eight-GPU resident routes, do not add `--enforce-eager`.
Warm the server once before measuring so regional compilation is outside the
measured request. The two-GPU DLO route intentionally remains eager.

For Ref2VA, stop the FL2VA server and restart the same command with
`MODEL="${MODEL_ROOT}/Ref2VA"`.

## Target-hardware validation

All three configurations were exercised on a PCIe-only, dual-socket host with
eight RTX PRO 5000 GPUs. The run used PyTorch 2.11.0+cu130, CUDA 13.0, driver
580.95.05, 1344x768 output, 124 frames, and two warmups.

The four-GPU route also received a five-step profiling validation for GPU
balance:

| Measurement | Result |
| --- | ---: |
| Maximum externally sampled peak | 69,219 MiB (67.60 GiB) per GPU |
| T2VA worker-reported peak | 65,314 MiB |
| First-frame FL2VA worker-reported peak | 66,780 MiB |
| T2VA maximum GPU kernel-time deviation | 0.56% |
| First-frame FL2VA maximum GPU kernel-time deviation | 0.09% |

The recommended routes produced the following 50-step results. MiniMax-H3
requested 50 denoise steps and executed 49 denoise updates, so per-step latency
is `denoise / 49`.

| GPUs | Workload | Parallelism | E2E (s) | Text encode (s) | Denoise (s) | VAE decode (s) | Per step (ms) | Peak memory (GiB) |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | T2VA | TP1 x Ulysses2, DLO resident 20 | 515.57 | 1.19 | 504.69 | 8.79 | 10,300 | 36.38 |
| 2 | FL2VA first-frame I2VA | TP1 x Ulysses2, DLO resident 20 | 553.45 | 1.27 | 541.90 | 8.76 | 11,059 | 36.38 |
| 4 | T2VA | TP2 x Ulysses2 | 284.01 | 0.03 | 278.73 | 4.36 | 5,688 | 67.58 |
| 4 | FL2VA first-frame I2VA | TP2 x Ulysses2 | 305.72 | 0.28 | 299.85 | 4.36 | 6,119 | 67.58 |
| 8 | T2VA | TP4 x Ulysses2 | 163.20 | 0.04 | 159.40 | 2.58 | 3,253 | 45.83 |
| 8 | FL2VA first-frame I2VA | TP4 x Ulysses2 | 171.62 | 0.27 | 167.25 | 2.57 | 3,413 | 45.83 |

Peak memory is the maximum per-GPU value sampled externally with
`nvidia-smi`. The two-GPU route leaves about 34.7 GiB below the reported
73,415 MiB device capacity, the four-GPU route leaves about 4.1 GiB, and the
eight-GPU route leaves about 25.9 GiB. Re-measure memory for longer reference
inputs, concurrency greater than one, or a different output shape.

## T2VA request example

```bash
export API_URL="http://127.0.0.1:${PORT}/v1/videos/sync"

curl -sS --max-time 1800 -X POST "${API_URL}" \
  -F 'prompt=At night, three cats march into a bedroom playing tiny brass instruments, then abruptly file out, with synchronized room ambience.' \
  -F 'width=1344' \
  -F 'height=768' \
  -F 'aspect_ratio=16:9' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":5.0,"audio_flow_shift":3.0}' \
  -o t2va.mp4
```
