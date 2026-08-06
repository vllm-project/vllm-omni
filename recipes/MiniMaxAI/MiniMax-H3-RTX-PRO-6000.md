# MiniMax-H3 on RTX PRO 6000 Blackwell GPUs

This recipe runs MiniMax-H3 in BF16 on 96 GiB RTX PRO 6000 Blackwell GPUs.
The additional HBM over the RTX PRO 5000 profile removes the four-GPU
minimum: two GPUs are enough to keep the model resident with TP2. Four and
eight GPUs raise the Ulysses degree to shard the attention sequence further.
CPU offload and distributed layerwise offload are not required in any of
these configurations.

Validated on:
- Host: <YLX Y762 >
- GPUs: 8 × RTX PRO 6000 Blackwell (96 GiB)
- Device order: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
- Driver / CUDA: 580.105.08 / 13.0
- vLLM-Omni : vllm/vllm-omni:minimax-h3


## Capacity requirements

| Resource | Two GPUs | Four GPUs | Eight GPUs |
| --- | ---: | ---: | ---: |
| GPU HBM | 96 GiB per GPU | 96 GiB per GPU | 96 GiB per GPU |
| Checkpoint storage | 135 GiB per partition | 135 GiB per partition | 135 GiB per partition |
| Available system RAM | 200 GiB minimum | 200 GiB minimum | 200 GiB minimum |
| Recommended system RAM | 384 GiB | 384 GiB | 384 GiB |

`FL2VA` and `Ref2VA` are separate checkpoint partitions. Start one server at
a time on a host sized for the minimum system-memory requirement, or pass
`--task-type fl2va` or `--task-type ref2va` to download and load a single
partition. The two-server layout described in the eight-GPU section doubles
both the storage and the host-memory requirement.

## Parallelism summary

| Configuration | TP | Ulysses | Text-encoder TP | VAE patch parallel |
| --- | ---: | ---: | ---: | ---: |
| Two GPUs | 2 | 1 | 2 | 2 |
| Four GPUs | 2 | 2 | 4 | 4 |
| Eight GPUs | 2 | 4 | 8 | 8 |
| Eight GPUs, headroom variant | 4 | 2 | 8 | 8 |

Per-GPU weight residency is set by the tensor-parallel degree alone. Raising
the Ulysses degree lowers activation memory and per-step latency but leaves
weight residency unchanged. Ring parallelism stays at 1, and H3 is
CFG-distilled, so `--cfg-parallel-size` must also remain 1.

## PCIe topology and GPU order

RTX PRO 6000 Blackwell does not provide NVLink. Before starting the server,
inspect the host topology:

```bash
nvidia-smi topo -m
nvidia-smi nvlink -s
```

Tensor-parallel groups are consecutive ranks. Ulysses groups are strided by
the tensor-parallel degree. Confirm this grouping against the server log
before tuning the device order on a new host.

For the two-GPU configuration, pick a single `PXB` pair on one NUMA node.
For the four-GPU configuration, pick two `PXB` pairs on the same node and
order them so that each Ulysses group lands inside one pair. If physical
GPUs `(0,1)` and `(2,3)` are the two local pairs, the order is
`CUDA_VISIBLE_DEVICES=0,2,1,3`. Do not copy these IDs blindly: reproduce the
same relationship on the target host.

## Two-GPU serving configuration

Two 96 GiB GPUs hold the BF16 model with TP2 alone. There is no Ulysses
group, so the DiT attention sequence is not sharded and activation memory is
higher than in the four-GPU profile at the same output shape. Re-measure
peak memory before increasing the output shape, the reference-input length,
or concurrency.

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
  --tensor-parallel-size 2 \
  --usp 1 \
  --ring 1 \
  --text-encoder-tp-size 2 \
  --vae-patch-parallel-size 2 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend CUDNN_ATTN
```

## Four-GPU serving configuration

Four GPUs keep TP2 for the weight shard and add Ulysses2, which shards the
attention sequence and lowers per-GPU activation memory. Text-encoder TP4
and VAE patch parallelism 4 spread the Qwen3-VL encoder and tiled decode
across all four GPUs. TP1 is not an option on this card: an unsharded BF16
partition does not fit in 96 GiB.

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

## Eight-GPU serving configuration

Eight GPUs extend the Ulysses degree to 4 while keeping TP2. This does not
lower per-GPU weight residency relative to the four-GPU profile; it lowers
activation memory and per-step latency by sharding the attention sequence
four ways.

On a dual-socket host, eight GPUs span both NUMA nodes. Do not bind the
server to a single node.

```bash
export MODEL_ROOT=/path/to/MiniMax-H3
export MODEL="${MODEL_ROOT}/FL2VA"
export PORT=8091

CUDA_VISIBLE_DEVICES=0,4,1,5,2,6,3,7 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
numactl --interleave=all \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 8 \
  --tensor-parallel-size 2 \
  --usp 4 \
  --ring 1 \
  --text-encoder-tp-size 8 \
  --vae-patch-parallel-size 8 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend CUDNN_ATTN
```

### Rank ordering across two sockets

With TP2 and Ulysses4 the two Ulysses groups are ranks `(0,2,4,6)` and
`(1,3,5,7)`, and the four tensor-parallel pairs are `(0,1)`, `(2,3)`,
`(4,5)`, and `(6,7)`. On a dual-socket host these two collectives cannot
both be socket-local. Choose the order that matches the measured
bottleneck:

| Device order | Socket-local collective | Crosses sockets |
| --- | --- | --- |
| `0,4,1,5,2,6,3,7` | Ulysses all-to-all | TP all-reduce |
| `0,1,2,3,4,5,6,7` | TP all-reduce | Ulysses all-to-all |

The first order assumes physical GPUs `0-3` are on NUMA node 0 and `4-7` on
node 1. Reproduce that relationship on the target host rather than copying
the IDs. Measure both orders once; the winner is host-specific.

### Headroom variant: TP4 with Ulysses2

Raising the tensor-parallel degree to 4 halves per-GPU weight residency and
leaves room for long Ref2VA references, larger output shapes, or
concurrency greater than one. It also adds a four-way all-reduce per DiT
block over a link without NVLink. Use it when the TP2 profile runs out of
headroom, not as the default. Replace the parallel flags above with:

```bash
  --tensor-parallel-size 4 \
  --usp 2 \
```

### Two independent four-GPU servers

An eight-GPU host can instead run one FL2VA server and one Ref2VA server
side by side, each pinned to its own NUMA node with the four-GPU
configuration. This removes every cross-socket collective and lifts the
one-server-at-a-time restriction, at the cost of doubling the requirements
to 400 GiB available system RAM and 270 GiB of model storage.

Use `CUDA_VISIBLE_DEVICES=0,2,1,3`, `--cpunodebind=0 --membind=0`, and
`--port 8091` for the FL2VA server. Use `CUDA_VISIBLE_DEVICES=4,6,5,7`,
`--cpunodebind=1 --membind=1`, and `--port 8092` for the Ref2VA server.

## Shared serving notes

Do not add `--enforce-eager` for a performance run. Warm the server once
before measuring so regional compilation is outside the measured request.

For Ref2VA on a single-server layout, stop the FL2VA server and restart the
same command with `MODEL="${MODEL_ROOT}/Ref2VA"`.

## Attention backend

RTX PRO 6000 Blackwell is an SM120 part, so the datacenter-Blackwell
`TRTLLM_ATTN` default does not apply here. Every configuration above selects
cuDNN BF16 attention explicitly. This keeps the recipe independent of
platform-default backend changes and does not depend on the experimental
SM120 kernel.

## Optional: online FP8

`--quantization fp8` quantizes eligible DiT linears at load time and is
compatible with tensor parallelism and VAE tiling. Use it when the resident
BF16 peak leaves too little headroom for long Ref2VA references, a larger
output shape, or concurrency greater than one. It cannot be combined with
layerwise offload.

## Target-hardware validation

<TODO: describe the validated host: socket and NUMA layout, PCIe-only or
not, CUDA version, driver version, PyTorch version, output shape, frame
count, warmup count, and the device order used for the eight-GPU run.>

| Measurement | Two GPUs | Four GPUs | Eight GPUs |
| --- | ---: | ---: | ---: |
| Client E2E (50-step T2VA) | ~10 min | Not measured| ~4 min |

<TODO: state the remaining headroom below the reported device capacity, and
report the measured difference between the two eight-GPU device orders.>
Re-measure memory for longer reference inputs, concurrency greater than one,
or a different output shape. The recorded run is a five-step profiling
validation; it is not a production 50-step latency claim.

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
Measured client E2E for the 50-step T2VA request above:
~10 min on 2× RTX PRO 6000, ~4 min on 8× (after one warmup request).
4-GPU latency was not measured on this host.
