# MiniMax-H3 on RTX PRO 6000 Blackwell GPUs

This recipe runs MiniMax-H3 in BF16 on 96 GiB RTX PRO 6000 Blackwell GPUs.
The additional HBM over the RTX PRO 5000 profile removes the four-GPU
minimum: two GPUs are enough to keep the model resident with TP2. Four and
eight GPUs raise the Ulysses degree to shard the attention sequence further.
CPU offload and distributed layerwise offload are not required in any of
these configurations.

Validated on:
- Host: YLX Y762
- GPUs: 8 × RTX PRO 6000 Blackwell (96 GiB)
- Device order: default (`CUDA_VISIBLE_DEVICES` not set)
- Driver / CUDA: 580.105.08 / 13.0
- Container image: `vllm/vllm-omni:minimax-h3`
- Workload: T2VA, 1344×768, `duration=5.0`, 50 steps, `flow_shift=12`, `seed=1101`

## Capacity requirements

| Resource | Two GPUs | Four GPUs | Eight GPUs |
| --- | ---: | ---: | ---: |
| GPU HBM | 96 GiB per GPU | 96 GiB per GPU | 96 GiB per GPU |
| Checkpoint storage | 135 GiB per partition | 135 GiB per partition | 135 GiB per partition |
| Available system RAM | 200 GiB minimum | 200 GiB minimum | 200 GiB minimum |
| Recommended system RAM | 384 GiB | 384 GiB | 384 GiB |
| Measured peak HBM per GPU | 77.49 GiB | 66.44 GiB | 61.07 GiB |

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

Device-order pinning and NUMA binding are tuning knobs, not prerequisites.
The measurements in this recipe were taken without either, so any gain from
them is additional to the numbers reported below and is host-specific.

## Two-GPU serving configuration

Two 96 GiB GPUs hold the BF16 model with TP2 alone. There is no Ulysses
group, so the DiT attention sequence is not sharded and activation memory is
higher than in the four-GPU profile at the same output shape. Re-measure
peak memory before increasing the output shape, the reference-input length,
or concurrency.

```bash
export MODEL_ROOT=/path/to/MiniMax-H3
export MODEL="${MODEL_ROOT}/FL2VA"
export PORT=8000

VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
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
export PORT=8000

VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
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

```bash
export MODEL_ROOT=/path/to/MiniMax-H3
export MODEL="${MODEL_ROOT}/FL2VA"
export PORT=8000

VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
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
both be socket-local. If you choose to pin the device order, the trade-off
is:

| Device order | Socket-local collective | Crosses sockets |
| --- | --- | --- |
| `0,4,1,5,2,6,3,7` | Ulysses all-to-all | TP all-reduce |
| `0,1,2,3,4,5,6,7` | TP all-reduce | Ulysses all-to-all |

The first order assumes physical GPUs `0-3` are on NUMA node 0 and `4-7` on
node 1. Reproduce that relationship on the target host rather than copying
the IDs. The reported measurements use the default order and no NUMA
binding; the two orders were not compared on this host.

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
This layout is described for completeness and was not part of the measured
runs.

## Shared serving notes

Do not add `--enforce-eager` for a performance run. Warm the server before
measuring so regional compilation is outside the measured request. On this
host the first request after startup ran 19% slower than the steady state
(107.58 s versus 90.48 s on eight GPUs); two warmup requests were enough to
converge.

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

T2VA, 1344×768, `duration=5.0`, `fps=24`, 50 steps, `flow_shift=12`,
`seed=1101`, BF16, `CUDNN_ATTN`, tiled VAE, one request at a time. Servers
were started with the commands above — default device order, no NUMA
binding — plus `--enable-diffusion-pipeline-profiler`. Two warmup requests
preceded each measured request.

| Measurement | Two GPUs | Four GPUs | Eight GPUs |
| --- | ---: | ---: | ---: |
| Topology | TP2 × USP1 | TP2 × USP2 | TP2 × USP4 |
| Text encode | 0.040 s | 0.044 s | 0.035 s |
| Denoise, 50 steps | 278.55 s | 168.73 s | 87.90 s |
| Per step | 5.571 s | 3.375 s | 1.758 s |
| VAE decode | 5.396 s | 2.791 s | 1.798 s |
| Client E2E | 284.76 s | 172.32 s | 90.48 s |
| Peak HBM per GPU | 77.49 GiB | 66.44 GiB | 61.07 GiB |
| Headroom below 96 GiB | 18.5 GiB | 29.6 GiB | 34.9 GiB |

Stage times are read from the `X-Stage-Durations` response header of
`/v1/videos/sync`. Peak memory is the maximum of `nvidia-smi
--query-gpu=memory.used` sampled at 1 Hz across every device for the
duration of the measured request. Per step is denoise wall time divided by
the 50 requested steps. Stage times sum to roughly 0.75 s less than
end-to-end in all three configurations; queueing, result transfer, and MP4
muxing sit outside the profiled stages.

### Scaling behaviour

Denoise speedup is not uniform across the sweep: 1.65× from two to four
GPUs (83% efficiency) but 1.92× from four to eight (96%). Two GPUs run
Ulysses1, so there is no sequence parallelism at all; going to four GPUs
pays the one-time cost of introducing the all-to-all, while going from four
to eight only widens an all-to-all that already exists. Expect the second
doubling to pay off better than the first on this topology.

VAE decode behaves in the opposite direction — 1.93× from two to four GPUs
but only 1.55× from four to eight — as patch parallelism produces smaller
tiles and tile-boundary overhead grows.

Text encoding is 0.04 s in every configuration and is not worth optimising;
denoise is 96-98% of end-to-end throughout.

### Per-GPU memory model

Peak memory falls by 11.05 GiB from two to four GPUs and by a further
5.37 GiB from four to eight — halving each time the Ulysses degree doubles,
because all three configurations shard the DiT with TP2 and hold identical
weights per GPU. Fitting the first two points gives:

```
peak HBM per GPU ≈ 55.4 GiB + 22.1 GiB / ulysses_degree
```

That fit predicts 60.9 GiB at Ulysses4 against 61.07 GiB measured, a 0.25%
error. The practical consequence is a floor near 55 GiB per GPU: adding
GPUs past eight will not reduce the per-GPU requirement further. To serve
on smaller cards, raise the tensor-parallel degree instead.

Re-measure memory for longer reference inputs, concurrency greater than
one, or a different output shape.

### Not yet measured

Ref2VA latency and memory have not been measured on this host. Ref2VA
drives the Qwen3-VL presentation from tens of tokens for text-only prompts
to several thousand with image or video references, so T2VA numbers do not
transfer to it. The eight-GPU device-order comparison in the section above
was also not run.

## T2VA request example

`t2va` requires an explicit `aspect_ratio` even when `width` and `height`
are supplied.

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

To collect the stage breakdown and peak memory for a request, add
`--enable-diffusion-pipeline-profiler` to the server command and `-D
headers.txt` to the curl invocation, then read `X-Stage-Durations` and
`X-Peak-Memory-MB` from the saved headers.

## Ref2VA request example

`ref2va` defaults to a 16:9 aspect ratio, so `aspect_ratio` is optional
here.

```bash
curl -X POST "http://127.0.0.1:${PORT}/v1/videos/sync" \
  --fail-with-body -w '\nHTTP %{http_code}\n' \
  --max-time 1200 \
  -F "input_reference=@/root/hand.jpg;type=image/jpeg" \
  -F "audio_reference=</root/audio_ref.json" \
  -F "prompt=2D动画融合在一起的影像。夕阳余晖残留在窗边，生活感十足的小厨房里有旧木桌、洗到一半的马克杯、起雾的玻璃瓶、悬挂的抹布。画面带有智能手机单手拍摄的手抖、近距离对焦的犹豫、逆光曝光波动。要像在家中慌忙拍下某个不可思议事件的自然质感，不要广告影像的精心整理。声音只用厨房环境声与手绘生物柔和的电子音、小小的叫声。" \
  -F 'width=1344' -F 'height=768' -F 'fps=24' \
  -F 'num_inference_steps=60' -F 'flow_shift=12' -F 'seed=1101' \
  -F 'extra_params={"task":"ref2va","duration":8,"audio_flow_shift":3.0}' \
  -o /root/out_ref2va.mp4
```
