# MiniMax-H3 on RTX 4090

This recipe uses BF16 weights, tiled VAE decode, tensor parallelism across two
GPUs, and distributed layerwise offload (DLO). It is the 24 GiB sibling of
[MiniMax-H3-5090.md](MiniMax-H3-5090.md): the topology is identical, but the
resident DiT-block count is lower so the per-rank peak fits in 24 GiB. Lower
resident counts reduce HBM use and increase CPU-to-GPU transfer time.

## Capacity requirements

| Resource | Two RTX 4090s |
| --- | ---: |
| GPU HBM | 24 GiB per GPU |
| Checkpoint storage | 135 GiB per partition |
| Available system RAM | 200 GiB minimum |
| Recommended system RAM | 384 GiB |

`FL2VA` and `Ref2VA` are separate 135 GiB checkpoint partitions. Start one
server at a time. DLO keeps rank-local weights in pinned host memory; increasing
`--dlo-resident-layers` improves latency but does **not** reduce host RAM in the
current implementation because resident layers retain pinned CPU master copies.

A single RTX 4090 is not covered here. The one-GPU DLO profile in the RTX 5090
recipe peaked at 26.50 GiB with 12 resident layers, which exceeds 24 GiB. On one
4090, use the model-level CPU offload command in
[MiniMax-H3.md](MiniMax-H3.md#single-gpu-accuracy-and-memory-first) instead, or
lower the resident count and re-measure before trusting it.

> **Modular H3:** after #5720 lands, preserve this recipe's one-partition
> behavior with `--task-type fl2va` or `--task-type ref2va`.

## Two RTX 4090s: 1024x576, 5 seconds

Use TP2 and 12 resident DiT layers. This is the `rtx4090` profile from the main
recipe. A 1024x576 capacity run of this profile peaked at 18,888 MiB per rank on
two B300 ranks; the B300 numbers are an allocation and correctness proxy, not an
RTX 4090 latency claim.

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400

CUDA_VISIBLE_DEVICES=0,1 vllm serve /path/to/MiniMax-H3/FL2VA \
  --omni --trust-remote-code --host 0.0.0.0 --port 8000 \
  --num-gpus 2 --tensor-parallel-size 2 --text-encoder-tp-size 2 \
  --usp 1 --ring 1 --vae-patch-parallel-size 2 \
  --vae-parallel-mode tile --vae-use-tiling \
  --enable-distributed-layerwise-offload --dlo-no-use-allgather \
  --dlo-resident-layers 12 --enforce-eager \
  --diffusion-attention-backend CUDNN_ATTN
