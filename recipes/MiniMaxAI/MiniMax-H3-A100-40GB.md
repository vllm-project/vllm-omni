# MiniMax-H3 on A100-SXM4-40GB

This recipe covers MiniMax-H3 FL2VA/Ref2VA serving on **NVIDIA A100-SXM4-40GB**
(sm80, 40 GiB per GPU). Two validated paths are provided:

- **TP4 + CPU offload** — the simplest path; the whole checkpoint streams
  through host memory, so per-GPU HBM stays at ~22 GiB for a synthetic
  reference and ~37.5 GiB for the official reference (with audio) under the
  reference-KV cache.
- **DLO (rank-local distributed layerwise offload)** — DiT blocks stream per
  layer; per-GPU HBM drops to ~11-14 GiB at the cost of wall-clock time.

A100 is sm80: **NVFP4 and hardware FP8 paths are unavailable** (FP8 tensor
cores require sm89+), so use BF16 throughout. The text encoder (Qwen3-VL) and
both VAEs are shared between the FL2VA and Ref2VA partitions; start one server
at a time, or use the combined FL2VA + Ref2VA layout from the main recipe.

## Capacity requirements

| Resource | Four A100-40GB (TP4 + offload) | Four A100-40GB (DLO) |
| --- | ---: | ---: |
| GPU HBM | 40 GiB per GPU | 40 GiB per GPU |
| Checkpoint storage | 135 GiB per partition | 135 GiB per partition |
| Available system RAM | 150 GiB minimum | 100 GiB minimum |
| Recommended system RAM | 384 GiB | 384 GiB |

`FL2VA` and `Ref2VA` are separate ~135 GiB checkpoint partitions. Host RAM
matters most for the CPU-offload path: the entire active partition (weights +
pinned staging) lives in host memory. DLO keeps rank-local weights in pinned
host memory as well.

## Four A100-40GB: TP4 + CPU offload (480x256, 4 seconds)

Use TP4 with model-level CPU offload. This is the same mechanism as the
4xL40S report in [issue #5700][l40s-comment]: per-GPU HBM peaks at ~22 GiB
(BF16, synthetic 4s reference) and the request completes in roughly
300-370 s for 50 denoising steps at 480x256 (machine-load dependent).

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /path/to/MiniMax-H3/FL2VA \
  --omni --trust-remote-code --host 0.0.0.0 --port 8000 \
  --num-gpus 4 --tensor-parallel-size 4 --text-encoder-tp-size 4 \
  --usp 1 --ring 1 --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile --vae-use-tiling \
  --enable-cpu-offload --cpu-offload-gb 50 \
  --offload-group-size 1 --offload-num-in-group 1 --offload-prefetch-step 1 \
  --enforce-eager \
  --diffusion-attention-backend CUDNN_ATTN
```

For Ref2VA, stop the server and restart with `/path/to/MiniMax-H3/Ref2VA`.
With the official reference video (1344x768, includes an audio reference) the
per-GPU peak rises to ~37.5 GiB when the reference-KV cache is active — still
within 40 GiB, but leave headroom for longer prompts or more references.

## Four A100-40GB: DLO rank-local (lower HBM)

If HBM headroom is needed (e.g. co-tenant workloads), the rank-local DLO path
peaks at ~11-14 GiB per GPU. It is slower per step because every denoising
step streams non-resident DiT blocks over NVLink/PCIe.

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /path/to/MiniMax-H3/FL2VA \
  --omni --trust-remote-code --host 0.0.0.0 --port 8000 \
  --num-gpus 4 --tensor-parallel-size 1 --text-encoder-tp-size 1 \
  --data-parallel-size 4 --vae-patch-parallel-size 1 \
  --usp 1 --ring 1 --vae-parallel-mode tile --vae-use-tiling \
  --enable-distributed-layerwise-offload --dlo-no-use-allgather \
  --dlo-resident-layers 0 --enforce-eager \
  --diffusion-attention-backend CUDNN_ATTN
```

## Eight A100-40GB

TP8 was not measured in this recipe (the validating node had a co-tenant GPU).
TP8 with the same CPU-offload flags is expected to fit (weights shard across
eight ranks, so per-GPU weight peak is lower than TP4), but re-measure peak
HBM before trusting it for production.

## Notes

- A100 is sm80: do not enable NVFP4 (`--quantization nvfp4`) or online FP8;
  the checkpoint and kernels assume BF16 on this generation.
- `--enforce-eager` avoids CUDA-graph memory overhead on 40 GiB cards; graph
  mode was not validated here.
- Model weights can be loaded from either Hugging Face (`MiniMaxAI/MiniMax-H3`)
  or ModelScope (`MiniMax/MiniMax-H3`); the pipeline resolves the partition
  directory (`FL2VA`/`Ref2VA`) automatically.

[l40s-comment]: https://github.com/vllm-project/vllm-omni/issues/5700#issuecomment-5187762935
