# MiniMax-H3 on A100-SXM4-40GB

This recipe covers MiniMax-H3 FL2VA/Ref2VA serving on **NVIDIA A100-SXM4-40GB**
(sm80, 40 GiB per GPU). Two paths are provided: TP4 with CPU offload (lower
latency) and distributed layerwise offload (DLO), which trades latency for a
much smaller denoising footprint:

- **TP4 + CPU offload** — the simplest path; the whole checkpoint streams
  through host memory. Per-GPU HBM holds ~22.8 GiB (22,827 MiB) through
  denoising and peaks at ~33.0 GiB (33,741 MiB) while the reference is
  processed — size for the peak, not for the denoising level.
- **DLO (rank-local)** — DiT blocks stream from host memory; per-GPU HBM during
  denoising drops to ~12.4 GiB (12,583–12,685 MiB) at roughly 5× the wall-clock
  time. The peak does not drop with it: 33-35 GiB transients on a single rank
  appeared in two of three measured runs.

A100 is sm80: **NVFP4 and hardware FP8 paths are unavailable** (FP8 tensor
cores require sm89+), so use BF16 throughout. The text encoder (Qwen3-VL) and
both VAEs are shared between the FL2VA and Ref2VA partitions; start one server
at a time, or use the combined FL2VA + Ref2VA layout from the main recipe.

## Capacity requirements

| Resource | Four A100-40GB (TP4 + offload) | Four A100-40GB (DLO) |
| --- | ---: | ---: |
| GPU HBM | 40 GiB per GPU | 40 GiB per GPU |
| Checkpoint storage | 135 GiB per partition | 135 GiB per partition |
| Available system RAM | 200 GiB minimum | 200 GiB minimum |
| Recommended system RAM | 384 GiB | 384 GiB |

`FL2VA` and `Ref2VA` are separate ~135 GiB checkpoint partitions. The
CPU-offload path holds the active partition in host memory; the rank-local DLO
path uses TP1 rank-local mmap
([#6213](https://github.com/vllm-project/vllm-omni/pull/6213)), where the four
DP replicas share one checkpoint page-cache copy instead of holding four private
copies. Both stay within the 200 GiB minimum, mostly as reclaimable page cache.
The AllGather variant keeps private shmem shards instead and needs more host
memory — see its section below.

## Four A100-40GB: TP4 + CPU offload (480x256, 4 seconds)

Use TP4 with model-level CPU offload (`--enable-cpu-offload`, the same
sequential-offload mechanism as the 4×L40S report in [issue #5700][l40s-comment],
without that report's vLLM-core offload flags — see the note below the command).
A 50-step 480x256 request completes in roughly 360-390 s, depending on machine
load.

`--vae-patch-parallel-size 4` is safe on current main: the 4×L40S report warned
about `ValueError: Found empty tasks on sp rank 3` when decoder tiles are fewer
than ranks, but [#6345](https://github.com/vllm-project/vllm-omni/pull/6345)
added the rank-local tiling fallback and is present in the recipe's base.

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /path/to/MiniMax-H3/FL2VA \
  --omni --trust-remote-code --host 0.0.0.0 --port 8000 \
  --num-gpus 4 --tensor-parallel-size 4 --text-encoder-tp-size 4 \
  --usp 1 --ring 1 --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile --vae-use-tiling \
  --enable-cpu-offload \
  --enforce-eager \
  --diffusion-attention-backend CUDNN_ATTN
```

> `--cpu-offload-gb`, `--offload-group-size`, `--offload-num-in-group` and
> `--offload-prefetch-step` are **no-ops for diffusion models** — they configure
> the autoregressive offloaders in `vllm/config/offload.py`, which `vllm_omni`
> never reads. MiniMax-H3 CPU offload is driven solely by
> `--enable-cpu-offload` (`enable_omni_model_cpu_offload` →
> `apply_sequential_offload`). A/B measurements show identical memory behavior
> with and without them.

For Ref2VA, stop the server and restart with `/path/to/MiniMax-H3/Ref2VA`.

Per-GPU HBM has two levels. With the official reference video (1344x768, with an
audio reference), the 480x256/96-frame run holds ~22.8 GiB (22,827 MiB) through
the 49 denoising steps but peaks at ~33.0 GiB (33,741 MiB) during reference
processing, on all four ranks, in repeated runs. That peak leaves ~7 GiB of the
40 GiB card free, so longer or multiple references can reach the limit.

The API reports `peak_memory_mb` = 32,098 for the same run. The field is the
**primary worker's (global rank 0) torch peak reserved memory in MiB**
(`max_memory_reserved`, bytes/1024²): it is not a worker sum, not an nvidia-smi
figure, and it includes allocator pool retention.

## Four A100-40GB: DLO rank-local (lower HBM)

If more HBM headroom is needed, the rank-local DLO path keeps weights in host
memory / shared page cache (~3.1 GiB per GPU after model loading) and streams
DiT blocks per denoising step. It is much slower per step than the TP4 path
(~38 s vs ~5.6 s per denoising step) because each rank streams complete blocks
over PCIe and uses no NVLink collectives; the AllGather variant sends only each
rank's shard and is ~1.7× faster overall — see the comparison below.

Validation: a full 50-step Ref2VA run with the same request as the TP4 path
(official 1344×768 reference video with audio, 480x256/96 frames, seed 0)
completed with `status=completed` in 1844.6 s (~38 s per denoising step); an
earlier run of the same command took 2012.8 s (~40 s per step). The API
reported `peak_memory_mb` = 19,108 (same metric as above).

Denoising per-GPU HBM is much lower than the TP4 path: 12,583–12,685 MiB, and
up to ~20.4 GiB on the rank that processes the reference. The peak is not lower
though — two of three measured runs showed a few-second, single-rank transient of
33-35 GiB during startup or reference processing, so a 40 GiB card keeps roughly
the same ~5-7 GiB of headroom as the TP4 path. An earlier DLO attempt OOM'd in
an activation layer before completing, so re-measure peak HBM on your own
hardware.

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

This is the official TP1 rank-local DLO topology
([docs](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion/offloader/distributed_layerwise_offload.md)):
`--data-parallel-size 4` with `--dlo-no-use-allgather` runs four independent DP
replicas, each streaming complete rank-local blocks. The loader's
direct-checkpoint mmap plan (`checkpoint_mmap`) supports TP1: the four replicas
map the same checkpoint file and share its page cache, so this is **not** four
private 135 GiB copies, and each worker keeps only two bounded pinned staging
slots.

> Note: DP-level request concurrency (one request per rank per denoising wave)
> is currently gated on the AllGather path. For rank-local no-AllGather it was
> the subject of [PR #5911](https://github.com/vllm-project/vllm-omni/pull/5911),
> which is closed, so treat this DP4 command as four replicas serving their own
> sequential requests (or a single request fanned out) rather than one batch of
> four concurrent requests.

**Why not AllGather?** The AllGather variant (`--dlo-use-allgather`, the
default) shards host weights across the DLO group (≈1/4 of each block per rank),
copies each shard to its device, and reconstructs the complete block with
`all_gather_into_tensor`. Both variants were measured with the same request:

| DLO variant | per denoising step | end-to-end | `peak_memory_mb` | denoising HBM | worst sample |
| --- | ---: | ---: | ---: | ---: | ---: |
| rank-local (`--dlo-no-use-allgather`) | ~38 s | 1844.6 s | 19,108 | 12,583–12,685 MiB | 33,299 MiB |
| AllGather (default) | ~22 s | **1069.3 s** | 19,724 | 13,829 MiB | 34,821 MiB |

Both "worst sample" values are few-second transients on a single rank; outside
them the curves stay near the denoising level (rank 0 reaching ~20-21 GiB while
the reference is processed).

AllGather is **1.7× faster end-to-end for a single request**, because each rank
copies only its ≈1/4 shard H2D instead of a complete block — the same mechanism
that gives SP ranks 1/N H2D transfer. It does not lower device HBM (the
reconstructed block still occupies the rotating device slots), and
`peak_memory_mb` barely separates the two modes. The host-memory footprint
does: rank-local shared-mmap keeps one reclaimable page-cache copy of the
checkpoint for all four replicas, while AllGather holds private shards in
shmem (~249 GiB across the group versus ~185 GiB for rank-local), so on a
shared host rank-local is the more memory-friendly layout.

Switch to `--dlo-use-allgather` (drop `--dlo-no-use-allgather`) when per-rank H2D
bandwidth or DP concurrency matters, at the cost of the larger host footprint
above. HSDP + AllGather is rejected by the offloader (double-sharding), so keep
TP1+DP (or TP-only) when switching.

## Eight A100-40GB (TP8 not validated)

Both commands above use **4 of the 8 GPUs** (`CUDA_VISIBLE_DEVICES=0,1,2,3`);
TP8 was not exercised for these paths. TP8 *with the same offload flags* is
expected to fit — weights shard across eight ranks, so the per-GPU weight peak
is lower than TP4 — but treat it as untested and re-measure peak HBM before
relying on it.

TP8 *fully resident* (no offload) is a different configuration, and it does not
fit: `--num-gpus 8 --tensor-parallel-size 8 --text-encoder-tp-size 8
--vae-patch-parallel-size 8` peaks above 40 GiB on one rank and then aborts with
an NCCL out-of-memory error at the start of generation.

## Notes

- A100 is sm80: do not enable NVFP4 (`--quantization nvfp4`) or online FP8;
  the checkpoint and kernels assume BF16 on this generation.
- `--enforce-eager` avoids CUDA-graph memory overhead on 40 GiB cards; graph
  mode was not validated here.
- Model weights can be loaded from either Hugging Face (`MiniMaxAI/MiniMax-H3`)
  or ModelScope (`MiniMax/MiniMax-H3`); the pipeline resolves the partition
  directory (`FL2VA`/`Ref2VA`) automatically.

[l40s-comment]: https://github.com/vllm-project/vllm-omni/issues/5700#issuecomment-5187762935
