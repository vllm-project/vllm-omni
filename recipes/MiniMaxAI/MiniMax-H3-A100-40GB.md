# MiniMax-H3 on A100-SXM4-40GB

This recipe covers MiniMax-H3 FL2VA/Ref2VA serving on **NVIDIA A100-SXM4-40GB**
(sm80, 40 GiB per GPU). Two validated paths are provided; the TP4 + CPU offload
path is the lower-latency one, and the DLO path trades latency for much lower
HBM:

- **TP4 + CPU offload** — the simplest path; the whole checkpoint streams
  through host memory, so per-GPU HBM stays at ~22 GiB (official reference with
  audio, reference-KV cache active; a synthetic reference measured the same
  ~22 GiB). The worker-aggregated torch peak reported by the API was ~31 GiB
  for the official-reference run.
- **DLO (rank-local distributed layerwise offload)** — DiT blocks stream per
  layer from host memory; per-GPU HBM peaks at ~12.6 GiB (measured), at the
  cost of wall-clock time (~5-7× slower per request on this node).

A100 is sm80: **NVFP4 and hardware FP8 paths are unavailable** (FP8 tensor
cores require sm89+), so use BF16 throughout. The text encoder (Qwen3-VL) and
both VAEs are shared between the FL2VA and Ref2VA partitions; start one server
at a time, or use the combined FL2VA + Ref2VA layout from the main recipe.

## Capacity requirements

| Resource | Four A100-40GB (TP4 + offload) | Four A100-40GB (DLO) |
| --- | ---: | ---: |
| GPU HBM | 40 GiB per GPU | 40 GiB per GPU |
| Checkpoint storage | 135 GiB per partition | 135 GiB per partition |
| Available system RAM | 150 GiB minimum | 150 GiB minimum |
| Recommended system RAM | 384 GiB | 384 GiB |

`FL2VA` and `Ref2VA` are separate ~135 GiB checkpoint partitions. Host RAM
matters most for the CPU-offload path: the entire active partition (weights +
pinned staging) lives in host memory. The DLO path uses TP1 rank-local mmap
([#6213](https://github.com/vllm-project/vllm-omni/pull/6213)): the four DP
replicas share the checkpoint's OS page cache on one node instead of holding
four private copies, and each worker keeps only two bounded pinned staging
slots (2×~1.2 GiB observed). The 150 GiB minimum for the DLO column is sized
for ~135 GiB of shared checkpoint pages plus staging; the completed DLO run
below was not instrumented for host-RSS, so keep 150 GiB unless you measure a
smaller footprint on your node.

## Four A100-40GB: TP4 + CPU offload (480x256, 4 seconds)

Use TP4 with model-level CPU offload (`--enable-cpu-offload`, the same
sequential-offload mechanism as the 4×L40S report in [issue #5700][l40s-comment],
minus that report's vLLM-core offload flags, which are no-ops for diffusion —
see the note below the command): per-GPU HBM peaks at ~22 GiB
(BF16, synthetic 4s reference) and the request completes in roughly
300-370 s for 50 denoising steps at 480x256 (machine-load dependent).

`--vae-patch-parallel-size 4` is safe on current main: the 4×L40S report
warned about `ValueError: Found empty tasks on sp rank 3` when decoder tiles
are fewer than ranks, but that was fixed by
[#6345](https://github.com/vllm-project/vllm-omni/pull/6345) (rank-local tiling
fallback), which is present in the recipe's base. The 480x256 runs below
exercise exactly this path.

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

> The vLLM-core flags `--cpu-offload-gb`, `--offload-group-size`,
> `--offload-num-in-group`, and `--offload-prefetch-step` are **no-ops for
> diffusion models** — they configure the autoregressive offloaders in
> `vllm/config/offload.py`, which `vllm_omni` never reads. MiniMax-H3's CPU
> offload is driven solely by `--enable-cpu-offload`
> (`enable_omni_model_cpu_offload` → `apply_sequential_offload`). The 4×L40S
> report in #5700 attributed its low VRAM to that flag combo, but the memory
> savings there also came from `--enable-cpu-offload`; the extra flags only
> emitted a vLLM-core warning (`offload_backend="auto"` with both UVA and
> prefetch fields set).

For Ref2VA, stop the server and restart with `/path/to/MiniMax-H3/Ref2VA`.
With the official reference video (1344x768, includes an audio reference) the
live per-GPU peak was ~22.3 GiB (nvidia-smi) during the 480x256/96-frame run —
within 40 GiB with headroom for longer prompts or more references. The
worker-aggregated torch peak reported by the API (`peak_memory_mb`) was
~31 GiB for the same run; this is a process-aggregate metric, not a per-GPU
figure.

## Four A100-40GB: DLO rank-local (lower HBM)

If HBM headroom is needed (e.g. co-tenant workloads), the rank-local DLO path
keeps weights in host memory / shared page cache (~3.1 GiB per GPU after model
loading) and streams DiT blocks per denoising step. It is slower per step
because every denoising step streams non-resident DiT blocks over NVLink/PCIe.

**Validation:** this DLO command was validated end-to-end with the same request
as the TP4 path (official 1344×768 reference video with audio, 480×256/96
frames, seed 0): a full 50-step Ref2VA run completed (`status=completed`) with
a live per-GPU peak of **~12.6 GiB** (nvidia-smi) and total inference time of
**2013 s** (~5.5× the TP4 path's 364 s). The API worker-aggregated
`peak_memory_mb` was 19,108 for the same run. An earlier DLO attempt on this
node OOM'd in an activation layer; the run below is the one that completed —
re-measure on your own node before trusting the numbers.

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
replicas, each streaming complete rank-local blocks. On current main the
loader's direct-checkpoint mmap plan (`checkpoint_mmap`) is supported for TP1:
the four replicas map the same checkpoint file, sharing the OS page cache, so
this is **not** four private 135 GiB copies. Each worker keeps only two bounded
pinned staging slots (`distributed_layerwise_backend.py` logs "checkpoint pages
are node-shared; each worker owns only two bounded host staging slots"). The
four replicas also give request-level concurrency for independent requests;
this is the rank-local DP routing shape from
[#5911](https://github.com/vllm-project/vllm-omni/issues/5911).

## Eight A100-40GB (not measured)

Both commands above use **4 of the 8 GPUs** on the testing node
(`CUDA_VISIBLE_DEVICES=0,1,2,3`; the node had a co-tenant GPU, so TP8 was not
exercised). TP8 with the same offload flags is *expected* to fit — weights
shard across eight ranks, so per-GPU weight peak is lower than TP4 — but treat
the 8-GPU row as an estimate and re-measure peak HBM before trusting it for
production.

## Notes

- A100 is sm80: do not enable NVFP4 (`--quantization nvfp4`) or online FP8;
  the checkpoint and kernels assume BF16 on this generation.
- `--enforce-eager` avoids CUDA-graph memory overhead on 40 GiB cards; graph
  mode was not validated here.
- Model weights can be loaded from either Hugging Face (`MiniMaxAI/MiniMax-H3`)
  or ModelScope (`MiniMax/MiniMax-H3`); the pipeline resolves the partition
  directory (`FL2VA`/`Ref2VA`) automatically.

[l40s-comment]: https://github.com/vllm-project/vllm-omni/issues/5700#issuecomment-5187762935
