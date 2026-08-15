# MiniMax-H3 on RTX 5090

This recipe uses BF16 weights, tiled VAE decode, tensor parallelism where a
second GPU is available, and distributed layerwise offload (DLO). It is a
memory-first serving configuration; lower resident counts reduce HBM use and
increase CPU-to-GPU transfer time.

## Capacity requirements

| Resource | One RTX 5090 | Two RTX 5090s |
| --- | ---: | ---: |
| GPU HBM | 32 GiB | 32 GiB per GPU |
| Checkpoint storage | 135 GiB per partition | 135 GiB per partition |
| Available system RAM | 200 GiB minimum | 200 GiB minimum |
| Recommended system RAM | 384 GiB | 384 GiB |

`FL2VA` and `Ref2VA` are separate 135 GiB checkpoint partitions. Start one
server at a time. By default, DLO keeps every rank-local streamed weight in
pinned host memory; increasing `--dlo-resident-layers` improves latency but does
**not** reduce host RAM because resident layers retain pinned CPU master copies.

### Optional bounded host staging

On a discrete-GPU host with local NVMe, add the following flags to either command
below to replace persistent pinned DiT shards with file-backed rank-local shards:

```bash
  --dlo-host-memory-budget-gib 8 \
  --dlo-host-cache-dir /local-nvme/vllm-omni-dlo \
  --dlo-pinned-staging-buffer-count 2 \
  --dlo-prefetch-depth 2
```

The 8 GiB value is a cap, not a reservation. Startup fails with the exact required
byte count if two buffers cannot hold the largest per-rank layer shard. Steady
private transfer memory is approximately:

```text
staging buffer count × sum(max layer-shard elements per dtype)
```

For the checked FL2VA BF16 checkpoint, safetensors metadata gives the following
single-rank capacity model for the 50 DiT blocks and two token-refiner blocks:

| DLO tensor payload | GiB |
| --- | ---: |
| Persistent pinned shards without bounded staging | 61.559 |
| Largest layer | 1.202 |
| Two bounded staging buffers | 2.405 |
| Pinned payload avoided | 59.154 (96.09%) |

These values describe tensor payload, not process RSS. Runtime TP sharding,
quantization, allocator overhead, and model revisions can change the per-rank
values, so the startup log reports the actual file-backed bytes,
private staging bytes, and pinned-payload reduction for the selected topology.

The background copy also prevents the new file-backed-to-pinned transfer from
remaining entirely on the layer critical path. In a host-only microbenchmark
using four 1.202 GiB file-backed BF16 shards, two pinned buffers, a 250 ms
per-layer compute window, local ext4 storage, and 16 PyTorch CPU threads, the
median of three alternating runs was:

| Bounded staging path | Exposed staging wait | Four-layer pipeline time |
| --- | ---: | ---: |
| Synchronous copy | 1.8149 s | 2.8159 s |
| Background one-layer-ahead copy | 1.0182 s (-43.90%) | 2.0196 s (1.394x) |

This is a staging-pipeline microbenchmark, not an end-to-end H3 latency claim.
Measure request latency and storage bandwidth on the production host.

This mode has important capacity boundaries:

- The cache directory needs space for one finalized rank-local streamed DiT copy
  per worker, in addition to the original checkpoint. Use local SSD/NVMe, not
  `tmpfs` or a network filesystem.
- The limit covers DLO's private transfer buffers only. OS page cache is
  reclaimable but can still grow, and encoders, VAEs, request data, and CUDA
  allocations remain outside the limit.
- The standard loader still constructs the rank-local model before DLO writes
  the file-backed cache. Therefore this first version does **not** lower the
  200 GiB startup requirement in the table. A reusable preprocessed cache or
  streaming model loader is needed to remove that peak.
- Cold requests can fault pages from NVMe. `--dlo-prefetch-depth 2` submits the
  current and following shard for OS readahead and copies the following shard
  into a pinned buffer on a background worker. Deeper values extend readahead;
  pinned background work remains bounded to at most one fewer than the configured
  staging-buffer count.
- Cache files are removed on clean shutdown. After `SIGKILL` or a host crash,
  remove stale `vllm-omni-dlo-*` directories manually.

> **Modular H3:** after #5720 lands, preserve this recipe's one-partition
> behavior with `--task-type fl2va` or `--task-type ref2va`.

## One RTX 5090: 1344x768, 5 seconds

Use 12 resident DiT layers. A 50-step B300 allocation test with this exact
single-rank topology peaked at 26.50 GiB; re-measure peak HBM on the target
card before increasing the resident count.

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/MiniMax-H3/FL2VA \
  --omni --trust-remote-code --host 0.0.0.0 --port 8000 \
  --num-gpus 1 --tensor-parallel-size 1 --text-encoder-tp-size 1 \
  --usp 1 --ring 1 --vae-patch-parallel-size 1 \
  --vae-parallel-mode tile --vae-use-tiling \
  --enable-distributed-layerwise-offload --dlo-no-use-allgather \
  --dlo-resident-layers 12 --enforce-eager \
  --diffusion-attention-backend CUDNN_ATTN
```

## Two RTX 5090s: 1344x768, 5 seconds

Use TP2 and 20 resident DiT layers. The two-rank B300 capacity run peaked at
27,726 MiB per rank for this shape and 50 steps. This is a memory/correctness
proxy, not a consumer-GPU latency claim.

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve /path/to/MiniMax-H3/FL2VA \
  --omni --trust-remote-code --host 0.0.0.0 --port 8000 \
  --num-gpus 2 --tensor-parallel-size 2 --text-encoder-tp-size 2 \
  --usp 1 --ring 1 --vae-patch-parallel-size 2 \
  --vae-parallel-mode tile --vae-use-tiling \
  --enable-distributed-layerwise-offload --dlo-no-use-allgather \
  --dlo-resident-layers 20 --enforce-eager \
  --diffusion-attention-backend CUDNN_ATTN
```

For Ref2VA, stop the FL2VA server and restart the same command with
`/path/to/MiniMax-H3/Ref2VA`. Ref2VA reference video count and prompt length
can increase activation memory; begin with one request at a time.
