# Cosmos3 — Distributed Layerwise Offload

> Sharding weights across DP ranks with H2D + AllGather overlap for multi-NPU/GPU deployment

## When to use this recipe

Use this recipe when deploying Cosmos3-Super (64B / 124 GB) or Cosmos3-Nano (17B / 33 GB) across multiple devices where the model does not fit on a single device's HBM, or when you need higher throughput via DP multi-concurrency.

**Key benefits:**
- Each rank stores only 1/dp_size of the model weights on host memory
- Only 2 transformer blocks reside on each device at any time (fixed double-buffer)
- DP multi-concurrency: N concurrent requests with near-linear throughput scaling
- mmap weight loading: shared page cache eliminates O(dp_size × model_size) RSS

## Prerequisites

- 2+ NPU or GPU devices (Ascend 910B3, NVIDIA B300, etc.)
- vLLM-Omni with distributed layerwise offload support
- Cosmos3-Nano or Cosmos3-Super checkpoint (local path or HuggingFace repo ID)

## Serve commands

### Cosmos3-Nano with DP=4 + AllGather (recommended for max throughput)

```bash
vllm serve /path/to/Cosmos3-Nano \
  --omni \
  --host 0.0.0.0 --port 8000 \
  --no-guardrails --init-timeout 3600 \
  --vae-use-tiling --vae-patch-parallel-size 1 \
  --enable-distributed-layerwise-offload \
  --data-parallel-size 4
```

### Cosmos3-Super with DP=2 + AllGather (for models > single-card HBM)

```bash
vllm serve /path/to/Cosmos3-Super \
  --omni \
  --host 0.0.0.0 --port 8000 \
  --no-guardrails --init-timeout 3600 \
  --vae-use-tiling --vae-patch-parallel-size 1 \
  --enable-distributed-layerwise-offload \
  --data-parallel-size 2
```

### Cosmos3-Super with DP=4 + AllGather

```bash
vllm serve /path/to/Cosmos3-Super \
  --omni \
  --host 0.0.0.0 --port 8000 \
  --no-guardrails --init-timeout 3600 \
  --vae-use-tiling --vae-patch-parallel-size 1 \
  --enable-distributed-layerwise-offload \
  --data-parallel-size 4
```

### Without DLO AllGather (rank-local weights, no DLO sharding)

Use `--dlo-no-use-allgather` when you want each rank to stream the standard
loader's rank-local weights independently — no DLO AllGather synchronization,
no concurrent request requirement, but no DLO weight sharding and higher host
memory use. In pure DP, this is equivalent to independent full-weight
layerwise-offload replicas; with TP or HSDP, each rank retains the standard
loader's local tensor layout.

```bash
vllm serve /path/to/Cosmos3-Nano \
  --omni \
  --host 0.0.0.0 --port 8000 \
  --no-guardrails --init-timeout 3600 \
  --vae-use-tiling --vae-patch-parallel-size 1 \
  --enable-distributed-layerwise-offload \
  --data-parallel-size 4 \
  --dlo-no-use-allgather
```

### With SP (sequence parallel) instead of DP

For long-sequence workloads where SP parallelism is preferred:

```bash
vllm serve /path/to/Cosmos3-Super \
  --omni \
  --host 0.0.0.0 --port 8000 \
  --no-guardrails --init-timeout 3600 \
  --vae-use-tiling --vae-patch-parallel-size 1 \
  --enable-distributed-layerwise-offload \
  --usp 4
```

## Request examples

### Text-to-Image (T2I)

```bash
curl -s -o output.png -X POST "http://localhost:8000/v1/images/generations" \
  -H "Content-Type: multipart/form-data" \
  -F "model=/path/to/Cosmos3-Nano" \
  -F "prompt=A robot arm cleaning a plate, cinematic" \
  -F "size=1024x1024" \
  -F "num_inference_steps=50" \
  -F "guidance_scale=6.0" \
  -F "seed=42"
```

### Text-to-Video (T2V)

```bash
curl -s -o output.mp4 -X POST "http://localhost:8000/v1/videos/sync" \
  -H "Accept: video/mp4" \
  -F "model=/path/to/Cosmos3-Nano" \
  -F "prompt=A robot arm cleaning a plate, cinematic shot" \
  -F "negative_prompt=blurry" \
  -F "size=832x480" \
  -F "num_frames=29" \
  -F "fps=24" \
  -F "num_inference_steps=35" \
  -F "guidance_scale=6.0" \
  -F "max_sequence_length=4096" \
  -F "flow_shift=10.0" \
  -F 'extra_params={"use_resolution_template":false,"use_duration_template":false,"guardrails":false}' \
  -F "seed=17"
```

> **Important:** When using DP multi-concurrency (AllGather mode), `num_inference_steps`
> must be specified explicitly and must be the same for all concurrent requests.
> `num_inference_steps=None` is not allowed because it may resolve differently
> per request mode (e.g., T2V=35 vs action_mode=30), which would deadlock AllGather.

### Multiple concurrent requests (DP multi-concurrency)

With `--data-parallel-size 4`, send 4 concurrent requests to maximize throughput:

```bash
for i in 1 2 3 4; do
  curl -s -o output_${i}.mp4 -X POST "http://localhost:8000/v1/videos/sync" \
    -H "Accept: video/mp4" \
    -F "model=/path/to/Cosmos3-Nano" \
    -F "prompt=A robot arm cleaning a plate, shot ${i}" \
    -F "negative_prompt=blurry" \
    -F "size=832x480" \
    -F "num_frames=29" \
    -F "fps=24" \
    -F "num_inference_steps=35" \
    -F "guidance_scale=6.0" \
    -F "max_sequence_length=4096" \
    -F "flow_shift=10.0" \
    -F 'extra_params={"use_resolution_template":false,"use_duration_template":false,"guardrails":false}' \
    -F "seed=17" &
done
wait
```

## CLI flags

| Flag | Description | Default |
|------|-------------|---------|
| `--enable-distributed-layerwise-offload` | Enable DLO with H2D + AllGather overlap | `false` |
| `--data-parallel-size N` | Number of DP ranks; DLO weight sharding + concurrent requests when AllGather is enabled | `1` |
| `--dlo-use-allgather` | Use shard + AllGather for weight reconstruction (recommended) | `true` |
| `--dlo-no-use-allgather` | Stream standard-loader rank-local weights independently (no DLO sharding, no DLO AllGather) | `false` |
| `--usp N` | Use SP instead of DP for weight sharding (long sequences) | `1` |

## Mode comparison

| Mode | CLI flags | CPU/rank | HBM/card | Throughput | Use case |
|------|----------|---------|---------|-----------|----------|
| DLO + AllGather (DP4) | `--enable-distributed-layerwise-offload --data-parallel-size 4` | 1/4 model | 2 blocks | **3.3× HSDP** | Max throughput, short sequences |
| DLO + AllGather (DP2) | `--enable-distributed-layerwise-offload --data-parallel-size 2` | 1/2 model | 2 blocks | 2× HSDP | Balanced throughput + memory |
| DLO + AllGather (SP4) | `--enable-distributed-layerwise-offload --usp 4` | 1/4 model | 2 blocks | 1× (single req) | Long sequences (720p+) |
| DLO no-AllGather | `--enable-distributed-layerwise-offload --dlo-no-use-allgather` | Standard-loader rank-local weights | 2 blocks | 0.4× HSDP | No DLO AllGather overhead, high CPU |
| HSDP | `--use-hsdp --hsdp-shard-size 4` | 0 | 1/4 model | 1× (baseline) | Weights fit in HBM |

## Memory expectations

| Model | Config | Host RAM (cgroup) | HBM/card |
|-------|--------|-------------------|----------|
| Nano (33 GB) | DP2 + AllGather | ~38 GB | ~10 GB |
| Nano (33 GB) | DP4 + AllGather | ~47 GB | ~10 GB |
| Super (124 GB) | DP2 + AllGather | ~157 GB | ~15 GB |
| Super (124 GB) | DP4 + AllGather | ~172 GB | ~10 GB |

> Host RAM is dominated by page cache (1× model_size, shared across ranks).
> HBM only holds 2 transformer blocks + framework + activations, independent of model size.

## Limitations

- `--dlo-use-allgather` (default) requires all concurrent requests to have the same
  `num_inference_steps` (AllGather is a collective that needs all ranks synchronized)
- Online per-tensor FP8 linears with DLO AllGather use the ordinary loader
  instead of direct checkpoint mmap, then shard the finalized FP8 weights and
  scales. Size host memory for a transient complete FP8 model on every rank at
  startup
- Tensor Parallel > 1 is rejected in the DLO+AllGather mmap path. TP with
  `--dlo-no-use-allgather` retains the standard TP loader and should be treated
  as an experimental compatibility path
- HSDP + AllGather is rejected (would double-shard weights). HSDP with
  `--dlo-no-use-allgather` is accepted at configuration level, but is not yet a
  broadly validated production combination
