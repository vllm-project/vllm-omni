# Submodule Offloading for MiniMax-H3

Submodule offloading is an opt-in weight-streaming mode for MiniMax-H3 DiT
blocks. It reuses the distributed layerwise offloading (DLO) transport but
streams three finer-grained weight units per block instead of one whole block:

```text
AdaLN projection -> attention -> MLP
```

This reduces the size of the device-side weight working set. It does not change
the attention or MLP math.

## Scope

This integration is implemented for **MiniMax-H3 only**. Other diffusion
models keep their existing offload behavior and are not part of this feature's
compatibility contract.

MiniMax-H3 token refiner blocks and encoders stay at block granularity. A block
that does not expose the required `adaln_proj`, `attn`, and `mlp` submodules
falls back to the ordinary DLO block hooks.

## Enable

Use the flag instead of the other offload strategy flags:

```bash
vllm serve /data/MiniMax-H3 --omni \
  --task-type ref2va \
  --num-gpus 8 \
  --usp 8 \
  --enable-submodule-offload
```

`--enable-submodule-offload` is a strategy selector. Do not combine it with
`--enable-distributed-layerwise-offload`, `--enable-layerwise-offload`, or
`--enable-cpu-offload`; use one strategy at a time. The runtime gives
`SUBMODULE` precedence over the legacy compatibility aliases, but mixed flags
are not a supported configuration.

The flag can be combined with the normal MiniMax-H3 serving options, including
USP/DP topology, VAE tiling, Cache-DiT, and reference-KV settings.

## Execution model

Submodule offload inherits DLO's two-slot device buffers, copy/communication
streams, events, host-weight staging, and failure cleanup. The difference is
only the streaming unit:

```text
Block-level DLO:  [ whole DiT block ]
Submodule DLO:    [ AdaLN ] [ attention ] [ MLP ]
```

The backend installs one hook group per DiT stack. While one submodule computes,
the next submodule's weights are prefetched through the same DLO transport.
Parameters outside the streamed leaves, such as norms, remain resident.

## Compatibility

### Cache-DiT

Physical block discovery understands Cache-DiT wrappers, so request-mode
Cache-DiT can use submodule offload without changing the wrapper's cache-hit
behavior. Cache state and policy remain owned by Cache-DiT.

### Step execution and continuous batching

Submodule offload is **not compatible with step execution**:

```text
--step-execution + submodule/DLO offload
```

is rejected because the DLO residency window spans a complete denoise loop,
while step execution would attempt to advance one scheduler step at a time.
Use request mode, or use data-parallel replicas if concurrent requests are
required.

### Reference KV

Reference-KV staging and refresh policy remain independently owned by the
reference-KV subsystem. Submodule offload only changes DiT weight residency; it
does not alter reference-KV semantics.

## Validation status

The feature has focused CPU coverage for hook ownership, skipped blocks,
configuration selection, and block-level fallback:

```bash
pytest -q tests/diffusion/offloader/test_submodule_offload.py
```

Current results: `5 passed`.

An Ascend 8-card service smoke test with a 15s / 720x1280 request completed with
HTTP 200 and produced a valid video/audio file. This is not CUDA coverage, and
the PR does not claim a specific device-memory reduction without a reproducible
matched comparison. Image-quality evaluation is also outside this validation.

## Troubleshooting

### The model still uses block-level hooks

Confirm that:

- the launched model is MiniMax-H3;
- `--enable-submodule-offload` is present in the server command;
- no conflicting offload strategy flag is set;
- the DiT block exposes the supported `adaln_proj`, `attn`, and `mlp` modules.

Incompatible auxiliary stacks intentionally fall back to block-level hooks.

### Startup fails with a step-execution error

Remove `--step-execution` when using submodule offload, or switch to
data-parallel serving for concurrent requests.

### Out of memory

Submodule offload lowers the weight working set but does not eliminate
activation, VAE, or communication memory. Reduce resolution, duration, or
batch concurrency, or increase the available device count.

## See also

- [Distributed layerwise offloading](distributed_layerwise_offload.md)
- [Layerwise offloading](layerwise_offload.md)
- [DLO design contract](../../../design/feature/offloader/distributed_layerwise_offload.md)
