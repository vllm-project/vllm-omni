# Layerwise Offloading

Layerwise, or blockwise, offloading keeps one transformer block on the
accelerator and prefetches the next block while the current block computes.
It is best suited to compute-heavy video DiTs whose block execution can hide
host-to-device transfers.

## Execution flow

Each block has a pre-forward and post-forward hook. Parameters are consolidated
in pinned host tensors and rematerialized for execution on a dedicated copy
stream.

| Block | Pre-forward hook | Forward | Post-forward hook |
| --- | --- | --- | --- |
| block 0 | Prefetch block 1 | Compute block 0 | Free block 0 |
| block 1 | Prefetch block 2 | Compute block 1 | Free block 1 |
| ... | ... | ... | ... |
| last block | Prefetch block 0 | Compute last block | Free last block |

Selected, plan-declared text-encoder blocks can use the same rank-local
streaming mechanism. Image encoders, VAEs, and non-block DiT modules remain
device resident.

## Usage

```python
from vllm_omni import Omni

omni = Omni(
    model="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    diffusion_offload_config={
        "mode": "layer",
        "components": ["dit"],
    },
)
```

```bash
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --omni \
  --diffusion-offload-config '{"mode":"layer","components":["dit"]}'
```

## Component selection

Add `dit`, `text_encoder`, or both to the `components` list. Omitting
`layer_options` uses safe rank-local defaults:

- `["dit"]` streams only DiT blocks.
- `["text_encoder"]` streams only declared text-encoder blocks.
- Listing both streams both components.

```bash
# DiT-only layer offload
vllm serve /path/to/model --omni \
  --diffusion-offload-config \
  '{"mode":"layer","components":["dit"]}'

# Stream a model-declared text encoder while keeping the DiT resident
vllm serve /path/to/model --omni \
  --diffusion-offload-config \
  '{"mode":"layer","components":["text_encoder"]}'
```

Encoder categories are resolved from `OffloadPlan.encoder_component_types`
first. A name-based fallback is retained for pipelines that predate
`OffloadPlan`.

## Model integration

Transformer classes declare containers of executable blocks:

```python
class WanTransformer3DModel(nn.Module):
    _layerwise_offload_blocks_attrs = ["blocks"]


class Flux2Transformer2DModel(nn.Module):
    _layerwise_offload_blocks_attrs = [
        "transformer_blocks",
        "single_transformer_blocks",
    ]
```

Auxiliary components use declarative pipeline metadata:

```python
from vllm_omni.diffusion.offloader import OffloadPlan


class MyPipeline(nn.Module):
    _encoder_modules = ["prompt_model"]
    _offload_plan = OffloadPlan(
        encoder_component_types={"prompt_model": "text_encoder"},
        encoder_block_attrs={"prompt_model": ("encoder.layers",)},
    )
```

See the [layerwise design](../../../design/feature/offloader/layerwise_offload.md)
for the discovery and hook invariants. Both ordinary and distributed layerwise
offload consume the same `OffloadPlan` metadata.

## Limitations

- The default weight transfer is `rank-local`. Set a selected component's
  `weight_transfer` to `allgather` to shard its host weights across a
  compatible multi-device group; backend selection is automatic.
- Setup consolidates and pins block parameters, increasing cold-start time.
- Performance depends on block compute time and host-to-device bandwidth;
  lightweight blocks may not hide transfers.

## NPU CPU binding

Set `additional_config={"enable_cpu_binding": true}` to bind each NPU worker
and its existing threads to permitted CPUs on its device's PCIe NUMA node.
`false` preserves incoming affinity; explicit `true` reports binding failure.
The DCMI adapter supports both legacy and Ascend 950 APIs and reordered device
visibility. Binding is independent of sparse precision and does not enable LO.

```bash
vllm serve /path/to/model --omni \
  --additional-config '{"enable_cpu_binding":true}'
```

Local spawned workers inherit locality before Torch imports; the launching
thread restores its affinity after startup, including failed starts. Worker
initialization and loading bind existing thread pools within their individual
CPU limits. IRQ affinity and process-wide page migration are unchanged.
Omit `enable_cpu_binding` to preserve the default placement. Consider enabling
it when host work and NPU transfers are running on different NUMA nodes. At
startup with `VLLM_LOGGING_LEVEL=DEBUG`, check the worker affinity messages for
the physical device, NUMA node, and the actual allowed CPUs. A5 uses explicit
binding with layerwise offload and AdaLN residency disabled.

`additional_config.hccl_expansion_mode` optionally selects the engine for
Ulysses and AllGather-KV groups: `2` for AICPU or `3` for AIV. Omit it to keep
the runtime's HCCL default. Use the same configuration on all ranks and check
the group creation debug logs for the effective override.
