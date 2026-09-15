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
streaming mechanism. Image encoders, unselected VAEs, and non-block DiT
modules remain device resident. Selected VAEs use pipeline-managed stage
transfers rather than block streaming.

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

Select `dit`, `text_encoder`, and/or `vae` in `components`. Omitting
`layer_options` uses ordinary rank-local layerwise loading:

- `["dit"]` streams only DiT blocks.
- `["text_encoder"]` streams only declared text-encoder blocks.
- Listing both streams both components.
- `"vae"` stages the model-declared VAEs around encode/decode.

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

### VAE stage lifecycle

To support `components: ["vae"]`, declare every VAE in `_vae_modules` and
`OffloadPlan.on_demand_component_paths`. Each wrapper must implement
`load_to_device()` and `offload_to_cpu()`. The pipeline must call these around
all encode/decode entry points and release the selected component on failure.
The resolver validates the declaration before installing hooks or moving
weights; it does not assume that `encode()` or `decode()` invokes `forward()`.

Use the same component selection for initial placement and runtime staging to
avoid materializing an offloaded VAE on the accelerator during startup.
Test repeated calls, exceptions, and unselected components. Module-mode VAE
offload additionally requires the pipeline-owned `SupportsModelCpuOffload`
lifecycle.

## Limitations

- Explicit `weight_transfer` selects the [bounded two-slot backend](distributed_layerwise_offload.md),
  including `rank-local` with zero resident layers. `allgather` additionally
  shards host weights across a compatible multi-device group.
- Setup consolidates and pins block parameters, increasing cold-start time.
- Performance depends on block compute time and host-to-device bandwidth;
  lightweight blocks may not hide transfers.
