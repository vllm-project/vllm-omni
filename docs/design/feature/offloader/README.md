# CPU Offloading

This document defines the shared architecture for diffusion CPU offloading.
The three strategies have separate user and design pages:

| Strategy | User guide | Design contract |
| --- | --- | --- |
| Model-level | [Guide](../../../user_guide/diffusion/offloader/cpu_offload.md) | [Design](cpu_offload.md) |
| Layerwise | [Guide](../../../user_guide/diffusion/offloader/layerwise_offload.md) | [Design](layerwise_offload.md) |
| Distributed layerwise | [Guide](../../../user_guide/diffusion/offloader/distributed_layerwise_offload.md) | [Design](distributed_layerwise_offload.md) |

## Strategy selection

`OffloadConfig.from_od_config()` converts diffusion configuration into one
`OffloadStrategy`. If callers set multiple flags, selection is deterministic:

1. `DISTRIBUTED_LAYERWISE`;
2. `LAYERWISE`;
3. `MODEL_LEVEL`; and
4. `NONE`.

The factory logs when a higher-priority strategy disables another. It derives
parallel and HSDP state from `DiffusionParallelConfig`; callers do not provide
a separate offload group size.

`get_offload_backend()` then validates platform offload support, resolves the
device, and creates exactly one backend. Returning `None` means offloading is
disabled or unsupported and must not leave partially installed hooks.

## Shared lifecycle

Every backend implements `OffloadBackend`:

- `enable(pipeline)` discovers modules, establishes initial residency, and
  installs hooks;
- `disable()` removes owned hooks and resources; and
- `is_enabled()` reports lifecycle state.

`disable()` does not promise to restore the pipeline's original device
placement. The caller owns any subsequent rematerialization.

Hooks are registered through `HookRegistry` and `ModelHook`; offload backends
must use distinct hook names and remove only hooks they own.

## Discovery and topology

Pipeline component discovery prefers `SupportsComponentDiscovery` declarations:

- `_dit_modules`;
- `_encoder_modules`;
- `_vae_modules`; and
- `_resident_modules`.

Dotted paths are supported. Legacy pipelines may use the fallback scan of
well-known attribute names, but new integrations should declare components
explicitly.

Ordinary layerwise offload obtains topology from declared
`_layerwise_offload_blocks_attrs` (with a deprecated singular-name
compatibility path). Distributed layerwise offload may instead use a pipeline
`OffloadPlan`, then falls back to declared block attributes and its documented
compatibility discovery. Discovery metadata describes structure only; the
backend remains responsible for transfer, synchronization, and storage
ownership.

## Cross-strategy invariants

- At most one strategy owns offload hooks for a pipeline.
- A parameter has one authoritative host representation while offloaded.
- Device storage is not freed until dependent compute or transfer work is
  complete.
- Non-persistent and model-specific buffers remain correct after movement or
  checkpoint rematerialization.
- Unsupported parallel or loading combinations fail before hooks mutate the
  model.
- Platform streams, events, synchronization, and cache management go through
  the vLLM-Omni platform abstraction.

The diffusion [Offloader module design](../../module/diffusion/offloader.md)
describes how these feature contracts fit into the larger diffusion runtime.
