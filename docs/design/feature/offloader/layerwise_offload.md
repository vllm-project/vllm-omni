# Layerwise Offload

Layerwise offload streams transformer blocks through one accelerator. For
commands and model support, see the
[layerwise user guide](../../../user_guide/diffusion/offloader/layerwise_offload.md).

## Hook ring

`LayerWiseOffloadBackend` discovers every streamable block and registers one
`LayerwiseOffloadHook` per block. Hooks form a ring: each hook owns the pinned
host representation of its registered block and a reference to the next block
it must prefetch; the final hook prefetches the first block for the next
denoising iteration.

During initialization, parameters and buffers are grouped by dtype and copied
into contiguous pinned CPU tensors. Original tensor storage is replaced with
an empty or meta placeholder while identity and module structure are retained.
Metadata records names, offsets, shapes, and dtypes for rematerialization.

## Forward lifecycle

For each block:

1. the previous hook's completion event ensures this block is materialized;
2. the copy stream asynchronously allocates and fills the next block's device
   tensors;
3. parameter and buffer storage is rebound to views of those tensors;
4. the current block computes on the normal stream; and
5. its post-forward hook replaces device storage with placeholders after all
   dependent work is safe.

The copy stream waits on the compute stream before reusing storage, and the
compute stream waits on the per-block prefetch event before forward. These
dependencies, not incidental global synchronization, define correctness.

## Topology contract

A DiT declares ordered block-container attributes using
`_layerwise_offload_blocks_attrs` or an `OffloadPlan`. Each resolved item must
be an executable `nn.Module`; ordering must match forward execution. Multiple
containers are concatenated in execution order.

Non-block DiT modules, encoders, VAEs, and declared resident modules are not
streamed by this backend.

## Invariants and limitations

- Each offloaded tensor has one pinned host backing store.
- Parameter identity is preserved when storage is rebound.
- Blocks with heterogeneous shapes and dtypes use their recorded metadata;
  implementations must not assume identical blocks.
- The ring is initialized before the first forward so block zero is available.
- This backend is single-device; distributed weight reconstruction belongs to
  [Distributed Layerwise Offload](distributed_layerwise_offload.md).

Shared selection and lifecycle behavior is defined in the
[CPU Offloading design](README.md).
