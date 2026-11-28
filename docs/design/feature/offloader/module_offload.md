# Model-Level Offload

Model-level offload enforces mutual exclusion between pipeline component
groups. For commands and model support, see the
[model-level user guide](../../../user_guide/diffusion/offloader/module_offload.md).

## Generic hook path

`ModelLevelOffloadBackend.enable()` discovers DiTs, encoders, and VAEs. It
makes encoders and VAEs device resident for initialization, then installs
`SequentialOffloadHook` instances:

- each DiT hook offloads selected encoders and other selected DiTs before
  loading itself;
- each encoder hook offloads selected DiTs before loading itself; and
- VAEs are not part of the mutual-exclusion hook set.

All execution-stage modules still receive a hook so an unselected resident
stage can evict a selected component before it runs. Only components listed in
`diffusion_offload_config.components` are eligible to move to CPU.

The hook moves parameters and buffers without recursively calling
`module.to()`. This avoids recursion through modules that retain references to
other transformer blocks. Host tensors are pinned when configured; DTensors
are not pinned through the ordinary tensor path. A module built in host memory
starts pageable and is pinned by its first offload, so later loads never stage
the whole module through pageable memory.

On ROCm, pinned host copies of dense local modules share allocator-aware slabs
when that reduces or preserves the estimated backing size. The packed path
preserves physical-storage aliases, offsets, strides, dtypes, and the values at
each offload. Ordinary sequential movement falls back before module storage
changes for unsupported layouts, mixed residency, distributed tensors, and
slab-allocation failures. A module that already sits pageable on the host is
packed by the same path on its first offload; a module whose host copies are
already pinned is left alone. Persistent module-residency stagers use the same
packing for their immutable pinned masters.

A pipeline can declare immutable DiTs in `OffloadPlan.cpu_master_dit_paths`.
For selected DiTs on CUDA with supported tensor storage, the sequential hook
stages parameters and buffers from one CPU master and releases device storage
on offload without a device-to-host copy. The usual move path handles HSDP and
unsupported tensor layouts. Decode graphs retain their own fixed device
staging policy so captured weight addresses stay valid.
Pipelines can also opt immutable encoder parameters in via
`OffloadPlan.cpu_master_encoder_paths`. Encoder buffers use regular device
moves so updates made during forward survive offload and reload.

Compatible CUDA hooks can retain a bounded subset of immutable storages across
swaps. The stager selects whole physical storages largest first and counts
aliases once. Uncached storages still load from the CPU master on demand. On a
load OOM, the stager releases its cached weights and retries once without
retention. Fixed-address staging for decode graphs uses a separate policy.

Before forward returns control to the model, the hook synchronizes the active
platform so compute cannot race the transfer. XPU uses blocking device-to-host
copies because its allocator cannot safely evict storage while an asynchronous
copy is pending.

## Split-model delegation

If a pipeline provides callable `enable_omni_model_cpu_offload`, the backend
delegates instead of installing generic encoder/DiT hooks. This supports models
whose mutually exclusive components are nested inside one transformer, such
as a reasoner and generator.

The delegated pipeline owns its internal contexts, but must preserve the same
contract:

- only the executing component is device resident;
- enable and disable are idempotent;
- transfers use the supplied device, pinning, and HSDP settings;
- explicit component selection is either honored or rejected before partial
  setup; and
- disabling removes all model-local offload hooks or contexts.

The delegation is currently duck typed. If another independent implementation
is added, formalize the enable/disable pair as a protocol rather than expanding
attribute-name heuristics.

## Extension invariants

- New ordinary pipelines declare both DiT and encoder groups through
  `SupportsComponentDiscovery`.
- Models with multiple DiTs offload the inactive DiTs as well as encoders.
- Empty or parameterless modules do not trigger transfers.
- Hook removal does not move modules; callers must not assume original
  residency is restored.

Shared selection and lifecycle behavior is defined in the
[CPU Offloading design](README.md).
