# Model-Level Offload

Model-level offload enforces mutual exclusion between pipeline component
groups. For commands and model support, see the
[model-level user guide](../../../user_guide/diffusion/offloader/cpu_offload.md).

## Generic hook path

`ModelLevelOffloadBackend.enable()` discovers DiTs, encoders, and VAEs. It
makes encoders and VAEs device resident for initialization, then installs
`SequentialOffloadHook` instances:

- each DiT hook offloads all encoders and other DiTs before loading itself;
- each encoder hook offloads every DiT before loading itself; and
- VAEs are not part of the mutual-exclusion hook set.

The hook moves parameters and buffers without recursively calling
`module.to()`. This avoids recursion through modules that retain references to
other transformer blocks. Host tensors are pinned when configured; DTensors
are not pinned through the ordinary tensor path.

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
- transfers use the supplied device, pinning, and HSDP settings; and
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
