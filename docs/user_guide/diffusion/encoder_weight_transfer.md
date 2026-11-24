# MiniMax-H3 Encoder Weight Transfer

The MiniMax-H3 video pipeline on Ascend NPU supports optional packing of
text encoder CPU master weights by module and dtype. Packing preserves
weight values, dtype, shape, stride, and aliases. It is disabled by default.

## Usage

Set `additional_config.encoder_weight_transfer` to `packed`:

```bash
vllm serve /path/to/MiniMax-H3 --omni \
  --diffusion-offload-config '{"mode":"layer","components":["dit","text_encoder"]}' \
  --additional-config '{"encoder_weight_transfer":"packed"}'
```

| Mode | Behavior |
| --- | --- |
| `off` (default) | Use the existing weight-transfer path. |
| `packed` | Reuse packed CPU masters for H2D transfer; offload rebinds the CPU masters without copying weights back from the device. |

The text encoder must run locally on Ascend NPU and be selected for
layerwise or distributed layerwise offload. Invalid values and unsupported
configurations fail at startup. The first encoder call builds the packs;
later requests reuse them. With per-layer offload hooks, packing applies to
non-block modules; decoder blocks retain their existing transfer path.

## Device memory

Packed staging releases live device weight tensors after encoding and
retains freed allocator blocks under the existing `BoundedAllocatorCache`
policy. Unused cached memory must stay within 25% of device capacity, and
at least 5% must remain physically free. A supplied component cache is
preserved; otherwise packed staging creates a component-local policy.
Memory pressure, missing telemetry, and forced cleanup release the cache.
Executor shutdown cleanup remains unconditional.

Weights are loaded from the CPU again on subsequent requests. Selecting
`off` restores the existing weight-transfer and allocator-cache behavior.

## Host memory

The original CPU storage and new pinned packs coexist during the first
packing call. Budget both, plus memory for other components and runtime.
Each source storage is aligned to 512 bytes inside a pack. The Ascend
pinned allocator rounds each pack allocation to the next power of two,
so reserved memory can exceed logical pack size. Partially built packs
can remain in the allocator cache after fallback to separate CPU masters.

## Hidden states

On Ascend NPU, the video pipeline keeps encoder hidden states on the device
for broadcast in both modes. Other backends retain the existing CPU return
path. Standalone encoder calls return CPU tensors by default.
