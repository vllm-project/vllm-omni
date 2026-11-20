# MiniMax-H3 Encoder Weight Transfer

The packed mode is an opt-in capability and remains disabled by default. No distinguishable encode-stage or end-to-end gain was measured at 8 × Ascend 910B4 / case01 in the supported layerwise configuration. Selecting `off` disables packing only; pipeline hidden remains on device in every mode.

The MiniMax-H3 video pipeline on Ascend NPU can pack the offloaded text
encoder's host weights by module and dtype. This reduces the number of H2D
copies when the encoder is loaded for a request. Packing preserves each
weight's values, dtype, shape, stride, and aliases.

## Usage

`additional_config.encoder_weight_transfer` is a deployment setting:

```bash
vllm serve /path/to/MiniMax-H3 --omni \
  --diffusion-offload-config '{"mode":"layer","components":["dit","text_encoder"]}' \
  --additional-config '{"encoder_weight_transfer":"packed"}'
```

| Mode | Behavior |
| --- | --- |
| `off` (default) | Use the existing weight-transfer path. |
| `packed` | Pack the encoder's CPU master weights by module and dtype for fewer H2D copies; offload rebinds those masters without a weight D2H copy. |

The first encoder call builds the packs; later requests reuse them.

## First-call memory requirement

Packing allocates and copies new CPU masters before rebinding the encoder's
weights. The original CPU storage and the new packs therefore coexist during
the first call. Keep enough host memory for the original weight storage plus
the new pinned packs, in addition to the other model components and runtime.

Each source storage is aligned to 512 bytes inside a pack. The Ascend pinned
allocator rounds each pack allocation up to the next power of two, so its
reservation can exceed the packs' logical byte size. For a successful cold
pack build, budget the original CPU storage plus the sum of those rounded
pack sizes. Partially built packs can remain in the allocator cache when
allocation falls back to separate masters.

Independent full-encoder packing measured with the BF16 MiniMax-H3 encoder at TP8 on 8 × Ascend 910B4 (50 retained decoder layers, replicated vision weights):

| Metric | Result |
| --- | ---: |
| First packing call, rank-max mean ± SD over three cold trials | 3582.5 ± 228.1 ms |
| Original CPU weight storage per rank | 6.967 GiB |
| Logical packed storage per rank | 6.967 GiB |
| Peak pinned allocator reservation per rank | 7.980 GiB |
| Original storage + rounded pack reservation per rank | 14.948 GiB |
| Peak encoder process RSS per rank | 9.232 GiB |

Measured over three trials with fresh encoder processes, after two complete
warmup trials. Packing time excludes checkpoint loading and validation.
RSS and native pinned reservation are reported separately. Budget the
original weights plus rounded packs, with runtime and DiT/VAE memory added
separately.

## Constraints

- The text encoder must run locally on Ascend NPU.
- The text encoder must be selected for layerwise or distributed layerwise
  offload.
- Invalid values and unsupported configurations fail at startup.
- `off` disables this optimization, not CPU offloading.

The video pipeline keeps encoder hidden states on the device for broadcast in
every mode. Standalone encoder calls still return CPU tensors by default.

## End-to-end evidence

The matched current-source measurement and exact launch/request logs are in [the benchmark record](../../assets/performance/minimax_h3_encoder_weight_transfer/README.md). The full-encoder packing figures above cover the independent path without per-layer offload hooks. A deployment with encoder layerwise hooks packs only its non-block modules.
