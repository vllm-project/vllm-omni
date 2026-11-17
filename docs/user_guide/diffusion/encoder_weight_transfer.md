# MiniMax-H3 Encoder Weight Transfer

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

## Constraints

- The text encoder must run locally on Ascend NPU.
- The text encoder must be selected for layerwise or distributed layerwise
  offload.
- Invalid values and unsupported configurations fail at startup.
- `off` disables this optimization, not CPU offloading.

The video pipeline keeps encoder hidden states on the device for broadcast in
every mode. Standalone encoder calls still return CPU tensors by default.
