# MiniMax H3 Online-FP8 Offload Compatibility Design

## Goal

Prevent `VLLM_OMNI_H3_TEXT_ENCODER_QUANTIZATION=fp8` from being silently
ignored when MiniMax H3 uses an offload mode whose parameter lifecycle is not
compatible with the text encoder's post-load online-FP8 conversion.

## Decision

Reject these combinations during `MiniMaxH3Pipeline` initialization:

- online FP8 plus `enable_cpu_offload=True`;
- online FP8 plus `enable_distributed_layerwise_offload=True`.

Continue to allow online FP8 with no offload and with
`enable_layerwise_offload=True` alone. Supporting online FP8 with model-level
or distributed layerwise offload requires a separate change that integrates
FP8 weights and scales with offloader parameter capture and movement.

## Initialization Flow

The pipeline parses the environment variable exactly once near the beginning
of `__init__`, before reading model metadata or constructing model components.
A small validation helper checks the parsed boolean against the two raw
offload flags. Raw flags are intentional: if CPU offload is requested together
with another strategy, the current encoder execution branch can still take the
CPU-offload path, so accepting the configuration would reintroduce the silent
fallback.

After validation, the parsed value is passed to
`MiniMaxH3Qwen3VLEncoder`. This avoids a second per-process environment read and
ensures the value that was validated is the value that is applied.

## Error Behavior

Each unsupported combination raises `ValueError` with an actionable message
that names online FP8, the incompatible offload mode, and the available
choices: disable that offload mode or disable online FP8. If both incompatible
flags are set, the message lists both modes.

The exception must occur before model files are read or heavyweight model
objects are created.

## Tests

CPU-only regression tests will cover:

- FP8 plus model-level CPU offload is rejected;
- FP8 plus distributed layerwise offload is rejected;
- FP8 plus both incompatible flags reports both modes;
- FP8 with no offload is accepted;
- FP8 with ordinary layerwise offload alone is accepted;
- BF16/default mode remains accepted with every offload flag.

One pipeline-construction test will use a deliberately nonexistent model path
and assert that the compatibility error is raised first, proving validation is
performed before heavyweight initialization.

## Non-goals

- Quantizing weights inside sequential-offload hooks;
- teaching distributed layerwise AllGather to manage FP8 weights and scales;
- changing offload strategy precedence;
- rebasing this PR onto current `main`.
