# Quantization

Primary design: [Quantization](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/module/quantization/).

Use for method registration, checkpoint interpretation, layer replacement,
reduced precision, platform capability, and runtime constraints.

## Contract checks

- Validate checkpoint metadata, selected method, supported layers, device
  capability, required dependencies, and parallel layout before loading.
- Preserve loader and weight-name mapping semantics, dtype, scale/granularity,
  zero point, padding, and buffer handling through the real kernel consumer.
- Fail clearly for unsupported combinations; do not silently choose a different
  method or precision.
- Keep online and pre-quantized checkpoint paths explicit and ensure patches or
  fallbacks do not bypass model-owned loaders.
- Require numerical comparison against a defined reference with explicit
  tolerance before treating performance results as valid.

Test registration, loading, representative layers, metadata, platform guards,
parallel combinations, and numerical correctness separately from performance.
