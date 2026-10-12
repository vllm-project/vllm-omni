# Model Integration

Primary design: [Model Integration](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/module/model_integration/).

Use for model registration, configuration adapters, preprocessing, checkpoint
loading, stage inputs, model-specific hooks, shared layers, runners, and model
execution. Use the diffusion family reference for diffusion models.

## Contract checks

- Declare explicitly how the model class, loader, input processor, stage
  configuration, and runtime capabilities are selected.
- Preserve weight-name mapping, dtype/device placement, optional-dependency
  errors, and checkpoint-specific loading semantics.
- Trace each typed input and output through preprocessing, runner state, model
  call, and its actual consumer.
- Reuse upstream vLLM behavior when sufficient; document and test the precise
  difference for every override or patch.
- Keep model code out of admission, cross-stage routing, public policy, and
  terminal lifecycle ownership.
- Place reusable behavior in shared modules and test it with more than one
  integration.

Require focused registry, loader, processor, runner, and representative model
execution tests. Apply the model-addition check for new or expanded support.
