# Execution Platforms

Primary design: [Execution Platforms](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/module/execution_platforms/).

Use for hardware capability detection, worker selection, patches, kernels,
attention backends, and vendor-specific configuration or connectors.

## Contract checks

- Guard hardware behavior through platform selection or capability detection
  and fail clearly when unsupported.
- Keep portable modules free of direct vendor imports when a common selector
  exists; verify portable imports without the target accelerator.
- Override only behavior that differs from common or upstream implementations,
  and compare sibling worker hooks when inheritance does not provide parity.
- Preserve dtype, layout, rank, stream, synchronization, workspace,
  graph-capture, and resource-lifetime contracts.
- Do not let simplified CPU doubles hide platform MRO, initialization, or
  dispatch failures.

Run portable checks first, then the smallest matching-device test in a fresh
environment. Report unavailable hardware as a validation gap, never simulated
device evidence.
