# Observability

Primary design: [Observability](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/module/observability/).

Use for metrics, logs, correlation fields, and cross-stage request visibility.

## Contract checks

- Preserve stable request and stage identity across entrypoints, orchestration,
  stages, workers, errors, and terminal cleanup.
- Treat metric names, units, labels, aggregation, and lifecycle as public
  contracts; document any semantic change.
- Keep labels bounded: never use request IDs, prompts, outputs, paths, or other
  unbounded user values.
- Register and reset metrics exactly once across repeated startup and shutdown,
  and keep disabled instrumentation inexpensive.
- Align emitted values with their actual timing and lifecycle owner rather than
  reconstructing them from ambiguous client strings.

Test registration, label sets, units, resets, error classification, and
representative multi-stage emission.
