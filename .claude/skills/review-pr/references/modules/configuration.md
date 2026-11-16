# vLLM-Omni Configuration

Primary design: [vLLM-Omni Configuration](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/module/vllm_omni_config/).

Use for config construction, deploy YAML, registries, schema, defaults,
aliases, CLI projection, pipeline config, and topology. Inspect the reviewed
page's status: while its contract remains deferred or draft, current code and
tests—not proposed precedence rules—are authoritative.

## Contract checks

- Resolve each supported structured, legacy, CLI, environment, default, alias,
  and stage-override producer through its actual runtime consumer.
- Reject unknown or owner-mismatched fields explicitly; keep process-local
  runtime objects out of transportable configuration.
- Verify default, explicit, and feature-off values reach every live factory and
  consumer without silent reinterpretation.
- Treat stage count, placement, connector, parallelism, device, and memory
  fields as one topology contract and reject impossible combinations early.
- Preserve supported construction-path parity until a documented migration
  removes a path.

Require normalization and schema tests, a live-consumer assertion, and docs for
public keys, defaults, constraints, precedence, or migration behavior.
