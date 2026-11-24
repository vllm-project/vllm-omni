# Profiling

Primary design: [Profiling](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/module/profiling/).

Use for opt-in instrumentation and trace collection across supported profiler
backends.

## Contract checks

- Keep expensive collection disabled by default and bounded by explicit start,
  stop, timeout, and cleanup behavior.
- Preserve scheduling, generated outputs, resource ownership, and concurrency
  semantics apart from documented synchronization overhead.
- Include stage, rank, worker, operation, and request correlation where useful
  without embedding user payloads or secrets.
- Make repeated start/stop and partial failure safe, including multi-rank trace
  naming and artifact ownership.
- Measure disabled and enabled overhead before making performance claims.

Test the disabled path, repeated lifecycle, failure cleanup, trace context, and
each affected backend.
