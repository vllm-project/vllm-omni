# Autoregressive Runtime

Primary design: [Autoregressive Runtime](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/module/ar_runtime/).

Use for AR request queues, scheduling state, token and memory budgets, KV/input
readiness, workers, runners, and upstream vLLM request/cache transitions.

## Contract checks

- Preserve upstream request-state, scheduling, cache, and preemption semantics
  unless an Omni-specific difference is documented and tested.
- Convert modality-specific data through explicit input or output adapters; do
  not inject it through unrelated scheduler state.
- Keep workers and model runners focused on assigned execution rather than
  admission or cross-stage routing.
- Synchronize token, sequence, batch, and memory budgets with worker-visible
  state and prevent starvation, double counting, or stale readiness.
- Preserve legal transitions and one terminal cleanup path across completion,
  abort, timeout, preemption, worker failure, and shutdown.

Test request lifecycle, scheduling decisions, abort, cache state, adapters, and
every affected worker execution mode against the pinned upstream contract.
