# Engine Orchestration

Primary design: [Engine Orchestration](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/module/engine_orchestration/).

Use for request-state creation, cross-stage routing, output ordering, companion
tracking, RPC correlation, cancellation propagation, and terminal convergence.
Orchestration does not own replica placement, stage-process startup, public
rendering, connector transport, or payload and error schemas.

## Contract checks

- Keep one owner for cross-stage routing; entrypoints, models, and stage clients
  must not independently select the next logical stage.
- Preserve request identity and output order across concurrent stages and
  correlate every control or RPC result to the originating request and wave.
- Make terminal state monotonic: no new forwarding after success, abort,
  request failure, or fatal engine failure.
- Propagate cancellation and failure to every active stage and release
  orchestrator-owned tasks, queues, companions, and request state.
- Preserve shutdown ordering across the orchestrator and stage runtime.

Test representative multi-stage routing, ordering, correlation, cancellation,
failure, and repeated shutdown.
