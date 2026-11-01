# Stage Runtime and Replica Lifecycle

Primary design: [Stage Runtime and Replica Lifecycle](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/module/stage_runtime/).

Use for local or distributed placement, startup, readiness, stage pools,
replica identity and membership, selection, affinity, liveness, draining, and
stage-client or process lifecycle. Cross-stage request policy belongs to engine
orchestration; model scheduling belongs to the owning runtime.

## Contract checks

- Acquire, replace, and retire replicas through one stage-runtime lifecycle.
- Preserve stable replica identity and request affinity until termination or a
  declared replica loss.
- Make readiness explicit before admission; surface partial-startup and
  membership failures without leaving half-initialized capacity.
- Keep local and distributed semantics aligned for supported modes.
- Make drain, shutdown, and cleanup idempotent across repeated signals and
  process failure.

Test startup, readiness, selection, affinity, membership changes, abort,
process loss, draining, and repeated shutdown on affected local/distributed
paths.
