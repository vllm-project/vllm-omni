# L1: complete request and semantic phases

## Prerequisites

Use the L0 boundary and [Omni execution map](../omni-execution.md). Read
[timing and overlap](timing-and-overlap.md) before assigning shares; for an online
or streaming objective also read [serving](serving.md).

## Procedure

1. Resolve entrypoint, `PipelineConfig`, deployment/placement, stage/replica and
   worker identities. Record actual process/device/rank relationships.
2. Map model semantics inside each stage: conditioning/encoding, initial work,
   repeated denoise or prefill/decode, scheduler, VAE/vocoder and output conversion.
   Semantic phases do not imply separate runtime stages.
3. Follow request/chunk IDs across admission, submission, execution, transport
   and completion. Include the orchestrator and output path, not only the worker.
4. Collect a bounded representative timeline using [Nsys collection](../nsys-collection.md).
   Check expected steps, branches, outputs and worker/rank coverage.
5. Associate ranges/kernels with phases using source and correlation IDs. Treat
   name-pattern matches as provisional until shape/count/path assertions agree.
6. Build a time ledger in one diagnostic window. Preserve unknown time and overlap
   buckets. Keep sums of per-kernel work in a separate population table.
7. Identify dependencies that determine request or chunk completion. Route exposed
   host delays to L2, excessive work to L3, expensive operators to L4 and transfers
   or remote readiness to L5.

## Root-cause branches

| Observation | Distinguishing check | Recommendation direction |
| --- | --- | --- |
| Fast DiT, slow request | Encoder, VAE, D2H, media encoding and client fetch | Diagnose missing phase; do not claim DiT tuning solves E2E |
| Large stage wall time, little stage GPU work | Input readiness, queues, CPU preparation, waits and missing events | Separate waiting from execution and collection gaps |
| Phase sums exceed request time | Nested ranges, concurrent stages/requests, inconsistent timebases | Rebuild non-overlapping accounting; do not normalize the sum to 100% |
| Huge first-step cost only | Compile/capture/prefill/conditioning and cache miss | Separate one-time versus repeated regimes |
| Trace ends before client result | Final output materialization/transfer/serialization | Extend capture or report the missing boundary explicitly |

## Required output

For every phase retain source owner, stage/replica, device/rank, start/end meaning,
wall span, device-work sum, union/exclusive attribution where available, dependency
on the next phase, and raw artifact reference. Distinguish measured fields from
unknowns. A phase table is complete only for its declared window and path.

L1 can establish where to focus and what is absent. It cannot certify kernel
efficiency, causality from timestamp proximity alone, or a global headroom bound.
