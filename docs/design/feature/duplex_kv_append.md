# Transactional KV Append for Native Duplex Sessions

## Status and scope

Status: implemented in the local cleanup, with acceptance of that exact
snapshot tracked separately. This design is not an E2E sign-off or a
performance result. Earlier H20/H200 runs describe earlier source snapshots.

The shared OpenAI duplex entry point routes MiniCPM-o 4.5, PersonaPlex, and
Nemotron VoiceChat Stage0 input through the same transactional append
contract. Each model still owns its input preparation, sampling policy, and
model state. Downstream generation stages and independent serving backends
have different contracts and are outside this unification.

The target dependency is vLLM 0.28.0. It retains requests and KV across
`StreamingUpdate` operations. Omni adds operation identity, retry receipts,
replica affinity, and lifecycle fencing to that mechanism. Retained KV was
already available through the previous resumable submission route; this
cleanup alone does not establish a reduction in GPU computation.

See [the validation runbook](../../validation/kv_append_v028.md) for environment
requirements and commands, and [engine orchestration](../module/engine_orchestration.md)
for shared stage ownership.

## Single supported lifecycle

`vllm_omni.engine.kv_append` checks the installed 0.28 contract without adding
methods or enum members to upstream classes. Session open also checks every
live Stage0 client and requires a positive context limit. Unsupported native
Stage0 submission fails instead of falling back to the previous update route.

| Phase | Owner and transition |
| --- | --- |
| First input | StagePool validates operation identity, binds a replica, and calls `admit_duplex_request_async`. The Omni client submits a resumable request through upstream `add_request_async`. |
| Initial acknowledgement | StagePool polls metrics until `num_computed_tokens >= num_prompt_tokens`. No last-token hold or external finalize operation is used. |
| Unit completion | The scheduler retains request/KV state in `WAITING_FOR_STREAMING_REQ`. The next append waits for this boundary and previous output-registration retirement. |
| Next input | The Omni client calls the explicitly declared `StageEngineCoreProc.append_streaming_prompt_unit` utility. The scheduler applies one `StreamingUpdate` through its Omni override. |
| Receipt publication | `_commit_native_append` marks the applied operation before its receipt is published. This is synchronous internal bookkeeping, not an RPC or another scheduling phase. |
| Close or cancel | The control plane fences pending work and aborts the bound physical request. Terminal cleanup releases its KV and request-scoped state. |

OutputProcessor registration remains per unit (`resumable=False`). Only after
that registration does the Omni client set `resumable=True` on the request
sent to the scheduler. These are different owners: the output registration
must retire at a unit boundary while the scheduler keeps the request and KV.

The scheduler update and receipt bookkeeping execute in one utility call on
the scheduler thread. A request may become runnable during that synchronous
call, but another scheduling iteration cannot interleave between its steps.
Append does not mutate the prompt halfway through an actively decoding unit.

An operation ID and complete fingerprint bind input tokens, model payload,
configuration, sampling parameters, and the final-input flag. Within the
retained receipt window, a same-operation retry returns the existing receipt;
window-expired operations are rejected rather than promising an indefinitely
available receipt. Different content with the same ID is
rejected. A pending marker after an interrupted receipt commit blocks other
operations. Retrying it finishes receipt bookkeeping without applying tokens
again, including when decoding has already completed that unit.

StagePool preserves the output registration and exact-operation guard after
an uncertain utility timeout. Sampling parameters cross the real serialized
utility boundary, with `SamplingParams` reconstructed on the Omni Core side.
Readiness polling, output retirement, and the utility await share the caller's
deadline. Acknowledgement means the documented engine operation completed;
it does not certify that the model produced a complete answer.

## Append and replay are separate capabilities

KV retention is also independent of frontend response lifetime. The
`response_lifecycle` capability defaults to `model_turn`; Nemotron opts into
`continuous_stream` to match NVIDIA NIM. Its utterance EOS does not terminate
the response, and graceful close requires accepted-input and delivered-output
watermarks to agree. Model-turn adapters retain their existing policy. See
the [official alignment and validation](../../validation/nemotron_official_alignment_20260907.md).

| Shared adapter | Transactional Stage0 append | `supports_prompt_replay` |
| --- | --- | --- |
| MiniCPM-o 4.5 | Advertised when the 0.28 contract is available | Enabled with append capability; bounded prompt journal and rebuild policy |
| PersonaPlex | Advertised when the 0.28 contract is available | `False` |
| Nemotron VoiceChat | Advertised when the 0.28 contract is available | `False` |

The engine validates these declarations against its actual stage clients.
Declaring append does not imply that replay can reconstruct codec, recurrent,
Mamba, or other model-owned state. PersonaPlex and Nemotron must not enter
automatic prompt-replay recovery or context rollover. Replica loss uses a
typed safe failure such as `native_kv_replica_lost`; clients must reopen a
session. Replaying application input after reopening remains a client policy,
not a guarantee of equivalent model state.

MiniCPM's bounded replay creates a new physical request and rebuilds from
committed input units. It does not transfer raw KV tensors, guarantee bitwise
history equivalence, or provide lossless arbitrary-context compaction.
Replay limits, candidate-fit checks, and terminal failure handling remain
part of that opt-in capability. None of the adapters claims live KV migration.

## Removed and protected boundaries

### Removed compatibility surface

The cleanup removes the following production paths and migrates their active
native Stage0 callers:

- `_streaming_prompt_compat.py` and the experimental forwarding module.
- Import-time additions to upstream `EngineCore` and `AsyncMPClient` for
  append/admission, and the synthetic `WAITING_FOR_STREAMING_PROMPT` alias.
- Dual legacy/0.28 streaming API selection, the old worker
  `new_prompt_token_ids` handling, and legacy held-finalize queue promotion.
- External `finalize_streaming_prompt` RPCs, including the separate call after
  initial prefill. `_commit_native_append` does not reintroduce that interface.
- The shared native Stage0 fallback to `StagePool.submit_update`.

Tests now exercise the 0.28 `StreamingUpdate` contract rather than keeping a
second fake scheduler that implements the retired API. Fault tests preserve
the relevant idempotency, context, deadline, and cancellation assertions.

### Protected active contracts

| Retained code | Production consumer and reason |
| --- | --- |
| `StagePool.submit_update` | Public streaming input, diffusion stage resubmission, PD decode, and downstream talker/code2wav routing still consume this segment-update contract. These are not all retained-KV append operations. |
| PersonaPlex independent browser backend | Its frame-stepper, session, and browser adapters have active consumers outside the shared OpenAI engine path. They are a separate implementation, not an unused import shim. |
| Upstream `_update_request_as_session` dependency | The Omni scheduler override delegates the actual retained-request update to vLLM. Its private API dependence remains concentrated at this boundary and must be checked on upgrades. |
| Other Omni request/worker extensions | Removing the append monkey patches does not remove unrelated multimodal request, serialization, or platform adaptations. |

Deleting any of these contracts requires a separate consumer migration and
validation plan. Renaming them to "KV Append" would not make their lifecycle
semantics equivalent. A future stable upstream append API can replace the
private update dependency after a contract comparison and regression run;
copying the upstream implementation into Omni is not the intended replacement.

### MiniCPM per-handoff boundaries

A resumable Talker request retains state across audio chunks. Its input
metadata is merged on update, so omitted boolean fields retain their previous
values. The Thinker-to-Talker bridge must explicitly publish `turn_end`,
`segment_end`, and `replace_streaming_prompt` for each native handoff, including
`False` on continuations. A previous turn's terminal flag must not turn later
nonterminal audio chunks into completed responses. Segment completion and
model-turn completion remain separate contracts; no WebSocket-side suppression
of extra `response.done` events substitutes for a correct stage boundary.

## Acceptance criteria

The cleanup is ready for review only when the exact candidate snapshot meets
the following checks:

1. A fresh-process import and capability probe leave the specified upstream
   classes and status enum unchanged. Missing methods or fields fail closed.
2. All three shared adapters route Stage0 admission and later append through
   the new contract. Positive tests demonstrate successful submission, not
   merely rejection of the retired route.
3. Initial acknowledgement waits for the complete prompt. Appends preserve
   the same physical request and replica until explicit teardown or supported
   MiniCPM rebuild.
4. Lost replies, commit-marker faults, changed fingerprints, evicted receipts,
   context exhaustion, pending cancel/close, and stale outputs preserve their
   stated isolation and idempotency guarantees.
5. PersonaPlex and Nemotron keep replay disabled and fail safely on model-state
   loss. MiniCPM replay/rollover tests retain their separate limits and checks.
6. Shared downstream, non-duplex, diffusion, and PD regression tests still pass.
7. Real multi-turn audio runs check continuity, audible output, session
   isolation, and complete answers across chunk boundaries. Receiving an EOS
   alone cannot certify answer completeness.

CPU contract passes do not substitute for model E2E. CUDA results do not sign
off H20/H100/NPU variants without matching runs. Performance comparisons need
pinned baselines, identical inputs/configuration, equivalent complete output,
and repeated measurements; this cleanup carries no measured speedup claim.
