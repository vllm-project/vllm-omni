# Runner-Owned Named Causal KV Branches

> **Status:** experimental model-runner capability. The current implementation
> supports one fixed-capacity causal branch on a CUDA AR runner. It is not a
> general scheduler-visible KV-cache extension.

## Motivation

Some autoregressive models need an additional causal language-model trajectory
for every parent request. VibeVoice classifier-free guidance is the first user:
the normal Qwen path produces the positive condition while a second Qwen path
produces the negative condition. The two paths share weights but must never
share attention KV.

Keeping that second KV in model-owned tensors would duplicate scheduler and
PagedAttention responsibilities. Reusing positive KV would cross conditioning
histories. A **named causal KV branch** therefore gives the model a narrow
append/reset/free interface while leaving allocation, attention metadata, and
KV-cache binding in the runner.

## Terminology

- **Parent request**: the scheduler request whose positive KV uses the normal
  vLLM cache.
- **Named KV branch**: a runner-owned PagedAttention store selected by a stable
  name such as `negative`.
- **Branch request state**: block IDs, sequence length, and block table for one
  parent request inside one named branch.
- **Model runtime state**: non-KV request data such as diffusion conditions,
  convolution caches, pending waveform copies, and feedback embeddings.
- **Executing request**: a request with a positive scheduled-token count in the
  current runner step. A finished request can still be executing that final
  step.

“Finished”, “scheduled”, and “executing” are not synonyms. Cleanup decisions
must use the actual executing set rather than assuming every finished request
is already idle.

## Interface

A model opts in declaratively:

```python
self.named_kv_branch_request = NamedKVBranchRequest(
    name="negative",
    memory_bytes=8 * 1024**3,
    activation_margin_bytes=512 * 1024**2,
)
```

After the normal KV cache is initialized, the runner:

1. validates the declaration;
2. constructs `NamedCausalKVBranch` from the selected attention layer group;
3. calls `model.bind_named_kv_branch(branch)`;
4. closes the new branch if binding fails;
5. publishes it in `runner.named_kv_branches` only after a successful bind.

Models without a declaration do not allocate a branch and do not enter this
path.

The branch interface is deliberately small:

```python
branch.reset(request_id)
branch.append_and_enter(request_id)
branch.append_and_enter_batch(request_ids)
branch.free(request_id)
branch.close()
```

The context managers append exactly one token per request and temporarily bind
the named caches to the shared attention layers. They restore the positive
cache bindings in `finally`, including when model execution fails.

## Ownership

| Resource | Owner | Lifetime |
| --- | --- | --- |
| Positive attention KV | normal vLLM runner | scheduler request |
| Named branch allocation and block tables | Omni model runner | branch/process |
| Per-request named KV blocks | named branch | parent request |
| Conditions, convolution caches, feedback embeddings | model state machine | parent request |
| Waveform after publication | output processor | output request |
| Pending waveform D2H event/buffer | model state machine | until drain or synchronized cleanup |

A model may retain the branch object after binding, but it must not retain an
entered forward context or replace branch-owned block tables/caches. The branch
may retain references to shared attention layers because `close()` runs before
upstream model teardown.

## Capacity and startup guards

The v1 implementation uses a fixed, non-overcommitted pool. Startup requires
capacity for every configured concurrent request at full model length:

```text
required branch tokens = max_num_seqs × max_model_len
```

The positive KV pool is checked against the same fixed-concurrency contract.
The branch additionally reserves the configured activation margin before
allocating its tensors.

Current constraints are explicit:

- CUDA runner;
- homogeneous `FullAttentionSpec` layer group;
- scheduler and kernel block sizes match;
- no quantized named KV;
- pipeline/context parallel size 1;
- no ubatching, speculative decode, sleep mode, prefix caching, or KV
  connector on the branch path;
- positive forward may use CUDA Graphs, but named-branch work stays eager and
  outside the captured model forward.

Pool exhaustion after successful startup is an invariant violation, not an
LRU-eviction signal.

## Request lifecycle

### Start or segment reset

The model calls `reset(request_id)` before the first branch append for an audio
segment. Reset frees any old branch state and creates an empty block table. A
later segment reset reuses the parent identity but not its previous negative
history.

### Step execution

`append_and_enter_batch()` validates the complete logical batch before
bookkeeping. It allocates one position per request, constructs one varlen
attention context, swaps every selected layer to the named cache, and restores
the positive cache after the shared forward.

Batch failure drops every touched branch request. A partially written causal
history cannot be rolled back safely and must never be reused.

### Finish notification

Models can implement:

```python
def on_requests_finished(
    finished_req_ids,
    *,
    scheduled_req_ids=(),
): ...
```

The AR runner computes `scheduled_req_ids` from strictly positive scheduled
counts. For compatibility, legacy one-argument hooks continue to receive only
`finished_req_ids`; dispatch is selected from the callback signature, not by
calling and catching `TypeError`.

The schedule-aware rule is:

```text
finished - scheduled  -> clean immediately
finished ∩ scheduled  -> defer until final postprocess
```

A zero-token early return supplies an empty executing set, so all finished
requests are immediately eligible for cleanup.

### Abort and failed output

Cleanup owns pending asynchronous copies. Before dropping pinned D2H buffers,
the model synchronizes their recorded events. This is an abort-path safety
operation, not a normal per-token synchronization.

Branch `free()` is idempotent after branch close. Internal forward faults use a
best-effort unchecked release so cleanup remains legal while the attention
context is entered and never masks the original exception.

### Shutdown

Shutdown order is contractual:

1. call `model.clear_runtime_state()`;
2. close every named branch;
3. clear the runner branch registry;
4. delegate to upstream runner shutdown.

An exception in model or branch cleanup is logged and does not prevent the
remaining teardown steps. Clearing model state first allows it to free
request-local branch entries while the branch is still open.

## Request identity

Runner-owned identity is carried in the reserved `_omni_req_id` metadata key.
Only models declaring `requires_omni_request_id` receive it. User-provided
`request_id` metadata cannot replace this key. The legacy `request_id` key is
still populated for models that have not migrated.

Branch and model state are keyed by the parent scheduler request ID. A named
branch name identifies a logical cache store, not a request namespace.

## Failure semantics

- Invalid declarations and insufficient capacity fail startup.
- Binding failure closes the unpublished branch.
- Metadata/bookkeeping failure frees every touched logical request.
- Forward failure invalidates every request in that logical branch batch.
- Freeing an unknown request is a no-op; freeing an unallocated block is an
  invariant error.
- Closing while a branch context is entered is rejected.
- Cleanup exceptions are logged during shutdown, and upstream teardown still
  runs.

## Non-goals

This capability does not provide:

- scheduler-visible cache affinity or preemption;
- cross-stage or cross-worker KV transfer;
- prefix sharing between positive and named branches;
- multi-tenant overcommit or eviction;
- arbitrary model-owned attention-cache mutation;
- a stable interface for AR-Diffusion sessions.

## Relationship to AR-Diffusion KV

[AR-Diffusion pipeline capability](ar_diffusion_pipeline_capability.md) owns
longer-lived diffusion sessions, logical branches, cross-attention KV, and LRU
session policy inside a dedicated experimental runner. Named causal KV instead
mirrors one parent AR request in the normal model runner and has fixed
scheduler-bounded capacity. They share ownership vocabulary but are not
interchangeable interfaces.

## Verification

The checked-in contract tests cover:

- capacity and unsupported-runner validation;
- fixed allocator accounting;
- single and batched append metadata;
- positive-cache restoration;
- whole-batch fault cleanup;
- original-exception preservation;
- transactional bind failure;
- legacy and schedule-aware finish hooks;
- zero-token cleanup notification;
- model-clear/branch-close/upstream-shutdown ordering;
- continued shutdown after cleanup exceptions;
- VibeVoice finish, abort, pending D2H, and request-isolation behavior.

Real GPU acceptance must additionally show positive/negative KV conformance,
request block counts returning to zero, and stable VRAM at the configured
concurrency.
