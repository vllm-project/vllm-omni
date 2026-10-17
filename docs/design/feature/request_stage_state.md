# Request-scoped producer state

Custom incremental producers invoked by `OmniChunkTransferAdapter` can store
typed state in a namespace under the adapter's existing `request_payload`
container. This is a core implementation candidate for RFC #6453; model
migrations and model-specific segment resets are separate work.

```python
from vllm_omni.distributed.omni_connectors.transfer_adapter.request_state import (
    StageRequestIdentity,
)

identity = StageRequestIdentity(request.request_id, request.external_req_id)
state = transfer_manager.request_state.get_or_create(
    identity, namespace="producer.stage_edge", factory=ModelState
)
```

The accessor is available during a queued custom producer invocation. Both IDs
must come from the supplied request. A namespace returns the same state until
whole-request release. Factories construct model state without connector I/O;
they run outside the sender lock and may be called concurrently, so they must
not perform external side effects.

The runtime owns the request container. Producers may update their namespace
but must not remove the container. Existing processors can continue using raw
payload entries, including tensor-valued entries. A request cannot mix legacy
raw entries and namespaced state: access fails instead of overwriting model
data.

| Outcome | State lifetime |
| --- | --- |
| Producer buffers data and returns no payload | Retained |
| Nonterminal send succeeds | Retained |
| Resumable segment completes | Retained |
| Whole-request terminal send succeeds | Released by canonical cleanup |
| Processor raises, or connector fails/raises | Retained until failure finalization |
| Cancellation or timeout finalizes the request | Released by canonical cleanup |
| Duplicate cleanup | Harmless |
| External ID is reused after release | Fresh namespaces |

Generation validation uses the existing sender token and sender lock. A
retained accessor cannot recreate state after cancellation or release, even
when a new request reuses the external ID. Factories that overlap cleanup also
fail before committing their result. This does not make a retained model state
object immutable: callers must not mutate it after their invocation finishes.

Processor exceptions are recorded against the scheduler's internal request ID.
They do not produce a successful empty terminal payload. Existing scheduler
failure finalization releases receiver and sender state; this API adds no retry
policy, timeout mechanism, or process-global request map.

The CPU contract tests include a synthetic producer through the production
save queue, sender, connector outcomes, failure collection, and finalization.
They do not establish model correctness or a latency improvement.
