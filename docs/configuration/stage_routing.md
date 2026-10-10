# Static stage transitions

Model developers can set `PipelineConfig.stage_transitions` to an explicit tuple
of `(source_stage_id, target_stage_id)` pairs. This is part of the registered,
developer-owned pipeline topology. Deployment YAML continues to select hardware,
replicas and connector backends.

With `stage_transitions=None`, requests follow the existing numeric sequence
`0 -> 1 -> ...`. Explicit transitions replace that sequence completely; omitted
edges do not gain a sequential fallback.

For example, a pipeline with three registered stages can run `0 -> 2`:

```python
PipelineConfig(
    model_type="example",
    stages=(
        StagePipelineConfig(stage_id=0, model_stage="encoder"),
        StagePipelineConfig(stage_id=1, model_stage="unused"),
        StagePipelineConfig(
            stage_id=2,
            model_stage="decoder",
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
        ),
    ),
    stage_transitions=((0, 2),),
)
```

The explicit route starts at stage 0 and has at most one successor and one
predecessor per stage. Stage IDs must remain contiguous array indices, even when
the route skips stages or orders them differently. Duplicate edges, cycles,
self edges, unknown IDs and disconnected edges are rejected. The last stage of
the route must declare `final_output=True`. An empty tuple explicitly selects
stage 0 alone, so stage 0 must declare a final output in that case.

`input_sources` still describes data dependencies. Every input source of an
active stage must precede that stage on the route. Changing a transition cannot
make a processor accept an incompatible payload or omit a required dependency.

Configuration extensions use the actual route tail. For example,
`--forced-aligner` appends its pooling stage after that tail, uses it as the
audio input source and extends explicit transitions. A default `None` route
remains `None`. Typed and compatibility launch configurations use the same
extended pipeline.

Requests execute a prefix of the configured route ending at `final_stage_id`.
Requested output stages must occur in that prefix and declare final outputs;
they must include the endpoint. Streaming updates retain the admitted endpoint
and output stages. Public sampling arrays retain one slot per registered stage,
including inactive stages; IPC admission checks coverage through the highest
stage ID in the prefix.
Output modalities select the last matching stage in route order. For example,
under `0 -> 2 -> 1`, stage 1 can finish a request after stage 2. Stage sampling
parameters remain indexed by stage ID.
Worker connector endpoints remain fixed for the full route. A request ending
early can leave unread outgoing payloads, which follow the backend cleanup
contract below; this feature does not prune all data-plane sends per request.
An explicitly requested registered output modality is rejected if all stages
that produce it are inactive. Omitting modalities selects from active outputs.

Async-chunk prewarming follows the same prefix and binds each upstream sender
through its StagePool before advertising its address to a receiver. The later
bridge submission reuses that binding. A lost bound replica fails its requests
instead of moving their senders after the receivers have been configured.
Connector payload send/receive endpoints use the projected transition,
including on headless workers. Backend projection respects `default_connector`;
request keys and chunk counters retain their existing per-request semantics.
Stages fed by an orchestrator bridge receive their real payload before dispatch.
Control edges and backend edges serve different purposes: a bridge-fed stage can
have an outgoing connector without an incoming one. A middle stage with both
connector directions requires the same backend on both edges, because one
worker owns one connector. Absent payload backends retain the existing local
IPC behavior; choosing a backend suitable for the deployment remains necessary.

Omitted stages receive no request submissions or prewarming. They are still
initialized, so this feature does not reduce startup memory or process counts.
Normal and empty terminal outputs share the same completion condition: all
selected outputs have finished, and non-streaming async requests also wait for
every submitted stage. The endpoint's terminal output is retained until then,
so delayed upstream usage and stop reasons arrive before completion.
Cleanup closes admission before waiting for aborts. For async-chunk transport
and native MRv2 payloads, it also waits for each backend's cleanup contract
across all live replicas before terminal success or successful abort acknowledgement.
Release markers use the existing transport queue after
pending puts; native workers also apply their existing cancellation fence.
V1 uses the scheduler's chunk adapter as its transport owner, while native MRv2
uses the worker data plane.
Legacy V1 full-payload and native KV transfer use their existing cleanup paths.
SHM implements per-prefix release. A backend exposing `wait_for_cleanup` also
confirms its release before the operation succeeds. Other backends retain their
existing per-payload lifecycle after the same queued publication fence; their
acknowledgement does not promise physical release. NIXL ownership changes are
reviewed separately from configured routing.
Native stages without a payload connector have no transport owner to release.

Concurrent cleanup calls share the same operation; cancelling a duplicate waiter
does not cancel its owner. A parent joins a companion's running cleanup without
replacing its operation. CFG relationships remain available until cleanup
succeeds, so retrying a failed parent also releases its companions. Replica waits
cannot submit a closed request. A failed
release reports a request-scoped error and retains closed request state for a
cleanup retry. That request ID remains unavailable for reuse until cleanup
succeeds. Chunk cancellation fences and per-request chunk counters remain owned
by the existing transport.

This first implementation supports static routes for turn-based requests.
Branches, joins, loops and token-dependent decisions need a separate execution
contract. Non-sequential duplex deployments are rejected before plugin loading;
their model-specific session contracts need a separate extension. Existing
native AR-to-DiT KV deployment restrictions still apply.
Streaming-input requests with a native MRv2 downstream receiver remain unsupported.
