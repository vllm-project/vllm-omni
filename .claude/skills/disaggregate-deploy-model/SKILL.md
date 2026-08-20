---
name: disaggregate-deploy-model
description: Design, implement, review, and validate an async_chunk=false multi-stage boundary for a vLLM-Omni model. Use when adding or changing non-async stage topology, completed-stage-output handoff, full-payload transport, KV transfer, scheduler readiness coordination, or cross-stage request and output semantics.
---

# Disaggregate Model Deployment

Define each stage boundary as a semantic contract, prove that the downstream
stage can reconstruct its inputs, and then map the contract to vLLM-Omni.

Use model-addition skills for the full model integration, `vllm-omni-test` for
test and CI routing, and `review-pr` for the final PR review. This skill owns the
non-async disaggregation boundary and its transfer responsibilities.

## Scope

`async_chunk=false` selects synchronous stage processors. A multi-stage edge
may hand off completed stage outputs, a terminal full payload, KV, or a declared
combination.

`async_chunk=true` streams incremental payloads during producer execution and
requires ordering, completion, and resumable-segment handling. This skill
currently covers only `async_chunk=false`. Async-chunk disaggregation requires
a separate workflow.

## Invariants

1. Each stage has an independent lifecycle and a reconstructible input
   contract.
2. Every cross-stage value has one authoritative transfer owner.
3. Completed stage outputs may flow through the orchestrator when they form the
   complete downstream input.
4. Lightweight control uses orchestrator requests; bulk tensors use full
   payload; attention cache uses KV transfer.
5. Every stage constructs independently and loads only its required state.
6. Independently arriving control and data join by stable request identity
   before execution.
7. Generic infrastructure changes require a demonstrated framework gap.

## Disaggregation Boundary

Disaggregation partitions an execution graph into independently deployable
stages connected by boundary state `B`. A valid boundary provides:

- semantic closure: `downstream_output = D(B, downstream_local_state)`;
- temporal closure: every value in `B` is stable at its handoff condition;
- process closure: each stage constructs, schedules, scales, fails, and cleans
  up independently.

```text
ValidBoundary = SemanticClosure & TemporalClosure & ProcessClosure
UsefulBoundary = ValidBoundary & DeploymentBenefitMeetsAcceptanceCriterion
```

Strong boundaries combine high compute per transferred byte, stable state, few
synchronization points, and useful placement or scaling independence.

## Workflow

Record each phase result in the task, issue, PR notes, or evidence mechanism
already supplied for the work.

### Phase 0: Establish The Baseline

Freeze the repository, base commit, model, deploy config, inputs, seeds,
sampling parameters, parallelism, endpoint, and acceptance criteria. Capture
trusted outputs, request cardinality, and stage-local tensor shapes.

**Output:** a reproducible validation contract.

**Gate:** later evidence uses the same contract or records a new run.

### Phase 1: Recover The Execution Graph

Inspect model construction, `forward()`, generation loops, cache lifecycle,
stage processors, and output materialization. Enumerate multimodal encoding,
autoregressive decoding, codec generation, waveform decoding, diffusion, and
client-visible outputs.

Choose execution type from runtime behavior:

| Type | Runtime behavior |
|---|---|
| `LLM_AR` | Iterative token sampling with decode or KV state across scheduler steps |
| `LLM_GENERATION` | One admission executes a complete non-autoregressive forward |
| `DIFFUSION` | Diffusion runtime owns denoising and output materialization |

**Output:** a graph with candidate stages, execution types, independent weights,
entry and exit conditions, and deployment objectives.

**Gate:** every candidate has an independent lifecycle and a complete list of
downstream dependencies.

### Phase 2: Define Edge Contracts

Record one edge contract in the issue, RFC, or PR notes for each candidate
boundary.

Companion requests are role-specific requests expanded from one parent, such as
CFG branches, and collected as one readiness bundle before consumer execution.

| Boundary item | Decision and evidence |
|---|---|
| Producer and execution type | Stage and handoff event |
| Consumer and execution type | Stage and reconstructed input |
| Identity and cardinality | Stable request mapping, source-to-target count, companion relations |
| Reconstruction claim | Selected transfer planes plus consumer-local state |
| Deployment objective | Placement, scaling, capacity, memory, latency, or throughput goal |

Add one row for every actual cross-stage dependency. Group values only when
their owner, stability point, transfer plane, and consumer binding are the same.

| Dependency | Owner and stable handoff | Transfer plane | Consumer binding | Requirement and validation |
|---|---|---|---|---|
| Semantic value or group | Producer stage and event | Completed output, control, full payload, or KV | Target field or cache | Readiness role and comparator |

Close the lifecycle explicitly:

| Lifecycle concern | Decision and evidence |
|---|---|
| Readiness and join | Boolean condition over required dependencies and companion roles |
| Completion and output | Terminal condition and client-visible semantics |
| Cleanup | Producer, transport, consumer, and companion cleanup conditions |
| Failure | Timeout/retry behavior, abort propagation, bounded terminal outcome, and permitted fallback |

Add mechanism-specific details only for planes selected by the dependency
table:

- **Full payload:** required keys, producer and consumer steps, bindings,
  accumulation rule, terminal row count, send condition, and validation. Record
  shape, dtype, device, concat axis, or transforms when they affect correctness.
- **KV:** cache ownership, transfer criterion, required metadata, role-to-cache
  binding, receive readiness, and cleanup.
- **CFG or companion requests:** parent-to-role identity, cardinality, role
  order when significant, expansion/collection ownership, missing-role failure,
  and bundle cleanup.

Record unresolved facts as blockers rather than placeholders.

See [Qwen3-Omni Worked Example](references/qwen3-omni-worked-example.md)
for a completed Thinker-to-Talker contract using control and full payload.

Trace every token, hidden state, embedding, codec row, latent, mask, position,
KV block, speaker/language field, sampling field, companion role, and output
field from producer to first consumer. Validate every payload key against
multi-step and terminal producer outputs.

**Output:** one reviewed edge contract record per edge.

**Gate:** every downstream dependency has one owner, stable identity, explicit
cardinality, consumer binding, readiness rule, terminal behavior, and bounded
failure outcome. Unknown dependencies block implementation.

### Phase 3: Apply The Feasibility Gate

Read [Current Runtime Mapping](references/current-runtime-mapping.md) before
accepting a boundary. Classify unsupported adjacency, branching, merging, or
transfer behavior as a framework gap.

A feasible boundary satisfies these conditions:

1. The producer exposes every boundary value at the declared handoff.
2. The consumer reconstructs its inputs in another process.
3. The selected transfer planes cover the complete boundary state.
4. Consumer execution starts only after its readiness condition.
5. Request and companion identities remain stable across the edge.
6. Boundary cost satisfies the deployment objective and acceptance limit.

Classify unresolved items:

| Class | Action |
|---|---|
| Hard blocker | Stop implementation and move or reject the boundary |
| Evidence gap | Continue investigation and report feasibility as unresolved |
| Framework gap | Define the smallest generic extension and its known users |

Measure only metrics relevant to the declared objective. For transfer-sensitive
objectives, consider serialized bytes, request rate, effective bandwidth, fixed
transport latency, serialization/deserialization, stage compute, and peak
memory:

```text
T_transfer ~= T_serialize + T_transport_fixed
              + bytes / effective_bandwidth + T_deserialize
```

When performance is out of scope, mark it non-gating and make no performance
claim. When no acceptance limit is supplied, report measurements and leave the
performance verdict unresolved.

**Output:** `accept`, `reject`, or `investigate`, with evidence for each edge.

**Gate:** implementation begins only for accepted boundaries.

### Phase 4: Build The Implementation Mapping

Map the accepted semantic contract to current vLLM-Omni responsibilities:

| Responsibility | Implementation mapping to record |
|---|---|
| Topology and deployment | Pipeline config, deploy config, stage placement |
| Completed output and control | Orchestrator processor, request scaffold, fields |
| Full payload and KV | Producer hook, connector edge, KV config, metadata, consumer binding |
| Receive readiness | Coordinator scope and runner-to-scheduler feedback |
| Model stages | Stage selector and emitted or consumed artifacts |
| Visible outputs | Stage, type, processor, ordering, completion |
| Framework gaps | Exact generic symbol, missing capability, and known users |

List exact files, symbols, reasons, and verification gates. Read
[Current Runtime Mapping](references/current-runtime-mapping.md) before mapping
runner, coordinator, connector, KV, or diffusion details, and re-resolve its
search anchors from the current checkout.

**Output:** a bounded write set and test matrix.

**Gate:** each contract field maps once, and every generic edit maps to a proven
framework gap.

### Phase 5: Implement The Bounded Change

Use this order:

1. Add topology, deployment placement, and required registrations.
2. Make stages independently constructible and implement edge processors.
3. Implement mapped orchestrator, scheduler, runner, connector, or KV gaps.
4. Wire every client-visible output and completion rule.

Keep changes local to the accepted contracts and their verification.

**Output:** the smallest implementation that realizes the mapping.

**Gate:** static, configuration, and focused unit tests pass before full-model
execution.

### Phase 6: Verify The Outcome

Run the applicable verification matrix below with fresh artifacts. Runtime or
performance changes require full-model parity and A/B evidence.

**Output:** commands, commits, configs, result signatures, artifact paths, and
comparison tables.

**Gate:** correctness matches the baseline and measured regressions stay within
declared limits.

### Phase 7: Diagnose Failures

Find the first transition without direct evidence and return to the earliest
phase contradicted by the result:

| Invalidated result | Resume phase |
|---|---|
| Validation contract or baseline | Phase 0 |
| Execution graph | Phase 1 |
| Edge contract | Phase 2 |
| Feasibility assumption | Phase 3 |
| Implementation mapping | Phase 4 |
| Implementation behavior | Phase 5 |
| Verification method, comparator, or artifact provenance | Phase 6 |

Rerun every downstream gate from the selected phase.

**Output:** a localized failed contract or transition with direct evidence.

**Gate:** the next action targets that result.

## Transfer Responsibilities

| Plane | Typical state | Runtime owner |
|---|---|---|
| Completed stage output | finished tokens, multimodal outputs, generated codes | orchestrator and stage client |
| Control | request identity, sampling, lightweight scaffold, companion roles | orchestrator and stage client |
| Full payload | terminal hidden states, embeddings, codec rows, latents | model runner mixin and `OmniConnector` |
| KV | attention blocks and interpretation metadata | scheduler, runner, `OmniKVTransferManager` |
| Output | visible text, audio, image, video, completion ordering | output processor and orchestrator |

The edge contract assigns every authoritative field to one plane. Mixed edges
declare their join and readiness conditions.

## Transition Hooks

| Field | Semantic responsibility |
|---|---|
| `sync_process_input_func` | Select the synchronous processor; it may convert completed outputs or build a connector-backed scaffold |
| `custom_process_input_func` | Convert completed upstream outputs for an ordinary or diffusion transition |
| `custom_process_next_stage_input_func` | Build the terminal worker-side full payload |
| `prompt_expand_func` | Expand a producer request into declared companion roles |
| `cfg_kv_collect_func` | Collect companion KV in declared role order |
| `omni_kv_config` | Configure KV direction, metadata, and transfer criterion |

Required full-payload hooks return a self-contained payload with authoritative
rows, batch/request alignment, transfer dtype/device, and interpretation
metadata. Completed-output processors return the exact request objects expected
by the target stage.

## Runtime Semantics To Verify

### Completed Stage Output

```text
producer completes -> orchestrator collects source output
                   -> target stage processor converts it
                   -> target stage request is submitted
```

Use this path when the completed source output contains the entire boundary and
no worker-resident payload or KV is required.

### Full Payload

```text
producer steps -> per-request accumulation -> completion observed
               -> materialize/pop -> producer hook -> enqueue send task
               -> background connector put/retry

downstream scaffold -> receive gate -> connector get -> local TP fanout
                    -> scheduler feedback -> request becomes runnable
                    -> runtime buffer injection -> stage execution
```

A flush materializes and removes eligible accumulator entries, then attempts
enqueue. It reports neither enqueue nor transport success; the background send
path and declared failure policy own the remaining outcome.

### KV And Companion Roles

KV edges declare sender/receiver configuration, transfer criterion, metadata,
role-to-cache bindings, reconstruction, and cleanup. Companion requests declare
their cardinality, stable identities, collection order, and abort propagation.

### Visible Outputs

Declare every client-visible stage, output type, processor, ordering, and
completion condition. Multi-stage output must preserve task semantics such as
text followed by downstream audio.

## Verification Matrix

| Area | Required evidence |
|---|---|
| Topology/startup | Config merge, registry resolution, routing, execution type, connector and KV roles |
| Stage processors | Shapes, dtypes, device, row/token alignment, metadata, cardinality, terminal behavior |
| Model/runner | Independent construction, weight prefixes, forward routing, iteration/one-shot behavior, TP fanout |
| Orchestrator/scheduler | Submission count, identity, arrival orders, receive gate, cleanup, timeout, abort propagation |
| Full-model parity | Representative modalities, deterministic seeds, multiple requests, exact task comparator |
| Deployment | Per-stage devices, replicas and TP; pipeline-wide DP/PP; primary connector backend |
| Performance | Metrics required by the declared deployment objective |

Include failure injection for required payload, processor, connector, KV, and
abort paths. Every failure reaches a bounded terminal outcome.

## Failure Diagnosis

### Completed-Output Transition

```text
source output complete?
  -> expected fields materialized?
  -> target processor selected for async_chunk=false?
  -> processor returned expected cardinality and request type?
  -> target request submitted once?
  -> visible output routed and completed once?
```

### Independently Arriving Control And Payload

| Control scaffold | Payload receive | Expected state |
|---|---|---|
| absent | absent | no downstream request state |
| present | absent | request waits; receive registration is active |
| absent | complete | backend retention and request key satisfy the declared contract |
| present | complete | join succeeds; request becomes runnable |

Trace a stalled request through scaffold creation, receive registration,
producer enqueue, background send, connector get, TP fanout, scheduler feedback,
coordinator restore, and runtime injection. Instrument the first unverified
transition.

### KV And Companion Bundle

```text
request expansion and stable role IDs?
  -> per-role KV transfer criterion reached?
  -> sender routing and metadata correct?
  -> primary and companion receives complete?
  -> collector binds roles in declared order?
  -> completion and abort clean every role?
```

### Output Mismatch

```text
wrong row count        -> accumulation, transform, terminal count, send condition
wrong shape or dtype   -> producer value, consumer binding, transfer conversion
single-rank divergence -> data-rank I/O and TP fanout
companion mismatch     -> cardinality, role ordering, identity, KV keys
decode mismatch        -> token, KV, position, and cleanup ownership
duplicate output       -> submission count, visible-output routing, completion
```
