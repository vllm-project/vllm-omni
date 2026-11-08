---
name: find-simplifications
description: Find evidence-backed simplification candidates in vLLM-Omni and, when requested, turn them into focused proposals or code changes. Use for audits of dead, duplicated, speculative, over-generalized, unnecessarily defensive, or hand-rolled code. Use review-pr for ordinary correctness review and diffusion-perf-opt for performance-first optimization.
---

# Find vLLM-Omni Simplifications

Find a small number of well-proven ways to reduce code, state, APIs, or maintenance cost without weakening supported behavior. Treat this as an architecture and ownership audit, not a dead-code search or style cleanup.

The default deliverable is a read-only report. Do not add TODOs, design documents, issues, or code unless the user asks for those changes.

## Establish the Contract

Before judging a candidate:

1. Read the repository instructions supplied by the host.
2. Read [`docs/design/architecture_overview.md`](../../../docs/design/architecture_overview.md).
3. Select only the relevant active module and feature documents from [`docs/design/index.md`](../../../docs/design/index.md). Archived design pages are historical context, not current contracts.
4. Identify the comparison base with `git merge-base HEAD origin/main`; do not assume the current branch represents the intended baseline.
5. If tests or CI coverage may change, use [`vllm-omni-test`](../vllm-omni-test/SKILL.md) for test placement, markers, and commands.

Separate an implementation accident from an intentional boundary. In particular:

- vLLM-Omni extends and sometimes overrides upstream vLLM. A similar upstream implementation is evidence for reuse only after comparing API, lifecycle, and multi-stage semantics.
- Registries, import strings, model configuration, deployment YAML, plugins, and optional dependencies create dynamic consumers that a static reference count can miss.
- Hardware backends, model families, execution modes, and connector implementations may be intentionally parallel. Do not propose collapsing one only because it is inactive in the local environment.
- Stage, replica, request, cache, and transport state can look redundant while representing different owners. Prove that two fields or barriers encode the same fact before merging them.
- A shorter hot path is not a simplification if it adds synchronization, tensor copies, host materialization, recompilation, or a measurable quality/performance regression.

## Strong Candidates

A strong candidate removes a meaningful surface and has a clear replacement or deletion boundary. Examples include:

- A public method, configuration option, registry entry, route, message field, or event has no production consumer and no compatibility obligation.
- Tests or documentation are the only consumers of behavior that is not part of a supported contract.
- Multiple layers mirror the same request, lifecycle, cache-validity, readiness, or topology state.
- AR, diffusion, platform, or model-specific implementations duplicate a stable common operation and can share it without hiding performance-critical differences.
- An inherited upstream vLLM primitive is reimplemented locally without an Omni-specific semantic difference.
- Compatibility or fallback branches protect environments the project no longer supports.
- Serialization, IPC, output formatting, or connector code materializes the same payload more than once or converts between equivalent representations.
- A helper, abstraction, package split, or extension point has only one real implementation and adds more indirection than policy.
- Hand-written infrastructure can be replaced by an existing project dependency, Python standard-library feature, PyTorch primitive, or upstream vLLM facility with net deletion and equal observability.
- Tests maintain large mocks, snapshots, or expected data solely for an unused API or representation.

Do not elevate typo fixes, formatting, isolated renames, or vague complexity complaints into simplification candidates. Bundle small cleanups only when they support one substantive removal.

## Survey by Ownership Boundary

For broad audits, cover the largest or riskiest production deltas first. Useful domains are:

- `vllm_omni/engine/`: orchestration, stage pools, request state, output convergence, cancellation, and replica lifecycle.
- `vllm_omni/worker/` and AR model execution: upstream vLLM overlays, scheduler/cache state, collective RPC, adapters, and weight loading.
- `vllm_omni/diffusion/`: request/step execution, batching, output materialization, IPC, offload, parallelism, and model pipelines.
- `vllm_omni/entrypoints/`: duplicated validation/defaulting, offline/online parity, streaming, and route compatibility.
- Connectors and distributed code: ownership, copies, barriers, handles, cleanup, and failure propagation across process/device/node boundaries.
- Platform backends: genuine vendor differences versus copied control flow that has drifted.
- Configuration, registries, recipes, and examples: duplicate sources of truth and options that never reach a runtime consumer.
- Tests and CI: redundant fixtures, mocks of obsolete contracts, and expensive coverage that can move to a cheaper level without losing the guarded behavior.

When the user explicitly requests a repository-wide or many-candidate audit and parallel agent work is available and authorized, divide these domains among agents and require the same evidence format from each. Otherwise survey them sequentially.

## Prove or Reject Each Candidate

Use `rg` first, then read every relevant caller and implementation. Search exact Python symbols plus serialized names, route paths, configuration keys, registry strings, YAML values, and CLI spellings.

Classify evidence:

- **Production:** `vllm_omni/`, runtime configuration, registries, serving entrypoints, connectors, recipes used for supported deployments, and executable loader paths.
- **Validation/support:** `tests/`, benchmarks, examples, docs, CI, and developer tooling.
- **External/dynamic:** upstream vLLM contracts, third-party plugins, model repositories, optional backends, and string-loaded objects. State what was checked and what remains unknown.

For each candidate, answer:

1. What exact surface would disappear or merge?
2. Who owns the state or behavior today?
3. Which production consumers exist, including dynamic consumers?
4. What supported behavior, compatibility, performance, memory, or observability could change?
5. What is the smallest validation that would catch a bad simplification?
6. Is the result net deletion after tests, adapters, migration glue, and documentation are counted?

Reject or downgrade the candidate when a live consumer exists and removal is actually a feature decision; when an active design contract explains the separation; when the change only moves complexity; or when hardware/model evidence is unavailable for a performance-sensitive claim.

## Lifecycle and Data-Movement Audits

For asynchronous or distributed code, map the ownership graph before recommending deletion:

```text
request admission -> orchestrator -> stage/replica -> worker/device
  -> connector or IPC boundary -> output aggregation -> terminal cleanup
```

Associate every lock, event, sentinel, queue, readiness flag, timeout, abort path, and cache marker with an owner and transition. Coalesce mechanisms only when they represent the same state and have the same failure boundary. Preserve separate mechanisms when they protect publication versus rollback, local versus remote ownership, first-terminal-outcome arbitration, or cleanup after partial failure.

For tensors and media, also trace device, process, and ownership transitions. Count GPU-to-CPU copies, shared-memory/object-store handles, serialization, decode/encode steps, and unlink/release responsibility. Prefer eliminating a transfer or representation over merely hiding it behind a helper.

## Dependency Replacements

Prefer facilities already available in the repository before adding a dependency. For a proposed replacement:

- Name the exact local behavior it covers and any semantics that remain in glue code.
- Check maintenance, license, optional-dependency boundaries, platform support, and transitive footprint.
- Compare net deletion, test burden, import/startup cost, and runtime behavior.
- Benchmark when the replaced code is on a model, scheduler, connector, serialization, or serving hot path.

A wrapper that retains the same state machine or conversion logic is not a simplification.

## Deliver the Result

Report only candidates supported by concrete evidence. For each one include:

- **Surface:** files and symbols involved.
- **Evidence:** production consumers found, dynamic paths checked, and relevant design contract.
- **Simplification:** exactly what to remove, merge, inline, or reuse.
- **Benefit:** deleted API/state/code, reduced data movement, or reduced maintenance/test burden.
- **Risk and validation:** behavior that could change and the narrowest useful tests or measurements.
- **Confidence:** high or medium, with the unresolved evidence named for medium-confidence items.

Also list representative ideas that were rejected when that prevents repeated investigation. Prefer three high-confidence candidates over a long speculative inventory.

If the user asks for a durable design proposal, place it under `docs/design/feature/` and register it in `docs/design/index.md`; use current design-document conventions and keep implementation details proportional to the decision. If the user asks for implementation, make the smallest coherent change, update affected tests/docs, and preserve public compatibility unless the user explicitly accepts a breaking change.

## Folding Work from Another Branch

Diff the source branch against its own merge base with `origin/main`, not against the current feature branch. Port only independently justified candidates. Consolidate overlaps into the candidate with stronger evidence, and do not preserve a candidate count by carrying weaker duplicates.

## Validation

Match validation to the outgoing diff:

- Documentation or skill changes: run the repository skill validator when available, check links/paths, search for stale source-specific terminology, and run `git diff --check`.
- Python changes: run the narrowest relevant pytest coverage through the repository environment, then lint/format the changed files.
- Distributed, model, hardware, performance, or accuracy changes: run available CPU/static checks and state the exact GPU/NPU/model gap; never present simulated evidence as device validation.
- Before submission, use [`precheck-pr`](../precheck-pr/SKILL.md) for the branch-level self-review.

When summarizing, distinguish checks that passed from checks that were not runnable, and state the snapshot or branch against which the audit was performed.
