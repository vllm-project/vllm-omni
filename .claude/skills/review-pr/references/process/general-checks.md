# General Review Checks

Apply these checks to every PR after freezing the review surface. A pattern is
not a finding until the diff introduces or exposes a reachable trigger, impact,
and smallest safe fix. Do not report unrelated backlog.

Official docs: [contributing guide](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/)
[feature compatibility](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/feature_compatibility/),
and [design documents](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/).

## Contract and scope

- Match the title/body claims to every changed production path, test, config,
  dependency, generated artifact, and user-visible document.
- Trace each changed value through public ingress, validation/defaulting,
  producer, transformations, process or device boundaries, final consumer, and
  terminal cleanup.
- Check every applicable sync/async, offline/online, streaming/non-streaming,
  feature-off, topology, and compatibility path without demanding unsupported
  modes merely for symmetry.

## Module, API, and class design

Apply these rules to new or materially changed production code; do not require
unrelated cleanup from untouched files.

1. Start each source module with a concise module docstring that defines its
   responsibility and boundaries and identifies its primary classes and key
   functions. Place it after any required license header and before imports.
2. Give every added or materially changed class and function a clear docstring
   covering its responsibility, parameters, return value, raised exceptions,
   and non-obvious invariants. Use explicit typed parameters at owned API
   boundaries; avoid opaque `**kwargs` and `Any`. Allow them only when an
   upstream or compatibility interface requires them, and document and validate
   the accepted keys and concrete types.
3. Move stable helpers shared by multiple features into an appropriately scoped
   utility module. Do not let feature modules accumulate large collections of
   unrelated private utility functions; keep a private helper local only when
   it is single-purpose and tightly coupled to that feature.
4. Keep every method on a key class within that class's stated responsibility,
   and document non-obvious design decisions and invariants. Before adding a
   method, prefer reuse, consolidation, or extension of an existing abstraction;
   split the class when new behavior creates a separate responsibility or makes
   the class unnecessarily long.
5. Require changes to user-facing behavior to update `docs/`. When an existing
   active module or feature design page owns the changed contract, update that
   page. Require a new design page only when the reviewed branch's repository
   policy, an accepted RFC/issue, or explicit maintainer acceptance criteria do
   so; otherwise suggest it as non-blocking. For a major architectural change,
   apply the branch's contributing policy for a design RFC or issue; the current
   policy uses more than 500 changed production-code lines, excluding kernel,
   data, config, and test changes. Resolve page status and authority through
   [design-contracts.md](design-contracts.md), and do not demand a duplicate page
   when an existing design owns the exact contract. When required, cover scope,
   data flow, public contracts, failure behavior, alternatives, validation,
   limitations, and rollout.
6. Require every performance PR to provide reproducible base-versus-head A/B
   results under the same hardware, software, model, workload, precision,
   topology, warmup, repetitions, and measurement scope. Include exact commands,
   variability, and a correctness or quality guard; one-sided or non-comparable
   numbers are incomplete evidence. For a new model absent from the frozen base,
   compare head with its pinned canonical reference implementation under rule 12
   instead. Follow
   [perf-verification.md](../checks/perf-verification.md) for the full protocol.
7. Require every addition, removal, default change, or semantic change to a
   user-facing inference API or CLI to explain why it is necessary, which
   alternatives were considered, and why the chosen interface is preferable.
   Record compatibility, deprecation or migration behavior, and the rationale
   in the PR. Update the corresponding module and feature design documents when
   they define the changed interface or workflow, or when policy requires one.
8. Require every bug-fix PR to add an automated regression test that reproduces
   the defect on the frozen base and passes on the fixed head. Exercise the
   production path or nearest stable boundary and assert the corrected behavior,
   not only process survival. Record the exact base/head command and result, and
   place hardware- or model-specific coverage in the correct CI lane. A manual
   reproduction supplements but does not replace the regression test.
9. Require every new or materially changed recipe to follow
   [`recipes/TEMPLATE.md`](../../../../../recipes/TEMPLATE.md) and the naming and
   layout conventions in
   [`recipes/README.md`](../../../../../recipes/README.md).
   Complete every applicable summary, use-case, reference, hardware,
   environment, exact-command, verification, and notes section with tested,
   reproducible details; explain any intentional deviation in the PR.
10. Require every new model to verify its supported modalities, platforms,
    execution modes, and features with production-path evidence. Update
    `docs/models/supported_models.md` and every applicable feature support or
    compatibility table; do not claim support that the PR does not validate.
    Follow
    [model-addition-checklist.md](../checks/model-addition-checklist.md) for the
    model-specific documentation checks.
11. Require every new model PR to add or update its in-repository model-family
    recipe under `recipes/` and its `recipes/README.md` index entry. Apply rule
    9, and keep the recipe's model identifiers, tasks, serving modes, hardware,
    commands, supported features, and limitations aligned with validated PR
    evidence. An external recipe link does not replace the in-repository recipe.
12. Require every new model to compare accuracy or output quality and performance
    against a pinned baseline or canonical reference implementation. Match the
    checkpoint, inputs, precision, hardware, workload, and measurement method;
    provide exact commands, results, and explanations for material differences.
    Prefer a detailed timing split for each stage and explain when it is
    unavailable. Follow
    [model-addition-checklist.md](../checks/model-addition-checklist.md) and
    [perf-verification.md](../checks/perf-verification.md) for the evidence
    contract.

## Blocking risk scan

| Risk | Prove before reporting |
| --- | --- |
| Correctness | A live input reaches a wrong output, exception, silent drop, shape/dtype/device mismatch, or partially initialized state. |
| Compatibility | A supported API, config key, default, serialization schema, model identifier, or caller breaks without validation, migration, or deprecation. |
| Lifecycle | Allocation or startup lacks cleanup on partial failure, timeout, cancellation, shutdown, or repeated requests. |
| Concurrency/distribution | Ordering, identity, rank, world size, completion, timeout, backpressure, or collective assumptions fail on a supported path. |
| Cache/shared state | Identity omits a correctness-affecting input, or isolation, invalidation, disabled behavior, eviction, or cleanup is incomplete. |
| Async behavior | Blocking I/O, sleep, a lock across `await`, or unnecessary serialization harms a reachable async path. |
| Security/data handling | Secrets, user payloads, unsafe deserialization/eval/shell, unvalidated paths, or unbounded metric labels are reachable. |
| Validation evidence | Tests bypass the production dispatcher, mock away the change, use unrealistic types/MRO, or cannot fail when the defect returns. |
| User contract | User-visible model, feature, API, CLI, config, default, or compatibility behavior changes without accurate docs or examples. |

Search bounded callers and sibling implementations before reporting dead code,
duplication, or a missed consumer. Keep new abstractions only when they have a
distinct live caller, invariant, or compatibility need.

## Finding bar

Anchor each finding to a changed `path:line`; name the trigger or call path,
current behavior, impact, and smallest fix direction. Treat pending CI, missing
hardware, or unsupported measurements as validation gaps unless the repository
contract makes that evidence a merge requirement.
