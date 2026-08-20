---
name: review-pr
description: Review pull requests and local branches for vllm-project/vllm-omni with a frozen snapshot, module-design ownership, feature-design overlays, targeted validation, and concise evidence-backed findings. Use for default, detailed, or repeat maintainer reviews; checking correctness, compatibility, tests, benchmarks, model additions, distributed changes, or breaking behavior; and identifying or explicitly requesting the most relevant code-owner reviewers. Use precheck-pr instead for an author's pre-submit self-check.
---

# Review vLLM-Omni Pull Requests

Review like a maintainer: direct, selective, and focused on issues that CI does
not prove. Prefer a few high-confidence findings over exhaustive commentary.
Zero findings is a valid result.

## Quality contract

Make every finding:

- **Correct:** prove a reachable failure, not a suspicion.
- **Prioritized:** lead with merge blockers and high-impact defects.
- **Actionable:** identify the smallest safe fix direction.
- **Evidence-based:** cite code, tests, docs, CI, or measurements.
- **Concise:** avoid review templates and repeated summaries.
- **Calibrated:** match severity to user and maintainer impact.

Do not report unrelated backlog, style already enforced by pre-commit, or a
missing test that would not protect changed behavior. Do report new allowlist
or budget entries in `check_forbidden_imports.py` / `check_torch_cuda.py` /
`check_tts_adapter.py` / `check_buildkite.py` unless the PR justifies them:
those are policy changes, not lint noise.

## Select the input and depth

Use `vllm-project/vllm-omni` as the base repository. Accept its forks and local
checkouts; use another skill for unrelated repositories.

| Input | Review surface |
| --- | --- |
| PR number or URL | Frozen PR metadata, full diff, and relevant threads. |
| Local branch/worktree | Frozen target-base SHA through committed, staged, unstaged, and in-scope untracked changes. |
| Pre-filled context | Reuse supplied metadata; fetch only missing facts and the full diff. |

Default to maintainer brevity. A detailed or audit request expands coverage and
lists `path:line` findings, but keeps the same confidence and severity bar.

## Reference guide

Load references in review order: process, one primary module contract, matching
feature designs, evidence checks, then delivery. Every file is linked directly
below; do not load unrelated module references.

Each concise reference links to the maintained
[vLLM-Omni documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/).
For branch-specific behavior, inspect the matching `docs/` file in the reviewed
checkout first; use the published latest docs for current guidance and discovery.
If docs and live code disagree, verify the code/tests and report the drift.

### Review process

| Reference | Read when |
| --- | --- |
| [review-execution.md](references/process/review-execution.md) | Every review; freeze inputs, inspect safely, and deliver against the same snapshot. |
| [general-checks.md](references/process/general-checks.md) | Every review; apply repository-wide correctness and evidence rules. |
| [design-contracts.md](references/process/design-contracts.md) | Every production review; resolve branch-local module and feature design status. |
| [review-routing.md](references/process/review-routing.md) | After the diff census; select one primary module and conditional overlays. |

### Primary module contract

| Reference | Read when |
| --- | --- |
| [entrypoints.md](references/modules/entrypoints.md) | Offline/CLI/API ingress, validation, rendering, streaming, or sessions change. |
| [configuration.md](references/modules/configuration.md) | Config construction, deploy/stage schema, defaults, registry, or topology changes. |
| [input-output-modality.md](references/modules/input-output-modality.md) | Requests, messages, serialization, output types, accumulation, or completion change. |
| [error-contracts.md](references/modules/error-contracts.md) | Error classification, fatality, propagation, sanitization, or rendering changes. |
| [engine-orchestration.md](references/modules/engine-orchestration.md) | Cross-stage routing, request state, output ordering, RPC correlation, or terminal convergence changes. |
| [stage-runtime.md](references/modules/stage-runtime.md) | Placement, startup, readiness, replica identity, affinity, membership, or shutdown changes. |
| [omni-connector.md](references/modules/omni-connector.md) | Cross-stage/process/device/node transport or synchronization changes. |
| [model-integration.md](references/modules/model-integration.md) | Registration, preprocessing, loading, runners, or model-specific execution changes. |
| [ar-runtime.md](references/modules/ar-runtime.md) | AR scheduling, request/cache state, adapters, workers, or upstream vLLM semantics change. |
| [diffusion.md](references/modules/diffusion.md) | Diffusion runtime, models, batching, parallelism, or offload changes. |
| [execution-platforms.md](references/modules/execution-platforms.md) | Hardware selection, capabilities, vendor workers, kernels, or patches change. |
| [cache-management.md](references/modules/cache-management.md) | Cache identity, reuse, validity, reset, eviction, or teardown changes. |
| [quantization.md](references/modules/quantization.md) | Quantization selection, checkpoint metadata, layer mapping, precision, or constraints change. |
| [observability.md](references/modules/observability.md) | Metrics, logs, units, labels, correlation, or lifecycle changes. |
| [profiling.md](references/modules/profiling.md) | Profiling instrumentation, traces, start/stop lifecycle, or overhead changes. |
| [benchmarking.md](references/modules/benchmarking.md) | Benchmark workload, metric calculation, CLI, or result metadata changes. |

### Feature-design overlays

| Reference | Read when |
| --- | --- |
| [runtime-stage-execution.md](references/features/runtime-stage-execution.md) | Disaggregated inference, async chunk/output/materialization, or prefix caching changes. |
| [communication.md](references/features/communication.md) | A concrete OmniConnector backend or its deployment contract changes. |
| [diffusion-acceleration.md](references/features/diffusion-acceleration.md) | Diffusion parallelism, attention, quantization, cache, batching, or offload changes. |
| [infrastructure-performance.md](references/features/infrastructure-performance.md) | Metrics infrastructure or documented speech optimization stacks change. |

### Evidence and quality checks

| Reference | Read when |
| --- | --- |
| [model-addition-checklist.md](references/checks/model-addition-checklist.md) | A model, architecture, loader, processor, registry, pipeline config, or deploy config is added. |
| [perf-verification.md](references/checks/perf-verification.md) | The PR makes a latency, throughput, memory, or quality claim. |
| [test-quality-evaluation.md](references/checks/test-quality-evaluation.md) | Tests change, are absent for risky code, or may not exercise production behavior. |
| [tests-docs-checklist.md](references/checks/tests-docs-checklist.md) | Coverage, CI markers, examples, user docs, or PR evidence need review. |
| [verification.md](references/checks/verification.md) | Hardware, a server, or a runnable affected path is available for active verification. |
| [examples-policy.md](../precheck-pr/references/examples-policy.md) | The PR adds, copies, or renames Python under `examples/`; apply the canonical policy shared with `precheck-pr`. |

### Delivery and reviewer coordination

| Reference | Read when |
| --- | --- |
| [maintainer-style-study.md](references/delivery/maintainer-style-study.md) | Findings are ready for concise maintainer-style delivery. |
| [review-requests.md](references/delivery/review-requests.md) | The user asks to identify, suggest, request, or ping code-owner reviewers. |

## Workflow

### 1. Freeze and report the snapshot

Pin the base and head before reading source or running validation. Within 60 seconds,
report the pinned head, CI, mergeability, and preliminary findings in the host
conversation. Do not wait for CI or post this update to GitHub.

If the target changes while fetching, discard the evidence and retry once. If
it changes again, report the churn and wait for a stable target.

For a trusted PR head, materialize the pinned head in an isolated detached
worktree. A worktree freezes identity but is not a security sandbox. Treat fork
heads as untrusted unless the user and environment policy explicitly establish
otherwise: execute them only in a disposable, secret-free sandbox with restricted
filesystem, network, and resources; without one, use static SHA-addressed reads
and CI evidence only. For a local review, freeze the committed, index, worktree,
and NUL-safe in-scope untracked contents. Follow
[review-execution.md](references/process/review-execution.md) for trust gates,
state fingerprints, and byte-for-byte staleness checks.

### 2. Build the diff census

Group files into production code, tests, docs, configuration, build/CI, and
generated artifacts. Map each changed production file and test group to the PR
goal. Compare the title/body claims with the actual diff; use linked issues only
when they define the contract or reproduction.

Mark unrelated scope and unexplained generated artifacts. Do not infer behavior
from the PR description without tracing the live code.

### 3. Route from the live behavior

Trace each claimed behavior through the changed producer to its live consumer,
then use [design-contracts.md](references/process/design-contracts.md) and
[review-routing.md](references/process/review-routing.md) to select one primary
module contract, a second only for a real documented cross-boundary call path,
and every matching feature-design and evidence overlay. Treat titles and paths
as hints; live behavior and the frozen head's current design metadata are
authoritative. For docs-, tests-, or CI-only changes, route to the production
contract they protect or use only the applicable evidence checks.

### 4. Run the blocker scan

Apply every category in [general-checks.md](references/process/general-checks.md) before
lower-priority comments.

If the diff census contains an added, copied, or renamed Python path under
`examples/`, read and apply the canonical
[examples policy](../precheck-pr/references/examples-policy.md). Treat a new
model-specific Python example as blocking. Do not flag model-specific example
debt that the PR only modifies or removes, and do not run the rest of the
author-oriented `precheck-pr` workflow.

For each changed value or behavior, trace:

```text
public ingress -> validation/defaulting -> producer -> transformations
  -> stage/worker/connector boundary -> final consumer -> terminal cleanup
```

Cover every applicable offline/online, streaming/non-streaming, sync/async,
feature-on/off, topology, and compatibility path. Search bounded callers and
sibling implementations rather than assuming the changed hunk is the only path.

### 5. Apply module and feature contracts

Apply the reference set selected in step 3 and any matching repo-local skill.
Read the exact module and feature pages in the frozen head, including status,
ownership boundary, dependencies, candidate invariants, safe-change guide, and
promotion gate. Candidate or draft rules are questions, not blockers, unless
current code, tests, or policy enforce them. Inspect both sides of any config,
registry, serialization, connector, cache, or stage boundary.

When a diff adds or expands a helper, class, fallback, compatibility branch, or
public behavior, run a subtraction pass: remove out-of-scope behavior and check
whether each new abstraction can be deleted, merged, moved, or inlined.

### 6. Verify the changed path

Before each validation group, verify the frozen SHA plus the tracked, index,
untracked, and ignored-file fingerprint, or recreate a pristine snapshot. On a
trusted head or inside the required sandbox, run an import/version preflight,
then the narrowest relevant tests and low-cost static checks. Bind every result
to the head SHA, snapshot fingerprint, and environment fingerprint. Never run
imports, tests, builds, hooks, or repo-configurable tooling from an untrusted
head on the reviewer host.

- Treat CI as status evidence; inspect only the first overlapping failure.
- For docs-only changes, use diff hygiene, links/build checks, and bounded live
  contract verification instead of dependency setup or pytest.
- For hardware-dependent paths, run available static/CPU checks and name the
  exact GPU/NPU gap. Never simulate device evidence.
- For performance or accuracy claims, require comparable base/head runs with
  the same environment, workload, warmup, repetitions, and quality criteria.

Stop when each changed semantic path has a supported finding or an explicit
no-issue conclusion. Do not search further only to increase confidence.

### 7. Consolidate and deliver

Verify each finding against the current diff, deduplicate by root cause, and
order by severity.

Re-read the remote head and reverify or recreate the pristine validation
snapshot immediately before delivery. If either changed, mark the review stale
and restart from the new snapshot.

Return findings first. Use
[maintainer-style-study.md](references/delivery/maintainer-style-study.md) to keep them
direct and brief. Each finding must include an exact `path:line`, trigger or
call path, current behavior, impact, and smallest fix direction. If there are
no findings, say so briefly and name material validation gaps.

Keep the review read-only unless the user explicitly authorizes posting. Do not
submit `APPROVE`, `COMMENT`, or `REQUEST_CHANGES`, add labels, edit code, or push
commits as an implied part of review.

### 8. Optionally request focused owner reviews

Only when the user asks to identify or request reviewers, read
[review-requests.md](references/delivery/review-requests.md). Rank path-matched
CODEOWNERS with the frozen module page's owners or required reviewers and
documented governance expertise; propose one to three focused reviewers with an
explicit contract rationale.

Identifying or suggesting reviewers is read-only. Requesting reviewers or
posting `@mention` comments changes external state and requires explicit user
authorization. When authorized, recheck the head, deduplicate existing
requests, and post at most one consolidated comment. Do not infer this
permission from a request to review the code.
