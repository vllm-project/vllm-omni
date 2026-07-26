---
name: run-minicpmo-ascend-perf-cycle
description: Run the repeatable daily workflow for MiniCPM-o 4.5 Ascend performance candidates in this vLLM-Omni repository. Use when starting, implementing, measuring, accepting, rejecting, documenting, committing, or pushing one performance experiment; comparing a candidate with a baseline; preparing an atomic perf commit; or resuming the next item in the Ascend competition TODO.
---

# Run a MiniCPM-o Ascend Performance Cycle

Move one optimization hypothesis from evidence to a reproducible decision and,
only when accepted, an atomic performance commit. Keep correctness, experiment
records, and Git history strong enough to compare or roll back every candidate.

## Required Context

For competition work, read the following before acting:

1. `../optimize-minicpmo-ascend/SKILL.md` for optimization and acceptance rules.
2. `../optimize-minicpmo-ascend/references/competition-rules.md` when official
   rules, metrics, datasets, or submission requirements affect the experiment.
3. `../optimize-minicpmo-ascend/references/repo-map.md` before changing an
   unfamiliar pipeline stage.
4. [references/candidate-record.md](references/candidate-record.md) before
   recording, committing, or pushing a candidate.

Prefer the repository tools in `benchmarks/competition/minicpmo_ascend/` for
environment capture, smoke tests, benchmarks, resource collection, and gates.
Treat `UNRESOLVED` official fields as blockers for formal claims, not as values
to infer.

## Daily Cycle

### 1. Protect the Starting State

- Inspect `git status`, current branch, HEAD, remotes, and recent commits.
- Assume existing changes belong to the user. Never discard, overwrite, or
  include unrelated paths in the candidate commit.
- Verify that the target remote contains the current HEAD before building new
  work on top of it.
- Create a candidate ID in the form `YYYYMMDD-short-hypothesis`.
- Record the baseline SHA, deploy config, model revision, and artifact root.
- Recheck official material when the experiment depends on competition rules.

If unrelated dirty changes overlap the candidate files, understand and work
with them. Stop only when a safe, reviewable candidate cannot be isolated.

### 2. Define the Candidate Before Editing

Write down:

- Evidence for the bottleneck.
- One primary variable to change.
- Target workload and metric expected to improve.
- Correctness, effect, memory, latency, and stability guardrails.
- Minimum gain needed to exceed baseline variance.
- Rollback condition.

Do not combine independent optimizations in one candidate. Compatibility fixes
needed to make the benchmark valid belong in separate `fix` commits.

### 3. Capture a Same-Session Baseline

Use fixed model revision, sampling, seed, prompt/media, output limits, warmup,
concurrency, and request count. Keep text-only and text-plus-audio separate.

At minimum:

1. Capture environment and exact commands.
2. Run deterministic smoke and correctness gates.
3. Run the target proxy or official benchmark after warmup.
4. Save raw results, server logs, NPU samples, and output artifacts.
5. Repeat enough to estimate run variance.

Prefer an A/B/A comparison in one server session or equivalent clean restarts
when configuration/model loading changes. Never use profiler timings as score
timings.

### 4. Implement One Primary Change

- Follow existing repository and platform patterns.
- Preserve GPU behavior unless the candidate explicitly targets it.
- Keep benchmark-specific logic outside model code.
- Add focused tests for changed state, shapes, dtypes, chunking, cleanup, or
  scheduling behavior.
- Avoid unrelated refactors and metadata churn.

### 5. Validate in Increasing Cost Order

Run the narrowest useful checks first:

1. Focused unit tests for changed modules.
2. Ruff/format, YAML or shell validation, and `git diff --check` as applicable.
3. Ascend server startup and deterministic smoke requests.
4. Machine-readable correctness/effect gate.
5. Target benchmark with raw resource collection.
6. Concurrency, cancellation, and stability checks when shared/request-local
   state or memory lifetime changes.

Stop performance evaluation when a correctness gate fails. Failed, truncated,
timed-out, empty, or invalid-output requests count as failures and never as
performance samples.

### 6. Compare and Decide

Compare baseline and candidate using the same workload and report central and
tail statistics. Include TTFT/first text, first audio, chunk cadence, E2E,
throughput, errors, NPU utilization, peak HBM, and host memory when applicable.

Choose exactly one decision:

- `keep`: gates pass and the gain exceeds noise without disqualifying tradeoffs.
- `reject`: the candidate loses, fails a gate, or adds unjustified complexity.
- `investigate`: evidence is inconclusive; do not claim or commit a performance
  win.

For `reject` or `investigate`, preserve the record when it prevents repeated
work. Remove only edits made for that candidate; do not revert user changes.

### 7. Commit an Accepted Candidate Atomically

- Review the complete diff and staged diff.
- Stage explicit candidate paths, never `git add -A` in a dirty worktree.
- Keep tests and config needed by the candidate in the same commit.
- Use `perf(minicpmo): <mechanism>` for a measured optimization.
- Use a separate `fix`, `test`, `bench`, or `docs` commit for non-performance
  prerequisites.
- Put baseline/candidate identifiers, gate result, key metric delta, artifact
  path, and rollback note in the commit body when useful.
- Verify the resulting commit and clean/expected worktree before pushing.

One accepted performance mechanism equals one performance commit. Do not amend
or squash an already published candidate when doing so would erase comparison
history.

### 8. Push Without Weakening Safety

- Push to the dedicated competition remote/branch configured for this project.
- Never store a token in a remote URL, tracked file, shell script, log, or
  artifact.
- Never force-push unless the user explicitly authorizes rewriting that exact
  remote branch.
- Verify the remote commit SHA after pushing.
- Report the commit URL, decision, validation, metrics, and remaining risk.

External writes require user authorization. A prior instruction to maintain
the dedicated competition repository authorizes normal non-force pushes for
accepted candidates, but not repository deletion, visibility changes, branch
protection changes, releases, or pull requests.

## Completion Gate

Finish a daily cycle only when all are true:

- Candidate record identifies baseline, candidate, environment, workload, and
  exact commands.
- Correctness/effect and stability status are explicit.
- Raw artifacts are traceable and formal/proxy metrics are labeled correctly.
- Decision is one of `keep`, `reject`, or `investigate` with evidence.
- An accepted change is isolated in one reviewable performance commit.
- Remote state is verified when a push was requested or authorized.
- The next highest-priority TODO or blocker is stated.
