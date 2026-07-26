# Candidate Record and Commit Policy

Use this reference for every performance candidate. Keep raw logs and large
audio/profiler artifacts outside Git unless the competition packaging rules
require them. Commit a compact report or record whose artifact paths and
checksums make the raw evidence traceable.

## Candidate Record

```markdown
# <candidate-id>: <short title>

- Decision: keep | reject | investigate
- Baseline SHA/config:
- Candidate SHA/diff:
- Official rules/toolkit version or retrieval date:
- Environment/NPU topology:
- Model revision/dtype:
- Hypothesis and mechanism:
- Target metric/workload:
- Guardrails and rollback condition:

## Commands

- Server:
- Warmup:
- Correctness/effect:
- Benchmark:
- Monitoring/profiling:

## Results

| Metric | Baseline | Candidate | Delta | Variance/confidence |
| --- | ---: | ---: | ---: | ---: |
| Gate pass | | | | |
| TTFT/first text p50/p95 | | | | |
| First audio p50/p95 | | | | |
| Chunk latency p50/p95 | | | | |
| E2E p50/p95 | | | | |
| Throughput | | | | |
| Stable sessions | | | | |
| NPU utilization | | | | |
| Peak HBM/host memory | | | | |
| Errors/timeouts | | | | |

## Evidence

- Raw result paths/checksums:
- Server log:
- Output artifacts:
- NPU samples/trace:
- Correctness/effect report:

## Decision Rationale

Explain why the result exceeds noise, which tradeoffs remain, and what would
cause rollback. Label local proxy metrics and unresolved official assumptions.
```

## Commit Boundaries

Use these boundaries so history remains bisectable:

- `perf(minicpmo)`: one accepted performance mechanism with its focused tests.
- `fix(npu)` or `fix(scheduler)`: compatibility/correctness prerequisite.
- `feat(bench)`: benchmark or observability capability, without model changes.
- `test(minicpmo)`: test-only gate expansion.
- `docs(minicpmo)`: records, reproduction notes, or rule snapshots.

Do not mix benchmark fixes, unrelated cleanup, and a performance mechanism in
one `perf` commit. If a candidate needs a prerequisite, land and validate the
prerequisite first, then rerun the baseline before the performance candidate.

## Pre-Commit Checklist

- The staged diff contains only candidate-owned paths.
- Focused tests pass.
- Static checks for touched files pass.
- Ascend smoke and applicable gates pass.
- Baseline and candidate commands/workloads match.
- The gain exceeds variance or the commit does not claim a performance win.
- GPU and unsupported workloads are unchanged or explicitly documented.
- Artifact paths contain no credentials, personal data, or machine secrets.

## Pre-Push Checklist

- Inspect `git status`, `git show --stat --oneline HEAD`, and the target remote.
- Confirm the local commit is based on the intended remote branch.
- Use a credential helper, `gh`, or an ephemeral environment token; keep the
  remote URL credential-free.
- Push normally without `--force`.
- Read back the remote branch SHA and compare it with local `HEAD`.
- Record the commit URL in the experiment report or handoff.
