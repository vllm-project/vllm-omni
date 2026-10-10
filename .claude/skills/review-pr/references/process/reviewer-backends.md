# Reviewer Backends

Use this reference when delegating defect discovery to Codex, Claude, or another reviewer. The parent `review-pr` workflow remains responsible for gates, domain routing, verification, prioritization, and any GitHub writes.

## Backend Selection

| Choice | Behavior |
|--------|----------|
| `auto` | Use one available reviewer backend; fall back to direct review |
| `codex` | Use an available Codex review agent or skill |
| `claude` | Use an available Claude review agent or CLI |
| `none` | Review directly without delegation |

Honor an explicit user choice. Do not block when the requested backend is unavailable; report the fallback locally and continue directly unless the user required that backend specifically.

Use one backend for ordinary code changes. Use at most two reviewers for large or high-risk changes, either with non-overlapping path scopes or as independent passes. Do not run multiple reviewers merely to vote on the same conclusion.

## Common Read-Only Handoff

Give every backend the same concrete target:

```yaml
repository: /absolute/path/to/vllm-omni
base_ref: origin/main
base_sha: <resolved-base-sha>
head_sha: <captured-pr-head-sha>
merge_base_sha: <resolved-merge-base-sha>
scope: complete-diff | path-partition
paths: []
focus: []
constraints:
  read_only: true
  no_github_writes: true
  no_delegation: true
output:
  finding: "[P0-P3] Imperative title — path:line"
  exhaustive: true
  include_assessment: true
  include_test_gaps: true
  include_residual_risks: true
```

Capture immutable SHAs before delegation. Ask the reviewer to:

1. Read the applicable `AGENTS.md`.
2. Inspect the complete assigned diff and enough surrounding code to understand it.
3. Check relevant tests and call sites.
4. Report only concrete regressions introduced by the change.
5. Cite the smallest changed-line range that demonstrates each finding.
6. Continue through the full assigned scope after finding the first issue.

The worker must not modify files, create commits, push branches, post comments, submit a review event, or delegate again.

## Backend Adapters

### Codex

Use an available Codex review agent or review skill; do not assume that a named system skill is installed. Give it the common handoff plus selected vLLM-Omni focus areas. Use it for a complete change or bounded path partition, not for a one-function context lookup.

### Claude

Pass the same handoff and output format to the available Claude review surface. Do not assume a particular command, model, or installation path. Explicitly include the read-only and no-GitHub-write constraints in the prompt.

### Direct fallback

If no backend is available, the parent performs the same checks. Keep the same evidence threshold and comment budget. Do not mention missing tooling in the GitHub review unless the user asks.

## Aggregate Results

The parent reviewer must:

1. Re-read the PR head SHA. If it changed, discard stale line mappings and rerun affected scopes.
2. Verify each finding against code, tests, and the captured diff.
3. Drop speculative, pre-existing, intentional, or style-only findings.
4. Deduplicate by root cause and affected call path rather than by line alone.
5. Classify `blocking` independently from `P0`-`P3`; priority alone does not determine the verdict.
6. Apply the vLLM-Omni comment budget after collecting the complete result.
7. Preserve material omitted findings, test gaps, and residual risks in the local report.

Only the parent may turn selected findings into GitHub comments, and only with user authorization.
