# Review Execution and Delivery

Use this reference for every review. It defines snapshot collection, gates,
validation records, delivery permissions, maintainer tone, and re-review.

Official docs: [contributing guide](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/)
and [CI failure triage](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/ci/failures/).

## Contents

- [Freeze the review surface](#freeze-the-review-surface)
- [Report status before analysis](#report-status-before-analysis)
- [Apply review gates](#apply-review-gates)
- [Run bounded validation](#run-bounded-validation)
- [Deliver maintainer-style findings](#deliver-maintainer-style-findings)
- [Re-review safely](#re-review-safely)

## Freeze the review surface

For a GitHub PR, read the base/head SHA before and after fetching metadata and
the diff:

```bash
gh api "repos/vllm-project/vllm-omni/pulls/<PR>" \
  --jq '{base_sha: .base.sha, head_sha: .head.sha}'
REVIEW_FIELDS="number,url,title,body,isDraft,baseRefName,headRefName,mergeable,mergeStateStatus,statusCheckRollup,files"
gh pr view <PR> --repo vllm-project/vllm-omni \
  --json "${REVIEW_FIELDS}"
gh pr diff <PR> --repo vllm-project/vllm-omni
gh api "repos/vllm-project/vllm-omni/pulls/<PR>" \
  --jq '{base_sha: .base.sha, head_sha: .head.sha}'
```

Discard the snapshot if either SHA changed. Do not mix comments or validation
from different heads.

After the SHAs stabilize, fetch and materialize a trusted `head_sha` in an
isolated detached worktree. Run every source read, `rg` search, import, and test
from that snapshot, not the caller's checkout. A detached worktree provides
snapshot isolation, not a security boundary.

Treat a fork head as untrusted unless the user and environment policy explicitly
establish trust. Never run its imports, tests, builds, hooks, package setup, or
repo-configurable linters/plugins on the reviewer host. Execute them only in a
disposable sandbox or VM with no credentials or inherited secrets, no host agent
or service sockets, minimal read-only host mounts, disabled or explicitly
allowlisted network, resource/time limits, and destruction after the run. If
that boundary is unavailable, limit the review to the remote diff, SHA-addressed
reads such as `git show <head_sha>:<path>`, and existing CI evidence; report all
executable validation as a gap.

Before the first source read, record a pristine snapshot fingerprint outside
the reviewed worktree. Include `HEAD`, the NUL-delimited index entries and blob
IDs, and a NUL-safe manifest of every worktree entry except Git metadata,
including tracked, untracked, and ignored paths with file type, mode, symlink
target, and content hash. Before each validation group and delivery, recompute
and byte-compare the fingerprint as well as asserting `HEAD == head_sha`. If a
tool changes or creates any entry, discard affected evidence and recreate the
snapshot before another group; do not rely on `HEAD` alone or clean an unknown
worktree in place. If an exact snapshot cannot be created, use SHA-addressed
reads and report filesystem-dependent validation as a gap.

For a local branch/worktree, determine the target ref from the user, current PR,
or configured upstream; never infer it from the branch name. Resolve the target
once and include all task-owned worktree state:

```bash
git status --porcelain=v2 -z
git rev-parse HEAD
git rev-parse <target-ref>
git merge-base <target-base-sha> HEAD
git diff --stat <comparison-commit>
git diff --name-status <comparison-commit>
git diff --binary <comparison-commit>
git diff --cached --binary <comparison-commit>
git diff --binary
git ls-files --others --exclude-standard -z
```

Freeze the exact bytes of every in-scope untracked file from the NUL-delimited
list, plus a NUL-safe path, file-type, mode, and content-hash manifest. Also
fingerprint `HEAD`, the target and merge base, porcelain status, and both binary
index and worktree patches. Recording only untracked names is insufficient.
Keep these snapshots in the evidence packet and do not mutate the reviewed
checkout during a read-only review.

For local changes without PR metadata, mark the title/body as synthetic. Base
them on the user request and commit messages without inventing validation claims.

If no PR, branch, or usable worktree is identifiable, ask for a PR URL/number or
explicit base and head rather than guessing.

## Report status before analysis

Within 60 seconds of starting, and before source searches or tests, send this
host update:

```text
Pinned head: <SHA>
Base/comparison: <ref and SHA>
CI: <pass/fail/pending/not applicable>
Mergeability: <state/not applicable>
Preliminary findings: <brief finding or none yet>
```

Mark early findings as preliminary and continue. This is not a GitHub comment.

## Apply review gates

- Report draft/WIP state. Continue a local review when explicitly requested,
  but do not publish review events without separate authorization.
- Record DCO, pre-commit, required CI, and mergeability. Pending or unknown gates
  do not block source review.
- Treat a failed gate as its own evidence. Do not restate its formatting/lint
  output as a new code-review finding.
- Open CI logs only when the first failing step overlaps the frozen diff or
  blocks the verdict. Start with the first error, not the last cascade.
- Treat the PR body as navigation. Commands, benchmark tables, and claimed tests
  become evidence only after their provenance and relevance are checked.

## Run bounded validation

Keep one evidence packet for files read, bounded searches, callers, tests, CI,
hardware, routes, and findings. Reuse it rather than refetching facts.

Before pytest, run a short import/version compatibility check. Record each
validation result with:

```text
repo, head SHA, command, result, Python/platform, dependency or lock fingerprint
```

After preflight, run targeted tests and low-cost static checks alongside source
inspection when possible. Imports, tests, builds, and linters may create caches
or rewrite files, so isolate each potentially mutating group in a disposable
snapshot or recreate the pinned snapshot before continuing. Map source symbols
to tests with bounded `rg` searches rather than assuming the test directory
mirrors production paths.

Classify failures as code, test, infrastructure, or flaky before reporting.
Skipped hardware tests are gaps, not passes. For available hardware, verify the
smallest representative unit/E2E path and compare actual output with PR claims.

## Deliver maintainer-style findings

Default to roughly 1-5 short comments, but treat that as calibration rather than
a quota: report every real blocker and report no finding when there is no issue.

Write each finding as:

```text
[P1] Short imperative title — path/to/file.py:<line>
<Trigger or call path>. <Current behavior and impact>. <Smallest fix direction>.
```

Use [maintainer-style-study.md](../delivery/maintainer-style-study.md) for
comment tone.
Keep rule IDs, grades, and audit matrices internal unless the user asks for the
complete audit.

Verify every inline `path:line` against the frozen diff before delivery. For a
PR, re-read the remote head and byte-compare the detached snapshot fingerprint,
or recreate it from `head_sha`, before asserting current lines. For a local
review, recompute and byte-compare the frozen `HEAD`, target, merge base, status,
index patch, worktree patch, and untracked-content manifest. Any mismatch makes
the review stale; discard the affected evidence and restart. Prefer one
root-cause comment to several symptom comments.

Local presentation is the default. Only explicit authorization permits GitHub
posting. When authorized, publish one consolidated final review after
validation; do not post preliminary or incremental comments. Submit a review
event (`APPROVE`, `COMMENT`, or `REQUEST_CHANGES`) only when the user explicitly
chooses that event.

Reviewer requests and owner `@mention` comments are separate external writes;
follow [review-requests.md](../delivery/review-requests.md) when the user asks
for them.

## Re-review safely

Freeze the new head and compare it with the previously reviewed SHA. Then:

1. inspect the delta and every file affected by conflict resolution or rebase;
2. revalidate earlier findings against current lines and behavior;
3. read unresolved, non-outdated threads and the author's evidence;
4. rerun only checks invalidated by the delta;
5. look for new regressions introduced by the fix;
6. do not repeat resolved or outdated comments.

If the target head changes during re-review, discard stale validation and start
again from the new snapshot.
