# Optional review workers

Use `auto` for one available backend, `codex` or `claude` for that backend,
and `none` for direct review. If unavailable, report the fallback locally and
review directly unless the user requires that backend. Do not assume a CLI,
model, installation path, or named skill exists.

Use one worker normally; at most two for large/high-risk changes, with bounded
path scopes or independent passes. Do not use reviewers as a vote.

Give each worker:

- Repository path, immutable base/head/merge-base SHAs, assigned paths and focus.
- Applicable trusted instructions and the parent's selected domain references.
- Read-only scope: no file changes, commits, pushes, GitHub writes, or delegation.
- The same execution/trust gates as the parent; a worktree is not a sandbox.
- A request to inspect the complete assigned diff, relevant callers and tests,
  and report concrete introduced regressions with priority, smallest changed-line
  range, trigger, impact and fix direction. Include test gaps and residual risks;
  finish the assigned scope after the first finding.

The parent rechecks the head, verifies findings against the frozen source,
drops speculative/pre-existing/style-only items, and deduplicates root causes.
Apply the normal severity and comment budget; priority alone does not determine
blocking status. Keep material omitted findings and validation gaps in the local
report. Only the parent posts, with separate user authorization.
