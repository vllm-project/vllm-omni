# Review and issue lessons

Check these lessons before requesting work; verify the current code and discussion.
Feedback → clarify/reproduce → community work → validate → reply → update lessons.
Maintainers retain CI admission and merge decisions.

## Keep this file small

- Keep the main file within **200 lines** and each entry concise. Merge duplicate
  patterns; revise obsolete guidance and retain only a useful replacement link.
- Each lesson needs a trigger, preventive check and evidence link. Keep incident
  details, reproduction commands and regression results in the linked issue/PR.
- For new incidents, establish what failed and why validation missed it; record
  unknown causes as unknown. Track risky change patterns, not author blacklists.
- This record grants no new authority to publish, run CI, merge or expand monitoring.

Checked **2026-10-14**, Omni `ed102c65aa46407c999a8d24e383ba98383669bb`.
All seeds are **source-only**: public code, reports or accepted guidance, not
independent reproductions. A similar historical report does not prove a new cause.

## L001: CUDA L2 responsibility

- **Trigger:** a contributor is asked for the project's CUDA L2/Buildkite result.
- **Check:** maintainers handle admission and inspect matching-head results. Ask
  contributors only for justified local evidence; preserve the existing merge gate.
- **Evidence:** [#8910](https://github.com/vllm-project/vllm-omni/pull/8910), accepted guidance. Recheck if admission policy changes.

## L002: Evidence must match the claim

- **Trigger:** benchmark numbers or model results are used to justify a change.
- **Check:** match the actual diff, model/path, workload, hardware and revisions.
  A workload correction does not establish a speedup or resolve an output bug.
- **Evidence:** [#8057](https://github.com/vllm-project/vllm-omni/pull/8057) states these limits. Revalidate when the workload changes.

## L003: Similar symptoms can need different fixes

- **Trigger:** a related issue/PR appears to explain the current report.
- **Check:** compare trigger, output field, backend and revision. NPU operator
  shapes and Python stacks differ; the missing-stack diagnosis remains unverified.
- **Evidence:** [#8877](https://github.com/vllm-project/vllm-omni/issues/8877) versus [#8770](https://github.com/vllm-project/vllm-omni/pull/8770). Confirm with the relevant trace before claiming a fix.

## L004: Merge does not establish full resolution

- **Trigger:** a fixing or prerequisite PR merges while an issue remains open.
- **Check:** verify the original acceptance criteria at the fixed revision/platform;
  retain unresolved scope. Read current discussion before recruiting parallel work.
- **Evidence:** [#7902](https://github.com/vllm-project/vllm-omni/issues/7902) separates [#7920](https://github.com/vllm-project/vllm-omni/pull/7920) from other milestones. Refresh dependencies and validation before closure.

## L005: Routing is not accepted ownership

- **Trigger:** a bot mentions a model owner or CODEOWNERS suggests reviewers.
- **Check:** resolve affected paths, read existing claims/replies, and ask the
  smallest relevant group a concrete question. A mention does not assign work.
- **Evidence:** [#8877 bot routing](https://github.com/vllm-project/vllm-omni/issues/8877#issuecomment-6141188620) and [CODEOWNERS](https://github.com/vllm-project/vllm-omni/blob/main/.github/CODEOWNERS). Recheck both before outreach.

## L006: Keep feedback actionable

- **Trigger:** a first issue reply or consolidated review has several next steps.
- **Check:** ask one clear question or use a short action checklist. Include only
  necessary evidence; omit routine process narration and already answered requests.
- **Evidence:** [#8910](https://github.com/vllm-project/vllm-omni/pull/8910), accepted delivery guidance. Remove resolved actions from later replies.
