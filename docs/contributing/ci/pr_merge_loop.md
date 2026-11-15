# PR merge loop (maintainer ops)

Daily loop for OSS and vendor PRs. Platform CI ship gate is **current-tip CUDA L2 ready**; L3/L4 feed quality watch and LESSONS — they are not pre-merge blockers.

## CI ladder (CUDA merge gate)

| Tier | Pipeline | Trigger | Role |
| ------ | ---------- | --------- | ------ |
| **GHA / GitHub checks** | pre-commit / DCO; wheel when path filters match | PR checks; wheel skips docs-only changes | Cheap filters |
| **L2 ready** | `.buildkite/cuda/test-ready.yml` (label `ready`) | pre-merge | **Ship Verifier** |
| **L3 merge-test** | `.buildkite/cuda/test-merge.yml` | `merge-test` **or post-merge on main** | Health hint → same-day LESSONS if red |
| **L4 nightly** | `.buildkite/cuda/test-nightly.yml` | `NIGHTLY=1` / `nightly-test` | Comprehensive + costly → deep LESSONS; never per-PR tax |

Details: [Test System Overview](./test_system_overview.md), [CI Settings](./ci_settings.md). Diff-aware skip-ci for docs-only still applies: record the skip decision rather than claiming an unrun GPU check passed. On docs-only PRs, label requests do not override that skip; scheduled main nightly coverage still runs.

**Merge-eligible:** tip-matched approval + applicable GitHub checks green + **current-tip CUDA L2 ready** green, or the recorded docs-only skip where applicable. Not L3/L4.

## High-priority selection

High-pri when **any** of:

1. Labels `high priority` / `priority` / `critical` / `bug`
2. Regression / crash / broken main or broken L2 after merge
3. Security / data-loss / API break
4. Review requested from `hsliuustc0106`
5. Release / required-CI / docs that **unblock ship or OSS contributors** (not blog polish)
6. RFC / roadmap only when it **unblocks ship** (linked issue + clear blocker)
7. **NV / AMD / Intel / Ascend** team PRs (platform tracks)

**Not high-pri alone:** claim-sensitive (perf numbers, marketing docs) — claim bar still applies, no queue jump.

**Queue order:** urgent regressions first; otherwise prioritize `ready` + `high priority`, then other high-pri PRs → OSS before internal → oldest tip-green among ties. `ready` is CI admission, not proof of a passing current-head run or merge eligibility.

**Current trial limits:** **30 regular triage slots plus 15 additional new-PR first-triage slots per Asia/Shanghai calendar day**, shared across all automated runs and manual passes; at most **3 substantive deep reviews per six-hour run**, selected according to complexity and evidence. These are ceilings, not completion targets.

The additional pool is reserved for newly opened PRs that have not previously been triaged, including an unhandled new-PR backlog carried forward from earlier days. An older PR with a new head is not a new PR. Record each PR number and its pool when first triaged that day; count it only once, never against both pools. Rechecks do not create extra slots, and new heads require fresh evidence rather than a new-PR allowance. Carry unhandled PRs forward, not unused daily slots. Manual passes do not reset either allowance. Keep the priority order above within both pools.

The 30 + 15 allocation is a configured trial cap, not a measured optimum or a Magpie requirement. vLLM-Omni has community contributors and shared maintainer responsibility; no throughput or review-quality improvement is established by these limits alone.

**Design question (PROPOSED):** should broad, lightweight metadata discovery find ready candidates before counted triage, while unchanged blocked heads wait for a meaningful recheck trigger? Define discovery versus triage and evaluate that boundary before changing accounting. This proposal must not become a way to perform uncounted triage. Keep the current 30 regular + 15 new-PR daily triage slots and 3-per-run deep-review limit until a further change is explicitly approved.

**Other platform evidence:** AMD, Intel, and NPU CI results are recorded separately; they are not independent platform CI merge gates for this loop. CUDA success does not establish correctness on those platforms. Demonstrated code defects, unresolved human review concerns, and applicable required GitHub checks still block merging. Record untested paths and route platform-specific failures for diagnosis rather than silently calling them passed. This operating policy does not modify branch protection or pipeline configuration.

For diagnosis, AMD uses `.buildkite/amd/test-amd-ready.yml`; NPU uses `.buildkite/npu/test-npu-ready.yml`; Intel uses the static `.buildkite/intel/pipeline-intel.yml`. Follow [CI Settings](./ci_settings.md#platform-configuration-style) for actual platform admission and tier labels. Do not request additional platform runs solely to satisfy the CUDA merge gate.

**Vendor soft-detect (evidence, not guess):** org membership if known · labels `npu` / `rocm` / `xpu` / `cuda` · `[Hardware][Vendor]` title prefix · CODEOWNERS touch. Triage drafts `vendor-high-pri?` with the evidence; maintainer applies labels.

## Daily clock (Asia/Shanghai)

Automated merge-readiness refresh runs every 6 hours, around **03:12 / 09:12 / 15:12 / 21:12**. These refreshes supplement the daily maintainer checkpoints below. The **30 regular + 15 new-PR triage slots are daily across runs and manual passes**; the **3-deep-review cap is per six-hour run**, with fewer reviews when complexity warrants.

1. **~06:15** — Overnight triage within the shared daily allowance, high-pri first, suggested labels for maintainer review
2. **~09:12** — Merge-ready list: tip SHA + applicable GitHub checks + current-tip CUDA L2 green (or documented docs-only skip) + tip approval; **Day-1 health** (main L3/L4 same-class reds)
3. **~10:11** — Squash-and-merge list for maintainer merge flip
4. Maintainer-controlled labels (`ready` + applicable platform admission labels), checklist, and deep reviews within the current run's allowance; approve only another author's PR at the reviewed tip
5. `hsliuustc0106` confirms and personally executes squash-and-merge
6. Post-merge L3 → same-day LESSONS if red; overnight L4 → deep LESSONS

## Community coordination

Classify the current head before spending a substantive deep review. Keep verified findings separate from suggested actions; an identified responsible party is not an assignment or a promise that they will act.

| Category | Next action | Responsible party | Recheck trigger |
| --- | --- | --- | --- |
| Conflicts | Identify conflicting paths and the base/head; propose an author update, then recheck the resulting head and gates. | PR author; maintainer if ownership is unclear | New head or base change |
| Missing evidence | List the exact missing test, command, environment, or result needed for this head. | PR author or relevant platform maintainer | Evidence or head update |
| CI pending | Monitor the current-head run; do not ask the author to fix a queued or running check. | CI/platform maintainer if scheduling needs investigation | Run completion or scheduling evidence |
| CI code failure | Link the failed job and verified cause; promptly request the concrete repair and local evidence below. | PR author | Repair evidence and new head |
| CI infrastructure / unknown | Preserve logs and classify the cause; an infrastructure diagnosis is not a passing gate. Do not demand a speculative code change. | CI/platform maintainer; unknown if not established | Diagnostic evidence or authorized retry result |
| Repaired, current-head CI missing | Verify the repair evidence, then hand off the exact head for maintainer-controlled retesting or the bounded fallback below. Old failures are not proof the new head still fails. | Author supplies local proof; maintainer controls project CI | Matching-head run or documented skip |
| Performance / reproduction hold | Keep the hold until the named model/path has the required reproduction or matched A/B evidence; do not infer a pass from unrelated tests. | PR author with relevant model/platform reviewer | Required evidence or claim/scope change |
| Changes requested | Read unresolved review findings and author replies; identify the remaining action without duplicating the review. | Author, then the requesting reviewer | Author response or updated head |
| Ready / approved | Revalidate tip approval, applicable GitHub checks, current-tip CUDA L2 (or recorded docs-only skip), and unresolved blockers; hand off to a maintainer. | Independent reviewer, then maintainer | Head, review, gate, or merge-state change |
| Merged | Record merge SHA and inspect applicable post-merge health; route confirmed CUDA regressions through the narrow revert rule below. | Maintainer and relevant CI/platform owner | Post-merge result or confirmed regression |

Keep one record per PR with the fields below. Use timestamps with a timezone; preserve the start of an unchanged wait rather than resetting it each refresh. Use `unknown` when the responsible party or wait start is not established.

| PR / exact head | Category | Verified evidence / proposed action | Missing evidence | Next action / responsible party | Waiting since | Recheck trigger |
| --- | --- | --- | --- | --- | --- | --- |
| PR link / SHA | Current category | Linked facts; clearly marked proposal | Specific gap or none verified | Concrete next step; verified owner or proposed recipient | Timestamp or unknown | Observable change |

### Actionable review TODOs

Present actionable findings in every PR review as a TODO checklist, including
single-action reviews. Each item records severity/problem, exact-head evidence,
a concrete next action, a verified or suggested responsible party, acceptance
criteria, and status. Keep completed items separate from outstanding work; mark
an item complete only after checking its acceptance evidence at the relevant
head. An author saying "fixed" is a recheck trigger, not verified completion.
Do not invent TODOs when no actionable finding remains: state the review scope,
testing limits, and that no actionable finding was identified.

Keep responsibilities explicit: contributors diagnose and fix their code and
supply the failed-job/root-cause/fix-SHA record plus local command, environment,
results, and redacted screenshots. The reviewer verifies the failure, sends an
actionable request, checks that evidence, and assesses retest readiness. The
maintainer normally authorizes and triggers project CI; the loop may use only
the verified CUDA fallback below when no maintainer has handled the retest.
The reviewer/maintainer then verifies current-head checks and unresolved concerns. Final merge
confirmation and execution of squash-and-merge remain with `hsliuustc0106`;
the loop prepares the handoff and never merges automatically.

For example (replace placeholders with verified facts before sending):

- [ ] **P1: <failure scenario>**. Evidence: <head SHA and file/line or failed job>.
  Action: <specific repair and named regression>. Owner: <author, verified;
  subsystem reviewer, suggested if needed>. Acceptance: <observable passing
  regression and required local repair record>. Status: awaiting author repair.

Completed items belong under a separate "Verified complete" list with the fix
SHA and acceptance evidence. Keep maintainer-only retest decisions and routine
CI monitoring in the coordination record rather than assigning them to the
contributor. A checklist changes presentation, not permission to publish a
review, comment, or request.

In the internal coordination checklist, keep these later steps separate from
contributor TODOs:

- [ ] **Reviewer/maintainer:** verify the repair record, decide and authorize a
  necessary current-head CUDA retest, and check its result (or documented skip)
  alongside applicable GitHub checks and unresolved review concerns. Acceptance:
  linked current-head evidence and no unresolved merge blocker. Status: awaiting
  evidence verification or checks; use the actual observed state.
- [ ] **Final merge confirmation — `hsliuustc0106`:** inspect the verified
  merge-ready handoff and personally execute squash-and-merge when satisfied. Acceptance: explicit
  confirmation and an observed squash merge by hs;
  record an actual merge only after observing it. Status: waiting for
  hs's final confirmation once the preceding gates are met.

### Bounded contributor feedback

For a PR that needs a conflict fix, a verified CI code-failure repair, or missing evidence, the configured loop may send one consolidated, actionable GitHub comment to its author or relevant maintainer. Before commenting, inspect existing comments, reviews, and replies; reuse the existing discussion and avoid duplicate requests or unchanged nudges. State the head, verified blocker, exact missing evidence or next step, and what will trigger recheck. An unchanged head/blocker is not by itself a reason to comment again.

For contributor-facing action lists, follow the canonical review skill's
[consolidated-reply format](https://github.com/vllm-project/vllm-omni/blob/main/.claude/skills/review-pr/references/delivery/maintainer-style-study.md#consolidated-replies-with-multiple-actions):
a few contributor-owned checkboxes with concrete deliverables and only necessary
context. Omit routine maintainer/CI status sections and process explanations.
Keep material testing limitations brief without implying unperformed validation.

Never ask contributors to provide the project's CUDA L2/Buildkite result.
Maintainers handle admission and inspect matching-head results; record actual
labels and run state in the review record. Do not require an untriggered result
before `ready` admission. Justified local evidence and the existing approval/merge
gates remain intact; only the bounded CUDA retest fallback below grants
CI-triggering authority.

This documented feedback scope is limited to conflict fixes, verified CI code-failure repairs, and missing evidence. Other outreach, including generic reminders and stale-approval nudges, remains proposed pending separate authorization. Documenting these operating settings or opening this documentation PR does not establish repository-wide adoption.

### CI failure → local repair evidence → controlled retest

Process verified code failures as soon as they are discovered in an authorized
check, especially for `ready` + `high priority` PRs. Do not wait for a full deep
review to communicate an already demonstrated blocker. This is a process target,
not continuous event monitoring or a guaranteed response deadline. Initial
feedback is not a completed substantive review or an approval.

1. **Diagnose before requesting a fix.** Read the actual failed job/log and its
   commit SHA. Separate code failures from infrastructure, network, capacity,
   flaky, and unknown failures. If the head has changed, check the relevant diff
   and latest discussion before deciding whether the old failure still applies.
   A red summary alone does not establish a code defect.
2. **Give one actionable author request.** Link the failed run/job, state the
   failing SHA and current head, verified cause or remaining uncertainty, and
   the smallest repair/reproducer needed. Name the author-owned next step;
   identify a maintainer only from verified ownership, or mark the routing as
   suggested. Check existing reviews/comments first and do not repeat an
   equivalent unresolved request. Re-fetch open/non-draft state and head
   immediately before posting; stop or revalidate if they changed.
3. **Require a local repair record before proposing another expensive run.**
   The author supplies the failed job and cause, fix commit, exact local test
   command, relevant environment/hardware/software versions, and actual result.
   Include a local screenshot showing the command and result, together with
   copyable text/log output; redact credentials and private data. For a bug,
   name a regression that fails without the repair and passes with it, or state
   precisely what could not be reproduced. A bare “fixed, please rerun” does
   not establish repair. If hardware or reproduction is unavailable, record
   that limitation and ask a maintainer to decide the smallest justified
   validation; never fabricate a local pass or require inaccessible hardware
   as an unresolvable prerequisite. Infrastructure-only failures need a
   diagnosis and maintainer retry decision, not an invented author code fix.
4. **Hand off a specific retest decision.** The maintainer checks the current
   head, repair evidence, and queued/running/completed results before admitting
   a necessary CUDA run. Deduplicate by PR/head/pipeline; coalesce superseded
   heads rather than repeatedly toggling labels. Use the smallest supported
   validation scope that still satisfies the gate. Do not promise per-test
   retry support without checking the actual pipeline. An absent status is
   unknown: verify whether a build exists rather than assuming none is running.
   **Bounded loop fallback:** if no other maintainer has handled the retest,
   the loop may trigger the necessary CUDA retest after verifying the repair
   commit and local evidence. Immediately re-fetch the open/non-draft state,
   exact head, latest discussion, and queued/running/completed builds. Proceed
   only when the evidence applies to that head and no equivalent PR/head/pipeline
   run or maintainer action already covers the request. Use the supported,
   smallest necessary trigger; never blindly toggle labels or duplicate runs.
   Record the trigger identity and verify the returned run URL, head, and state.
   If sending is uncertain, read back before retrying. If head/evidence changes,
   access is denied, or duplicate-run status cannot be established, stop and
   leave the retest pending. This exception covers a necessary CUDA retest only,
   not additional platform runs, discretionary paid compute, or broader retries.
5. **Close the loop on matching-head evidence.** Record the new run and result,
   or the applicable documented docs-only skip. A screenshot, an old-head pass,
   or an infrastructure classification cannot replace the current-tip CUDA
   gate. Keep unresolved code/review blockers open and revalidate approval and
   other required checks before the maintainer's merge decision.

Keep the request identity, head, blocker, feedback time, repair evidence,
retest decision, responsible party, and next recheck trigger in the PR record.
Do not reset an unchanged waiting-since timestamp. New feedback needs a material
change in the blocker or evidence; an unchanged head or elapsed time is not a
reason for another nudge. Monitor pending CI rather than asking the author to
resubmit evidence while the same run is still executing.

`ready` remaining on a PR does not prove that a new head has a new CI result.
[Diff-aware CI](./ci_settings.md#diff-aware-ci) selects work after a build is
scheduled; repository-side pipeline selection does not by itself create a
Buildkite build. Diagnose missing-build admission separately and verify the
actual trigger configuration. This guide documents the bounded retest permission;
it does not implement a trigger, automatic label toggling, webhook handler, or
paid-compute expansion. A deployed loop must have the corresponding explicit
permission in its configuration; this documentation PR does not update an
existing automation.

### Action boundaries

The readiness loop does not approve its own PRs or merge ordinary PRs. `hsliuustc0106` retains final confirmation and personally executes squash-and-merge. Apart from the bounded, verified CUDA retest fallback above, do not trigger or rerun workflows, apply CI-triggering labels, or launch paid compute without separate authorization. Observe existing results and propose the next action otherwise. The only standing revert exception is the confirmed, causally attributed CUDA merge-test regression described below; it does not authorize speculative reverts or unrelated merges.

## Quality watch (first-class)

Watch **same-class** L3/L4 reds (area / platform / claim), not raw volume.

If the same class stacks: hold/skip auto-`ready` on that class until cleared; tighten next queue; write LESSONS same day. Fast-promote into precheck if main bleeds the **same area twice**.

**Day-1 health line:** merges landed · same-class L3/L4 reds · claim slips.

### CUDA merge-test regressions

If CUDA merge-test fails after a merge, promptly determine whether the newly merged change caused the failure. When the regression is confirmed to come from that change, revert it immediately; do not wait for a same-class pattern or the next daily review. Distinguish infrastructure failures and flaky tests from code regressions, and identify the causal commit before reverting. Record the failing check, causal change, revert, and follow-up in LESSONS. L3 remains a post-merge health check, not an added pre-merge gate.

## Stale-approval nudge

Near-misses are often tip-stale after rebase. Propose a request to the relevant independent reviewer to re-approve tip `SHA`; check the existing discussion first. Stale-approval outreach remains proposed until separately authorized; the conflict/evidence feedback permission does not cover it. A status-only recheck is not a substantive deep review; any substantive re-review counts toward the current run's allowance.

## LESSONS → code regulation + skills

Record reusable findings in [Review and issue lessons](./lessons.md). Check the
linked evidence and current code before applying a historical lesson.

| Pattern ≥3 days (or same-area twice on main) | Where it lands |
| --------------------------------------------- | ---------------- |
| CI / precheck mechanics | Auto → this tree + `.claude/skills/precheck-pr` (+ bot precheck sync); **≤1 regulation PR/day** |
| Review bars (non-claim) | Auto → `.claude/skills/review-pr` (+ bot review sync) |
| Claim / docs language | **Draft for maintainer yes** → then `docs/contributing` / [Documentation Build Guide](../DOCS_GUIDE.md) + skills |

One shared bible for humans and agents — not a bot-only shadow.

## First-comment checklist (template)

Type: bug | perf | API | docs

Acceptance (product):

- Bug: named regression that fails without the change; green on tip; no new claim
- Perf: matched A/B (same HW + prompt set, tip vs baseline) — no number without that pair
- API: matches documented surface at tip; breaking change has migrate note
- Docs: matches live code at pinned Omni SHA; future work labeled future

Verifier: say what was run at **this tip SHA** — a check not run is listed “not run” / “not tried on <GPU/model>”, never omitted. Merge only the reviewed tip; rebase → re-review.

Perf/model claims need matched A/B on the **named** model path; fix the class, not only the sample.
