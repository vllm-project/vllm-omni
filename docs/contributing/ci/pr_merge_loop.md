# PR merge loop (maintainer ops)

Daily loop for OSS and vendor PRs. Ship gate is **L2 ready**; L3/L4 feed quality watch and LESSONS — they are not pre-merge blockers.

## CI ladder (CUDA example; use each platform’s actual gate)

| Tier | Pipeline | Trigger | Role |
|------|----------|---------|------|
| **GHA / GitHub checks** | pre-commit / DCO; wheel when path filters match | PR checks; wheel skips docs-only changes | Cheap filters |
| **L2 ready** | `.buildkite/cuda/test-ready.yml` (label `ready`; use the matching platform gate below) | pre-merge | **Ship Verifier** |
| **L3 merge-test** | `.buildkite/cuda/test-merge.yml` | `merge-test` **or post-merge on main** | Health hint → same-day LESSONS if red |
| **L4 nightly** | `.buildkite/cuda/test-nightly.yml` | `NIGHTLY=1` / `nightly-test` | Comprehensive + costly → deep LESSONS; never per-PR tax |

Details: [Test System Overview](./test_system_overview.md), [CI Settings](./ci_settings.md). Diff-aware skip-ci for docs-only still applies: record the skip decision rather than claiming an unrun GPU check passed. On docs-only PRs, label requests do not override that skip; scheduled main nightly coverage still runs.

**Merge-eligible:** tip-matched approval + applicable GitHub checks green + **matching L2 ready** green, or the recorded docs-only skip where applicable. Not L3/L4.

## High-priority selection

High-pri when **any** of:

1. Labels `priority` / `critical` / `bug`
2. Regression / crash / broken main or broken L2 after merge
3. Security / data-loss / API break
4. Review requested from `hsliuustc0106`
5. Release / required-CI / docs that **unblock ship or OSS contributors** (not blog polish)
6. RFC / roadmap only when it **unblocks ship** (linked issue + clear blocker)
7. **NV / AMD / Intel / Ascend** team PRs (platform tracks)

**Not high-pri alone:** claim-sensitive (perf numbers, marketing docs) — claim bar still applies, no queue jump.

**Queue order:** high-pri first → OSS before internal → oldest tip-green among ties.

**Current trial limits:** **30 regular triage slots plus 15 additional new-PR first-triage slots per Asia/Shanghai calendar day**, shared across all automated runs and manual passes; at most **3 substantive deep reviews per six-hour run**, selected according to complexity and evidence. These are ceilings, not completion targets.

The additional pool is reserved for newly opened PRs that have not previously been triaged, including an unhandled new-PR backlog carried forward from earlier days. An older PR with a new head is not a new PR. Record each PR number and its pool when first triaged that day; count it only once, never against both pools. Rechecks do not create extra slots, and new heads require fresh evidence rather than a new-PR allowance. Carry unhandled PRs forward, not unused daily slots. Manual passes do not reset either allowance. Keep the priority order above within both pools.

The 30 + 15 allocation is a configured trial cap, not a measured optimum or a Magpie requirement. vLLM-Omni has community contributors and shared maintainer responsibility; no throughput or review-quality improvement is established by these limits alone.

**Design question (PROPOSED):** should broad, lightweight metadata discovery find ready candidates before counted triage, while unchanged blocked heads wait for a meaningful recheck trigger? Define discovery versus triage and evaluate that boundary before changing accounting. This proposal must not become a way to perform uncounted triage. Keep the current 30 regular + 15 new-PR daily triage slots and 3-per-run deep-review limit until a further change is explicitly approved.

**Vendor L2:** use the **matching** platform gate — never mark Ascend/AMD “green” on CUDA-only L2. CUDA uses `.buildkite/cuda/test-ready.yml`; AMD uses `.buildkite/amd/test-amd-ready.yml`; NPU uses `.buildkite/npu/test-npu-ready.yml`; Intel uses the static `.buildkite/intel/pipeline-intel.yml`. Follow [CI Settings](./ci_settings.md#platform-configuration-style) for platform admission and tier labels, including NPU’s `ready` + `npu-test`. The trees are not identical: NPU has L2/L4 but no separate L3, and Intel has a static L2 pipeline rather than separate L3/L4 files.

**Vendor soft-detect (evidence, not guess):** org membership if known · labels `npu` / `rocm` / `xpu` / `cuda` · `[Hardware][Vendor]` title prefix · CODEOWNERS touch. Triage drafts `vendor-high-pri?` with the evidence; maintainer applies labels.

## Daily clock (Asia/Shanghai)

Automated merge-readiness refresh runs every 6 hours, around **03:12 / 09:12 / 15:12 / 21:12**. These refreshes supplement the daily maintainer checkpoints below. The **30 regular + 15 new-PR triage slots are daily across runs and manual passes**; the **3-deep-review cap is per six-hour run**, with fewer reviews when complexity warrants.

1. **~06:15** — Overnight triage within the shared daily allowance, high-pri first, suggested labels for maintainer review
2. **~09:12** — Merge-ready list: tip SHA + applicable GitHub checks + matching L2 green (or documented docs-only skip) + tip approval; **Day-1 health** (main L3/L4 same-class reds)
3. **~10:11** — Squash-and-merge list for maintainer merge flip
4. Maintainer-controlled labels (`ready` + applicable platform admission labels), checklist, and deep reviews within the current run's allowance; approve only another author's PR at the reviewed tip
5. Maintainer squash/merge
6. Post-merge L3 → same-day LESSONS if red; overnight L4 → deep LESSONS

## Community coordination

Classify the current head before spending a substantive deep review. Keep verified findings separate from suggested actions; an identified responsible party is not an assignment or a promise that they will act.

| Category | Next action | Responsible party | Recheck trigger |
| --- | --- | --- | --- |
| Conflicts | Identify conflicting paths and the base/head; propose an author update, then recheck the resulting head and gates. | PR author; maintainer if ownership is unclear | New head or base change |
| Missing evidence | List the exact missing test, command, environment, or result needed for this head. | PR author or relevant platform maintainer | Evidence or head update |
| CI pending / failed | Link the current-head run; distinguish queued/running, code failure, and infrastructure/flake evidence. Propose the smallest needed fix or investigation. | Author for code; CI/platform maintainer for infrastructure | Run completion, diagnostic evidence, or new head |
| Performance / reproduction hold | Keep the hold until the named model/path has the required reproduction or matched A/B evidence; do not infer a pass from unrelated tests. | PR author with relevant model/platform reviewer | Required evidence or claim/scope change |
| Changes requested | Read unresolved review findings and author replies; identify the remaining action without duplicating the review. | Author, then the requesting reviewer | Author response or updated head |
| Ready / approved | Revalidate tip approval, applicable GitHub checks, affected-platform L2 (or recorded docs-only skip), and unresolved blockers; hand off to a maintainer. | Independent reviewer, then maintainer | Head, review, gate, or merge-state change |
| Merged | Record merge SHA and inspect applicable post-merge health; route confirmed CUDA regressions through the narrow revert rule below. | Maintainer and relevant CI/platform owner | Post-merge result or confirmed regression |

Keep one record per PR with the fields below. Use timestamps with a timezone; preserve the start of an unchanged wait rather than resetting it each refresh. Use `unknown` when the responsible party or wait start is not established.

| PR / exact head | Category | Verified evidence / proposed action | Missing evidence | Next action / responsible party | Waiting since | Recheck trigger |
| --- | --- | --- | --- | --- | --- | --- |
| PR link / SHA | Current category | Linked facts; clearly marked proposal | Specific gap or none verified | Concrete next step; verified owner or proposed recipient | Timestamp or unknown | Observable change |

### Bounded contributor feedback

For a PR that needs a conflict fix or missing evidence, the configured loop may send one consolidated, actionable GitHub comment to its author or relevant maintainer. Before commenting, inspect existing comments, reviews, and replies; reuse the existing discussion and avoid duplicate requests or unchanged nudges. State the head, verified blocker, exact missing evidence or next step, and what will trigger recheck. An unchanged head/blocker is not by itself a reason to comment again.

This permission is limited to conflict fixes and missing evidence. Other outreach, including generic reminders and stale-approval nudges, remains proposed pending separate authorization. Documenting these operating settings or opening this documentation PR does not establish repository-wide adoption.

### Action boundaries

The readiness loop does not approve its own PRs or merge ordinary PRs. Maintainers retain the merge decision. Do not trigger or rerun workflows, apply CI-triggering labels, or launch paid compute without separate authorization. Observe existing results and propose the next action instead. The only standing revert exception is the confirmed, causally attributed CUDA merge-test regression described below; it does not authorize speculative reverts or unrelated merges.

## Quality watch (first-class)

Watch **same-class** L3/L4 reds (area / platform / claim), not raw volume.

If the same class stacks: hold/skip auto-`ready` on that class until cleared; tighten next queue; write LESSONS same day. Fast-promote into precheck if main bleeds the **same area twice**.

**Day-1 health line:** merges landed · same-class L3/L4 reds · claim slips.

### CUDA merge-test regressions

If CUDA merge-test fails after a merge, promptly determine whether the newly merged change caused the failure. When the regression is confirmed to come from that change, revert it immediately; do not wait for a same-class pattern or the next daily review. Distinguish infrastructure failures and flaky tests from code regressions, and identify the causal commit before reverting. Record the failing check, causal change, revert, and follow-up in LESSONS. L3 remains a post-merge health check, not an added pre-merge gate.

## Stale-approval nudge

Near-misses are often tip-stale after rebase. Propose a request to the relevant independent reviewer to re-approve tip `SHA`; check the existing discussion first. Stale-approval outreach remains proposed until separately authorized; the conflict/evidence feedback permission does not cover it. A status-only recheck is not a substantive deep review; any substantive re-review counts toward the current run's allowance.

## LESSONS → code regulation + skills

| Pattern ≥3 days (or same-area twice on main) | Where it lands |
|---------------------------------------------|----------------|
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
