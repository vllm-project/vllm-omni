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

**Queue order:** high-pri first → OSS before internal → oldest tip-green among ties. Cap: ≤10 triage / 1 deep review per day.

**Vendor L2:** use the **matching** platform gate — never mark Ascend/AMD “green” on CUDA-only L2. CUDA uses `.buildkite/cuda/test-ready.yml`; AMD uses `.buildkite/amd/test-amd-ready.yml`; NPU uses `.buildkite/npu/test-npu-ready.yml`; Intel uses the static `.buildkite/intel/pipeline-intel.yml`. Follow [CI Settings](./ci_settings.md#platform-configuration-style) for platform admission and tier labels, including NPU’s `ready` + `npu-test`. The trees are not identical: NPU has L2/L4 but no separate L3, and Intel has a static L2 pipeline rather than separate L3/L4 files.

**Vendor soft-detect (evidence, not guess):** org membership if known · labels `npu` / `rocm` / `xpu` / `cuda` · `[Hardware][Vendor]` title prefix · CODEOWNERS touch. Triage drafts `vendor-high-pri?` with the evidence; maintainer applies labels.

## Daily clock (Asia/Shanghai)

Automated merge-readiness refresh runs every 6 hours, around **03:12 / 09:12 / 15:12 / 21:12**. These refreshes supplement the daily maintainer checkpoints below; the caps remain **≤10 triage / 1 deep review per day**, not per run.

1. **~06:15** — Overnight triage ≤10, high-pri first, suggested labels in-room
2. **~09:12** — Merge-ready list: tip SHA + applicable GitHub checks + matching L2 green (or documented docs-only skip) + tip approval; **Day-1 health** (main L3/L4 same-class reds)
3. **~10:11** — Squash-and-merge list for maintainer merge flip
4. Labels (`ready` + applicable platform admission labels) + first-comment checklist + deep review (1/day) + approve on tip
5. Maintainer squash/merge
6. Post-merge L3 → same-day LESSONS if red; overnight L4 → deep LESSONS

## Quality watch (first-class)

Watch **same-class** L3/L4 reds (area / platform / claim), not raw volume.

If the same class stacks: hold/skip auto-`ready` on that class until cleared; tighten next queue; write LESSONS same day. Fast-promote into precheck if main bleeds the **same area twice**.

**Day-1 health line:** merges landed · same-class L3/L4 reds · claim slips.

### CUDA merge-test regressions

If CUDA merge-test fails after a merge, promptly determine whether the newly merged change caused the failure. When the regression is confirmed to come from that change, revert it immediately; do not wait for a same-class pattern or the next daily review. Distinguish infrastructure failures and flaky tests from code regressions, and identify the causal commit before reverting. Record the failing check, causal change, revert, and follow-up in LESSONS. L3 remains a post-merge health check, not an added pre-merge gate.

## Stale-approval nudge

Near-misses are often tip-stale after rebase. Same-day light nudge: “re-approve tip `SHA`” — does not spend the deep-review token.

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
