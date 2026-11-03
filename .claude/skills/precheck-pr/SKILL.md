---
name: precheck-pr
description: Self-check your branch before creating a PR — catch dead code, prevent new model-specific Python examples, verify accuracy/perf claims, validate PR title format, and confirm merge readiness. Use before implementing a contribution and when the user says "precheck", "self review", "pre-submit check", or "check my PR before I open it." Never posts to GitHub.
---

# PR Pre-Check

Self-review your branch before creating a PR against `vllm-project/vllm-omni`. Two modes: **quick** catches showstoppers, **full** does a thorough maintainer-grade review. Never posts to GitHub; the report is for the contributor's terminal only.

## Contributor intent gate

Before self-checking, read the relevant entries in
[Review and issue lessons](../../../docs/contributing/ci/lessons.md). Verify their
linked evidence against the current change; apply only relevant checks.

Run this gate before implementing a contribution and again before submission
if its intent or scope has changed. It applies to contributor self-review,
not independent review of somebody else's PR or automated review selection.

The human contributor must explicitly answer these four questions:

1. What outcome do you want to achieve?
2. What is the highest-leverage variable for that outcome?
3. How will this work move that variable?
4. Whose customer or user needs does this address, and what are those needs?

Reuse explicit human answers already supplied for this task or in its issue;
do not ask the contributor to repeat them. An agent must not invent answers,
infer approval from a diff, or fill gaps with its own rationale. Ask the human
for any missing or materially ambiguous answers and wait before implementing
or submitting the contribution. Read-only investigation may continue to make
the questions concrete.

Before submission, verify that the PR description answers all four required
questions in the [PR template](../../../.github/PULL_REQUEST_TEMPLATE.md),
or provides public-safe references to the human's explicit answers. Blank
sections, unchanged placeholders, and agent-written guesses do not satisfy
the gate. Missing or ambiguous answers block submission and readiness for
review; ask the human and wait. This is a contributor workflow requirement,
not a claim that GitHub or CI mechanically enforces the template.

Once the answers are clear, ask whether achieving the outcome actually needs
a PR. Prefer an existing capability, configuration change, documentation or
support answer, or no change when that meets the need. Documentation changes
may themselves need a PR. If no PR is needed, stop the submission workflow and
explain the smaller next step. If a PR is needed, connect its narrow scope and
validation plan to the human's stated outcome and variable.

For a bug fix, establish an actual failure: record expected versus observed
behavior, the affected revision and environment, and supporting evidence such
as a minimal reproducer, failing regression test, or observed failure trace.
Do not treat a plausible diagnosis or an unexecuted reproducer as verified
failure evidence. If the failure is not established, report the evidence gap
and continue investigation rather than claiming a confirmed fix.

Keep the answers and their source in the self-review report. Do not copy
private task conversations into a public issue or PR; share an intent summary
only within the audience and information-sharing scope the human authorized.

## Mode Selection

| Mode | When | Time |
|------|------|------|
| **Quick** | About to push, final sanity check | ~3 min |
| **Full** | Ready for review, want maintainer-level scan | ~10 min |

Default to quick if unsure. Run full before marking a PR "ready for review."

## Workflow

### Step 1: Detect Base Branch

```bash
BASE_SHA=$(git merge-base HEAD origin/main 2>/dev/null \
         || git merge-base HEAD main 2>/dev/null \
         || echo origin/main)
echo "diffing against ${BASE_SHA}"
git diff --name-only ${BASE_SHA}...HEAD
```

### Step 2: Validate PR Title

Check the most recent commit message (or branch name if no commit yet) against the project convention. Valid prefixes:

| Prefix | Applies to |
|--------|-----------|
| `[Bugfix]` | Bug fixes |
| `[CI/Build]` | Build or CI improvements |
| `[Doc]` | Documentation changes |
| `[Model]` | New/improved models (include model name) |
| `[Frontend]` | Frontend changes (API server, OmniLLM class, etc.) |
| `[Kernel]` | CUDA/kernel changes |
| `[Core]` | Core logic changes (OmniProcessor, OmniARScheduler, etc.) |
| `[Hardware][Vendor]` | Hardware-specific (e.g., `[Hardware][Ascend]`) |
| `[Misc]` | Other changes (use sparingly) |

✗ if: missing prefix, wrong case (`[bugfix]`), or WIP/Draft in title.
⚠ if: `[Model]` prefix without the model identifier (e.g., `[Model] Add new model` — should be `[Model] Add <ModelName> ...`).

### Step 3: Categorize the PR

| Diff contains | PR type |
|---------------|---------|
| New files under `vllm_omni/model_executor/models/<name>/` | **New Model** |
| Changes to `vllm_omni/diffusion/` | **Diffusion Model** |
| `[Bugfix]` prefix or single-file fix | **Bug Fix** |
| Perf/benchmark/throughput claims in commit msg or diff | **Performance** |
| Everything else | **General** |

If multiple rows apply (e.g., a diffusion model is also a new model), union the checklists.

### Step 4: Run Checklist

Ask: "Quick mode or full mode?" Then walk the checklist for the detected PR type from [references/checklists.md](references/checklists.md). Each item produces ✓, ✗, or ⚠.

**Also resolve affected design contracts** using the canonical review
[design-contract resolver](../review-pr/references/process/design-contracts.md).
Use the candidate head's existing module and feature pages; update an affected
active contract in the same PR. Check user guides and examples when public
configuration, defaults, or supported behavior changes. Treat draft invariants
as review questions, not automatic blockers. In the PR description, summarize
before/after behavior, the affected documentation (or why it is unchanged),
and the supporting regression evidence. Report checks not run explicitly.

**Also run the Code-Quality sweep on every PR**, regardless of type or mode: the five diff-scoped checks in [references/code-quality.md](references/code-quality.md) — kwargs fragility, broad-except swallow, `Any`/wrong type hints, hot-path `.clone()`/`deepcopy`, and event-loop blocking — plus the advisory conventions (log level, structured logging, synchronization, cleanup, dependencies, naming) in [checklists.md](references/checklists.md). These count only lines the PR *adds* — the pre-existing backlog across the repo is out of scope.

**Also run the Examples-Policy check on every PR** using [references/examples-policy.md](references/examples-policy.md). Inspect only Python paths introduced by the diff. Treat a new model-, checkpoint-, vendor-, or family-specific Python example as blocking; do not report pre-existing example debt.

**Also run the Simplification pass on every PR** using [find-simplifications](../find-simplifications/SKILL.md). Keep it diff-scoped: inspect added or modified public APIs, state, abstractions, fallback and compatibility paths, data movement, and only the adjacent ownership needed to prove a candidate. Do not expand the pass into a repository-wide audit. Zero candidates is a valid pass; report only evidence-backed opportunities. Mark a candidate as blocking only when it proves a correctness or merge-readiness problem, such as newly unreachable code. Otherwise report it as a warning.

**Also confirm local pre-commit gates** from [docs/contributing/README.md](../../../docs/contributing/README.md#linting). GitHub Actions `SKIP` does not prove they passed. Full new-hook list: Omni SPDX (`vLLM-Omni project`); forbidden imports (stdlib `re`/`base64`, pickle, Hugging Face Hub API, Triton/TileLang); no new `torch.cuda` call sites; shellcheck (macOS/Windows must install the binary); mypy-3.10; test files have CI level + hardware marks; TTS `_tts_model_type` branches in `serving_speech.py` must not increase; markdownlint on `docs/`; Buildkite YAML schema. ✗ for growing `allowed_files` / `ALLOWED_FILES` or raising `MAX_MODEL_TYPE_BRANCHES` without review.

### Step 5: Print Report

```
Pre-check report for <branch>

  Mode: quick | full
  Type: <new-model | diffusion-model | bug-fix | perf | general>

  Dimension          Result
  ─────────────────  ──────
  PR title format    ✓
  Code quality       ⚠ 1 broad except, 2 Any hints
  Examples policy    ✗ new model-specific Python example
  Simplification     ⚠ helper duplicates an existing owner
  PR desc integrity  ✓
  Registry/config    ✓
  Dead code          ⚠ 2 warnings
  Accuracy           ✓
  Benchmark          ✗ missing software versions

  Verdict: 1 blocking | 2 warnings | recommend fixing ✗ before PR
```

**Severity:**

| Mark | Meaning |
|------|---------|
| ✗ | Blocking — fix before opening PR |
| ⚠ | Warning — consider fixing |
| ✓ | Pass |
| — | Skipped (not applicable) |

### Stop Here

Do not post comments, open PRs, or modify files. The report is for the contributor's terminal only.
