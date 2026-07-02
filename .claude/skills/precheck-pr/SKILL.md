---
name: precheck-pr
description: Self-check your branch before creating a PR — catch dead code, verify accuracy/perf claims, validate PR title format, and confirm merge readiness. Use when the user says "precheck", "self review", "pre-submit check", or "check my PR before I open it." Never posts to GitHub.
---

# PR Pre-Check

Self-review your branch before creating a PR against `vllm-project/vllm-omni`. Two modes: **quick** catches showstoppers, **full** does a thorough maintainer-grade review. Never posts to GitHub; the report is for the contributor's terminal only.

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

**Also run the Code-Quality sweep on every PR**, regardless of type or mode: the five diff-scoped checks in [references/code-quality.md](references/code-quality.md) — kwargs fragility, broad-except swallow, `Any`/wrong type hints, hot-path `.clone()`/`deepcopy`, and event-loop blocking — plus the advisory conventions (log level, structured logging, synchronization, cleanup, dependencies, naming) in [checklists.md](references/checklists.md). These count only lines the PR *adds* — the pre-existing backlog across the repo is out of scope.

### Step 5: Print Report

```
Pre-check report for <branch>

  Mode: quick | full
  Type: <new-model | diffusion-model | bug-fix | perf | general>

  Dimension          Result
  ─────────────────  ──────
  PR title format    ✓
  Code quality       ⚠ 1 broad except, 2 Any hints
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
