---
name: vllm-omni-CI-Inspection
description: Triages vLLM-Omni CI failures from Buildkite API or pasted logs. Single-job mode extracts first error, classifies failures (build/install, pytest, infra/resource, config/credentials, timeout/perf), attributes PR vs pre-existing, and outputs evidence-based hypotheses with <=5-minute verification. Batch mode inventories many nightly logs (full_moon_*, nightly-*), clusters root causes across GPU pools (H800/H100/A100), detects incomplete logs, and supports cross-run comparison. Buildkite auto-fetch supports PR dedup (latest build per branch), scheduled READY/MERGE triage, Feishu alert forwarding, and GitHub issue prefill. Use when CI is red/flaky/slow, user pastes nightly log paths, asks nightly red/green summary, compares H800 vs A100, or mentions L1-L5, timeout, pytest, build. By default excludes main branch builds unless user explicitly asks to include them.
---

# CI Failure & Duration Triage

## Mode Selection

| Scenario | Workflow |
|----------|----------|
| User provides **many** nightly log paths (`full_moon_*.log`, `nightly-*`), wants red/green overview or H800 vs A100 comparison | **Nightly Batch Triage Mode** |
| User provides **one** log, Buildkite URL, or asks to deep-dive **one red job** | **Single Job Mode** (Quick Start below) |
| Batch finds red jobs, user asks to deep-dive specific job(s) | Batch first → **Single Job Report Template** for selected jobs |
| User provides Buildkite URL / token, wants failed builds fetched automatically | **Buildkite Auto-Fetch Mode** |

## Quick Start (Single Job Mode)

When the user reports CI **failure (red) / flakiness / significant slowdown** and provides logs or a Buildkite URL:

1. Focus only on the **first error** (the earliest fatal signal); do not get pulled into downstream cascading errors.
2. Extract: job name/level, stage/step name of the first error, error code/exception stack, whether it timed out, per-stage duration (if available).
3. Classify into one of 5 categories: **build/compile/dependency**, **test failure**, **infrastructure/resource**, **timeout/performance regression**, **config/permission/credentials**.
4. Provide **1–3 root-cause hypotheses**: each must include a "verbatim log evidence snippet", sorted by "verification cost from low to high".
5. Provide a **≤5 minute** minimal verification for **hypothesis 1 only**: specify **environment + command + expected result** (no full pipeline required).

## Nightly Batch Triage Mode

When the user pastes **many** nightly log paths (or a directory of `*.log`), run these four phases **in order**:

### Phase 1: Inventory

Parallel-grep pytest closing lines across all logs (patterns in [references/grep-patterns.md](references/grep-patterns.md)):

```text
=+ .* failed|=+ .* passed|=+ .* error|short test summary
```

Record one row per log: job name (filename minus `.log`), result (`passed` / `failed` / `error` / `skipped` / **incomplete**), counts from summary line, duration from summary parentheses.

**Incomplete** if any of: no pytest summary and no closing `=+ .* passed|failed`; log ends mid-test (`--- Running test:`); line count abnormally short vs peers. See [references/incomplete-logs.md](references/incomplete-logs.md). Never count incomplete as passed.

### Phase 2: First error scan (red/yellow jobs only)

For **failed / error / incomplete** jobs only, grep earliest fatal signal using [references/grep-patterns.md](references/grep-patterns.md). Do not chase downstream cascading tracebacks. Match against [references/vllm-omni-signatures.md](references/vllm-omni-signatures.md) for known nightly patterns.

### Phase 3: Classify & cluster

Classify each first error into the same 5 categories as Single Job Mode. **Cluster** identical error snippets across jobs (e.g. HF mirror 404, SSL self-signed, GatedRepo 403); tag P0/P1/P2 priority.

### Phase 4: Output batch report

Default to batch overview (not per-green-job analysis). Use template below. For user-named red jobs, expand with **Single Job Report Template**.

**Agent execution rules**:

1. Prefer `Grep` over full file reads; parallelize independent paths.
2. Green jobs with clear summary: **do not** read full log.
3. Red/yellow jobs: read first-error context only (±50–100 lines).
4. No PR/SHA → write "unknown"; do not fabricate.
5. User requests Chinese → respond in Chinese.

```markdown
# Nightly Batch Triage (<date/batch/GPU pool>)

- **Log count**: <N>
- **Change**: <PR/SHA or unknown>
- **Shared environment**: <Python/CUDA/path prefix/mirror from logs>

## Overview

| Job | Result | Duration | Notes |
|-----|--------|----------|-------|

## Root-cause clusters

| Priority | Theme | Affected jobs | First error summary |
|----------|-------|---------------|---------------------|

## Red job highlights

Expand **failed / error / incomplete** only: 2–4 lines each (first error excerpt + category).

## Cross-batch comparison (if two batches given)

| Dimension | Batch A | Batch B |
|-----------|---------|---------|

## Recommended actions

P0→P1 actionable items (ops / data prep / code fix / re-run).
```

## Buildkite Auto-Fetch Mode

When the user provides a Buildkite URL or asks to analyze CI failures, automatically fetch logs via API—no manual paste required.

### Environment Requirements

- `BUILDKITE_TOKEN` is set (provided by user or already in env)
- `curl` + `python3` + `jq` available

### Fetch Workflow

1. **Determine time range**:

   **Important**: Always run `date` first to get the current real time, then compute the UTC time range for API queries. Do not rely on conversation timestamps or assumed trigger times.

   Time-slot matching rules (local time CST = UTC+8):

   | Trigger time (CST) | Analysis window (CST) | API created_from (UTC) | API created_to (UTC) |
   |---------|------------|----------------------|---------------------|
   | 7:30 / 8:30 | Yesterday 18:00 → Today 7:30/8:30 | Yesterday 10:00 UTC | Today 00:30/01:30 UTC |
   | 13:30 / 14:30 | Today 7:30/8:30 → Today 13:30/14:30 | Today 00:30/01:30 UTC | Today 06:30/07:30 UTC |
   | 17:30 / 18:30 | Today 13:30/14:30 → Today 17:30/18:30 | Today 06:30/07:30 UTC | Today 10:30/11:30 UTC |

   **UTC conversion formula**: `UTC time = CST time - 8 hours`

   **Safety margin**: Query time range must include **4 hour buffer before** the analysis window and **1 hour buffer after**. This is critical because window matching uses `finished_at` (not `created_at`) — a build whose `created_at` is in window A but `finished_at` is in window B belongs to window B; without the larger front buffer, such builds are missed:
   ```
   API created_from = analysis window start (CST) - 4h → convert to UTC
   API created_to   = analysis window end (CST) + 1h → convert to UTC
   ```

   **Filter then trim precisely**: After the API returns all builds, filter using each build's `finished_at` field (not `created_at`) to the exact target window (convert to CST for comparison). Then keep only builds with `state` in `['failed', 'timed_out']`. This avoids the gap where a build is still `running` when the earlier window triggers (and thus invisible to `state=failed` API filter) but its `created_at` falls before the later window's start (and thus excluded by `created_at`-based window matching). Do not rely solely on API `created_from/created_to` parameters.

2. **List failed builds**:
   ```bash
   # Example: analyze MERGE CI from yesterday 18:00 to today 8:30 (CST)
   # CST 18:00 yesterday = UTC 10:00 yesterday
   # CST 08:30 today = UTC 00:30 today
   # With 4h front buffer + 1h back buffer: from=UTC 06:00 yesterday, to=UTC 01:30 today
   # NOTE: do NOT use state=failed in API query — a build may still be "running"
   # when the trigger fires but finish (as failed) within the target window.
   # Fetch without state filter, then filter by finished_at + state in Python.
   curl -s -H "Authorization: Bearer $BUILDKITE_TOKEN" \
     "https://api.buildkite.com/v2/organizations/vllm/pipelines/vllm-omni/builds?per_page=50&created_from=<FROM_TS>&created_to=<TO_TS>" \
     | python3 -c "
   import sys, json
   from datetime import datetime, timezone, timedelta
   cst = timezone(timedelta(hours=8))
   builds = json.loads(sys.stdin.read())
   # Precise filter: use finished_at (not created_at) for window matching
   target_start = datetime(2026, 6, 28, 18, 0, tzinfo=cst)  # yesterday 18:00 CST
   target_end = datetime(2026, 6, 29, 8, 30, tzinfo=cst)    # today 8:30 CST
   filtered = []
   for b in builds:
       if b.get('state') not in ('failed', 'timed_out'):
           continue
       finished_at = b.get('finished_at')
       if not finished_at:
           continue  # still running, skip
       finished = datetime.fromisoformat(finished_at.replace('Z','+00:00')).astimezone(cst)
       if finished >= target_start and finished <= target_end:
           filtered.append(b)
   ..."
   ```
   - `created_from`/`created_to`: computed from table above, with 4h front + 1h back buffer
   - **Do NOT add `state=failed` to the API URL** — fetch all states, then filter by `finished_at` + `state` in Python
   - After API returns, use Python to filter precisely: `finished_at` within target window AND `state in ['failed', 'timed_out']`

2. **For each build, fetch all jobs and group by CI type**:
   Use `classify_jobs_by_ci_type()` to partition jobs into READY/MERGE/NIGHTLY/WEEKLY segments
   - Check `Upload Xxx Pipeline` step state to determine which CI type segments are active
   - Based on user-specified CI type filter, keep only failed/timed_out jobs in the matching segment

3. **Extract failed job IDs**:
   From filtered jobs, take `id`, filter `state in ['failed', 'timed_out']`
   - **Exclude jobs with `state='broken'`** (skipped segments, not actually executed)

4. **Fetch job log (trimmed mode, save tokens)**:

   **Important**: Do not feed the full job log to the model! A job log may be 5000+ lines (10k+ tokens), but triage only needs the first error vicinity and pytest summary. Preprocess in shell first; extract key snippets before text analysis.

   **Extraction steps** (process immediately after curl with Python):
   ```bash
   curl -s -H "Authorization: Bearer $BUILDKITE_TOKEN" \
     "https://api.buildkite.com/v2/organizations/vllm/pipelines/vllm-omni/builds/<NUM>/jobs/<ID>/log" \
     | python3 -c "
   import sys, json, re
   data = json.loads(sys.stdin.read())
   content = data.get('content', '')
   content = re.sub(r'\x1b\[[0-9;]*m', '', content)  # strip ANSI
   content = re.sub(r'\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\]\s*', '', content)  # strip BK timestamps

   lines = content.split('\n')

   # 1. Find pytest summary (short test summary info through ====)
   summary_lines = []
   in_summary = False
   for line in lines:
       if 'short test summary info' in line.lower():
           in_summary = True
       if in_summary:
           summary_lines.append(line.strip())
           if '====' in line and ('passed' in line.lower() or 'failed' in line.lower()):
               in_summary = False
               break

   # 2. Find first error (first matching keyword line ± 50 lines)
   error_keywords = ['FAILED', 'AssertionError', 'Traceback (most recent call last)',
       'ModuleNotFoundError', 'ImportError', 'RuntimeError', 'EOFError',
       'OOM', 'Out of memory', 'Killed', 'SIGKILL', 'Timeout', 'timed out',
       'Permission denied', 'AccessDenied', 'FileNotFoundError', 'CMake Error',
       'KeyError', 'TypeError', 'ValueError', 'httpx.RemoteProtocolError']

   first_error_lines = []
   first_error_idx = None
   for i, line in enumerate(lines):
       clean = line.strip()
       if any(kw in clean for kw in error_keywords):
           # exclude warnings
           if any(w in clean for w in ['Warning', 'warning', 'Deprecation', 'RuntimeWarning']):
               continue
           first_error_idx = i
           break

   if first_error_idx:
       start = max(0, first_error_idx - 50)
       end = min(len(lines), first_error_idx + 50)
       first_error_lines = [lines[j].strip() for j in range(start, end) if lines[j].strip()]

   # 3. Output only these two sections, not full log
   print('=== short test summary ===')
   for l in summary_lines:
       print(l)
   print()
   print('=== first error context (50 lines before/after) ===')
   for l in first_error_lines[:100]:  # max 100 lines
       print(l)
   "
   ```
   - **Use only the above output (pytest summary + first error context) for text analysis**
   - Do not store full job log in variables or pass to the model
   - pytest summary is usually < 30 lines; first error context ≤ 100 lines; total ~130 lines ≈ 2000 tokens (vs full 5000+ lines ≈ 16000 tokens)

5. **Locate first error**: In cleaned log, search for these keywords; take the earliest matching line:
   - `FAILED`, `AssertionError`, `AssertionError`, `E   assert`
   - `Traceback (most recent call last)`
   - `ModuleNotFoundError`, `ImportError`
   - `RuntimeError`, `EOFError`
   - `OOM`, `Out of memory`, `Killed`, `SIGKILL`
   - `Timeout`, `timed out`
   - `Permission denied`, `AccessDenied`, `401`, `403`
   - `FileNotFoundError`, `CMake Error`, `nvcc fatal`
   - `httpcore.RemoteProtocolError`, `httpx.RemoteProtocolError`
   - Exclude `RuntimeWarning`, `WARNING`, `DeprecationWarning` (non-fatal signals)

6. **Extract pytest summary**: Search from `short test summary info` to the next `====` for all `FAILED` / `ERROR` lines

7. **Failure attribution** (for PR branch builds): Determine whether failure was introduced by the PR, is a pre-existing repo issue, or is infrastructure-related. Workflow:

   a. **Get PR diff**: Infer PR info from build JSON `commit` and `branch` (branch name is usually `<user>:<pr-name>`), fetch diff file list via GitHub API or Buildkite API:
   ```bash
   # If branch format is user:pr-name, use GitHub API for diff
   curl -s "https://api.github.com/repos/vllm-project/vllm-omni/pulls?head=<user>:<pr-name>&state=open" \
     | python3 -c "import sys,json; prs=json.load(sys.stdin); ..."
   # Or compare commit SHA directly against main
   curl -s "https://api.github.com/repos/vllm-project/vllm-omni/compare/main...<SHA>" \
     | python3 -c "import sys,json; d=json.load(sys.stdin); files=[f['filename'] for f in d['files']]; print(files)"
   ```

   b. **Check whether main nightly in the same window has the same error**: Fetch main-branch builds in the same time window (without `state=failed` filter, then filter by `finished_at` + state in Python — same pattern as step 2); compare error patterns (exception type + deduplicated error message):
   ```bash
   curl -s -H "Authorization: Bearer $BUILDKITE_TOKEN" \
     "https://api.buildkite.com/v2/organizations/vllm/pipelines/vllm-omni/builds?branch=main&created_from=<FROM_TS>&created_to=<TO_TS>&per_page=20"
   | python3 -c "
   ...
   for b in builds:
       if b.get('state') not in ('failed', 'timed_out'): continue
       finished_at = b.get('finished_at')
       if not finished_at: continue
       finished = datetime.fromisoformat(finished_at.replace('Z','+00:00')).astimezone(cst)
       if finished >= target_start and finished <= target_end: filtered.append(b)
   ..."
   ```
   - Compare first error exception type and key message (ignore line number differences) to decide "same error"

   c. **Check whether error falls within diff scope**: Extract source file paths from first error traceback; check if they appear in diff file list

   d. **Combined attribution**:

   | Condition | Attribution | Output tag |
   |------|------|----------|
   | error in diff scope + main has no such error | **PR-introduced** | `🔴 PR-introduced` |
   | main nightly also has same error (regardless of diff scope) | **Pre-existing repo issue** | `🟡 Pre-existing` |
   | error unrelated to code (OOM/timeout/permission/network) + main also has it | **Infrastructure issue** | `🔵 Infra/env` |
   | error in diff scope + main also has same error | **PR may worsen pre-existing issue** | `🟠 PR worsens existing` |
   | Cannot determine (no main comparison / unclear traceback) | **Uncertain** | `⚪ Uncertain` |

8. **Output triage report**: For each failed job, generate a report per this skill's output template (including attribution field)

8a. **Pre-fill the issue-filing fields from the build** (the user-reported gap: the existing template did not pull `commit` or `vllm-omni version`, and "Describe the bug" had no log fragments). Before writing the Single Job Report:

   - **vllm-omni version / commit** — pull from `build.commit` (full SHA, prefer short-SHA `commit[:7]`). The same SHA is also the canonical source for "Change" and for the "Has main nightly in same window also failed?" diff (see `ci-github-ci-failure-issues.md` in the `vllm-omni-test-report` skill for the matching issue data source rules). Render as:
     - Markdown: `[vllm-omni @ abc1234](https://github.com/vllm-project/vllm-omni/commit/abc1234)` (short SHA + GitHub commit URL)
     - HTML: same link in an `<a href="https://github.com/vllm-project/vllm-omni/commit/abc1234">` tag

   - **Describe the bug** — populate from the **First error** section's `Stage / step name` + `Excerpt` (which already captures the verbatim log evidence from the trimmed-mode job log fetch in step 4). The flow is:
     1. Take the **Stage / step name** as the bug's location (e.g. `:full_moon: Omni · Function Test with H100`).
     2. Take the **Excerpt** (verbatim lines, ANSI stripped) as the bug's reproduction snippet.
     3. Combine into a one-line headline (≤120 chars) + the snippet block.
     This guarantees the description maps the log to a precise position in the build pipeline, which is what the user expects when filing a real `vllm-project/vllm-omni` issue.

   - **Cross-skill consistency** — the issue's `label:bug` + `label:ci-failure` filter, time window and title-prefix rules match `references/ci-github-ci-failure-issues.md` from the `vllm-omni-test-report` skill. The triage report's "Recommended actions" should call out whether the same error has appeared in `branch=main` in the same window (attribution flow above); if so, the issue should be cross-referenced to the pre-existing main-nightly ticket rather than filed as new.

8b. **One-line summary** at the top of the report (right under the build header) is auto-generated as `<build.commit[:7]> · <job name> · <stage> · <first-error exception class>` (≤160 chars) so the maintainer scanning a daily report batch can match it to a single commit / job pair without scrolling.

### Parallel Fetch Optimization

Multiple job logs can be fetched with parallel curl calls. Recommend batches of 5.

### CI Type Identification & Filtering

**Important**: One build (even on a PR branch) may contain jobs of multiple CI types. vLLM-Omni's Buildkite pipeline uses **`Upload Xxx Pipeline` steps as segment boundaries** to partition CI types.

| CI type | Boundary step | Jobs included | Notes |
|---------|-----------|-----------|------|
| 🟢 **READY CI** | After `Upload Ready Pipeline` until next Upload step | Lightweight PR validation tests | Runs on all PR builds |
| 🟡 **MERGE CI** | After `Upload Merge Pipeline` until next Upload step | Medium validation after merge to main | Jobs run only when `branch=main` + `source=webhook` |
| 🔴 **NIGHTLY CI** | After `Upload Nightly Pipeline` until next Upload step | Full deep tests (full_moon), long-running | PR builds may trigger (if Upload Nightly Pipeline passed) |
| 🟣 **WEEKLY CI** | After `Upload Weekly Pipeline` | Weekly full tests | Rarely used |

#### Core Pattern

**The jobs array is ordered**; CI type is determined by `Upload Xxx Pipeline` step positions:

```
[init steps]
Upload Ready Pipeline ──→  🟢 READY segment (all jobs until next Upload)
Upload Merge Pipeline ──→  🟡 MERGE segment (all jobs until next Upload)
Upload Nightly Pipeline ──→  🔴 NIGHTLY segment (all jobs until next Upload)
Upload Weekly Pipeline ──→  🟣 WEEKLY segment (until end)
```

- **`Upload Xxx Pipeline` `state`** determines whether that segment actually ran:
  - `passed` → jobs in that segment executed
  - `broken` → segment skipped (jobs not executed; state is also `broken`)
  - If an Upload step is missing → that CI type is not in this build

#### Classification Rules (code implementation)

```python
def classify_jobs_by_ci_type(jobs):
    """Partition each job into the corresponding CI type segment by Upload Xxx Pipeline boundaries."""
    pipeline_boundaries = {
        'upload-ready-pipeline': 'READY',
        'upload-merge-pipeline': 'MERGE',
        'upload-nightly-pipeline': 'NIGHTLY',
        'upload-weekly-pipeline': 'WEEKLY',
    }

    result = []
    current_type = 'INIT'  # steps before Upload (init, build image, etc.)

    for job in jobs:
        name = job.get('name', '')
        step_key = job.get('step_key', '') or ''

        # detect boundary step
        is_boundary = False
        for boundary_key, ci_type in pipeline_boundaries.items():
            if step_key == boundary_key or (boundary_key.replace('upload-', '').replace('-pipeline', '').title() + ' Pipeline') in name:
                current_type = ci_type
                is_boundary = True
                break

        if is_boundary:
            # boundary step itself tagged with current type
            result.append({'job': job, 'ci_type': current_type, 'is_boundary': True})
        else:
            result.append({'job': job, 'ci_type': current_type, 'is_boundary': False})

    return result

def classify_single_job_ci_type(job, all_jobs):
    """For a single job, determine CI type from its position in all_jobs."""
    classified = classify_jobs_by_ci_type(all_jobs)
    job_id = job.get('id', '')
    for item in classified:
        if item['job'].get('id', '') == job_id:
            return item['ci_type']
    return 'UNKNOWN'
```

#### Build-Level CI Type Activation

| `Upload Ready Pipeline` state | `Upload Merge Pipeline` state | `Upload Nightly Pipeline` state | Notes |
|------|------|------|------|
| `passed` | `broken` | `broken` | Only 🟢 READY segment ran (typical PR build) |
| `passed` | `broken` | `passed` | 🟢 READY + 🔴 NIGHTLY ran (PR build triggered full_moon) |
| `broken` | `passed` | `passed` | 🟡 MERGE + 🔴 NIGHTLY ran (main merge build) |
| `broken` | `broken` | `passed` | Only 🔴 NIGHTLY ran (scheduled nightly build) |

**Note**: When `Upload Xxx Pipeline` state is `broken`, jobs in that segment are also `broken` (skipped)—**do not include in failure analysis**.

#### User CI Type Filter Parameters

User can specify analyzing failures for specific CI types only:

- `/vllm-omni-CI-Inspection ... ready CI only` — only 🟢 READY segment failed jobs
- `/vllm-omni-CI-Inspection ... merge CI only` — only 🟡 MERGE segment failed jobs
- `/vllm-omni-CI-Inspection ... nightly CI only` — only 🔴 NIGHTLY segment failed jobs
- `/vllm-omni-CI-Inspection ... analyze ready and nightly CI` — multiple types
- Unspecified → **default to READY CI only** (PR authors care most about READY CI failures)

#### Filter Implementation

1. Fetch all jobs for the build (including Upload steps)
2. Use `classify_jobs_by_ci_type()` to partition each job into a CI type segment
3. Keep only user-specified types with `state=failed` or `state=timed_out` (exclude `broken`)
4. Keep only jobs in segments where `Upload Xxx Pipeline` state is `passed` (exclude skipped segments)

```python
# fetch all jobs, group by CI type
classified = classify_jobs_by_ci_type(all_jobs)

# determine active CI type segments (Upload step state=passed)
active_types = set()
for item in classified:
    if item['is_boundary'] and item['job'].get('state') == 'passed':
        active_types.add(item['ci_type'])

# user specified READY CI only
target_types = {'READY'}  # parsed from user command

# filter: active segment + specified type + truly failed jobs only
filtered_jobs = [
    item['job'] for item in classified
    if item['ci_type'] in target_types
    and item['ci_type'] in active_types  # segment must be active
    and not item['is_boundary']          # exclude Upload steps themselves
    and item['job'].get('state') in ['failed', 'timed_out']  # truly failed only
]
```

### Branch Filtering & Build Exclusion

#### Branch Filter Rules

**Core definitions**:
- 🟢 **READY CI** = analyze failed jobs in `Upload Ready Pipeline` segment on **non-main** builds
- 🟡 **MERGE CI** = analyze failed jobs in `Upload Merge Pipeline` segment on **main** builds
- 🔴 **NIGHTLY CI** = analyze failed jobs in `Upload Nightly Pipeline` segment on nightly builds

| User-specified CI type | Branch filter | CI segment |
|------------------|----------|---------|
| `ready CI only` | **Non-main branch** (`branch != 'main'`) | 🟢 READY segment |
| `merge CI only` | **Main branch** (`branch = 'main'`) | 🟡 MERGE segment |
| `nightly CI only` | Main branch nightly builds | 🔴 NIGHTLY segment |
| Unspecified | Default READY CI only (non-main branch) | 🟢 READY segment |

#### Build Exclusion Rules

**Must exclude the following builds** (even if branch matches):
- build `message` is `Scheduled nightly build` → scheduled nightly trigger, not MERGE CI
- build `message` is `Scheduled weekly build` → scheduled weekly trigger, not MERGE CI

Check build JSON `message` field; exclude builds where `message == 'Scheduled nightly build'` or `message == 'Scheduled weekly build'`.

#### Implementation Example

```python
# branch filtering
def filter_builds_by_branch(builds, ci_type):
    if ci_type == 'READY':
        return [b for b in builds if b.get('branch') != 'main']
    elif ci_type == 'MERGE':
        # main branch, but exclude scheduled nightly/weekly builds
        return [b for b in builds
                if b.get('branch') == 'main'
                and b.get('message', '') not in ('Scheduled nightly build', 'Scheduled weekly build')]
    elif ci_type == 'NIGHTLY':
        return [b for b in builds if b.get('message', '') == 'Scheduled nightly build']
    return builds
```

| User specifies a specific build number | No filtering; follow user request |

#### PR Deduplication (highest priority for READY CI)

The same PR (identified by `branch`, usually `user:pr-name`) may trigger multiple builds from repeated commits. **Analyze only the latest build per branch** (`created_at` most recent); skip older builds for that branch.

- If the **latest** build for a branch is `state=passed`, skip that PR entirely (even if earlier builds failed).
- Implementation: after fetching builds in the time window, group by `branch`. For each branch, find the latest build (any state). If latest is passed → drop the branch. If latest is failed → keep only that build for analysis; discard older failed builds on the same branch.

```python
def dedupe_builds_by_branch_latest(all_builds_in_window, failed_builds):
    """Keep at most one failed build per branch: the latest build, only if it failed."""
    from collections import defaultdict
    by_branch = defaultdict(list)
    for b in all_builds_in_window:
        by_branch[b.get('branch', '')].append(b)
    keep = []
    for branch, builds in by_branch.items():
        if not branch or branch == 'main':
            continue
        latest = max(builds, key=lambda x: x['created_at'])
        if latest.get('state') == 'passed':
            continue
        if latest.get('state') in ('failed', 'timed_out'):
            keep.append(latest)
    return keep
```

Fetch `all_builds_in_window` with `state=` omitted (or separate query per branch) so passed latest builds can be detected—not only `state=failed`.

## Input Requirements (list missing items if insufficient)

### Mode A: User provides log text

- Job name and type (deploy/L1–L5/other)
- Log: **50–100 lines before and after first error** (or full log)
- Change: PR link or commit SHA (mark "unknown" if not available)
- CI environment info (runner/image/Python/CUDA, etc.—cite from log if present)

### Mode B: User provides Buildkite URL or token

- Buildkite pipeline URL (e.g. `https://buildkite.com/vllm/vllm-omni/builds`)
- `BUILDKITE_TOKEN` (API access token)
- Time range (default last 24 hours)
- Branch filter preference (default exclude main)

## Decision Tree (branch on first error)

- **Build / compile / dependency install failure**
  - Common signals: `SyntaxError`, compiler `error:`, `ld:`/`linker`, `undefined reference`, `CMake Error`, `nvcc fatal`, `ModuleNotFoundError`, `ImportError`, `pip install` failure, `Failed building wheel`, `No matching distribution found`
- **Test case failure (assertion or in-test exception)**
  - Common signals: pytest `FAILED`, `AssertionError`, `E   assert ...`, traceback pointing to `tests/...` line numbers, or flaky retry still failing
- **Resource/infrastructure anomaly (non-business logic)**
  - Common signals: `Timeout`/`timed out`, `Killed`/`SIGKILL`, `OOM`/`Out of memory`, `No space left on device`, `Disk quota exceeded`, `Connection reset`, `TLS handshake`, `503`/`429`, image pull failure
- **Duration/performance regression (job did not fail but noticeably slower)**
  - Common signals: stage/step duration significantly increased in log/summary; near timeout threshold; or long silence with no output
- **Config / permission / credentials / env var issues**
  - Common signals: `Permission denied`, `AccessDenied`, `401/403`, `missing required env`, `KeyError: <ENV>`, `secret`/`token` not injected, `could not read credentials`, `FileNotFoundError: No such file or directory: 'wget'` and other missing tools

## Output (must use this template)

Rules:

- Root-cause hypotheses must be based on log facts; if no evidence, write "cannot attribute from current log" and list info needed
- Duration issues must specify: **stage name + baseline vs current**
- Multiple hypotheses must be prioritized (usually verification cost low to high)

### Single Job Report Template

> Pre-fill rule (see **Step 8a** above): every placeholder tagged with
> `[from build]` MUST be populated from the fetched `build` / `job` JSON
> before the report is emitted. The commit + Describe-the-bug block is the
> data the user actually files into `vllm-project/vllm-omni` issues, so a
> blank field there is the same as a blank issue body.

```markdown
## CI Triage Report

- **Build #**: <build.number>  `[from build.number]`
- **Build URL**: https://buildkite.com/vllm/vllm-omni/builds/<build.number>  `[from build.web_url]`
- **Branch**: <build.branch>  `[from build.branch]`
- **Build state**: <build.state>  `[from build.state]`
- **vllm-omni version / commit**: [vllm-omni @ `<build.commit[:7]>`](https://github.com/vllm-project/vllm-omni/commit/<build.commit>)  `[from build.commit]`
- **One-line summary**: `<build.commit[:7]> · <job name> · <stage> · <exception class>`  (≤160 chars)
- **Job**: <job.name> / <L1|L2|…> / <trigger type>
- **CI type**: <🟢 READY | 🟡 MERGE | 🔴 NIGHTLY> — determined by `:full_moon:` emoji and `nightly-` step_key
- **Change**: `<PR link or build.commit>`  `[from build.commit + build.pull_request or build.branch]`
- **Diagnosis category**: <build failure | test failure | infrastructure | config/env | timeout/perf regression | other>
- **Attribution**: <🔴 PR-introduced | 🟡 Pre-existing | 🔵 Infra/env | 🟠 PR worsens existing | ⚪ Uncertain>
- **Attribution rationale**: <one sentence, e.g. "error in diff file scope, main nightly has no such error" or "main nightly #11612 in same window also has KeyError: 'op9'">

### Describe the bug

- **One-line headline** (≤120 chars): `<stage / step name>: <exception class>: <message excerpt>`  `[from job.log + build.commit]`
- **Reproduction snippet** (verbatim from the job log, ANSI stripped, ≤40 lines):
```
<extracted from `log content` field via the trimmed-mode fetch in Step 4; minimum 1, maximum 5 lines around the first error context>
```

### First error

- **Location**: <stage or step name>  `[from job.log]`
- **Excerpt**:
```
<verbatim log lines>  `[from log content — same fragment used in "Describe the bug → Reproduction snippet"]`
```

### Duration (if relevant)

- **Abnormal stage**: <stage name>  `[from job.timings or log timestamps]`
- **Baseline vs current**: <e.g. 2min → 10min> or <multiplier>  `[from main nightly history]`
- **Notes**: <log-based only: serial wait/retry/cache miss, etc.>

### Root-cause hypotheses (sorted by verification priority)

**Hypothesis 1 (verify first)**
- **Description**: <one sentence>
- **Evidence**:
```
<log snippet>
```

**Hypothesis 2**
- **Description**: …
- **Evidence**: …

(Hypothesis 3 optional)

### Minimal verification (for hypothesis 1)

- **Environment**: <local / specific image container / minimal CI workflow name>
- **Steps**:
  1. `<command or action A>`
  2. `<command or action B>`
- **Expected result**: <success | reproduce error | duration within range>

### Recommended actions

- If verification **succeeds**: <fix direction or suggested rollback>
- If verification **fails**: <next hypothesis or expand log scope / contact ops>
- If **infra/env**: <adjust parallelism/timeout/resources or ops ticket points>
```

### Batch Summary Template (multiple jobs)

When analyzing multiple failed jobs, output **cluster summary table** first, then **per-job detail table**, then global recommendations.

```markdown
## 🔍 Failure Cluster Summary

### Grouped by root-cause pattern

| Cluster | Builds | Jobs | Core signal |
|------|-------------|-----------|----------|

### Priority ranking (verification cost low to high)

| P | Issue | Impact | Recommended action |
|---|------|--------|----------|

### Per-build per-job analysis

> One build may contain multiple failed jobs; this table expands each failed job's core diagnosis. The `Commit` column is the **vllm-omni** SHA pulled from `build.commit` (short 7-char form) and **must** be filled from the build JSON — see Step 8a / 8b in the workflow above.

| Build # | Commit | Branch | CI type | Job name | Diagnosis | Attribution | First error summary | Hypothesis 1 | Minimal verification |
|---------|--------|--------|--------|----------|----------|------|-------------------|------------|----------|
| <11315> | [vllm-omni @ `<commit[:7]>`](https://github.com/vllm-project/vllm-omni/commit/<commit>) | <branch> | 🟢 READY | <Simple · Diffusion & Model Executor Test> | <test failure> | <🔴 PR-introduced / 🟡 Pre-existing / 🔵 Infra / 🟠 PR worsens / ⚪ Uncertain> | <AssertionError: ...> | <...> | <run pytest xxx locally> |
| <11315> | [vllm-omni @ `<commit[:7]>`](https://github.com/vllm-project/vllm-omni/commit/<commit>) | <branch> | 🔴 NIGHTLY | <:full_moon: Omni · Function Test with H100> | <test failure> | <🟡 Pre-existing> | <...> | <...> | <...> |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

- **CI type**: 🟢 READY / 🟡 MERGE / 🔴 NIGHTLY — per `classify_job_ci_type()` rules
- **Diagnosis category**: build failure / test failure / infrastructure / config/env / timeout/perf regression / other
- **First error summary**: ≤120 chars key info (exception type + first line)
- **Hypothesis 1**: one-sentence highest-priority hypothesis
- **Minimal verification**: one-sentence ≤5 minute verification step

### Global recommendations

- <cross-build common fixes, e.g. add package to Docker image>
- <PRs needing rebase on main> (attributed 🟡 Pre-existing—rebase after main fix)
- <tests with thresholds too strict>
- <🔴 PR-introduced fix direction suggestions>
- <🔵 Infra/env ops intervention points>
```

## Scheduled Tasks & Automation Rules

Use these rules when running CI triage on a **cron schedule** so output format, issue filing, Feishu alerts, and CSV logging stay consistent.

### Scheduled Task Configuration

| Task | Cron | Trigger time | Buildkite pipeline | CI type |
|------|------|--------------|-------------------|---------|
| READY CI | `0 9,11,17,19 * * *` | 09:00 / 11:00 / 17:00 / 19:00 | vllm-omni | Ready CI |
| NPU READY CI | `10 9,11,17,19 * * *` | 09:10 / 11:10 / 17:10 / 19:10 | vllm-omni-npu-ci | Ready CI |
| MERGE CI | `20 9,11,17,19 * * *` | 09:20 / 11:20 / 17:20 / 19:20 | vllm-omni | Merge CI |

- **session_mode**: `new-per-run`
- **timeout_mins**: 60
- **BUILDKITE_TOKEN**: must be set in env or prompt
- **Alert group session_key**: `feishu:oc_929f070b14744291bef4150c0d62deb0:ou_b5fe2c5e00ae0619cf6ae7e0c21321e1`

Apply **PR deduplication** (see above) for READY CI runs.

### Issue Filing & Alert Forwarding

| CI type | 🔴 PR-introduced | Other attribution (pre-existing / infra / uncertain) |
|---------|------------------|------------------------------------------------------|
| READY CI | Do not file issue; do not forward to alert group | File issue; forward to alert group |
| NPU READY CI | Do not file issue; do not forward to alert group | File issue; forward to alert group |
| MERGE CI | File issue; forward to alert group | File issue; forward to alert group |

**Summary**:
- **READY CI**: only PR-introduced failures are silent (PR author fixes); all other attributions → issue + alert
- **MERGE CI**: all failures → issue + alert (merged-to-main problems must be tracked)

### Issue Template

> **Reuse vllm-omni-test-report's Submit-issue format.** The constants and
> field structure live in
> `vllm-omni-test-report/scripts/nightly_local_log_report.py:1129-1148`
> (`VLLM_OMNI_REPO`, `VLLM_OMNI_BUG_ISSUE_TEMPLATE`,
> `VLLM_OMNI_BUG_ENV_FIELD_ID`, `VLLM_OMNI_BUG_CODE_VERSION_FIELD_ID`,
> `VLLM_OMNI_BUG_DESCRIBE_FIELD_ID`, `VLLM_OMNI_BUG_ISSUE_LABELS_CI`)
> and the URL builder at `nightly_local_log_report.py:1816-2004`
> (`_github_issue_submit_script`). Match it exactly so issues filed from
> the alert land in the same shape as issues filed from the nightly /
> release / development HTML reports.

**Title format** (same shape as `issueTitle()` in
`nightly_local_log_report.py:1927-1933`):

```
[Bug]: Nightly / CI failed - <test node id> - <first error keyword summary>
```

NPU context swaps the prefix to `[Bug][NPU]:` — same rule as the
test-report script (the `data-report-context` attribute contains
`npu` ⇒ NPU prefix).

Examples:
- `[Bug]: Nightly / CI failed - tests/tools/test_check_tts_adapter.py::test_gate_passes_on_current_tree - assert 1 == 0`
- `[Bug]: Nightly / CI failed - tests/config/test_omni_config.py::test_diffusion_config_field_classification_covers_current_fields - AssertionError: Extra items: fa_deterministic`

**Labels** (always all three — same as `VLLM_OMNI_BUG_ISSUE_LABELS_CI`):
`bug`, `ci-failure`, `high priority`.

**URL params** (URL-encode each value with `python3 urllib.parse.quote`):

| Param | Source constant | Value |
|-------|-----------------|-------|
| `template` | `VLLM_OMNI_BUG_ISSUE_TEMPLATE` | `400-bug-report.yml` |
| `labels` | `VLLM_OMNI_BUG_ISSUE_LABELS_CI` | `bug,ci-failure,high%20priority` |
| `title` | derived above | URL-encoded title |
| `current-environment` | `VLLM_OMNI_BUG_ENV_FIELD_ID` | the `ci env` `<details>` block (see body section) |
| `code-version` | `VLLM_OMNI_BUG_CODE_VERSION_FIELD_ID` | two `<details>` blocks (vllm + vllm-omni), each with a `text` fenced commit line |
| `bug-description` | `VLLM_OMNI_BUG_DESCRIBE_FIELD_ID` | the Describe-the-bug content (build URL, step URL, failure kind, test node, log reason, analysis, error log excerpt in ```text fenced block, footer) |

> The legacy `body=` param is **not** used — the GitHub issue form
> `400-bug-report.yml` only auto-populates when you set the per-field
> URL params by `id`, not when you set the legacy free-form body. This
> is why the test-report button works and the old alert link did not.

**`current-environment` value** (identical to
`_vllm_omni_bug_env_ci_prefill()`):

```text
<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
ci env
```

</details>
```

**`code-version` value**:

```text
<details>
<summary>The commit id or version of vllm</summary>

```text
<build.vllm_version or "(not found in Buildkite step log)">
```

</details>
<details>
<summary>The commit id or version of vllm-omni</summary>

```text
<build.commit[:7] or "(not found in Buildkite step log)">
```

</details>
```

**`bug-description` value** (matches
`buildDescribeFieldValue()` in
`nightly_local_log_report.py:1887-1925`):

```text
**Buildkite build:** <build.web_url>

**Buildkite step:** [<job.bkStepName>](<job.bkStepUrl>)    (omit if no step URL)

**Failure kind:** pytest FAILED   (or pytest ERROR)
**Test node:** `<job.test_node>`

**Log reason:**
<reason text>

**Analysis:**
<analysis text>

**Error log excerpt:**

```text
<verbatim log lines, ANSI stripped, ≤40 lines>
```

---

*Generated from a CI-Inspection alert. Redact secrets before submitting; complete the checkboxes on GitHub.*
```

**Issue link column rules** (Feishu summary, table column 9):
- Attribution = 🔴 PR-introduced **and** CI type = READY CI → show `N/A`
- Any other attribution, or CI type = MERGE CI → generate clickable prefill link
- Link format: `[File issue](https://github.com/vllm-project/vllm-omni/issues/new?template=400-bug-report.yml&labels=bug,ci-failure,high%20priority&title=<URLEncodedTitle>&current-environment=<URLEncoded>&code-version=<URLEncoded>&bug-description=<URLEncoded>)`
- URL-encode each param with `python3 urllib.parse.quote`

### Diagnosis Taxonomy (root cause, not symptom)

**Required for scheduled runs**: classify by **root cause**, not error symptom.

| Category | Description | Examples |
|----------|-------------|----------|
| 🔴 Product code defect | API/signature change, logic bug, memory leak | Caller breaks after API change |
| 🔴 Test code defect | Insufficient test resources, wrong assertion, bad test config | Mock missing attribute; stale expected count |
| 🔴 Config defect | Field validation, whitelist drift, yaml misconfiguration | New field not added to classification set |
| 🟡 Pre-existing repo defect | Bug already on base branch | Same error on main in same window |
| 🟢 Infrastructure | CI env, network, port conflict | buildkitd disconnect; agent allocation failure |

OOM / TypeError / Timeout are **symptoms**—put them in the First Error summary column, not Diagnosis category.

### Summary Output & Alert Forwarding

The summary table is the **single source of truth**: it is produced once, already filtered, and used for both the triage session output and the alert-group forwarding. There is no separate "full vs filtered" copy.

**Filtering applied at summary-table generation time** (before any output):

| CI type | 🔴 PR-introduced rows | Effect |
|---------|----------------------|--------|
| READY CI / NPU READY CI | **Excluded** from the summary table | PR-author fixes; silent on these |
| MERGE CI | **Kept** — MERGE CI forwards everything | Merged-to-main problems must all be tracked |

If filtering removes every row (all failures were PR-introduced), **send nothing** to the alert group (stay silent).

**Format — do not use markdown tables.** Feishu splits long messages into multiple bubbles; table headers and rows land in separate bubbles and become unreadable. Use one record per line, fields separated by `|`, each line self-contained:

```
📋 CI Triage Summary

Build#11824 | https://buildkite.com/... | Simple · Diffusion Test | test_xxx | 🔴 Product code defect | 🟡 Pre-existing | AssertionError: ... | Hypothesis 1 | [File issue](https://github.com/vllm-project/vllm-omni/issues/new?template=400-bug-report.yml&labels=bug,ci-failure,high%20priority&title=...&current-environment=...&code-version=...&bug-description=...)
Build#11825 | https://buildkite.com/... | Engine Test | test_yyy | 🟢 Infrastructure | 🔵 Infra/env | OOM killed | Hypothesis 2 | [File issue](https://github.com/vllm-project/vllm-omni/issues/new?template=400-bug-report.yml&labels=bug,ci-failure,high%20priority&title=...&current-environment=...&code-version=...&bug-description=...)
```

> Every `[File issue]` link uses the **same** URL shape as the
> `vllm-omni-test-report` skill's Submit-issue button
> (`template=400-bug-report.yml` + form-field IDs `current-environment`,
> `code-version`, `bug-description`) so the GitHub issue form auto-fills
> identically whether the alert or the report fires it.

**Column order**:
1. Failed build number
2. Build URL
3. Failed job name
4. Failed test case (pytest node id)
5. Diagnosis (root-cause taxonomy above)
6. Attribution (🔴 PR-introduced / 🟡 Pre-existing / 🔵 Infra/env / 🟠 PR worsens existing / ⚪ Uncertain)
7. First error summary (≤120 chars)
8. Top hypothesis (one sentence)
9. File issue (clickable link or `N/A`)

**Alert-group forwarding** — forward the (already-filtered) summary table + per-job details directly:

```bash
cc-connect send -s "feishu:oc_929f070b14744291bef4150c0d62deb0:ou_b5fe2c5e00ae0619cf6ae7e0c21321e1" --stdin <<EOF
<filtered summary table and per-job analysis>
EOF
```

No second filtering pass at forward time — filtering already happened when the summary table was built. If the filtered table is empty, send nothing.

### CSV Persistence

Append each scheduled run to CSV. The CSV directory resolves in priority order:

1. `CI_TRIAGE_DATA_DIR` environment variable (if set)
2. Default: `<current working directory>/ci_triage_data/` (created if missing; cron runs unattended, so fall back to this default without prompting)

File name by CI type:

- **READY CI**: `ready_ci.csv`
- **NPU READY CI**: `ready_npu_ci.csv`
- **MERGE CI**: `merge_ci.csv`

CSV columns: `time_window,failed_build,build_url,failed_job,failed_test,diagnosis,attribution,first_error_summary,hypothesis`

- Create file with header if missing; append rows only (no duplicate headers)
- Time window format: `YYYYMMDD_HHMM-HHMM`, e.g. `20260707_1800-0900`
- CSV records **all** analyzed failures (pre-filter, unfiltered), since the CSV is the audit trail — only the summary table and alert forwarding apply the PR-introduced filter

### Root-Cause Evidence Validation

Evidence cited in a hypothesis must be **unique to the failing build**. Before stating a hypothesis, fetch a **passing** build in the same window with the same CI type and job name; confirm the cited log signal appears only in the failure, not in the pass.

### Silent Intermediate Steps

During scheduled runs, do not stream intermediate tool output to chat. Emit the **complete report once** after all analysis finishes.

## File a GitHub issue (reuses `ci-github-ci-failure-issues.md` data rules)

Use this template when the user asks to file a real `vllm-project/vllm-omni` issue
from a Triage Report. The data rules (`bug` + `ci-failure` AND, time window
`created ∈ [stats-from, stats-to] UTC`, exclude PRs, table columns
`Issue #` / `Title` / `Status`) come from the `vllm-omni-test-report` skill's
`references/ci-github-ci-failure-issues.md` — this template is its direct
mirror for the **outgoing** (filing) side.

### Pre-flight checks (do these BEFORE writing the title)

1. **Label set is fixed** — matches `VLLM_OMNI_BUG_ISSUE_LABELS_CI` in
   `vllm-omni-test-report/scripts/nightly_local_log_report.py:1147`:
   always pass `-l bug -l ci-failure -l high priority` to
   `gh issue create`. NPU context drops `high priority` and keeps
   `bug` + `ci-failure` (the test-report script uses `[Bug][NPU]:`
   prefix without the priority label).
2. **Time window** — the issue's `created_at` UTC must fall in
   `[--stats-from, --stats-to]` (same window as Metrics overview + Open
   issues). For an auto-filed issue from a single build, this is the
   `build.created_at`.
3. **Skip PRs** — `gh issue create` only creates issues (not PRs); no extra
   filter needed. But when cross-checking via search, drop any result with
   `.pull_request` non-null.
4. **Duplicate guard** — before filing, run a search to confirm no open
   ticket already covers the same first-error exception + first-error
   message:
   ```bash
   gh search issues "repo:vllm-project/vllm-omni is:issue label:bug label:ci-failure <EXCEPTION_CLASS> \"<FIRST_ERROR_MESSAGE_KEY>\""
   ```
   - If a match exists → don't file; add the new evidence (latest run #,
     latest commit) to the existing ticket's comment and link the build URL.

### Title (canonical form)

> **Reuse the test-report `issueTitle()` shape exactly** —
> `nightly_local_log_report.py:1927-1933`. Title prefix and length
> budget must match so issues filed from the alert are visually
> indistinguishable from issues filed from the nightly/release/dev HTML
> reports. (`<220` chars; `data-report-context` containing `npu` ⇒
> `[Bug][NPU]:` prefix.)

```
[Bug]: Nightly / CI failed - <test node id> - <first error keyword summary>
```

Example: `[Bug]: Nightly / CI failed - :full_moon: Omni · Function Test with H100 — AssertionError on test_qwen_image_matches_diffusers: tensor mismatch at index 0`

### `gh issue create` command (with all three labels)

```bash
gh issue create \
  --repo vllm-project/vllm-omni \
  --template 400-bug-report.yml \
  --title "[Bug]: Nightly / CI failed - <test node id> - <first error summary>" \
  --label bug \
  --label ci-failure \
  --label high priority \
  --body "$(cat <<'EOF'
<issue body — fields below map to 400-bug-report.yml `id`s>
EOF
)"
```

> `gh` does not support `--template` directly; pass it via the
> `--template` CLI flag (added in gh 2.18+) **or** open the URL from the
> Feishu alert summary instead — both paths land on the same form.

### Issue body template (Markdown)

> The body has to **map field-by-field to `400-bug-report.yml` `id`s** so
> the GitHub form auto-populates when the user opens the prefill URL.
> Mirrors `buildEnvFieldValue()` / `buildCodeVersionFieldValue()` /
> `buildDescribeFieldValue()` in
> `vllm-omni-test-report/scripts/nightly_local_log_report.py:1847-1925`.

#### `current-environment` value (id = `current-environment`)

```markdown
<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
ci env
```

</details>
```

#### `code-version` value (id = `code-version`)

```markdown
<details>
<summary>The commit id or version of vllm</summary>

```text
<build.vllm_version or "(not found in Buildkite step log)">
```

</details>
<details>
<summary>The commit id or version of vllm-omni</summary>

```text
<build.commit[:7] or "(not found in Buildkite step log)">
```

</details>
```

#### `bug-description` value (id = `bug-description`)

```markdown
**Buildkite build:** <build.web_url>
**Buildkite step:** [<job.bkStepName>](<job.bkStepUrl>)    (omit line if no step URL)

**Failure kind:** pytest FAILED    (or pytest ERROR)
**Test node:** `<job.test_node>`

**Log reason:**
<reason text>

**Analysis:**
<analysis text>

**Error log excerpt:**

```text
<verbatim log lines, ANSI stripped, ≤40 lines>
```

---

*Generated from a CI-Inspection alert. Redact secrets before submitting; complete the checkboxes on GitHub.*
```

> The previous template (legacy `body=` blob with `### Your current
> environment` / `### Your code version` / `### 🐛 Describe the bug`
> sections) is **removed** because the GitHub form ignores it — only
> per-field URL params by `id` actually populate the form.

### Output table after filing

Per `ci-github-ci-failure-issues.md` table columns (Issue # / Title / Status):

```markdown
| Issue # | Title | Status |
|---------|-------|--------|
| [#<n>](https://github.com/vllm-project/vllm-omni/issues/<n>) | `[Bug]: Nightly / CI failed - <test node id> - <first error summary>` | Open |
```

If a duplicate is found via the search step above: do not file; produce the
**same table row pointing at the existing ticket** with a one-line note
(`referenced: <new build #> / <new commit>`).

## Additional Resources

> Cross-skill pointers — the GitHub-issue data rules are owned by the
> `vllm-omni-test-report` skill. Reuse them when filing from this skill.

- **GitHub issue data source (labels / time / exclude PR / table columns):**
  [../vllm-omni-test-report/references/ci-github-ci-failure-issues.md](../vllm-omni-test-report/references/ci-github-ci-failure-issues.md)
  — the *incoming* side (what to pull for the report's
  `### Analysis (CI Failure)` table) and the *outgoing* side (the labels /
  time window / PR filter for new issues filed from this skill) share the
  same rules — this skill's `## File a GitHub issue` section is the direct
  mirror of that file for the outgoing flow.
- Grep patterns for batch inventory & first error: [references/grep-patterns.md](references/grep-patterns.md)
- Incomplete/truncated log detection: [references/incomplete-logs.md](references/incomplete-logs.md)
- vLLM-Omni nightly common error signatures: [references/vllm-omni-signatures.md](references/vllm-omni-signatures.md)
- L1–L5 levels and directory conventions: `https://github.com/vllm-project/vllm-omni/blob/main/docs/contributing/ci/CI_5levels.md`
