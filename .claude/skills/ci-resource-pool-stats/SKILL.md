---
name: ci-resource-pool-stats
description: "Fetch Buildkite build data for vllm-omni and vllm-omni-npu-ci pipelines and compute per-resource-pool queue wait time and occupancy statistics for the previous day in **Beijing Time (CST, UTC+8)**. The **Per-Pool Detail** section is computed statically from the local vllm-omni git repo (after `git pull`) by parsing `.buildkite/test-*.yml` against `.buildkite/common/ci_mirror_hardwares.yml` — no Buildkite API calls for that section. **Default output is always an HTML file** unless the user explicitly asks for Markdown or JSON. When the user explicitly says '归档报告', save the generated report under vllm-omni-kanban/data/pool_stats_report, commit it, and push it. Use when the user says 'resource pool stats', 'pool occupancy', 'queue time', 'CI pool stats', 'wait time stats', or 'resource pool report'."
---

# CI Resource Pool Statistics

Compute per-resource-pool **queue wait time** and **occupancy** statistics for Buildkite CI pipelines, covering the previous day in **Beijing Time (CST, UTC+8)** by default. The `--from` / `--to` arguments are interpreted as **CST calendar dates** (each covers 00:00–23:59 CST, which maps to `(date-1) 16:00 UTC` → `date 15:59:59 UTC`).

## What this skill does

For each Buildkite pipeline (`vllm-omni` and `vllm-omni-npu-ci`), the script:

1. **Per-Pool Detail section — static YAML (no Buildkite calls).** Runs
   `git pull --ff-only` in the local vllm-omni git repo (default
   `/home/wy/vllm-omni`) and parses `.buildkite/test-*.yml` plus
   `.buildkite/common/ci_mirror_hardwares.yml` to compute, per pipeline ×
   category, which resource pools the test YAML intends to use. This is
   the source of truth for *what pools are wired up* — independent of what
   actually ran on Buildkite.
2. Fetches all builds created in the specified CST date range (default: yesterday CST). Internally this becomes a UTC window of `(yesterday-1) 16:00 UTC` → `yesterday 15:59:59 UTC`.
3. For each build, fetches job details (including `scheduled_at`, `started_at`, `finished_at`, and `agent_query_rules`)
4. Groups jobs by **resource pool** (derived from `agent_query_rules` → `queue=<name>`)
5. Computes per-pool statistics:
   - **Queue wait time**: `started_at - scheduled_at` — how long a job waited before an agent picked it up (avg, max, p50, p90)
   - **Job duration**: `finished_at - started_at` — how long a job ran on the agent (avg, total occupancy)
   - **Job count**: number of jobs routed to each pool
6. Builds **hourly time-series** (per-pool avg wait and job count for each **CST** hour 0-23) from each job's `scheduled_at` timestamp
7. Renders **two inline SVG trend charts**: Avg Queue Wait per Hour and Job Count per Hour — each pool drawn as a separate colored polyline with area fill

## Default output (HTML)

**Unless the user explicitly asks for Markdown or JSON** (e.g. "generate md", "markdown", "json"), **always produce HTML**. The HTML report uses the editorial dashboard theme (CSS variables, dark-mode support, summary cards, styled tables) consistent with the vllm-omni test report suite.

## Intent keywords

- `resource pool stats`, `pool occupancy`, `queue time`, `CI pool stats`
- `wait time stats`, `resource pool report`, `pool statistics`
- `排队时间`, `资源池`, `占用率`

## Prerequisites

- **`BUILDKITE_API_TOKEN`** (or `BUILDKITE_TOKEN`) must be set in the environment.
- Python 3.10+ and `requests` package (`pip install requests`).

## Usage

### Default: yesterday CST, both pipelines, HTML file

```bash
export BUILDKITE_API_TOKEN="..."
python scripts/resource_pool_stats.py
# Writes pool-stats-2026-07-26.html (covering 2026-07-26 00:00 — 23:59 CST) in the current directory
```

### Custom output path

```bash
python scripts/resource_pool_stats.py --output /path/to/report.html
```

### Custom date range (CST calendar dates)

```bash
python scripts/resource_pool_stats.py --from 2026-07-20 --to 2026-07-22
# Each date covers a full CST day (00:00 — 23:59 CST)
```

### Single pipeline

```bash
python scripts/resource_pool_stats.py --pipeline vllm-omni-npu-ci
```

### Markdown output (stdout, only when explicitly requested)

```bash
python scripts/resource_pool_stats.py --format markdown
```

### JSON output (stdout, only when explicitly requested)

```bash
python scripts/resource_pool_stats.py --format json
```

### Verbose (show per-build job counts)

```bash
python scripts/resource_pool_stats.py --verbose
```

### Override the local vllm-omni repo path

The Per-Pool Detail section runs `git pull` and reads `.buildkite/test-*.yml`
from a local vllm-omni clone. Override the default (`/home/wy/vllm-omni`)
if your clone lives elsewhere:

```bash
python scripts/resource_pool_stats.py --repo-path /path/to/vllm-omni
```

### Skip the `git pull` (use whatever's currently on disk)

```bash
python scripts/resource_pool_stats.py --skip-git-pull
```

Useful for offline runs or when the network is restricted. The HEAD SHA
shown in the source meta line will be whatever was checked out locally.

## HTML report structure

The HTML report includes:

1. **Top bar** — title with date range, purple accent (Buildkite CI color)
2. **Summary cards** — five top-level metrics (sourced from Buildkite):
   - Total Jobs (across all pools)
   - Avg Queue Wait (across all jobs with wait data)
   - Total Occupancy (sum of all job runtimes — wall-clock time, not
     accelerator-weighted)
   - **Device-Hours** — Σ duration × accel_count across all jobs,
     broken down by chip family as a sub-line (e.g. `h100 46.0h ·
     l4 10.0h · npu 58.1h`). This is the **workload-normalized**
     counterpart to Total Occupancy: a multi-card job (h100_4 = 4
     cards, 60 min runtime) contributes 4h here vs. 1h for Total
     Occupancy. Use this metric to track day-over-day CI resource
     consumption independent of job-count variance.
   - Resource Pools count (with pipeline names)
3. **Per-Pool Detail panel** — **sourced from local YAML**, not Buildkite:
   - **Pool Usage by CI Category** subcards — one card per CI category
     (ready / merge / nightly / weekly), each listing the per-pool step
     counts for each pipeline within that category. Pipeline subcards show
     the test YAML filename (e.g. `test-nightly.yml`) and the step count,
     and a **`Total`** row at the bottom of every subcard shows the
     pipeline × category-scoped totals. The chips are pipeline-specific:
       - **vllm-omni** → `h100` (Σ h100_1·1 + h100_2·2 + h100_3·3 + h100_4·4)
         and `l4` (Σ l4_1·1 + l4_4·4 + any direct `gpu_*_queue` step)
       - **vllm-omni-npu-ci** → `A2` (Σ a2b3_npu_1·1 + … + a2b3_npu_8·8)
         and `A3` (Σ a3_npu_2·2 + … + a3_npu_16·16)
     Sourced from `.buildkite/test-*.yml` parsed against
     `.buildkite/common/ci_mirror_hardwares.yml`.
   - **Daily Resource Pool Usage (Buildkite)** — daily per-pool table
     restored from the original view: one row per (pipeline × pool) with
     Jobs · Builds · Avg Cards / Build · Device-Hours ·
     Device-Hours / Build · Total Occupancy · Avg Duration ·
     Total Wait · Avg Wait · Max Wait · P50 Wait · P90 Wait. Sourced
     from the Buildkite API for the date window. Two metrics here
     intentionally diverge from "job count":
       - **Avg Cards / Build** = Σ accel_count across the pool's jobs
         ÷ distinct build numbers that touched the pool. A multi-card
         job (e.g. h100_4 = 4 cards) counts as 4 — a pool running
         few-but-large jobs reads heavier than a pool running
         many-but-single-card jobs.
       - **Device-Hours / Build** = Σ duration × accel_count ÷ builds.
         Same weighting idea as Avg Cards / Build but on time instead
         of count.
   - **Device-Hours by Preset** — sits between the Daily table and
     Job-Level Detail. Jobs are grouped by inferred preset name
     (h100_1..h100_4, l4_1, l4_4, a2b3_npu_*, a3_npu_*), then split
     into **two sub-tables by device type**:
       - **GPU sub-table** — h100_* + l4_* presets, share computed
         against the GPU subtotal only.
       - **NPU sub-table** — a2b3_npu_* + a3_npu_* presets, share
         computed against the NPU subtotal only.
     Each sub-table has columns: Preset · Device-Hours · Share · Jobs ·
     Cards · Distribution (a horizontal bar). **Share is intentionally
     computed within each device type** (not against a day-wide total)
     so the smaller NPU footprint doesn't get drowned by GPU totals
     and you can spot within-family drift (e.g. is h100_4 share
     shrinking while h100_2 grows?). Each subhead shows
     `<type> · <total> total · <n> jobs · <m> cards`. Each preset's
     `accel_count` is reconstructed from `(queue, accel_count)` since
     `upload_pipeline.py` strips `mirror_hardwares` after upload.
   - **Job-Level Detail** — split into **one sub-card per pipeline**
     (`vllm-omni`, `vllm-omni-npu-ci`) so each pipeline's busiest jobs
     get their own panel. Within each card, runs are **grouped by Job
     Name** so repeated runs collapse into a single line showing the
     **average** duration (primary sort), total duration, max duration,
     run count, and the set of CI categories (`ready` / `merge` /
     `nightly` / `weekly`) that ran the job. Click a row (or focus +
     Enter/Space) to expand the group and reveal every individual run
     inside an inner table — columns: build #, duration, wait,
     started/finished (CST), the CI category for that specific run, and
     the state. Each pipeline is capped at 50 groups by default
     (`JOB_LEVEL_TABLE_LIMIT`). A **per-pipeline CI-category filter**
     sits on the right side of each card head — four compact checkboxes
     (ready / merge / nightly / weekly, all on by default), scoped to
     that card only so toggling `vllm-omni`'s chips never affects
     `vllm-omni-npu-ci`. Unchecking a chip hides every outer group in
     that card whose union of CI categories has no intersection with
     the remaining checked set; the card sub-line rewrites to reflect
     the new visible-group count (or "no groups match the selected CI
     categories"). CI categories are inferred from the build's branch
     (non-main → `ready`) and source (non-scheduled main → `merge`;
     scheduled main with Sunday CST `created_at` → `weekly`, else
     `nightly`).
   - Source meta line below the table: local repo path, files used, and
     the post-`git pull` HEAD SHA.
4. **Hourly Trends panel** — two inline SVG line charts (Buildkite-driven),
   each showing a 24-hour (CST, UTC+8) time series:
   - **Avg Queue Wait per Hour** — colored polyline per pool, x-axis = 00:00–23:00 CST, y-axis = seconds
   - **Job Count per Hour** — same layout, y-axis = count
   Each pool has a distinct color (purple, blue, green, amber, etc.) with
   semi-transparent area fill and data-point dots. A shared color legend
   row beneath the charts maps pool names to swatches.
5. **Legend** — definitions for Wait, Duration, Occupancy, Total Wait,
   Device-Hours, Resource Pool (notes that the Per-Pool Detail section
   is YAML-derived).
6. **Source metadata** — pipeline names, the CST date range, and the
   corresponding UTC window.

The report supports **dark mode** via `prefers-color-scheme: dark`.

## Output naming (required)

**Always use the report date (`--from` date) in the filename**, following the same convention as the test report suite:

| Do | Don't |
|----|-------|
| `pool-stats-2026-07-22.html` (date from `--from`) | `pool-stats-20260722.html` (no hyphens) |
| Let the script default to `pool-stats-YYYY-MM-DD.html` | Manually pick a different date for the filename |

## Report archival (`归档报告`)

Only activate this behavior when the user explicitly says **`归档报告`**:

1. Generate the report in the requested format (HTML by default) using the required `pool-stats-YYYY-MM-DD.<ext>` filename.
2. Locate the `vllm-omni-kanban` repository. Prefer `/home/wy/vllm-omni-kanban` when it exists; otherwise locate the repository within the current workspace.
3. Write the report to `vllm-omni-kanban/data/pool_stats_report/`. Create the destination directory if it does not exist.
4. Before committing, inspect the repository status. Do not overwrite unrelated existing reports or include unrelated changes in the commit.
5. Stage only the generated report, commit it with a descriptive message such as `data: archive pool stats report for YYYY-MM-DD`, and push the current branch to its configured remote.
6. Report the archived path, commit hash, branch, and push result to the user. If generation, commit, or push fails, report the failure accurately and do not claim that archival completed.

The user's explicit `归档报告` request authorizes the report-specific commit and push. Do not ask for an additional confirmation unless the repository state, target branch, remote, or required operation differs materially from the instructions above.

Without the exact `归档报告` intent, preserve the normal behavior: generate the report in the current directory and do not commit or push anything.

## Resource pool identification

### Buildkite-driven sections (summary cards, hourly trends)

Each Buildkite job has an `agent_query_rules` array. The standard convention is `queue=<pool-name>` entries. The script extracts the pool name from these rules:

- If `agent_query_rules` contains `{"rule": "include", "query": "queue=gpu-h200"}`, the pool is `gpu-h200`.
- If `agent_query_rules` is a simple string list like `["queue=gpu-h200"]`, the pool is `gpu-h200`.
- If no queue rule is found, the job is assigned to the `default` pool.
- Some API responses also include a convenience `queue` field, which is used as a fallback.

See [references/buildkite_api.md](references/buildkite_api.md) for details on the Buildkite API endpoints and job object fields used.

### Per-Pool Detail section (static YAML — no Buildkite API calls)

The Per-Pool Detail panel is computed entirely from the local vllm-omni git
repo. The script:

1. Runs `git pull --ff-only` in the local repo (default
   `/home/wy/vllm-omni`, override with `--repo-path`; disable with
   `--skip-git-pull`).
2. Loads `.buildkite/common/ci_mirror_hardwares.yml` to map each
   `mirror_hardwares: <preset>` name to its `agents.queue` and infers
   the GPU count from the preset name (e.g. `h100_4` → 4, `l4_1` → 1,
   `gpu_1_queue` queue → 1, `gpu_4_queue` queue → 4). NPU presets
   (`a2b3_npu_*`, `a3_npu_*`) are tracked but excluded from the GPU totals.
3. For each pipeline in `PIPELINE_YAML_MAP`, walks every step in the
   pipeline's category YAML files:

   | Pipeline              | Category → YAML file                                              |
   |-----------------------|--------------------------------------------------------------------|
   | `vllm-omni`           | ready → `.buildkite/cuda/test-ready.yml`                          |
   |                       | merge → `.buildkite/cuda/test-merge.yml`                          |
   |                       | nightly → `.buildkite/cuda/test-nightly.yml`                      |
   |                       | weekly → `.buildkite/cuda/test-weekly.yml`                        |
   | `vllm-omni-npu-ci`    | ready → `.buildkite/npu/test-npu-ready.yml`                       |
   |                       | nightly → `.buildkite/npu/test-npu-nightly.yml`                   |

4. For each leaf step (recursing into `group:` → nested `steps:`), the
   resolved pool is determined by precedence:
   - A direct `agents.queue` on the step (used by some custom-pipeline
     steps like the `Custom Pipeline Test` in `test-ready.yml`).
   - Otherwise, look up `mirror_hardwares: <preset>` in
     `ci_mirror_hardwares.yml` and read `agents.queue` from the preset.

   AMD-style list values like `mirror_hardwares: [amdproduction]` are
   skipped (the AMD pipeline uses a Jinja template to derive the queue
   from `agent_pool`, which we don't have here).

The result is a `pipeline × category × preset` step count, rendered as
the **Pool Usage by CI Category** subcards (keyed by preset when set,
else queue) and the per-pool table. The table breaks `mithril-h100-pool`
into its individual `h100_1`, `h100_2`, `h100_3`, `h100_4` rows so the
GPU count is visible, then appends two summary rows per pipeline:
- `h100_total` = h100_1·1 + h100_2·2 + h100_3·3 + h100_4·4
- `gpu_total`  = l4_1·1 + l4_4·4 + h100_total

Because this section reads only files on disk, it works even when
`BUILDKITE_API_TOKEN` is missing — useful for diagnosing "what does the
test YAML *intend* to use" without waiting for the API.

## Workflow

1. Confirm `BUILDKITE_API_TOKEN` is set if you want the Buildkite-driven
   sections (summary cards, hourly trends). The Per-Pool Detail section
   works without the token since it sources from local YAML.
2. Confirm the local vllm-omni repo is reachable at
   `/home/wy/vllm-omni` (or whatever `--repo-path` points to). The script
   runs `git pull --ff-only` here to pick up the latest test YAML.
3. Determine the date range. Default: yesterday **CST (UTC+8)**,
   interpreted as a full Beijing-time calendar day (00:00 — 23:59 CST).
   The user may specify `--from` / `--to` (also CST dates).
4. Run `scripts/resource_pool_stats.py` with the desired options.
5. **Default**: the script writes an HTML file to the current directory.
   Tell the user the file path and suggest they open it in a browser.
6. If the user explicitly asks for Markdown or JSON, pass
   `--format markdown` or `--format json`.
7. If the user asks about a specific pipeline, pass `--pipeline <slug>`.
8. If the user wants more detail, pass `--verbose`.

## Error handling

- If `BUILDKITE_API_TOKEN` is not set: print an error and exit with code 1.
  (Per-Pool Detail still works without the token — only the Buildkite
  summary cards and hourly trends will be empty.)
- If the Buildkite API returns a 429 (rate limit): the script retries with the `Retry-After` header value.
- If a pipeline has no builds in the date range: the HTML table shows "No builds found in the specified date range."
- If builds lack job details: the script refetches each build individually to include `jobs[]`.
- If the local vllm-omni repo is missing or `git pull` fails: the Per-Pool
  Detail panel falls back to whatever's already on disk (still useful) and
  prints a warning to stderr. The script never aborts the whole report
  over a static-YAML failure.
