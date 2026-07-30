---
name: ci-resource-pool-stats
description: "Fetch Buildkite build data for vllm-omni and vllm-omni-npu-ci pipelines and compute per-resource-pool queue wait time and occupancy statistics for the previous day in **Beijing Time (CST, UTC+8)**. **Default output is always an HTML file** unless the user explicitly asks for Markdown or JSON. When the user explicitly says '归档报告', save the generated report under vllm-omni-kanban/data/pool_stats_report, commit it, and push it. Use when the user says 'resource pool stats', 'pool occupancy', 'queue time', 'CI pool stats', 'wait time stats', or 'resource pool report'."
---

# CI Resource Pool Statistics

Compute per-resource-pool **queue wait time** and **occupancy** statistics for Buildkite CI pipelines, covering the previous day in **Beijing Time (CST, UTC+8)** by default. The `--from` / `--to` arguments are interpreted as **CST calendar dates** (each covers 00:00–23:59 CST, which maps to `(date-1) 16:00 UTC` → `date 15:59:59 UTC`).

## What this skill does

For each Buildkite pipeline (`vllm-omni` and `vllm-omni-npu-ci`), the script:

1. Fetches all builds created in the specified CST date range (default: yesterday CST). Internally this becomes a UTC window of `(yesterday-1) 16:00 UTC` → `yesterday 15:59:59 UTC`.
2. For each build, fetches job details (including `scheduled_at`, `started_at`, `finished_at`, and `agent_query_rules`)
3. Groups jobs by **resource pool** (derived from `agent_query_rules` → `queue=<name>`)
4. Computes per-pool statistics:
   - **Queue wait time**: `started_at - scheduled_at` — how long a job waited before an agent picked it up (avg, max, p50, p90)
   - **Job duration**: `finished_at - started_at` — how long a job ran on the agent (avg, total occupancy)
   - **Job count**: number of jobs routed to each pool
5. Builds **hourly time-series** (per-pool avg wait and job count for each **CST** hour 0-23) from each job's `scheduled_at` timestamp
6. Renders **two inline SVG trend charts**: Avg Queue Wait per Hour and Job Count per Hour — each pool drawn as a separate colored polyline with area fill

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

## HTML report structure

The HTML report includes:

1. **Top bar** — title with date range, purple accent (Buildkite CI color)
2. **Summary cards** — four top-level metrics:
   - Total Jobs (across all pools)
   - Avg Queue Wait (across all jobs with wait data)
   - Total Occupancy (sum of all job runtimes)
   - Resource Pools count (with pipeline names)
3. **Per-pool detail table** — one row per pipeline × pool, with columns:
   Pipeline · Resource Pool · Jobs · Avg Wait · Max Wait · P50 Wait · P90 Wait · Avg Duration · Total Occupancy · Total Wait
4. **Hourly Trends panel** — two inline SVG line charts, each showing a 24-hour (CST, UTC+8) time series:
   - **Avg Queue Wait per Hour** — colored polyline per pool, x-axis = 00:00–23:00 CST, y-axis = seconds
   - **Job Count per Hour** — same layout, y-axis = count
   Each pool has a distinct color (purple, blue, green, amber, etc.) with semi-transparent area fill and data-point dots. A shared color legend row beneath the charts maps pool names to swatches.
5. **Legend** — definitions for Wait, Duration, Occupancy, Total Wait, Resource Pool
6. **Source metadata** — pipeline names, the CST date range, and the corresponding UTC window

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

Each Buildkite job has an `agent_query_rules` array. The standard convention is `queue=<pool-name>` entries. The script extracts the pool name from these rules:

- If `agent_query_rules` contains `{"rule": "include", "query": "queue=gpu-h200"}`, the pool is `gpu-h200`.
- If `agent_query_rules` is a simple string list like `["queue=gpu-h200"]`, the pool is `gpu-h200`.
- If no queue rule is found, the job is assigned to the `default` pool.
- Some API responses also include a convenience `queue` field, which is used as a fallback.

See [references/buildkite_api.md](references/buildkite_api.md) for details on the Buildkite API endpoints and job object fields used.

## Workflow

1. Confirm `BUILDKITE_API_TOKEN` is set. If not, tell the user to set it and stop.
2. Determine the date range. Default: yesterday **CST (UTC+8)**, interpreted as a full Beijing-time calendar day (00:00 — 23:59 CST). The user may specify `--from` / `--to` (also CST dates).
3. Run `scripts/resource_pool_stats.py` with the desired options.
4. **Default**: the script writes an HTML file to the current directory. Tell the user the file path and suggest they open it in a browser.
5. If the user explicitly asks for Markdown or JSON, pass `--format markdown` or `--format json`.
6. If the user asks about a specific pipeline, pass `--pipeline <slug>`.
7. If the user wants more detail, pass `--verbose`.

## Error handling

- If `BUILDKITE_API_TOKEN` is not set: print an error and exit with code 1.
- If the Buildkite API returns a 429 (rate limit): the script retries with the `Retry-After` header value.
- If a pipeline has no builds in the date range: the HTML table shows "No builds found in the specified date range."
- If builds lack job details: the script refetches each build individually to include `jobs[]`.
