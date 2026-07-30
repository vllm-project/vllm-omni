---
name: buildkite-ci-daily-analysis
description: "Fetch yesterday's Buildkite builds (in **Beijing Time, CST, UTC+8**) for vllm-omni and vllm-omni-npu-ci pipelines (the default date window is the previous full CST calendar day, 00:00 — 23:59 CST) and analyze per-job and per-build success/failure and duration (with default infra-job filtering for `:pipeline: init`, `:docker: Build image`, `:buildkit: Build and Push`, `:github: Resolve skip-ci`, `Upload … Pipeline`, `Collect results`), then emit an HTML report with interactive **Pipeline**, **Branch**, **CI**, **State**, and **Job Name** filter dropdowns plus a **CI Aggregate** panel broken down by `ready` / `merge` / `nightly` / `weekly` buckets × pipeline (each card has both job-level and build-level stats). **Default output is always an HTML file** unless the user explicitly asks for Markdown or JSON. When the user explicitly says '归档报告' (or 'archive' / '提交报告' / 'push report'), copy the report to `vllm-omni-kanban/data/ci_monitor/`, commit it, and push to `origin`. Use when the user says 'daily CI analysis', 'today's CI jobs', 'nightly status', 'CI health check', '今日 CI 分析', '今日 job 分析', '今日 build 分析', 'Buildkite 当日分析', or '归档报告'."
---

# Buildkite Daily CI Analysis

Analyze **yesterday's** Buildkite builds in **Beijing Time (CST, UTC+8)**
across `vllm-omni` and `vllm-omni-npu-ci` and emit a self-contained HTML
report with interactive **Pipeline** / **Branch** / **CI** / **State** /
**Job Name** filter dropdowns plus a **CI Aggregate** panel broken down by
`ready` / `merge` / `nightly` / `weekly`.

> **Why "yesterday" by default?** "Yesterday" is interpreted in **Beijing
> Time (CST)** — i.e. the previous full CST calendar day, 00:00 — 23:59 CST,
> which maps to `(yesterday-1) 16:00 UTC` → `yesterday 15:59:59 UTC`. So
> "today's report" naturally maps to "yesterday CST" so the report is
> reproducible as a daily artifact. Pass `--today` for the current CST day.
>
> **Note on the 18:00 UTC nightly run:** under the old UTC default, the
> nightly job triggered at 18:00 UTC fell into "yesterday UTC". Under the
> new CST default it falls at 02:00 CST on the next calendar day and is
> therefore part of that day's window, not yesterday's. If you need to
> inspect the previous night's nightly in isolation, pass an explicit
> `--date` covering the relevant CST day.

## What this skill does

For each pipeline (`vllm-omni` and `vllm-omni-npu-ci`), the script:

1. Fetches every build created in the requested **CST calendar day**,
   which maps to UTC `(date-1) 16:00 UTC` → `date 15:59:59 UTC` (default:
   **yesterday CST**; pass `--today` or `--date YYYY-MM-DD` to override).
2. Refetches each build that didn't return `jobs[]` inline.
3. Walks every `script` / `command` job and records:
   - pipeline slug, branch, build number, commit
   - job name, job state, exit status, job URL
   - `started_at`, `finished_at`, computed `duration = finished_at − started_at`
4. Classifies job state into five buckets: `passed`, `failed`, `canceled`,
   `running`, `other` (`scheduled`/`blocked`/`skipped`/`not_run`/`broken`).
5. Builds three layers of aggregates:
   - **Top-level summary cards** — total jobs, passed, failed, success rate,
     average duration across all pipelines.
   - **Per-pipeline aggregate** — one card per pipeline with counts, success
     rate, and avg / P50 / P90 / max duration.
   - **Per-branch aggregate** — one card per branch sorted by job count.
6. Renders a **filterable job table** — every `<tr>` carries
   `data-pipeline` and `data-branch` attributes so vanilla JS can filter
   live in the browser.

## Default output (HTML)

**Unless the user explicitly asks for Markdown or JSON** (e.g. "生成 md",
"markdown", "json"), **always produce HTML**. The HTML report uses the
editorial dashboard theme (CSS variables, dark-mode support, summary cards,
styled tables) consistent with the vllm-omni test report suite and the
`ci-resource-pool-stats` skill.

## Intent keywords

- `daily CI analysis`, `today's CI jobs`, `nightly status`, `CI health check`
- `daily job analysis`, `today's builds`, `today's Buildkite`
- `今日 CI 分析`, `今日 job 分析`, `今日 build 分析`, `Buildkite 当日分析`
- `当日 CI`, `当日 job`, `nightly 状态`, `CI 跑得怎么样`

> Saying "today's report" or "今日" still routes here — the script will
> actually fetch **yesterday CST** (00:00 — 23:59 CST). Pass `--today` if
> you really want today's (in-progress) data.

## Prerequisites

- **`BUILDKITE_API_TOKEN`** (or `BUILDKITE_TOKEN`) must be set in the
  environment.
- Python 3.10+ and the `requests` package (`pip install requests`).

## Usage

### Default: yesterday CST, both pipelines, HTML file

```bash
export BUILDKITE_API_TOKEN="..."
python scripts/ci_daily_analysis.py
# Writes ci-daily-2026-07-26.html (covering 2026-07-26 00:00 — 23:59 CST) in the current directory
```

### Today CST (instead of yesterday)

```bash
python scripts/ci_daily_analysis.py --today
# Writes ci-daily-2026-07-27.html in the current directory
```
```

### Custom output path

```bash
python scripts/ci_daily_analysis.py --output /path/to/report.html
```

### Custom date

```bash
python scripts/ci_daily_analysis.py --date 2026-07-22
```

### Single pipeline

```bash
python scripts/ci_daily_analysis.py --pipeline vllm-omni-npu-ci
```

### Markdown output (stdout, only when explicitly requested)

```bash
python scripts/ci_daily_analysis.py --format markdown
```

### JSON output (stdout, only when explicitly requested)

```bash
python scripts/ci_daily_analysis.py --format json
```

### Verbose (show per-build job counts)

```bash
python scripts/ci_daily_analysis.py --verbose
```

### Keep infrastructure jobs (disable the default skip list)

```bash
python scripts/ci_daily_analysis.py --include-infra
```

### Extend the skip list with custom patterns

```bash
python scripts/ci_daily_analysis.py --exclude-jobs "^:docker: Build image$,^:buildkit: .*"
```

### Infrastructure job filtering

By default, Buildkite orchestration jobs that aren't real CI work are
filtered out so they don't pollute success rate / duration aggregates.
The default skip patterns (case-insensitive regex):

| Pattern                            | Matches                                                |
|------------------------------------|--------------------------------------------------------|
| `^:pipeline:\s*init\s*$`           | `:pipeline: init`                                      |
| `^:docker:\s*Build image\s*$`      | `:docker: Build image`                                 |
| `^:buildkit:\s*Build and Push\b`   | `:buildkit: Build and Push …` (any suffix)             |
| `resolve\s+skip-ci`                | `:github: Resolve skip-ci ...`                         |
| `upload[^\n]*\bpipeline\b`         | any `Upload … Pipeline` job                            |
| `collect[\s_\-]*results?`          | nightly / weekly aggregator steps                      |

Skipped jobs are absent from every aggregate card, the Job-Level Detail
table, and the dropdowns. The source metadata footer at the bottom of
the HTML report shows how many of each were filtered, and a one-line
summary is printed to stderr:

```
Filtered 7 infrastructure job(s): :pipeline: init(4), :github: Resolve skip-ci (docs / skip marks) & upload pipeline(2), :pipeline: Upload pipeline(1)
```

Use `--include-infra` to keep them, or `--exclude-jobs "regex1,regex2"`
to extend the skip list with additional regex patterns (the patterns are
matched as `re.search`, case-insensitive).

## HTML report structure

The HTML report includes:

1. **Top bar** — title with date and pipeline list, purple accent
   (Buildkite CI color).
2. **Summary cards** — five top-level metrics:
   - Total Jobs
   - Passed (with share of all jobs)
   - Failed (with share of all jobs)
   - Success Rate (`passed / (passed + failed)`)
   - Avg Duration (across all jobs with timing data)
3. **Per-Pipeline Aggregate** — one card per pipeline:
   total / passed / failed / canceled / running / other counts;
   success rate; avg / P50 / P90 / max duration.
4. **CI Aggregate** — one card per CI bucket, in fixed order
   (`ready` · `merge` · `nightly` · `weekly`). Each job is classified from
   its parent build via the rules below; same shape as the per-pipeline
   cards.
5. **Job-Level Detail panel** — the filterable table:
   - **Pipeline** dropdown — choose a single pipeline or "All pipelines".
   - **Branch** dropdown — branches update automatically when the pipeline
     filter changes.
   - **CI** dropdown — one of the four CI buckets (`ready`, `merge`,
     `nightly`, `weekly`) or "All CI buckets". Lists only buckets that have
     at least one job surviving Pipeline + Branch.
   - **State** dropdown — raw Buildkite job state (`passed`, `failed`,
     `running`, `canceled`, `scheduled`, `blocked`, `skipped`, `not_run`,
     `broken`, `unknown`) or "All states". Lists only states that survive
     Pipeline + Branch + CI.
   - **Job Name** dropdown — lists only job names that survive the current
     Pipeline + Branch + CI + State combination, with a per-name count
     (`unit-tests (12)`).
   - **Reset** button — clears all five filters.
   - **Visible / total** counter on the right.
   - Table columns: Pipeline · Branch · CI · Build # · Job · State ·
     Duration · Started (CST) · Finished (CST) · Link
   - State is shown as a colored pill (passed/failed/canceled/running/other).
   - The **CI** column shows the per-row CI bucket as a colored pill
     (ready=blue, merge=green, nightly=amber, weekly=purple).
   - Link opens the job in Buildkite in a new tab.
6. **Legend** — definitions for Pipeline, Branch, Job Name, State,
   Duration, CI Bucket, Build #.
7. **Source metadata** — script name, pipeline names, the CST date, and the corresponding UTC window.

The page supports **dark mode** via `prefers-color-scheme: dark`. All
filtering is done client-side with vanilla JS — no external dependencies.

## CI bucket classification

Each job is bucketed using its parent build's `branch` and `message`
fields (logic adapted from the `vllm-omni-test-report` skill):

| Bucket     | Rule |
|------------|------|
| `ready`    | `branch != "main"` |
| `merge`    | `branch == "main"`, ordinary run, not scheduled nightly / weekly |
| `nightly`  | `branch == "main"` AND (`source == "schedule"` OR message contains `"nightly"` OR (`"scheduled"` AND `"build"` in message)), excluding scheduled weekly |
| `weekly`   | `branch == "main"` AND message matches `scheduled\s+weekly` (regex, case-insensitive) |

The `weekly` check runs before `nightly`, so a `"Scheduled weekly"`
build is never double-counted as nightly.

## Output naming (required)

**Always use the report date (`--date`, `--today`, or default yesterday
CST) in the filename**, following the same convention as the test report
suite and `ci-resource-pool-stats`:

| Do | Don't |
|----|-------|
| `ci-daily-2026-07-26.html` (date from `--date` / default) | `ci-daily-20260726.html` (no hyphens) |
| Let the script default to `ci-daily-YYYY-MM-DD.html` | Manually pick a different date for the filename |

## State classification

| API state   | Bucket       | Counts in success rate? |
|-------------|--------------|-------------------------|
| `passed`    | **passed**   | ✅ yes |
| `failed`    | **failed**   | ✅ yes |
| `canceled`  | **canceled** | ❌ excluded (reported separately) |
| `running`   | **running**  | ❌ excluded (still in flight at fetch time) |
| `scheduled` / `blocked` / `skipped` / `not_run` / `broken` | **other** | ❌ excluded |

Excluding canceled/running/other from the denominator prevents preemption
and in-flight jobs from dragging the success rate down.

## Workflow

1. Confirm `BUILDKITE_API_TOKEN` is set. If not, tell the user to set it and stop.
2. Determine the date. **Default: yesterday CST** (the previous full
   Beijing-time calendar day, 00:00 — 23:59 CST). Pass `--today` for the
   current CST day, or `--date YYYY-MM-DD` for an arbitrary CST day.
3. Run `scripts/ci_daily_analysis.py` with the desired options.
4. **Default**: the script writes an HTML file to the current directory.
   Tell the user the file path and suggest they open it in a browser.
5. If the user explicitly asks for Markdown or JSON, pass
   `--format markdown` or `--format json` (printed to stdout).
6. If the user asks about a specific pipeline, pass `--pipeline <slug>`.
7. If the user wants more detail, pass `--verbose`.
8. If the user explicitly says **"归档报告"** (or equivalent — see below),
   follow the [Report archival](#report-archival-归档报告) procedure instead
   of leaving the report in the current directory.

## Intent keywords (archival)

- `归档报告`, `归档`, `archive`, `commit report`, `push report`
- `提交报告`, `推送到 kanban`

## Report archival (`归档报告`)

Only activate this behavior when the user explicitly says **`归档报告`**
(or one of the [intent keywords above](#intent-keywords-archival)):

1. Generate the report in the requested format (HTML by default) using the
   required `ci-daily-YYYY-MM-DD.html` filename.
2. Locate the `vllm-omni-kanban` repository. Prefer `/home/wy/vllm-omni-kanban`
   when it exists; otherwise locate the repository within the current
   workspace.
3. Write the report to `vllm-omni-kanban/data/ci_monitor/`. Create the
   destination directory if it does not exist.
4. Before committing, inspect the repository status. Do not overwrite
   unrelated existing reports or include unrelated changes in the commit.
5. **Fetch / rebase first** if the local branch has diverged from
   `origin/main` (other agents or dashboards may have pushed daily syncs
   since you last pulled). Replay your commit on top of the remote HEAD
   before pushing.
6. Stage only the generated report, commit it with a descriptive message
   such as `data: archive job monitor report for YYYY-MM-DD`, and push the
   current branch to its configured remote.
7. Report the archived path, commit hash, branch, and push result to the
   user. If generation, commit, fetch, rebase, or push fails, report the
   failure accurately and do not claim that archival completed.

The user's explicit `归档报告` request authorizes the report-specific
commit and push. Do not ask for an additional confirmation unless the
repository state, target branch, remote, or required operation differs
materially from the instructions above.

Without the exact `归档报告` (or equivalent) intent, preserve the normal
behavior: generate the report in the current directory and do not commit
or push anything.

## Error handling

- If `BUILDKITE_API_TOKEN` is not set: print an error and exit with code 1.
- If the Buildkite API returns 429 (rate limit): the script retries with the
  `Retry-After` header value.
- If a pipeline has no builds in the date range: the table shows "No script
  jobs found in the specified date range." and the summary cards reflect
  zero counts.
- If builds lack job details: the script refetches each build individually
  to include `jobs[]`.

## Reference

See [references/buildkite_api.md](references/buildkite_api.md) for the
Buildkite REST API endpoints, build / job object fields, and how state and
duration are derived.
