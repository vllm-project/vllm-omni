---
name: vllm-omni-test-report
description: Three report kinds; **default output is always HTML** unless the user explicitly asks for Markdown (.md). **Release** — `scripts/compose_full_report.py` (**Test conclusion**, Buildkite metrics, **Test Result** = Common stack + optional `--log-dir-h*` nightly-style summaries + H100/CI block, Open bugs); use `--format markdown` only when the user wants .md or `patch_report_*.py`. **Development** — `compose_full_report.py --kind development` — same **Test Result** layout as release, but **drops Test conclusion** + the top-level Open issues section and replaces **Metrics overview** with a 3-row snapshot (Outstanding DI, Open Critical Issue, DI Top10 + CI Failure) followed by **DI Top10 + CI Failure** sub-tables and a top-level **遗留事项 (Next Steps)** action table (combined with editable Assignee/Status columns in HTML, persisted via `localStorage`). Each snapshot row turns red (`<span class="dev-snapshot-alert">`) when its threshold is breached: DI > 30; open critical issue > 0; DI Top10+CI Failure row alerts when DI > 30 or any open ci-failure issue exists. Each H200/H800/A100 section under Test Result gets a `#### Performance Data Comparison` subsection (reuses nightly Local Test perf comparison logic; read-only against kanban `docs/assets/charts/*_history.json`; no `prepare_kanban_before_report.py` / no push). **Nightly** — `scripts/nightly_local_log_report.py` from local `nightly_jobs` (fetch: vllm-omni-local-test) plus optional latest Buildkite scheduled nightly when token is set; use `--markdown-report` / `--to-stdout markdown` only when the user asks for Markdown. **Archive (opt-in)** — when the user asks to archive/commit/push, run **`scripts/push_report_to_kanban.py`** then **`scripts/push_kanban_report.py`** (push only in the second step; **requires [gh CLI](https://cli.github.com/)**; prompt install if missing). Use when generating Buildkite release summaries, parsing local nightly_jobs with CI cross-check, or opening https://buildkite.com/vllm/vllm-omni/builds?branch=main for CI documentation.
---

# vLLM-Omni Test Report

## Report types

| Kind | Output | When to use |
|------|--------|-------------|
| **release** | **HTML** (default): `compose_full_report.py` (or `--kind release`) | **Test conclusion** + **Metrics** (only the **bugs (first response)** row is rendered; the `ready` / `merge` / `nightly` / `weekly` CI-category buckets and the `ut` / `ut (exclude models)` rows are dropped so the section stays focused on bug response times + the appended CI issue detection rate row + the appended operator-editable **Device-Hours / Build (7-day avg)** row — see [Device-Hours / Build (7-day avg)](#device-hours--build-7-day-avg--operator-editable-metric-row)) + **Quality Defense Radar** (per-model 5-axis coverage radar across 9 flagship models in a 3×3 grid — Qwen3-Omni, MiniCPM, Qwen-TTS, Qwen-Image, HunyuanImage, HunyuanVideo, Wan, MinimaxH3, Cosmos. Each radar carries 8 clickable segments — 2 single + 3 axes split into GPU/NPU halves of the same circle; GPU halves turn green and NPU halves turn blue on click; NPU halves also carry a dashed outline in the default state; state kept in `localStorage` and mirrored to a `data-quality-on` attribute for Save-As persistence) + **Test Result** (matrix Common stack; H200/H800/A100 optional log roots; H100 = CI nightly) + **Failure Analysis** (per-GPU failure detail with interactive **Status** column — Filed / Not an issue) + **Open issues** (open `label:bug` in stats window, narrowed to `critical` / `high priority` / `medium priority`) + **Next Steps / Outstanding Items** (manual-entry action table with **Add Item** button, persisted via `localStorage` — same implementation as the Development variant). |
| **development** | **HTML** (default): `compose_full_report.py --kind development` | Same **Test Result** layout as release, but **drops** *Test conclusion* + the top-level *Open issues* section; **Metrics overview** is replaced with a 3-row snapshot (Outstanding DI · Open Critical Issue · **DI Top10 + CI Failure**) followed by the **DI Top10** + **CI Failure Issues** sub-tables and a top-level **遗留事项 (Next Steps)** action table (combines DI Top10 + open CI-Failure with editable **Assignee** / **Status** columns; HTML upgrades persist via `localStorage`). Each snapshot row turns red when its threshold is breached (see Development Quick Path below). Use when the audience is development rather than release-gating. |
| **nightly** | **HTML** (default): `nightly_local_log_report.py --html-report …` | **Local** `nightly_jobs` tree (see vllm-omni-local-test) **and** optional **Buildkite** latest scheduled nightly (same log analysis as local; needs token unless `--no-buildkite`). |

## Default output (HTML)

**Unless the user explicitly asks for Markdown** (e.g. “generate md / markdown”, hand-editing, or `patch_report_*.py`), **always produce HTML**: `compose_full_report.py` (default) and `nightly_local_log_report.py` (writes dated HTML by default). Use `--format markdown` or `--markdown-report` / `--stdout` **only** after that explicit request.

**Buildkite `--buildkite-build` policy:** Default behavior of `nightly_local_log_report.py` is to look up the latest **scheduled nightly** build number **per pipeline** (CUDA `vllm-omni` + NPU `vllm-omni-npu-ci` each get their own number). Do **NOT** pass `--buildkite-build` unless the agent has manually diagnosed a stale-build issue — pinning one build number applies it to BOTH pipelines, and NPU has a different build counter than CUDA so the NPU chapter silently 404s. The script's `resolve_latest_scheduled_nightly_number()` now also falls back to the most recent green/red `main` build when no `Scheduled nightly` message matches in the first 50 builds (covers API cache edge cases like the 07-29 incident).

## Report naming (required)

**Always use the UTC calendar date when the report is generated** (`YYYY-MM-DD`), **not** timestamps from synced log directories.

| Do | Don't |
|----|-------|
| `nightly-report-buildkite-latest-2026-06-26.html` (today UTC) | `nightly-report-buildkite-latest-20250628-143022.html` |
| Omit `--html-report` / `--out` — scripts default to dated filenames | Parse `logs/nightly_jobs_20250628-143022` suffix for the report name |
| Optional override: `--report-date YYYY-MM-DD` | Use log dir mtime or Buildkite build `created_at` for the filename |

Shared helpers live in `scripts/report_naming.py`. `push_report_to_kanban.py` archives non-canonical filenames under **today UTC** unless `--date` is set.

**Exception — kanban `manual_*`:** only **`nightly_jobs_local_*`** perf JSON (via **`logs/.kanban_perf_source`**) is copied to kanban; stability **`nightly_jobs_stability_*`** and general **`nightly_jobs_YYYYMMDD-*`** perf stay in **`logs/nightly_jobs`** for the report only. See [kanban-pre-report-prep.md](references/kanban-pre-report-prep.md).

## Agent Quick Path

**Laptop path defaults (required before sync / prep / report):** Before log sync, kanban prep, or nightly HTML on the **laptop**, show and confirm local **`REPO_ROOT`** (`~/vllm-omni`) and **`KANBAN_REPO_ROOT`** (`~/vllm-omni-kanban`) — see [references/confirm-laptop-path-defaults.md](references/confirm-laptop-path-defaults.md). Wait for user **confirm / use defaults** or custom paths before proceeding. (Cluster **`REPO_ROOT`** `/rebase/vllm-omni` is confirmed separately via [vllm-omni-local-test](../vllm-omni-local-test/SKILL.md).)

**Intent keywords:**
- **Nightly**: `nightly`, `nightly report`, `nightly_jobs`, `scheduled nightly`, `generate nightly`, `daily report`.
- **Release**: `release report`, `test report`.
- **Development**: `development report`, `dev report`, `development test report`. When the user explicitly asks for a *development* variant of the test report, pass `--kind development` to `compose_full_report.py`.
- **Markdown opt-in only**: `markdown`, `.md`, `generate md`, `generate markdown`.
- **Archive / push to kanban (opt-in only)**: `archive`, `commit`, `push`, `kanban`, `upload report`. **Do not** push unless the user prompt includes one of these (or an explicit equivalent).

### Archive to [vllm-omni-kanban](https://github.com/hsliuustc0106/vllm-omni-kanban) (after report is written)

When archive/push intent is present, **after** HTML generation:

1. Verify **`gh --version`** and **`gh auth status`**; if `gh` is missing, **stop** and tell the user to install [GitHub CLI](https://cli.github.com/) (`winget install --id GitHub.cli` on Windows) then `gh auth login`.
2. Require local clone of [vllm-omni-kanban](https://github.com/hsliuustc0106/vllm-omni-kanban) — default **`~/vllm-omni-kanban`** ([confirm with user](references/confirm-laptop-path-defaults.md); override via `KANBAN_REPO_ROOT` or `--kanban-repo-root`).
3. Run **`scripts/push_report_to_kanban.py`** to copy + stage **report HTML** and, when prep wrote `.last_manual_dir`, the matching **`data/local_nightly_raw/manual_*`**. Its stdout includes a **Kanban push preview** block (repo, branch, commit message, staged files, diff stat).

   > ⚠️ **`.last_manual_dir` is a local-only marker file** — written by `prepare_kanban_before_report.py` to record which `manual_*` directory should be included in the next archive push. It is **not** part of the report or raw test data and **must NOT be staged or pushed**. The script stages `manual_<date>/` content directly, so the marker stays untracked. If you (or the agent) accidentally `git add` it, run `git restore --staged data/local_nightly_raw/.last_manual_dir` before committing.
4. **Paste the full push preview to the user** (do not summarize as “ready to push”). Ask whether to proceed. Then run **`scripts/push_kanban_report.py`** — the only script that commits and pushes. Use **`--preview-only`** to re-print the preview without attempting push. In a terminal it prompts `[y/N]`; in agent/non-interactive mode it exits with code 3 and prints the full preview again — **ask the user in chat**, then re-run with **`--yes`** after they confirm.
5. Confirm push succeeded; report filenames and paths: [references/kanban-report-archive.md](references/kanban-report-archive.md).

```bash
gh auth status   # required before push
export KANBAN_REPO_ROOT="${KANBAN_REPO_ROOT:-~/vllm-omni-kanban}"   # confirm with user first
export KANBAN_ASSETS_DIR="${KANBAN_ASSETS_DIR:-$KANBAN_REPO_ROOT/docs/assets/charts}"   # required for Days failing column

# 1) Generate report (no push) — default output uses UTC today in filename/title
python scripts/nightly_local_log_report.py \
  --kanban-repo-root "$KANBAN_REPO_ROOT" \
  --kanban-assets-dir "$KANBAN_ASSETS_DIR"

# Or pin archive date explicitly:
# python scripts/nightly_local_log_report.py \
#   --report-date 2026-06-26 \
#   --kanban-repo-root "$KANBAN_REPO_ROOT" \
#   --kanban-assets-dir "$KANBAN_ASSETS_DIR"

# 2) Archive + stage (separate command; prints preview, no push)
python scripts/push_report_to_kanban.py \
  --report ./nightly-report-buildkite-latest-YYYY-MM-DD.html \
  --kanban-repo-root "$KANBAN_REPO_ROOT" \
  --kind nightly

# 3) Push after user confirms (separate command; --yes only after chat confirmation)
python scripts/push_kanban_report.py \
  --kanban-repo-root "$KANBAN_REPO_ROOT"

# Release: compose_full_report.py --out ... then steps 2–3 with --kind release
#   → archives to vllm-omni-kanban/data/release_test_report/
# Development: compose_full_report.py --kind development --out ... then steps 2–3 with --kind development
#   → archives to vllm-omni-kanban/data/development_test_report/
```

Standalone (report already on disk): run `push_report_to_kanban.py` then `push_kanban_report.py` (add `--yes` to the push script only after user confirms).

**Git commit scope:** push the HTML file under `data/nightly_test_report/`, `data/release_test_report/`, or `data/development_test_report/` (depending on `--kind` flag: `nightly` / `release` / `development`) **and**, when nightly prep created one, the matching **`data/local_nightly_raw/manual_*`** directory (perf JSON + job logs). **`docs/assets/test_reports/` is gitignored** in kanban — MkDocs regenerates it from `data/` at `mkdocs serve` / `mkdocs build`; **never** `git add` that directory. Details: [references/kanban-report-archive.md](references/kanban-report-archive.md).

### Nightly Quick Path

Ask for or infer:
- **Report date for filename/title:** UTC **today** when generating (scripts default automatically). **Never** copy from `nightly_jobs_YYYYMMDD-HHMMSS` log dir suffixes.
- Buildkite token in the environment (`BUILDKITE_TOKEN` or `BUILDKITE_API_TOKEN`) unless the user explicitly wants `--no-buildkite`.
- Kanban repo root for baseline comparison: default **`~/vllm-omni-kanban`** ([confirm with user](references/confirm-laptop-path-defaults.md)) or `--kanban-repo-root`.
- **Kanban assets dir for `*_history.json`** (drives the **Daily focus "Days failing" column** and per-model perf baseline table): **must be passed explicitly** as `--kanban-assets-dir "$KANBAN_REPO_ROOT/docs/assets/charts"`. `--kanban-repo-root` does **not** auto-derive `assets_dir`; if omitted, `_compute_history_fail_lookup` reads an empty dir and every focus row shows `Days failing = —`. (Discovered 2026-07-09: a full nightly run had 113 fail-status focus rows but all showed `—` because `kanban_cfg.assets_dir` was `None`.)
- Optional pinned Buildkite build: `--buildkite-build N`.
- Optional local logs: default **`REPO_ROOT=~/vllm-omni`** ([confirm with user](references/confirm-laptop-path-defaults.md)) with `logs/nightly_jobs`, or pass `--log-dir`.

**Before generating the report** (when using kanban for **performance baseline comparison**), run [references/kanban-pre-report-prep.md](references/kanban-pre-report-prep.md) **`scripts/prepare_kanban_before_report.py`**: pull kanban → optional `manual_*` sync → `mkdocs build`.

```bash
export KANBAN_REPO_ROOT="${KANBAN_REPO_ROOT:-~/vllm-omni-kanban}"
export KANBAN_ASSETS_DIR="${KANBAN_ASSETS_DIR:-$KANBAN_REPO_ROOT/docs/assets/charts}"
export REPO_ROOT="${REPO_ROOT:-~/vllm-omni}"
export BUILDKITE_TOKEN=...   # optional; omit with --no-buildkite
python scripts/prepare_kanban_before_report.py
python scripts/nightly_local_log_report.py \
  --kanban-repo-root "$KANBAN_REPO_ROOT" \
  --kanban-assets-dir "$KANBAN_ASSETS_DIR"
```

### Release Quick Path

Ask for or infer:
- Buildkite token in the environment (`BUILDKITE_TOKEN` or `BUILDKITE_API_TOKEN`) - required unless using `--preview`.
- GitHub token (`GITHUB_TOKEN` or `GH_TOKEN`) - recommended for stable issue data.
- Optional stats window: `--stats-from YYYY-MM-DD --stats-to YYYY-MM-DD`.
- Optional GPU local logs: `--log-dir-h200`, `--log-dir-h800`, `--log-dir-a100`.
- Optional output path: `--out ./vllm-omni-test-report-YYYY-MM-DD.html`.

```bash
export BUILDKITE_TOKEN=...  # or BUILDKITE_API_TOKEN
export GITHUB_TOKEN=...     # optional but recommended
python scripts/compose_full_report.py \
  --out ./vllm-omni-test-report-YYYY-MM-DD.html
```

With optional GPU nightly summaries:

```bash
python scripts/compose_full_report.py \
  --log-dir-h200 /path/to/nightly_jobs_h200 \
  --log-dir-h800 /path/to/nightly_jobs_h800 \
  --log-dir-a100 /path/to/nightly_jobs_a100 \
  --out ./vllm-omni-test-report-YYYY-MM-DD.html
```

### Development Quick Path

The **Development** variant shares the **Test Result** layout with `--kind release` but **drops Test conclusion + Open issues + the H100 (CI — Buildkite scheduled nightly) chapter** and replaces **Metrics overview** with a 3-row snapshot focused on outstanding defect inventory + top-DI / open CI-failure triage. Each H200/H800/A100/A3 section in **Test Result** also gains a `#### Performance Data Comparison` subsection (read-only kanban usage — see below).

**Document body layout (Development variant):** The Development HTML/Markdown **starts directly at the `## Metrics overview` section** — there is no leading multi-bullet descriptive preamble (no release/development comparison paragraph, no section-ordering bullet list, no CSS-class explainer). Only a single `* **Report date (UTC):** YYYY-MM-DD*` line sits between the H1 title and the first section so the date is unambiguous in the rendered output. Section ordering remains: Metrics overview → Test Result → Failure Analysis → Performance Data Comparison → **Skip Test Case Monitoring** → **遗留事项 (Next Steps)**.

**Skip Test Case Monitoring (Development variant):** static AST scan of `<vllm-omni>/tests/**` for pytest skips whose reason references a GitHub issue. **`Issue #` is the first column** (then Issue Title / State / Updated, then Test File / Test / Skip Mark / Skip Reason), rows are sorted by issue number, and in **HTML** every site that shares an issue number is folded under **one collapsible group row** (`▸ #N · N sites`) — click the group row or its caret to expand, or use the *Expand all* / *Collapse all* buttons above the table. All groups start collapsed. Markdown output stays a flat, Issue-#-first table. Implementation: `skip_issue_monitor.SKIP_MONITOR_HEADERS` + `release_md_to_html._group_skip_monitor_table_by_issue` / `_SKIP_GROUP_SCRIPT` (CSS in `report_html_theme.RELEASE_MARKDOWN_DOC_CSS`).

Ask for or infer:
- Buildkite token in the environment (`BUILDKITE_TOKEN` or `BUILDKITE_API_TOKEN`) — required unless using `--preview`.
- GitHub token (`GITHUB_TOKEN` or `GH_TOKEN`) — recommended for stable issue data.
- Optional GPU local logs: `--log-dir-h200`, `--log-dir-h800`, `--log-dir-a100`, `--log-dir-a3` (same as release; A3 follows the H200/H800/A100 generation pattern).
- Optional kanban assets source for the per-GPU perf subsection: `--kanban-repo-root <vllm-omni-kanban>`. **Resolution order:** (1) explicit `--kanban-repo-root`, (2) `$KANBAN_REPO_ROOT` env var, (3) `$VLLM_OMNI_KANBAN_ROOT` env var, (4) **default `~/vllm-omni-kanban`** (if exists). Alternatively use `--perf-assets-dir <kanban>/docs/assets/charts` to directly specify the assets directory.
- Optional output path: `--out ./vllm-omni-test-report-development-YYYY-MM-DD.html`.

```bash
export BUILDKITE_TOKEN=...  # or BUILDKITE_API_TOKEN
export GITHUB_TOKEN=...     # optional but recommended
python scripts/compose_full_report.py \
  --kind development \
  --out ./vllm-omni-test-report-development-YYYY-MM-DD.html
```

With optional GPU nightly summaries + perf comparison (Test Result layout is identical to `--kind release` except the H100 / Buildkite scheduled nightly chapter is dropped; per-GPU `#### Performance Data Comparison` is added under H200/H800/A100 when both `--log-dir-h*` and a kanban assets source are supplied):

```bash
python scripts/compose_full_report.py \
  --kind development \
  --log-dir-h200 /path/to/nightly_jobs_h200 \
  --log-dir-h800 /path/to/nightly_jobs_h800 \
  --log-dir-a100 /path/to/nightly_jobs_a100 \
  --log-dir-a3 /path/to/nightly_jobs_a3 \
  --kanban-repo-root /path/to/vllm-omni-kanban \
  --out ./vllm-omni-test-report-development-YYYY-MM-DD.html
```

The 3-row **Metrics overview** snapshot (per spec) is:
1. **Outstanding DI** — sum of per-issue SLO DI for **all** open `label:bug` issues using the same model as the nightly Daily focus (`DI = base × ⌈days_open / slo_days⌉`; see the **Per-issue DI formula (SLO-escalating)** table below); no date filter (cumulative snapshot).
2. **Open Critical Issue** — count of open issues that carry **both** labels `bug` **and** `critical` (AND filter; RFC / Feature tickets tagged only with `critical` are intentionally excluded) + the first 10 issue numbers.
3. **DI Top10 + CI Failure** — replaces the former "Unassigned Open Issue" row. The cell summarises the **top 10 open `label:bug` issues ranked by DI** (SLO-escalating model, same as nightly Daily focus) and **all open `label:bug` + `label:ci-failure` issues**. Full tables render immediately below the snapshot under `### DI Top10 (SLO-escalating: ...)` and `### CI Failure Issues (...)`.

> The previous **merge CI result** and **nightly CI result** rows (Buildkite latest finished merge / scheduled nightly) were removed in this revision. The development variant drops the H100 / Buildkite scheduled nightly chapter from **Test Result** entirely; H100 still appears in the release variant under Test Result.

**Red-alert rules** (each row's cell is wrapped in `<span class="dev-snapshot-alert">…</span>` so it renders red via `.release-doc .dev-snapshot-alert` in `release_html_theme.RELEASE_MARKDOWN_DOC_CSS`):

| Row | Red Threshold |
|-----|---------|
| Outstanding DI | DI > 30 (i.e. `total_tenths > BUG_DI_THRESHOLD_TENTHS = 300` tenths) |
| Open Critical Issue | open `bug` + `critical` count > 0 |
| DI Top10 + CI Failure | total DI > 30 OR any open `label:bug` + `label:ci-failure` exists |

#### Performance Data Comparison — collapsible per-GPU subsections

In `--kind development`, the **`## Performance Data Comparison`** section contains collapsible **`#### H200` / `#### H800` / `#### A100`** subsections whenever:

- The corresponding `--log-dir-h*` is supplied (so the local perf JSON under `nightly_jobs/<result_root>/result_test_*.json` exists), **and**
- A kanban assets dir is reachable. **Resolution order:** (1) `--kanban-repo-root`, (2) `$KANBAN_REPO_ROOT`, (3) `$VLLM_OMNI_KANBAN_ROOT`, (4) **default `~/vllm-omni-kanban`** (if exists). Assets path resolves to `<repo>/docs/assets/charts/*_history.json`.

Each GPU subsection (e.g., `#### H200`) is rendered as a collapsible `<details>` block in HTML, allowing users to expand/fold the performance baseline comparison table per GPU. **Per-model rows (`##### {model_name}`) render as nested `<details>` inside their parent GPU** — so the structure is `#### H200 → ##### BAGEL → ##### Qwen-Image → ##### ...` rather than flat siblings. Implementation reuses the nightly Local Test perf logic end-to-end (`compose_full_report.render_dev_perf_baseline_local_md` → `nightly_local_log_report._buildkite_perf_rows` → `_filter_perf_summary_for_local`; the perf helper accepts a `model_heading_level=` kwarg so model headings stay one level deeper than their parent) but is **strictly read-only**:

- No `scripts/prepare_kanban_before_report.py` is run.
- No `mkdocs build` / no staging / no commit.
- No `push_report_to_kanban.py` / `push_kanban_report.py`.
- The kanban repo (if supplied) is only used for its on-disk `docs/assets/charts/*_history.json` files; `build_assets_perf_summary(..., kanban_repo_root=None)` skips git inspection entirely.

If a GPU has no `--log-dir-h*` the subsection is omitted entirely (no placeholder). If kanban assets are missing the subsection renders a one-line skip note (`"*--kanban-repo-root / --perf-assets-dir not provided — skipping perf baseline comparison …*"`); it never blocks report generation.

After **Skip Test Case Monitoring** the report includes **`## 遗留事项 (Next Steps)`** (top-level, replaces the release variant's *Open issues* section) — a single action table combining:

- all issues in **DI Top10** (top-10 open `label:bug` ranked by DI, SLO-escalating model) and
- all open issues with both `label:bug` + `label:ci-failure` (de-duplicated against DI Top10).

Columns: **Issue** (link) · **Title** · **Priority** · **DI** · **Assignee** · **Status**. In HTML the **Assignee** column becomes an inline editable input (click to edit; persisted via `localStorage`) and the **Status** column becomes a `<select>` with options `Open / In Progress / Blocked / Won't fix / Fixed`. Markdown output keeps `—` placeholders. Implementation: `compose_full_report.render_next_steps_section` / `NEXT_STEPS_HEADERS` / `NEXT_STEPS_STATUS_OPTIONS` + `release_md_to_html._upgrade_next_steps_cells` / `_NEXT_STEPS_ACTION_SCRIPT` (CSS in `report_html_theme.RELEASE_MARKDOWN_DOC_CSS`).

Preview-only (no Buildkite / GitHub / pytest calls): `--preview --kind development` writes `vllm-omni-test-report-development-preview-YYYY-MM-DD.html` (default). The per-GPU subsections are shown as placeholder notes in preview mode; remove `--preview` to populate from real data.

## Nightly report (local logs + optional Buildkite nightly, HTML)

**Prerequisite:** `LOG_DIR` on disk - paths and pytest rules in [references/nightly-local-log-layout.md](references/nightly-local-log-layout.md). To **produce** logs on cluster, follow [vllm-omni-local-test](../vllm-omni-local-test/SKILL.md) (**H200** or **H800**). To **copy** logs to your laptop, use [../vllm-omni-local-test/references/nightly-local-log-fetch.md](../vllm-omni-local-test/references/nightly-local-log-fetch.md) — **required before each sync:** **`rm -rf` local `$REPO_ROOT/logs`**, then pull per **sync scope** (`local` → latest **`nightly_jobs_local_*` only**; `stability` → latest **`nightly_jobs_stability_*` only**; `default` → latest **`nightly_jobs_YYYYMMDD-*` only**; `all` → local + stability + general nightly; auto-detected from the run's `--test-type` when the user started by running tests first) → **merge** into **`logs/nightly_jobs`**.

**Daily focus summary:** Nightly HTML and Markdown place **Daily focus** immediately after the title. It summarizes Buildkite / Local job failures and lists all baseline-backed performance regressions whose normalized `Status` is `fail` across Buildkite and Local; the HTML table supports Model checkbox filtering. A **major performance regression** is any baseline comparison row whose normalized `Status` is `fail` (current threshold: worse than baseline by more than 6%, using metric direction from `kanban_assets_perf_summary.py`). If there are no `fail` rows, the summary may show the worst `normal` rows as observation items; if baseline data is missing, it reports that explicitly instead of guessing.

**Performance baseline comparison (Local + Buildkite):** The Buildkite section reads kanban **`docs/assets/charts/*_history.json`** but **excludes rows already shown under Local Test** when local perf JSON exists; the **Local** section shows only synced `logs/nightly_jobs` cases. Run **`prepare_kanban_before_report.py`** before generating the report (pull → optional `manual_*` sync → `mkdocs build`).

**Full local logs (HTML):** Per-failed-row **View full log** in the Local **Failure analysis** table opens the raw log excerpt for that row in the in-page modal. The previously used top-of-fold "View full log" toggle (which embedded the entire concatenated job log) has been removed from the Local Test section — use the per-row excerpt instead. The Buildkite CUDA / NPU chapter **Failure analysis** rows still ship their **View full log** excerpts (same modal) plus the top-level "Use **View full log** to open the complete step log in a dialog" hint under each Buildkite chapter.
By default the script also pulls **main** latest **scheduled nightly** from Buildkite (vllm/vllm-omni), downloads each reportable step log, and adds **reason / heuristic analysis / excerpts** for failures (same parsing as local). Set **`BUILDKITE_TOKEN`** or **`BUILDKITE_API_TOKEN`** in the environment; use **`--no-buildkite`** for local-only. Optional **`--buildkite-build N`** to pin a build number.

**Buildkite performance baseline comparison (kanban assets):** Nightly report reads precomputed history from `docs/assets/charts/*_history.json`. Refresh via [kanban-pre-report-prep.md](references/kanban-pre-report-prep.md) **`prepare_kanban_before_report.py`** before rendering. Daily report commands should use `--kanban-repo-root <vllm-omni-kanban>` (resolved to `<repo>/docs/assets/charts`).
Optional source checks:
- `--kanban-expected-remote` / `--kanban-expected-branch` add warnings when current/upstream config differs.

**Kanban raw fallback:** By default the report remains read-only against kanban assets. The performance baseline comparison block keeps Data source / Local filter / History / generated_at / Raw data fallback diagnostics out of the rendered output and focuses on per-model baseline rows. To explicitly regenerate kanban assets from raw perf artifacts before rendering, add `--kanban-refresh-from-raw` with `--kanban-repo-root <vllm-omni-kanban>`; optional `--kanban-raw-root <path>` overrides the raw root. This runs kanban-side `scripts/sync_buildkite_raw_model_results.py` for known model groups and then `scripts/generate_charts.py`, so it mutates the kanban checkout under `data/results/` and `docs/assets/charts/`.

**Kanban raw model sync mapping:** Keep `KANBAN_RAW_MODEL_SYNCS` in `scripts/nightly_local_log_report.py` aligned with [vllm-omni-kanban `scripts/mkdocs_hooks.py`](https://github.com/hsliuustc0106/vllm-omni-kanban/blob/main/scripts/mkdocs_hooks.py). Current mapping: `qwen3omni -> qwen3_omni`, `qwen3tts -> qwen3_tts`, `qwen_image -> qwen_image`, `qwen_image_edit -> qwen_image_edit`, `qwen_image_edit_2509 -> qwen_image_edit_2509`, `wan22 -> wan22`. When adding a model, update kanban first, then update both the report script constant and this note.

From **this** skill directory (after [fetch](../vllm-omni-local-test/references/nightly-local-log-fetch.md) and [kanban prep](references/kanban-pre-report-prep.md) into **`$REPO_ROOT`** / **`$KANBAN_REPO_ROOT`**):

```bash
export REPO_ROOT="${REPO_ROOT:-~/vllm-omni}"
export KANBAN_REPO_ROOT="${KANBAN_REPO_ROOT:-~/vllm-omni-kanban}"
export KANBAN_ASSETS_DIR="${KANBAN_ASSETS_DIR:-$KANBAN_REPO_ROOT/docs/assets/charts}"
python scripts/prepare_kanban_before_report.py
export BUILDKITE_TOKEN=...   # optional; omit with --no-buildkite
python scripts/nightly_local_log_report.py \
  --kanban-repo-root "$KANBAN_REPO_ROOT" \
  --kanban-assets-dir "$KANBAN_ASSETS_DIR"
python scripts/nightly_local_log_report.py \
  --kanban-repo-root "$KANBAN_REPO_ROOT" \
  --kanban-assets-dir "$KANBAN_ASSETS_DIR" \
  --kanban-refresh-from-raw
python scripts/nightly_local_log_report.py \
  --kanban-repo-root "$KANBAN_REPO_ROOT" \
  --kanban-assets-dir "$KANBAN_ASSETS_DIR" \
  --no-buildkite
```

Other flags: `--title`, `--buildkite-build`, `--kanban-repo-root`, `--kanban-assets-dir` (required for the Days failing column — see Nightly Quick Path), `--kanban-raw-root`, `--kanban-refresh-from-raw`, `--kanban-expected-remote`, `--kanban-expected-branch`. **Markdown** (only if the user explicitly asks): `--markdown-report`, `--to-stdout markdown`. See `python scripts/nightly_local_log_report.py --help`.

**Daily focus DI card (SLO-escalating model):** The nightly Daily focus HTML grid now includes an **`Outstanding DI`** card as the 5th focus tile, mirroring the development snapshot's first row with a **time-based** SLO-escalation formula. The calculation lives in `nightly_local_log_report.py::_compute_outstanding_di(gh_token)` and is fully decoupled from the development-snapshot implementation in `compose_full_report.py`.

**Source:** paginated `GET /repos/vllm-project/vllm-omni/issues?state=open&labels=bug&per_page=100&page=…` (PRs excluded). Requires `GITHUB_TOKEN` / `GH_TOKEN` env var; if absent the call falls back to unauthenticated REST (more aggressively rate-limited but still functional — the card degrades to `0 open bug(s)` if the API blocks the request). `_resolve_github_token()` reads both env names in order.

**Per-issue DI formula (SLO-escalating):**

| priority         | base | SLO (days) | order |
|------------------|------|------------|-------|
| critical         | 10.0 | 1          | 0     |
| high priority    | 3.0  | 5          | 1     |
| medium priority  | 1.0  | 10         | 2     |
| low priority     | 0.1  | 14         | 3     |
| invalid          | 0.0  | —          | 4     |

For each open issue:
1. **Highest-priority** label wins (invalid → DI=0, unlabelled → DI=0).
2. **DI = base × ⌈days_open / slo_days⌉**, where `days_open` is the number of full days from `issue.created_at` (UTC) to report-time. `⌈·⌉` is `math.ceil`; the per-issue DI is always ≥ 0 — for a freshly created bug `days_open = 0` ⇒ DI = 0 (no SLO fully elapsed yet).

Total Outstanding DI = sum of per-issue DI across all open bugs. **Severity** flips to `focus-card--fail` when the total > 30 (matches the development-snapshot red-alert rule: "DI > 30 ⇒ Pass becomes Fail").

**Example ladder for a `critical` issue (SLO = 1 day):**

| days_open (since `created_at`) | ⌈days/SLO⌉ | per-issue DI |
|--------------------------------|------------|--------------|
| 0                              | 0          | 0            |
| 0.5                            | 1          | 10           |
| 1.0                            | 1          | 10           |
| 1.001                          | 2          | 20           |
| 2.0                            | 2          | 20           |
| 3.0                            | 3          | 30           |

**Card value:** decimal string, trailing zeros stripped (e.g. `12.3`, `0`, `110`). **Card detail:** `N open bug(s); critical=K, high priority=K, medium priority=K, low priority=K; top: priority(DI=X, Nd), priority2(DI=Y, Md), priority3(DI=Z, Kd)...` — the per-issue snippet is sorted by DI descending.

The DI card is implemented in `nightly_local_log_report.py` via `_compute_outstanding_di(gh_token)` + `_render_focus_metric_card("Outstanding DI", …)` in `_render_daily_focus_html`, and the matching `- **Outstanding DI**: \`X\` — N open bug(s); …` line in `_append_daily_focus_markdown` (with a `(alert)` suffix when the total > 30). Both branches share the same `_resolve_github_token()` helper that reads `GITHUB_TOKEN` / `GH_TOKEN` from the process env.

## Release report (Buildkite, HTML)

**Automated (recommended):** from **this** skill directory with `BUILDKITE_TOKEN` or `BUILDKITE_API_TOKEN` set:

```bash
python scripts/compose_full_report.py
# Optional: embed the same grouped Summary as nightly (see nightly-local-log-layout.md):
# python scripts/compose_full_report.py \
#   --log-dir-h200 /path/to/nightly_jobs_h200 \
#   --log-dir-h800 /path/to/nightly_jobs_h800 \
#   --log-dir-a100 /path/to/nightly_jobs_a100
# default: vllm-omni-test-report-YYYY-MM-DD.html in this skill directory
```

**Markdown (opt-in only):** if the user explicitly asks for a `.md` file or needs `scripts/patch_report_*.py`:

```bash
python scripts/compose_full_report.py --format markdown --out ./vllm-omni-test-report-YYYY-MM-DD.md
```

`--format markdown` is for hand-editing or `scripts/patch_report_*.py` (those tools expect `.md`). HTML is produced via `scripts/release_md_to_html.py` internally.

## Overview

Generate a **human-readable test report** ordered as:

1. **Test conclusion** — Checklist table: only **UT coverage…**, **requirements**, **performance**, and **NPU CI** (4 items) are manual **Pass / Fail** in HTML; **Latest GPU CI(L1-L5) pass rate is 100%**, **Remaining DI < 30**, and **No remaining critical issues** are **automatic** (Buildkite: same **ready** (non-main) and **merge** (main non-nightly/weekly) latest **finished** builds as Metrics — any `failed`/`broken` job fails the row; GitHub: open **`label:bug`** whose `created_at` ≤ `--stats-to` (start date unbounded; issues created after the stats window are excluded) weighted by priority labels **DI < 30**; **no** open **`critical`**). Archive/plain Markdown matches HTML.
2. **Metrics overview** — `buildkite_build_stats.py --markdown` generates the main table (Bug avg first response, aligned with **`--stats-from`..`--stats-to`**). The release Metrics overview intentionally drops the CI-category buckets (`ready` / `merge` / `nightly` / `weekly`) and the `ut` / `ut (exclude models)` rows so the section stays focused on bug response times — only the **`bugs (first response, …)`** row is rendered from the upstream script. Below that, `compose_full_report.append_ci_issue_detection_rate_row` adds a **CI issue detection rate** row (share of bugs in the stats window that carry the `ci-failure` label), and `compose_full_report._append_device_hours_build_row` adds a final **Device-Hours / Build (7-day avg)** row whose value is operator-editable — see [Device-Hours / Build (7-day avg)](#device-hours--build-7-day-avg--operator-editable-metric-row) below.
3. **Test Result** — `### Common stack (all rows)` from [references/local-test-matrix.md](references/local-test-matrix.md); `### H200` / `### H800` / `### A100` use the same grouped tables as nightly local **Summary** (pass `--log-dir-h200` / `--log-dir-h800` / `--log-dir-a100`; directories must match [references/nightly-local-log-layout.md](references/nightly-local-log-layout.md)); `### H100 (CI — Buildkite scheduled nightly)` includes **Build** (build number/branch/commit), reportable job **Summary**, **Failed test jobs** (**excludes** per-job pytest detail and **Analysis (CI Failure)**; maintain separately via hand edits or `nightly_job_pytest_table.py` / `patch_report_ci_failure.py` when needed).
4. **Failure Analysis** — Top-level section with one collapsible subsection per GPU (H200 / H800 / A100 from local nightly logs; H100 from Buildkite scheduled nightly). Each failure table has an interactive **Status** column with two buttons — **Filed** / **Not an issue** — backed by `localStorage`. Clicking **Filed** opens an in-page modal where the user enters the GitHub issue number (and an optional note); the cell renders as `Filed #<n>` with a link to the GitHub issue. Clicking **Not an issue** opens the same modal with the issue-number field hidden so the user can record an optional note. The modal has its own **Save** / **Cancel** / **Reset** buttons; Esc closes it (iframe-safe). State is keyed by the row's `data-row-id` (derived from the nearest preceding section heading + row index) so reloads keep the chosen status. Mirrors the Development variant's Failure Analysis layout.
5. **Open issues (stats window)** — Paginated **`label:bug`**, **open**, `created_at` UTC date in **`--stats-from`..`--stats-to`**, further filtered to issues whose highest-priority label is `critical` / `high priority` / `medium priority` (drops `low priority`, `invalid`, and unlabelled-priority bugs; the **Remaining DI < 30** auto row is intentionally **not** narrowed — it sums across every open `label:bug` issue via `slo_open_bug_di_total` to preserve the existing threshold semantics). Precompute daily DI from the same issues: `critical` = 10, `high priority` = 3, `medium priority` = 1, `low priority` = 0.1, `invalid` = 0. The table ends with two **manual triage** columns shared by the release *and* development variants: **Follow-up action** (HTML `<select>`: *Fix in a later iteration* / *Blocked by dependency* / *Won't fix (evaluated)*; empty = not set) and **Remarks** (click the cell to open an inline textarea; Save / Cancel, Ctrl+Enter saves, Esc cancels). Both persist in `localStorage` keyed by the row's **issue number** (`open-issue-followup:#N` / `open-issue-note:#N`), so triage survives report regeneration and is shared between the two report kinds. Markdown output keeps `—` placeholders. Implementation: `compose_full_report.OPEN_ISSUES_HEADERS` + `OPEN_ISSUES_RELEASE_PRIORITIES` (the priority filter set) + `github_open_bug_rows_in_range(..., priority_filter=)` + `release_md_to_html._upgrade_open_issue_action_cells` / `_OPEN_ISSUE_ACTION_SCRIPT` (CSS in `report_html_theme.RELEASE_MARKDOWN_DOC_CSS`).
6. **Next Steps (Outstanding Items)** — Manual-entry action table for the **release** variant. Three columns (**Item** / **Assignee** / **Status**) plus a per-row delete button, seeded with a single placeholder row. Click **Add Item** below the table to append a row; every cell is inline-editable in HTML (`<input>`). All edits persist via `localStorage` keyed by `outstanding-items:<row-uuid>`, surviving report regeneration and reloads on the same origin. H2 appears between Open issues and Data source so it matches the Development variant's ordering; `_release_section_theme` substring match (`"outstanding items"`) auto-applies the `--outstanding` card theme (clipboard SVG + red accent). Implementation: `compose_full_report.render_next_steps_section()` (already used by the Development variant) + `release_md_to_html._upgrade_next_steps_outstanding_cells` / `_NEXT_STEPS_OUTSTANDING_SCRIPT` (already injected unconditionally for both variants).
7. **Quality Defense Radar** — **Release variant only**. A 3×3 CSS Grid of nine inline SVG radars / pentagons — one per flagship model: **Qwen3-Omni, MiniCPM, Qwen-TTS, Qwen-Image, HunyuanImage, HunyuanVideo, Wan, MinimaxH3, Cosmos**. Each radar has **5 axes** (Functionality / Performance / Documentation / Stability / Reliability) arranged clockwise from the top. Three of them (Functionality / Performance / Stability) are split into **GPU + NPU halves of the same circle**, so the GPU half and the NPU half share one circle but are independently clickable; Documentation and Reliability are single full circles. Each radar therefore carries **8 clickable segments** (`func-gpu` / `func-npu` / `perf-gpu` / `perf-npu` / `doc` / `stab-gpu` / `stab-npu` / `rel`), giving **72 clickable targets** across the 9-model grid (segment keys are namespaced as `<model-id>:<segment-id>`). GPU and NPU are visually distinguished three ways so reviewers can tell them apart at a glance: (1) **GPU halves carry a solid outline, NPU halves carry a dashed outline** in the default gray state; (2) on click, **GPU halves turn green** (`#4ade80` fill, `#22c55e` stroke) and **NPU halves turn blue** (`#7dd3fc` fill, `#0284c7` stroke); (3) each split segment carries a `data-qd-side="gpu"|"npu"` attribute for CSS targeting. State is mirrored to a `data-quality-on="1"|"0"` attribute on each `<g>` (so `Ctrl+S` Save-Page-As preserves state across origins) **and** to `localStorage["quality-defense:<model>:<segment-id>"]` for reload persistence; an in-memory `mem = {}` fallback covers Chrome `file://` reloads. H2 appears right after the **Metrics overview** section (and before Test Result); `_release_section_theme` substring match (`"quality defense"` / `"quality radar"`) auto-applies the `--quality-defense` card theme (shield SVG + green accent). Keyboard support: focus a segment via Tab, press Space / Enter to toggle. Implementation: `compose_full_report.render_quality_defense_section()` + `release_md_to_html._upgrade_quality_defense_block` / `_quality_defense_block_html()` (per-model SVG generator with `_qd_model_radar_svg` / `_QUALITY_DEFENSE_MODELS`) + `_QUALITY_DEFENSE_SCRIPT` + CSS in `report_html_theme.RELEASE_MARKDOWN_DOC_CSS` (`.qd-grid`, `.qd-cell`, `.qd-radar`, `.qd-half`, `.qd-circle`, `.qd-segment[data-qd-side="..."][data-quality-on="1"]`).

## When to Apply

- User asks for a **release** / **Buildkite** / **CI** test report for vllm-omni — **default to HTML** via `compose_full_report.py`; use `--format markdown` **only** if they explicitly want Markdown
- User pastes a Buildkite build URL or build number
- User wants to summarize failures, flaky steps, or duration from Buildkite
- User needs **Common stack** or optional **H200/H800/A100** nightly log summaries in the **release** report — set **`--log-dir-h*`** on `compose_full_report.py` when logs are available
- User asks for **nightly** report from **local** `nightly_jobs` — **default to HTML** (`nightly_local_log_report.py --html-report`); Markdown **only** if they explicitly ask (`fetch` in vllm-omni-local-test)
- User asks for a **development** / **dev** variant of the test report (or wants to skip Test conclusion and focus on outstanding defects / latest CI verdicts / unassigned owners) — `compose_full_report.py --kind development`; same Test Result layout as release
- User asks to **archive / commit / push** the report to [vllm-omni-kanban](https://github.com/hsliuustc0106/vllm-omni-kanban) — run [references/kanban-report-archive.md](references/kanban-report-archive.md) **after** HTML is written

## Definitions

| Term | Meaning |
|------|---------|
| **Scheduled nightly build** | A build whose **message/title** contains `Scheduled nightly build` (Buildkite scheduled job). It is **not** the same as arbitrary `main` commits. |
| **Latest** | Among matching builds, the one with the **most recent** `finished_at` or `created_at` (prefer finished if both present). |
| **Reportable job** | Any `jobs[]` entry whose **name** does **not** match `(?i)^Upload .+ Pipeline$` (exclude `Upload Ready/Nightly/Merge Pipeline` and similar upload-only steps). |

## Buildkite authentication (environment only)

- All Buildkite REST calls and skill scripts read **`BUILDKITE_TOKEN`** or **`BUILDKITE_API_TOKEN`** from the **process environment** (either name works for the scripts; use one consistently in CI).
- **Do not** paste the secret into chat, pass it on the command line, or embed it in files committed to git. If unset, ask the user to **export it in their local shell** or **configure it as a CI/secret env var**, then retry.
- Before `curl` or Python helpers, confirm the variable is set in the same shell (e.g. bash: `[ -n "${BUILDKITE_TOKEN:-}" ] || [ -n "${BUILDKITE_API_TOKEN:-}" ]`; PowerShell: `if (-not $env:BUILDKITE_TOKEN -and -not $env:BUILDKITE_API_TOKEN) { ... }`).

## Workflow (release only)

**Test Result (Common stack):** Maintain **`## Common stack (all rows)`** in [references/local-test-matrix.md](references/local-test-matrix.md); H200/H800/A100 sections depend on synced `nightly_jobs` paths passed to `compose_full_report.py` via **`--log-dir-h*`**.

**Metrics overview, H100 (CI), Open issues:** See Steps 2–3 below; **nightly** HTML uses only the **Nightly report** section at the top of this doc.

### Step 1: Resolve the target build (CI testing)

**Option A - API (recommended)**

1. Ensure **`BUILDKITE_TOKEN` or `BUILDKITE_API_TOKEN`** is set in the environment for the session that runs `curl` / scripts.
2. If missing, **do not** ask for the raw token in chat. Prompt the user to set it locally (or in CI), e.g. *"Set read-only Buildkite API token in the environment as `BUILDKITE_TOKEN` or `BUILDKITE_API_TOKEN`, then ask again."* Offer **Option C** (web-only fallback) if they cannot use env-based auth.
3. With the env var set, list builds on `main`:

```bash
curl -s -H "Authorization: Bearer $BUILDKITE_TOKEN" \
  "https://api.buildkite.com/v2/organizations/vllm/pipelines/vllm-omni/builds?branch=main&per_page=30" \
  | jq '.'
```

4. Select the **first** build in the array where `.message` matches `(?i)scheduled nightly` **or** the build is clearly labeled as scheduled in the UI message.
5. If none match, report that no scheduled nightly was found in the page and optionally fall back to the **most recent green/red `main` build** only if the user agrees.

**Option B - User-provided build**

If the user gives a URL like `https://buildkite.com/vllm/vllm-omni/builds/<number>`:

```bash
curl -s -H "Authorization: Bearer $BUILDKITE_TOKEN" \
  "https://api.buildkite.com/v2/organizations/vllm/pipelines/vllm-omni/builds/<number>" | jq '.'
```

**Option C - No env token / user prefers web-only**

- Clearly state that without **`BUILDKITE_TOKEN` / `BUILDKITE_API_TOKEN`** in the environment you can only produce a **summary-level** report (build status/duration), not full step/job breakdown.
- Open the [builds list](https://buildkite.com/vllm/vllm-omni/builds?branch=main) and locate the topmost **Scheduled nightly build** entry.
- Ask the user to paste the **build number** or **full build URL** (and optional failing log snippets) for fallback reporting.

### Step 2: Fetch jobs and steps (exclude Upload pipelines)

For the chosen `build_number`:

```bash
curl -s -H "Authorization: Bearer $BUILDKITE_TOKEN" \
  "https://api.buildkite.com/v2/organizations/vllm/pipelines/vllm-omni/builds/<build_number>" \
  | jq '{
      state, message, commit, branch, created_at, finished_at,
      reportable_jobs: [.jobs[] | select(.name | test("^Upload .+ Pipeline$"; "i") | not)
        | {name, state, id, raw_log_url, log_url}]
    }'
```

- **Summary counts** (passed / failed / skipped / broken): count **only** `reportable_jobs`, not upload steps.
- **Failed steps** tables: include only reportable jobs (upload failures must **not** pollute "test failure" narrative unless the user explicitly asks for pipeline hygiene).

### Step 2b: Per-job pytest results (detailed table)

**Not included in `compose_full_report.py` release output.** Use when authoring a separate CI appendix or nightly-style doc:

1. For **each reportable job**, GET `raw_log_url` (fallback `log_url`) with `Authorization: Bearer` using the same credential as in [Buildkite authentication (environment only)](#buildkite-authentication-environment-only).
2. Parse pytest output from the log (session `=== ... ===` footer; `FAILED` / `ERROR` lines). See [references/buildkite-api.md](references/buildkite-api.md).
3. **Helper:** from the skill directory (requires `BUILDKITE_TOKEN` or `BUILDKITE_API_TOKEN` already exported in that shell):

```bash
python scripts/nightly_job_pytest_table.py              # latest scheduled nightly
python scripts/nightly_job_pytest_table.py --build 4708
```

Paste the emitted **Per-job test execution (pytest)** table where needed. The script **skips** `Upload * Pipeline` jobs.

### Step 2c: Metrics overview (bug avg first response)

1. From the skill directory, run [scripts/buildkite_build_stats.py](scripts/buildkite_build_stats.py) with `BUILDKITE_TOKEN` or `BUILDKITE_API_TOKEN` set. Optional `GITHUB_TOKEN` (or `GH_TOKEN`) for the **Bug avg first response** column. The script uses `requests` (`pip install requests` if needed).
2. Optional: pass `--from` / `--to` as `YYYY-MM-DD` (UTC, inclusive) for a custom window. If **both are omitted**, the script uses **the current UTC calendar month through today** (month-to-date). To override one past month, pass both dates (e.g. `--from 2025-01-01 --to 2025-01-31`).
3. Add `--markdown` to print a ready-to-paste **Metrics overview** block: Source line plus the full metrics table (**Success rate/UT coverage**; **Bug avg first response** on **bugs (first response, YYYY-MM-DD..YYYY-MM-DD)** row from GitHub - same date window as **`--from` / `--to`**; **ut** / **ut (exclude models)** from **Simple Unit Test** log parsing - implementation detail stays in `buildkite_build_stats.py`, not in the pasted report prose). The release Metrics overview filters out the **CI category** rows (`ready` / `merge` / `nightly` / `weekly`) and the **`ut`** / **`ut (exclude models)`** rows so only the **`bugs (first response, …)`** row remains; the **CI issue detection rate** row is appended below it (see Step 2d).

```bash
pip install requests   # if not already installed
python scripts/buildkite_build_stats.py --markdown
# Or an explicit UTC window (must pass both --from and --to together):
# python scripts/buildkite_build_stats.py --from YYYY-MM-DD --to YYYY-MM-DD --markdown
```

4. Paste the full script output (the `## Metrics overview` section through the main metrics table — the release report drops `ready` / `merge` / `nightly` / `weekly` / `ut` / `ut (exclude models)` rows via `compose_full_report.replace_ut_coverage_with_manual_edit`, leaving only `bugs (first response, ...)`; the `append_ci_issue_detection_rate_row` row is appended below; the `_append_device_hours_build_row` row is appended last — see [Device-Hours / Build (7-day avg)](#device-hours--build-7-day-avg--operator-editable-metric-row)). **Do not** hand-edit numbers or coverage; they must match the script run.

See [references/buildkite-api.md](references/buildkite-api.md) for how builds are classified into **ready** / **merge** / **nightly** buckets.

### Step 2d: CI testing - Analysis (CI Failure) GitHub issues (stats date window)

1. Enumerate issues with **`label:bug`** **and** **`label:ci-failure`** (exact names; see [references/ci-github-ci-failure-issues.md](references/ci-github-ci-failure-issues.md)) and **`created`** (UTC) in **`--stats-from`..`--stats-to`** (same `YYYY-MM-DD` inclusive range as **Metrics overview** / `compose_full_report.py`).
2. Prefer **GitHub Search API**: `GET /search/issues?q=repo:vllm-project/vllm-omni+is:issue+label:bug+label:ci-failure+created:YYYY-MM-DD..YYYY-MM-DD`. Include **open** and **closed** issues returned by Search (no title-based filter).
3. Optional: `GITHUB_TOKEN` / `GH_TOKEN` in the environment for higher rate limits (do not paste tokens into chat).
4. Add **#### Analysis (CI Failure)** (optional hand section; **not** generated by `compose_full_report.py`) with columns **Issue #** | **Title** | **Status** (`Open` / `Closed`). If none match, state that explicitly.
5. **compose_full_report.py** does **not** emit this subsection; use `scripts/patch_report_ci_failure.py` on a hand-maintained `.md` if you need it inside a report file.

### Step 2e: Device-Hours / Build (7-day avg) — operator-editable metric row

The release Metrics overview always ends with a final row whose first column
reads **`**Device-Hours / Build (7-day avg)**`** and whose **Success rate/UT
coverage** cell is a stub — the report does **not** compute this number
itself. Compute-burn lives in a separate spreadsheet; rather than wire the
source into the report (which would tie the script to that sheet's format),
the cell ships as a manually-editable input and persists in `localStorage`.

**How it shows up**

* **Markdown export** (`.md`, generated by `--format markdown`): the cell is
  rendered as the literal marker text `@@DEVICE_HOURS_PER_BUILD_CELL@@` so
  the round-trip is stable and visible in a plain text diff. To fill the
  value in Markdown, replace the marker with the chosen value (for example
  `132.4 h`) before publishing the `.md`.
* **HTML export** (default): the marker is replaced at render time by an
  inline `<input class="dhpb-input">` whose `placeholder` reads
  `click to fill (e.g. 132.4 h)`. Click the cell, type the value, press
  Tab/Enter. The input keeps the value while you type (no submit step) and
  mirrors it to a `data-dhpb-value` attribute so a browser *Save Page As*
  download captures the user's edits across origins.

**Persistence**

| Storage | Key | Use |
|---------|-----|-----|
| `data-dhpb-value` attribute on the `<input>` | inline DOM | Survives Save-Page-As download (preferred source on reload) |
| `localStorage` | `"device-hours-per-build"` | Survives reload / re-open on the same origin |
| in-memory `mem` | (transient, per-page-load) | Fallback when `localStorage` throws (Chrome `file://`) |

**Implementation**

* `compose_full_report._append_device_hours_build_row` — appends the row
  beneath the CI issue detection rate row with the marker placeholder.
* `compose_full_report.DEVICE_HOURS_PER_BUILD_MARKER` — the literal
  marker string (`@@DEVICE_HOURS_PER_BUILD_CELL@@`).
* `release_md_to_html._upgrade_device_hours_cell` — substitutes the
  marker for the editable `<input>` during HTML conversion.
* `release_md_to_html._DEVICE_HOURS_BUILD_SCRIPT` — JS handler that wires
  the input to `localStorage["device-hours-per-build"]`, mirrors the value
  to the `data-dhpb-value` attribute on every input/blur, and reads the
  attribute on hydration so a saved copy retains the value across origins.
* `report_html_theme.RELEASE_MARKDOWN_DOC_CSS` — `.release-doc .dhpb-input*`
  styles (dashed border, accent on focus, solid border once persisted).

**Verification**

```bash
PYTHONPATH=scripts python3 - <<'PY'
from compose_full_report import (
    _append_device_hours_build_row, DEVICE_HOURS_PER_BUILD_MARKER,
    append_ci_issue_detection_rate_row, replace_ut_coverage_with_manual_edit,
)
sample = '''
| CI category | Success rate/UT coverage | Avg duration | Other finished count | Bug avg first response |
|-------------|--------------------------|--------------|----------------------|------------------------|
| bugs (first response, 2026-08-01..2026-09-01) | - | - | - | 12.3h |
'''
out = append_ci_issue_detection_rate_row(
    replace_ut_coverage_with_manual_edit(sample), None, "2026-08-01", "2026-09-01"
)
assert DEVICE_HOURS_PER_BUILD_MARKER in out
assert "Device-Hours / Build (7-day avg)" in out
print(out)
PY
```

### Step 3: Fetch **all** open bug issues (paginated), filter by stats window

Do **not** rely on the GitHub web UI first page for counts or tables - it is incomplete when there are many issues.

1. Prefer **GitHub REST API** pagination: `GET /repos/vllm-project/vllm-omni/issues?state=open&labels=bug&per_page=100&page=...` until a page has **fewer than 100** items (or empty). Merge pages; **exclude** entries with `pull_request` (PRs masquerading as issues).
2. If `GITHUB_TOKEN` is missing and you hit rate limits or need stable automation, **prompt the user** (e.g. `Provide GITHUB_TOKEN to paginate all open bugs reliably and avoid unauthenticated rate limits.`).
3. After the full list is assembled, **keep only** issues whose **`created_at` UTC calendar date** (**`YYYY-MM-DD`**) falls in **`--stats-from`..`--stats-to`** (inclusive), matching **Metrics overview** / `compose_full_report.py` - not "current calendar month" unless those flags happen to bound the month.
4. Commands and a bash/jq example: [references/github-issues-pagination.md](references/github-issues-pagination.md).
5. To refresh **Open issues** in an existing report without re-running the full composer: `python scripts/patch_report_open_issues.py --report <file.md> --stats-from YYYY-MM-DD --stats-to YYYY-MM-DD` (from the skill directory; `GITHUB_TOKEN` / `GH_TOKEN` recommended).

### Step 4: Produce the report

**Full document:** `python scripts/compose_full_report.py` (HTML). For a Markdown file (patch scripts, hand merge): `python scripts/compose_full_report.py --format markdown --out report.md`.

Use the **Report structure** below when assembling manually. Fill:

- **Test conclusion** — Checklist table + Go/Rejected (interactive HTML; Markdown static default Go)
- **Metrics overview** from Step 2c — immediately after **Test conclusion**
- **Test Result** — Common stack from [references/local-test-matrix.md](references/local-test-matrix.md); H200/H800/A100 nightly-style grouped tables (when log dirs exist); **H100** embeds **Build** (build link, branch, commit only), Summary, failed table (excludes Step 2b pytest, Step 2d **Analysis (CI Failure)**)
- *(Optional)* **Test content (job scope)** — not generated by compose; use `patch_report_scope_local.py` or hand-author
- **Open issues** from Step 3
- **Unknown** if data was incomplete

## Report structure (same content as compose_full_report HTML / Markdown)

Hand-authored or review-only; automation emits the same sections in HTML by default.

```markdown
# vLLM-Omni Test Report - Scheduled Nightly

## Test conclusion

| Check item | Result |
| ... | Pass / Fail (HTML: **UT, requirements, performance, NPU CI** clickable; **GPU CI / critical issues / bug assignees** three rows auto-locked; **Test conclusion:** Go or Rejected) |

## Metrics overview

(`buildkite_build_stats.py --markdown`, aligned with `--stats-from`..`--stats-to`. The release report keeps only the **`bugs (first response, ...)`** row from the upstream table; the `ready` / `merge` / `nightly` / `weekly` CI-category buckets and the `ut` / `ut (exclude models)` rows are dropped via `compose_full_report.replace_ut_coverage_with_manual_edit`. `append_ci_issue_detection_rate_row` appends a **CI issue detection rate** row beneath it.)

## Test Result

### Common stack (all rows)

(Body of that section in `references/local-test-matrix.md`.)

### H200

(Optional: `--log-dir-h200`, same grouping as nightly local Summary.)

### H800

(Optional: `--log-dir-h800`.)

### A100

(Optional: `--log-dir-a100`.)

### H100 (CI — Buildkite scheduled nightly)

#### Build
…
#### Summary (reportable jobs only)
…
#### Failed test jobs (if any)
…

## Failure Analysis

Per-machine failure detail. Click the *Failed* cell in the Test Result summary table to jump to the matching subsection below.

#### H200 failures
(Per-job **Failures & errors** table with interactive **Status** column — Filed / Not an issue.)

#### H800 failures
(Same layout as H200.)

#### A100 failures
(Same layout as H200.)

#### H100 (CI — Buildkite scheduled nightly) failures
(Buildkite failed/broken steps with interactive **Status** column.)

## Open issues (stats window)

(REST pagination `label:bug`; `created_at` in stats window; DI from priority labels. Last two columns **Follow-up action** / **Remarks** are manual triage cells — dropdown + note box in HTML, `—` in Markdown.)

## Data source

(Buildkite, GitHub, `--log-dir-*`, etc.)
```

## Constraints

1. **Do not invent** pass/fail counts: use API JSON or user-confirmed paste.
2. If the API returns 401/403, state that **`BUILDKITE_TOKEN` or `BUILDKITE_API_TOKEN` must be set in the environment** (read-only token) and fall back to Option C if the user cannot do that.
3. Prefer **Scheduled nightly** explicitly; do not label a random `main` build as nightly without matching the message/pattern.
4. For **open bugs**, **paginate until done**; do not report "all issues" from a single HTML page.
5. Keep prose concise; tables over long bullet lists.
6. **Always omit** `Upload * Pipeline` jobs from **test** summaries and the per-job pytest table unless the user explicitly requests pipeline upload health.

## Related

- CI pipeline concepts (Buildkite L1–L4, test-ready/merge/nightly): [vllm-omni-create-testcase](../vllm-omni-create-testcase/SKILL.md)
- Cluster nightly run + docker (produces logs): [vllm-omni-local-test](../vllm-omni-local-test/SKILL.md)

## Reference

- Buildkite REST API: [Buildkite API documentation](https://buildkite.com/docs/apis/rest-api)
- Optional detail: [references/buildkite-api.md](references/buildkite-api.md)
- GitHub issues (pagination + month filter): [references/github-issues-pagination.md](references/github-issues-pagination.md)
- CI job test scope (what each nightly job tests): [references/ci-job-test-scope.md](references/ci-job-test-scope.md)
- Local / release **Common stack** + compose `--log-dir-*` notes: [references/local-test-matrix.md](references/local-test-matrix.md)
- CI Failure GitHub issues (`label:bug` + `label:ci-failure`, stats `created` range): [references/ci-github-ci-failure-issues.md](references/ci-github-ci-failure-issues.md)
- HTML from Markdown (release): [scripts/release_md_to_html.py](scripts/release_md_to_html.py) (used by compose_full_report)
- **Nightly** log tree / pytest parsing: [references/nightly-local-log-layout.md](references/nightly-local-log-layout.md) (fetch off cluster: [vllm-omni-local-test](../vllm-omni-local-test/SKILL.md))
- Buildkite performance summary from kanban assets: [scripts/kanban_assets_perf_summary.py](scripts/kanban_assets_perf_summary.py)
- Archive HTML reports to kanban + **gh** push: [references/kanban-report-archive.md](references/kanban-report-archive.md), [scripts/push_report_to_kanban.py](scripts/push_report_to_kanban.py), [scripts/push_kanban_report.py](scripts/push_kanban_report.py)
- Laptop path defaults (confirm before sync/prep/report): [references/confirm-laptop-path-defaults.md](references/confirm-laptop-path-defaults.md)
- Kanban prep before report (pull, `manual_*`, mkdocs build): [references/kanban-pre-report-prep.md](references/kanban-pre-report-prep.md), [scripts/prepare_kanban_before_report.py](scripts/prepare_kanban_before_report.py)
