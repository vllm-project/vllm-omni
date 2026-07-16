# Kanban prep before test report

Run **before** `scripts/nightly_local_log_report.py` or `scripts/compose_full_report.py` when the report needs fresh **Buildkite performance baseline comparison** data from [vllm-omni-kanban](https://github.com/hsliuustc0106/vllm-omni-kanban) `docs/assets/charts/*_history.json`.

Automated entry point: **`scripts/prepare_kanban_before_report.py`**.

## Prerequisites

- Local clone: `git clone https://github.com/hsliuustc0106/vllm-omni-kanban` → default **`KANBAN_REPO_ROOT=~/vllm-omni-kanban`** ([confirm with user](confirm-laptop-path-defaults.md))
- **`gh`** installed and authenticated (`gh auth login`) — used for `git pull --rebase`
- Kanban Python env with **`mkdocs`** (and kanban deps) for step 3
- After cluster sync: default **`REPO_ROOT=~/vllm-omni`** ([confirm with user](confirm-laptop-path-defaults.md)) with **`logs/nightly_jobs`** (job logs + perf JSON in one tree)

## Steps (in order)

### 1. Pull latest kanban

```bash
cd "$KANBAN_REPO_ROOT"
git pull --rebase origin main   # or your tracking branch
```

The script runs this via **`gh auth git-credential`** (same as report archive push).

### 2. Sync local perf + logs → `data/local_nightly_raw/manual_*`

**Only `nightly_jobs_local_*` perf** goes to kanban. After log sync, **`prepare_kanban_before_report.py`** reads **`$REPO_ROOT/logs/.kanban_perf_source`** (populated from the latest local run during fetch merge). **Stability `nightly_jobs_stability_*` and general `nightly_jobs_YYYYMMDD-*` perf JSON are not copied to kanban** (they may still exist under **`logs/nightly_jobs`** for the HTML report).

**When** `.kanban_perf_source` (or a local-only sync) contains perf JSON (`result_test_*.json`, `diffusion_result_*.json`, `benchmark_results_*.json`):

1. Sync into **`$KANBAN_REPO_ROOT/data/local_nightly_raw/manual_YYYYMMDD/`**:
   - **`YYYYMMDD`** from **`nightly_jobs_local_*`** in **`.nightly_jobs_source`**, not from general nightly
   - If that directory already exists for the same run date, it is **cleared and repopulated**
2. Copy into it:
   - All matching perf JSON from **`.kanban_perf_source`** (unchanged basenames)
   - **Only** one Hunyuan Image job log from the same tree → **`test_hunyuan_image3.log`**

Example kanban layout (committed over time):

```
data/local_nightly_raw/manual_20260622/
  diffusion_result_test_hunyuan_image_tp4_20260622-111338.json
  test_hunyuan_image3.log
```

If **`.kanban_perf_source`** has **no perf JSON** (no local run synced), skip this step (still run step 3 to refresh charts from existing raw data).

### 3. MkDocs build (regenerate chart history)

From the kanban repo root:

```bash
cd "$KANBAN_REPO_ROOT"
python -m mkdocs build
```

`scripts/mkdocs_hooks.py` **on_startup** will:

- Sync **`data/buildkite_nightly_raw`** → **`data/results/`** (Buildkite perf JSON)
- Sync **`data/local_nightly_raw`** → **`data/results/`** for configured local models (e.g. Hunyuan Image 3)
- Run **`scripts/generate_charts.py`** → update **`docs/assets/charts/*_history.json`**

Then generate the HTML report with **`--kanban-repo-root "$KANBAN_REPO_ROOT"`**. Local Test and Buildkite **performance baseline comparison** both read **`docs/assets/charts/*_history.json`** from that checkout.

## One command (from this skill directory)

```bash
export KANBAN_REPO_ROOT="${KANBAN_REPO_ROOT:-~/vllm-omni-kanban}"
export REPO_ROOT="${REPO_ROOT:-~/vllm-omni}"

python scripts/prepare_kanban_before_report.py

python scripts/nightly_local_log_report.py \
  --html-report ./nightly-report.html \
  --kanban-repo-root "$KANBAN_REPO_ROOT"
```

## Flags

| Flag | Effect |
|------|--------|
| `--skip-pull` | Do not `git pull --rebase` |
| `--skip-manual-sync` | Do not create `manual_*` or copy perf/logs |
| `--skip-mkdocs` | Do not run `mkdocs build` |
| `--log-dir` | Override `$REPO_ROOT/logs/nightly_jobs` (job logs + perf JSON scan root) |

## Notes

- **`manual_*` directories are data artifacts** — `prepare_kanban_before_report.py` copies locally and writes `.last_manual_dir`; **`push_report_to_kanban.py`** stages that `manual_*` together with the HTML report when the user archives/pushes (see [kanban-report-archive.md](kanban-report-archive.md)).
- **`--kanban-refresh-from-raw`** on `nightly_local_log_report.py` is a lighter alternative (sync + `generate_charts.py` only, no pull / no `manual_*` / no full mkdocs). Prefer **`prepare_kanban_before_report.py`** for the full workflow above.
