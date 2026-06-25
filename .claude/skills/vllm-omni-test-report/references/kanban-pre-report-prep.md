# Kanban prep before test report

Run **before** `scripts/nightly_local_log_report.py` or `scripts/compose_full_report.py` when the report needs fresh **Buildkite performance baseline comparison** data from [vllm-omni-kanban](https://github.com/hsliuustc0106/vllm-omni-kanban) `docs/assets/charts/*_history.json`.

Automated entry point: **`scripts/prepare_kanban_before_report.py`**.

## Prerequisites

- Local clone: `git clone https://github.com/hsliuustc0106/vllm-omni-kanban` → default **`KANBAN_REPO_ROOT=~/vllm-omni-kanban`** ([confirm with user](confirm-laptop-path-defaults.md))
- **`gh`** installed and authenticated (`gh auth login`) — used for `git pull --rebase`
- Kanban Python env with **`mkdocs`** (and kanban deps) for step 3
- After cluster sync: default **`REPO_ROOT=~/vllm-omni`** ([confirm with user](confirm-laptop-path-defaults.md)) with **`logs/nightly_jobs`** and optional **`tests/dfx/perf/results/`**

## Steps (in order)

### 1. Pull latest kanban

```bash
cd "$KANBAN_REPO_ROOT"
git pull --rebase origin main   # or your tracking branch
```

The script runs this via **`gh auth git-credential`** (same as report archive push).

### 2. Sync local perf + logs → `data/local_nightly_raw/manual_*`

**When** `$REPO_ROOT/tests/dfx/perf/results` contains perf JSON (`result_test_*.json`, `diffusion_result_*.json`, `benchmark_results_*.json` — including under a latest timestamp subdirectory):

1. Create a **new** directory under **`$KANBAN_REPO_ROOT/data/local_nightly_raw/`**:
   - Name: **`manual_YYYYMMDD`**
   - If that name already exists: **`manual_YYYYMMDD_HHMMSS`** (or with numeric suffix)
2. Copy into it:
   - All matching perf JSON from **`$REPO_ROOT/tests/dfx/perf/results`** (flat copy, original basenames — unchanged)
   - **Only** **`$REPO_ROOT/logs/nightly_jobs/local_pytest_hunyuan_image.log`** → **`test_hunyuan_image3.log`** (other logs under `nightly_jobs` are not copied)

Example kanban layout (committed over time):

```
data/local_nightly_raw/manual_20260622/
  diffusion_result_test_hunyuan_image_tp4_20260622-111338.json
  test_hunyuan_image3.log
```

If the results directory is **empty or missing**, skip this step (still run step 3 to refresh charts from existing raw data).

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
| `--perf-result-root` | Override `$REPO_ROOT/tests/dfx/perf/results` |
| `--log-dir` | Override `$REPO_ROOT/logs/nightly_jobs` |

## Notes

- **`manual_*` directories are data artifacts** — `prepare_kanban_before_report.py` copies locally and writes `.last_manual_dir`; **`push_report_to_kanban.py`** stages that `manual_*` together with the HTML report when the user archives/pushes (see [kanban-report-archive.md](kanban-report-archive.md)).
- **`--kanban-refresh-from-raw`** on `nightly_local_log_report.py` is a lighter alternative (sync + `generate_charts.py` only, no pull / no `manual_*` / no full mkdocs). Prefer **`prepare_kanban_before_report.py`** for the full workflow above.
