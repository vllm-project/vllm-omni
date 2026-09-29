# `nightly_jobs` log layout

## Default path

- On disk: **`.../logs/nightly_jobs`** (copy from cluster per [../../vllm-omni-local-test/references/nightly-local-log-fetch.md](../../vllm-omni-local-test/references/nightly-local-log-fetch.md)).
- **On your laptop**, `...` should be **`$REPO_ROOT`** (default **`~/vllm-omni`** — [confirm with user](confirm-laptop-path-defaults.md)) so the tree is **`$REPO_ROOT/logs/nightly_jobs`** — the same path **`nightly_local_log_report.py`** uses by default for job logs **and** perf JSON (recursive scan).

**Performance baseline comparison:** **Buildkite** section reads kanban **`docs/assets/charts/*_history.json`** (all models). **Local** section reads the same history but **filters to tests with perf JSON under synced `logs/nightly_jobs/`**. Run [prepare_kanban_before_report.py](../scripts/prepare_kanban_before_report.py) before generating the report (see [kanban-pre-report-prep.md](kanban-pre-report-prep.md)).

**Before each sync (required):** delete local **`$REPO_ROOT/logs`** before `scp` / `rsync` / tarball extract ([clear local trees](../../vllm-omni-local-test/references/nightly-local-log-fetch.md#clear-local-trees)).

- **Nightly HTML / Markdown** (`scripts/nightly_local_log_report.py`): **Summary** / **Failure analysis** use `nightly_jobs/`; **Local performance baseline comparison** reads kanban history but **only rows matching perf JSON under `logs/nightly_jobs`**; **Buildkite performance baseline comparison** shows all models from kanban history.

## Discovery rules (`scripts/nightly_local_log_report.py`)

1. **Job subdirectories (preferred)**  
   If `LOG_DIR` contains **subdirectories**, each name is the **job name**. Concatenate all `*.log`, `*.out`, `*.txt` in that directory (sorted by name).

2. **Flat log files**  
   If `LOG_DIR` has **no** subdirectories, each `*.log` / `*.out` / `*.txt` at the top level is **one job** (stem = job name).

3. **Hidden** names (leading `.`) are ignored.

4. **Infrastructure sub-directories are skipped.** `run_nightly_jobs.sh` writes raw nohup output under a sibling `logs/` folder and stores generated `.sh` scripts under `jobs/`, perf JSON under `perf_results/` / `results/`. These folders aren't test jobs and would otherwise surface as a bogus row named `logs` / `jobs` / `perf_results`. The discovery helper (`scripts/nightly_job_log_discovery.py`, `discover_job_logs`) skips any sub-directory whose name (case-insensitive) is in `{logs, jobs, perf_results, perf-results, results, raw, nohup, tmp, __pycache__}`. Flat files at the top level are still picked up by rule (2).

5. **HTML Summary grouping** — jobs are placed under **Omni / TTS / Diffusion** × **Perf, Acc, Function, doc, stability** when the name matches either:
   - **Prefix:** ``<omni|tts|diffusion|diff>_<perf|acc|function|doc|stability>`` (or the same two tokens in reverse order), case-insensitive, with spaces/hyphens like underscores; or
   - **Keywords** anywhere in the folder / stem. **Pillar** substrings: ``diffusion``, ``hunyuan``, ``hunyuan_image``, ``qwen-image``/``qwen_image``, ``wan``/``wan2.2``, ``bagel``, ``glm-image``/``glm_image``, ``longcat``, ``flux``, ``tts``, ``omni``. **Dimension** substrings: ``accuracy`` / ``acc``, ``performance`` / ``perf``, ``function`` / ``functional``, ``documentation`` / ``docs`` / ``doc``, ``stability`` / ``stable`` (see ``_classify_local_nightly_job`` in `scripts/nightly_local_log_report.py`).  
   Examples: ``full_moon_Diffusion_X2I_A_T_Accuracy_Test`` → **Diffusion · Acc**; ``full_moon_HunyuanImage3-DIT_Accuracy_Test`` → **Diffusion · Acc** (sub-model keywords also roll up under Diffusion); ``nightly-hunyuan-image3-performance`` → **Diffusion · Perf**. Names that do not resolve to both a pillar and a dimension appear under **Other**.

## Summary read order: timing_summary first, failed-job log second

The **Test Result** per-GPU pillar×dim Summary tables are built in two tiers —
this is the rule the update follows (先读 `timing_summary.log` 获得summary信息，失败的job再去读对应的日志):

1. **Summary table — read `timing_summary.log` first.** Each run's
   `timing_summary.log` is a rollup that lists **every job** with an
   OK / FAILED status and a wall-clock duration. It is the source of truth
   for the **job list** and the **pass/fail status**, which is why a run
   whose per-job `.log`s were cleaned up still shows all its jobs — they
   surface as **`(manifest only)`** rows via
   `_augment_groups_with_manifest_only`. The manifest is discovered by
   `discover_stability_manifests` (`scripts/stability_log_manifest.py`),
   which globs `**/timing_summary.log` under subdirs named with the
   nightly prefix (`nightly_stability_jobs_*` / `nightly_jobs_local_*` /
   `nightly_jobs_*`); per-job logs are discovered by `discover_job_logs`
   (`scripts/nightly_job_log_discovery.py`).

2. **FAILED jobs — then read the per-job `.log`.** Only **FAILED** jobs
   have their per-job `.log` parsed by `parse_pytest_log` to fill the
   **counts** column (Total / Passed / Failed / Skipped / Errors) and to
   extract the failure/error **excerpts** used in **Failure Analysis**.
   If a FAILED job's `.log` is missing, the row keeps its manifest
   status but shows no counts/excerpts — a "log not pulled" warning
   surfaces instead of silently downgrading the row to OK.

3. **OK jobs — never read the `.log` for the Summary.** Only FAILED jobs
   do. This is why `selective_stability_pull.py` phase-1 packs just
   `timing_summary.log` + the FAILED jobs' `.log`s (not every OK job's
   log): the manifest already covers the OK job list and status, so the
   OK `.log`s would be pulled for nothing.

## Perf JSON (under same `LOG_DIR`)

- Patterns: `result_test_*.json`, `diffusion_result_*.json`, `benchmark_results_*.json` — found recursively under **`logs/nightly_jobs`** (e.g. run root or **`results/`** subdir).
- **`local_perf_results.py`** and **`prepare_kanban_before_report.py`** scan this tree; no separate **`tests/dfx/perf/results`** path on the laptop.

## Pytest parsing

- Expect `FAILED ...`, `ERROR ...`, and a session footer with `N passed`, `N failed`, etc.

## `run_nightly_jobs.sh`

If logs live elsewhere, pass `--log-dir` to the report script or symlink into `logs/nightly_jobs`.
