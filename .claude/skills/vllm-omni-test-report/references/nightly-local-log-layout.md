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

## Perf JSON (under same `LOG_DIR`)

- Patterns: `result_test_*.json`, `diffusion_result_*.json`, `benchmark_results_*.json` — found recursively under **`logs/nightly_jobs`** (e.g. run root or **`results/`** subdir).
- **`local_perf_results.py`** and **`prepare_kanban_before_report.py`** scan this tree; no separate **`tests/dfx/perf/results`** path on the laptop.

## Pytest parsing

- Expect `FAILED ...`, `ERROR ...`, and a session footer with `N passed`, `N failed`, etc.

## `run_nightly_jobs.sh`

If logs live elsewhere, pass `--log-dir` to the report script or symlink into `logs/nightly_jobs`.
