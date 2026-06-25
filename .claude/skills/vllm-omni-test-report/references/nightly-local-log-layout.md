# `nightly_jobs` log layout

## Default path

- On disk: **`.../logs/nightly_jobs`** (copy from cluster per [../../vllm-omni-local-test/references/nightly-local-log-fetch.md](../../vllm-omni-local-test/references/nightly-local-log-fetch.md)).
- **On your laptop**, `...` should be **`$REPO_ROOT`** (default **`~/vllm-omni`** — [confirm with user](confirm-laptop-path-defaults.md)) so the tree is **`$REPO_ROOT/logs/nightly_jobs`** — the same path **`nightly_local_log_report.py`** uses by default.

**Performance baseline comparison:** **Buildkite** section reads kanban **`docs/assets/charts/*_history.json`** (all models). **Local** section reads the same history but **filters to tests present in synced `tests/dfx/perf/results/`** perf JSON. Run [prepare_kanban_before_report.py](../scripts/prepare_kanban_before_report.py) before generating the report (see [kanban-pre-report-prep.md](kanban-pre-report-prep.md)).

**Before each sync (required):** delete local **`$REPO_ROOT/logs`** and **`$REPO_ROOT/tests/dfx/perf/results`** before `scp` / `rsync` / tarball extract ([clear local trees](../../vllm-omni-local-test/references/nightly-local-log-fetch.md#clear-local-trees)).

- **Nightly HTML / Markdown** (`scripts/nightly_local_log_report.py`): **Summary** / **Failure analysis** use `nightly_jobs/`; **Local performance baseline comparison** reads kanban history but **only rows matching synced `tests/dfx/perf/results` JSON**; **Buildkite performance baseline comparison** shows all models from kanban history.

## Discovery rules (`scripts/nightly_local_log_report.py`)

1. **Job subdirectories (preferred)**  
   If `LOG_DIR` contains **subdirectories**, each name is the **job name**. Concatenate all `*.log`, `*.out`, `*.txt` in that directory (sorted by name).

2. **Flat log files**  
   If `LOG_DIR` has **no** subdirectories, each `*.log` / `*.out` / `*.txt` at the top level is **one job** (stem = job name).

3. **Hidden** names (leading `.`) are ignored.

4. **HTML Summary grouping** — jobs are placed under **Omni / TTS / Diffusion** × **Perf, Acc, Function, doc, stability** when the name matches either:
   - **Prefix:** ``<omni|tts|diffusion|diff>_<perf|acc|function|doc|stability>`` (or the same two tokens in reverse order), case-insensitive, with spaces/hyphens like underscores; or
   - **Keywords** anywhere in the folder / stem: pillar substrings **diffusion**, word **omni**, word **tts**; dimension **accuracy** / **acc**, **performance** / **perf**, **function** / **functional**, **documentation** / **docs** / **doc**, **stability** / **stable** (see ``_classify_local_nightly_job`` in `scripts/nightly_local_log_report.py`).  
   Example: ``full_moon_Diffusion_X2I_A_T_Accuracy_Test`` → **Diffusion · Acc**. Names that do not resolve to both a pillar and a dimension appear under **Other**.

## Pytest parsing

- Expect `FAILED ...`, `ERROR ...`, and a session footer with `N passed`, `N failed`, etc.

## `run_nightly_jobs.sh`

If logs live elsewhere, pass `--log-dir` to the report script or symlink into `logs/nightly_jobs`.
