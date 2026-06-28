# Confirm laptop path defaults with user

**Before** log sync, kanban prep, or nightly HTML report on the **laptop**, show the planned local paths and **ask whether to use them**. Do not silently assume defaults when the user has not already confirmed or overridden values in the same conversation.

This is separate from [cluster run defaults](../../vllm-omni-local-test/references/nightly-local-environment.md#confirm-run-defaults-with-user) (`/rebase/vllm-omni` on H200/H800).

## Default paths (laptop)

| Variable | Default | Purpose |
|----------|---------|---------|
| **`REPO_ROOT`** | `~/vllm-omni` | Local vLLM-Omni checkout — sync target for **`logs/nightly_jobs`** (job logs + perf JSON) |
| **`KANBAN_REPO_ROOT`** | `~/vllm-omni-kanban` | Local [vllm-omni-kanban](https://github.com/hsliuustc0106/vllm-omni-kanban) clone — pull, **`manual_*`**, **`mkdocs build`**, **`--kanban-repo-root`** |

Scripts resolve env overrides first; when unset they use the defaults above (`scripts/laptop_path_defaults.py`).

## When to confirm

Confirm **once per session** before the first laptop-side step among:

- Clearing / syncing logs ([nightly-local-log-fetch](../../vllm-omni-local-test/references/nightly-local-log-fetch.md))
- **`prepare_kanban_before_report.py`**
- **`nightly_local_log_report.py`**
- Kanban archive push (`push_report_to_kanban.py` / `push_kanban_report.py`)

Skip re-prompting only if the user already said **use defaults** or gave explicit paths in the same conversation.

## Agent prompt template

> Planned **local (laptop)** path defaults:
> - **`REPO_ROOT`**: `~/vllm-omni` (local vLLM-Omni checkout — sync target for `logs/nightly_jobs`)
> - **`KANBAN_REPO_ROOT`**: `~/vllm-omni-kanban` (local kanban clone — pull, `manual_*`, performance baseline JSON)
>
> Use these defaults? To override, reply with custom paths. After you **confirm**, I will run sync, kanban prep, or report generation.

## Rules

1. **Always display** both paths before the first laptop sync / prep / report step (unless the user already confirmed or overrode).
2. **Wait for confirmation** or explicit overrides before `rm -rf`, `scp`/`tar`, `prepare_kanban_before_report.py`, or `nightly_local_log_report.py`.
3. If the user confirms defaults → `export REPO_ROOT="${REPO_ROOT:-~/vllm-omni}"` and `export KANBAN_REPO_ROOT="${KANBAN_REPO_ROOT:-~/vllm-omni-kanban}"` (or rely on script defaults without export).
4. If the user gives custom paths → use those for the rest of the session.
5. **Cluster vs laptop:** cluster **`REPO_ROOT`** default is **`/rebase/vllm-omni`** — confirm separately when running on H200/H800; laptop **`REPO_ROOT`** is the sync destination on your machine.

## Example (after confirmation)

```bash
export REPO_ROOT="${REPO_ROOT:-~/vllm-omni}"
export KANBAN_REPO_ROOT="${KANBAN_REPO_ROOT:-~/vllm-omni-kanban}"

cd skills/vllm-omni-test-report
python scripts/prepare_kanban_before_report.py
python scripts/nightly_local_log_report.py \
  --html-report ./nightly-report.html \
  --kanban-repo-root "$KANBAN_REPO_ROOT"
```
