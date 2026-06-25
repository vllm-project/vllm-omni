# Fetch nightly logs to your laptop

Part of **vllm-omni-local-test**. Pull **`nightly_jobs`** and optional **`tests/dfx/perf/results/`** from the cluster/container **before** running **`vllm-omni-test-report`** `scripts/nightly_local_log_report.py` on your machine.

**Before sync:** confirm laptop path defaults with the user — **`REPO_ROOT=~/vllm-omni`**, **`KANBAN_REPO_ROOT=~/vllm-omni-kanban`** — see [confirm-laptop-path-defaults.md](../../vllm-omni-test-report/references/confirm-laptop-path-defaults.md).

**H200 and H800 use the same laptop-side workflow** (clear local **`logs/`** + **`results/`** → remote tarball → extract → report). Only the **remote pack command** differs: **H200** = direct **`ssh`** (already in container); **H800** = **`ssh` + `srun --overlap docker exec`**.

**Default destination on your laptop:** mirror the cluster layout under your **local** vLLM-Omni checkout root (**`$REPO_ROOT`**):

- **`$REPO_ROOT/logs/nightly_jobs/`**
- **`$REPO_ROOT/tests/dfx/perf/results/`** — DFX perf JSON; copied into kanban **`data/local_nightly_raw/manual_*`** via [prepare_kanban_before_report.py](../../vllm-omni-test-report/scripts/prepare_kanban_before_report.py) before **`mkdocs build`** updates **`docs/assets/charts/*_history.json`**

That matches **`nightly_local_log_report.py`** defaults: **`$REPO_ROOT/logs/nightly_jobs`** for job logs; **`$REPO_ROOT/tests/dfx/perf/results`** for baseline comparison (when **`REPO_ROOT`** is set).

## What to copy

Cluster/container paths are relative to the **confirmed cluster repo root** (**`CLUSTER_REPO_ROOT`**, default **`/rebase/vllm-omni`**):

| Path (under repo root) | Required | Local destination (under laptop **`$REPO_ROOT`**) |
|------------------------|----------|---------------------------------------------------|
| **`logs/nightly_jobs/`** | Yes | **`logs/nightly_jobs/`** — layout: [../../vllm-omni-test-report/references/nightly-local-log-layout.md](../../vllm-omni-test-report/references/nightly-local-log-layout.md) |
| **`tests/dfx/perf/results/`** | Optional (if present) | **`tests/dfx/perf/results/`** — DFX perf JSON / artifacts from local nightly runs |

- If your site bind-mounts the repo tree on the host, you may use **`scp` / `rsync`** against that host path instead of tarball (see [Optional: scp / rsync](#optional-scp--rsync)).
- Missing optional paths must **not** fail the sync — tarball uses **`--ignore-failed-read`**; **`scp` / `rsync`** steps below are best-effort when the remote tree exists.

**Remote pack inner logic** (shared by H200/H800 tarball — runs with **`cd "$CLUSTER_REPO_ROOT"`**):

```bash
ROOT="${CLUSTER_REPO_ROOT:-/rebase/vllm-omni}"
cd "$ROOT" || exit 1
tar czf - --ignore-failed-read logs/nightly_jobs tests/dfx/perf/results
```

<a id="log-sync-workflow"></a>

## Log sync workflow (H200 and H800)

Run these steps **in order** on your laptop **after the cluster run finishes** and **before** pulling logs or perf JSON. **H200** and **H800** share steps **1**, **3**, and **4**; step **2** picks the machine-specific remote command.

<a id="clear-local-trees"></a>

### 1. Clear local `logs/` and `results/` (required)

**Always delete the existing *local* `logs` directory and DFX `results` directory before each sync.** Otherwise old job folders or stale perf JSON can **merge with the new pull** and skew the nightly report.

```bash
REPO_ROOT="${REPO_ROOT:-~/vllm-omni}"
rm -rf "$REPO_ROOT/logs"
rm -rf "$REPO_ROOT/tests/dfx/perf/results"
```

- **Scope:** your **laptop** checkout only; nothing on the cluster/container is removed.
- **What is cleared:** the entire **`$REPO_ROOT/logs`** tree (including **`nightly_jobs`** and any other log artifacts) and **`$REPO_ROOT/tests/dfx/perf/results`** (perf JSON for kanban **`manual_*`** sync).
- **Archive instead:** if you need a backup, rename before delete, e.g. `mv "$REPO_ROOT/logs" "$REPO_ROOT/logs.bak.$(date +%Y%m%d%H%M)"`.

Run this **before** remote tarball download (step 2) and again **before** extract (step 3) if step 1 was skipped earlier.

### 2. Remote tarball (machine-specific)

Pack **`logs/nightly_jobs`** and **`tests/dfx/perf/results`** in one archive. On **GNU tar**, missing optional paths do not abort the archive when **`--ignore-failed-read`** is set.

**H800** — **`ssh` + Slurm + `docker exec`** ([nightly-local-h800.md](nightly-local-h800.md)):

```bash
ssh -o BatchMode=yes "<SSH_CONNECTION_NAME>" \
  "bash -lc 'type module >/dev/null 2>&1 && module load slurm 2>/dev/null; srun --jobid=\"<JOBID>\" --overlap docker exec \"<CONTAINER_NAME>\" bash -lc \"
    ROOT=\\\${CLUSTER_REPO_ROOT:-/rebase/vllm-omni}
    cd \\\"\\\$ROOT\\\" || exit 1
    tar czf - --ignore-failed-read logs/nightly_jobs tests/dfx/perf/results
  \"'" \
  > nightly_logs.tgz
```

**H200** — direct **`ssh`** (session already in container; no Slurm, no **`docker exec`**) ([nightly-local-h200.md](nightly-local-h200.md)):

```bash
ssh -o BatchMode=yes "<SSH_CONNECTION_NAME>" \
  "bash -lc '
    ROOT=\${CLUSTER_REPO_ROOT:-/rebase/vllm-omni}
    cd \"\$ROOT\" || exit 1
    tar czf - --ignore-failed-read logs/nightly_jobs tests/dfx/perf/results
  '" \
  > nightly_logs.tgz
```

Use the **confirmed cluster repo root** in **`ROOT` / `-C`** when the user overrode the default **`/rebase/vllm-omni`**.

### 3. Extract on laptop (same for H200 and H800)

```bash
REPO_ROOT="${REPO_ROOT:-~/vllm-omni}"
# Required — same as step 1 if not already done:
rm -rf "$REPO_ROOT/logs" "$REPO_ROOT/tests/dfx/perf/results"
mkdir -p "$REPO_ROOT" && tar xzf nightly_logs.tgz -C "$REPO_ROOT"
```

- Job logs land at **`$REPO_ROOT/logs/nightly_jobs`**.
- DFX perf JSON (if present) lands at **`$REPO_ROOT/tests/dfx/perf/results/`** — feed into kanban via **`prepare_kanban_before_report.py`**, not read directly by the HTML report.

### 4. Verify, prepare kanban, and generate report (same for H200 and H800)

1. Confirm **`logs/nightly_jobs`** and **`tests/dfx/perf/results/`** (if applicable) under your local checkout.
2. **Kanban prep (before HTML report)** — [../../vllm-omni-test-report/references/kanban-pre-report-prep.md](../../vllm-omni-test-report/references/kanban-pre-report-prep.md):

```bash
export KANBAN_REPO_ROOT="${KANBAN_REPO_ROOT:-~/vllm-omni-kanban}"
export REPO_ROOT="${REPO_ROOT:-~/vllm-omni}"
cd ~/vllm-omni-skills/skills/vllm-omni-test-report   # or your skills checkout path
python scripts/prepare_kanban_before_report.py
```

This pulls latest kanban, copies perf JSON + job logs into **`data/local_nightly_raw/manual_*`** when results exist, and runs **`mkdocs build`** to refresh **`docs/assets/charts/*_history.json`**.

3. HTML nightly report — from **`skills/vllm-omni-test-report/`** ([../../vllm-omni-test-report/SKILL.md](../../vllm-omni-test-report/SKILL.md), report kind **nightly**):

```bash
export REPO_ROOT="${REPO_ROOT:-~/vllm-omni}"   # must match the tree you synced into
export KANBAN_REPO_ROOT="${KANBAN_REPO_ROOT:-~/vllm-omni-kanban}"
python scripts/nightly_local_log_report.py --html-report ./nightly-report.html \
  --kanban-repo-root "$KANBAN_REPO_ROOT"
# log-dir default: $REPO_ROOT/logs/nightly_jobs
# performance baseline comparison: --kanban-repo-root → docs/assets/charts/*_history.json
```

4. Release / combined report: **`--log-dir-h200`** or **`--log-dir-h800`** on **`compose_full_report.py`** as appropriate.

<a id="optional-scp--rsync"></a>

## Optional: scp / rsync

When the repo tree is visible on a **host bind-mount** (not only inside the container), you can sync without tarball. Apply **[step 1](#clear-local-trees)** first.

**Remote repo root** on the host: **`REMOTE_REPO="user@remote_host:/path/on/host/vllm-omni"`** (adjust to your bind-mount).

### scp (recursive)

```bash
REPO_ROOT="${REPO_ROOT:-~/vllm-omni}"
REMOTE_REPO="user@remote_host:/path/on/host/vllm-omni"
rm -rf "$REPO_ROOT/logs" "$REPO_ROOT/tests/dfx/perf/results"
mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/tests/dfx/perf"
scp -r "${REMOTE_REPO}/logs/nightly_jobs" "$REPO_ROOT/logs/"
scp -r "${REMOTE_REPO}/tests/dfx/perf/results" "$REPO_ROOT/tests/dfx/perf/" 2>/dev/null || true
```

### rsync

```bash
REPO_ROOT="${REPO_ROOT:-~/vllm-omni}"
REMOTE_REPO="user@remote_host:/path/on/host/vllm-omni"
rm -rf "$REPO_ROOT/logs" "$REPO_ROOT/tests/dfx/perf/results"
mkdir -p "$REPO_ROOT/logs/nightly_jobs" "$REPO_ROOT/tests/dfx/perf"
rsync -avz -e ssh "${REMOTE_REPO}/logs/nightly_jobs/" "$REPO_ROOT/logs/nightly_jobs/"
rsync -avz -e ssh "${REMOTE_REPO}/tests/dfx/perf/results/" "$REPO_ROOT/tests/dfx/perf/results/" 2>/dev/null || true
```

Then continue with [Verify and generate report](#4-verify-and-generate-report-same-for-h200-and-h800) (step 4 above).
