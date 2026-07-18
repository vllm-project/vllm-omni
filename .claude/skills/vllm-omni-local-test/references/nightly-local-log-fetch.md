# Fetch nightly logs to your laptop

Part of **vllm-omni-local-test**. Pull run directories from the cluster/container **before** running **`vllm-omni-test-report`** `scripts/nightly_local_log_report.py` on your machine.

**⚠️ `/rebase/vllm-omni/logs/` is on shared NFS storage that is only visible on the compute nodes, NOT on the SSH login node.** Every `ssh` command in this file that touches `$REPO_ROOT` (or runs `ls … /rebase/vllm-omni/logs/nightly_jobs_*`, the pick-latest block, or the `tar czf - …` pack) **MUST be wrapped in an slurm allocation**:

```bash
# H800: pick a running job (or `salloc -w <node>` if none) and `srun --overlap` everything
ssh -o BatchMode=yes -o ConnectTimeout=60 "<SSH_CONNECTION_NAME>" \
  "bash -lc 'type module >/dev/null 2>&1 && module load slurm 2>/dev/null; \
     srun --jobid=\"<JOBID>\" --overlap --gres=gpu:0 bash -lc \"\
       export CLUSTER_REPO_ROOT=\${CLUSTER_REPO_ROOT:-/rebase/vllm-omni} && \
       export SYNC_SCOPE=… && <the pick + pack block from below> \
     \"'" > nightly_logs.tgz
```

If `ssh <host> "ls -d /rebase/vllm-omni"` returns "No such file or directory", you are on the login node and MUST slurm-attach first.

**Before sync:** confirm laptop path defaults with the user — **`REPO_ROOT=~/vllm-omni`**, **`KANBAN_REPO_ROOT=~/vllm-omni-kanban`** — see [confirm-laptop-path-defaults.md](../../vllm-omni-test-report/references/confirm-laptop-path-defaults.md).

**H200 and H800 use the same laptop-side workflow** (clear local **`logs/`** → resolve remote run dir(s) → tarball → extract → merge into **`logs/nightly_jobs`** → report). Only the **remote pack command** differs: **H200** = direct **`ssh`** (already in container); **H800** = **`ssh` + `srun --overlap docker exec`**.

**Default destination on your laptop:** everything lands under **`$REPO_ROOT/logs/nightly_jobs/`** — job tee logs, optional perf JSON (`result_test_*.json`, …), and generated **`jobs/`** scripts. Report scripts and kanban prep read perf JSON recursively from that tree.

## Which remote directories to pull

Cluster/container paths are relative to **`CLUSTER_REPO_ROOT`** (default **`/rebase/vllm-omni`**).

Three **suffix families** exist under **`$CLUSTER_REPO_ROOT/logs/`**:

| Pattern | Meaning |
|---------|---------|
| **`nightly_jobs_local_*`** | Local-test nightly runs (`--test-type local`, etc.) |
| **`nightly_stability_jobs_*`** | Long-stability nightly runs (`--test-type stability`, etc.) — **NOTE the order: `…stability_jobs…` not `…jobs_stability…`** |
| **`nightly_jobs_YYYYMMDD-HHMMSS`** | General / full nightly runs (**not** local/stability) |

```text
$CLUSTER_REPO_ROOT/logs/
  nightly_jobs_local_20250628-143022/      ← local-test run (example)
  nightly_stability_jobs_20250628-143022/  ← stability-test run (example)
  nightly_jobs_20250628-091530/            ← general nightly run (example)
  nightly_jobs_20250627-091530/
    …
```

**Do not** sync the legacy fixed path **`logs/nightly_jobs`** (no suffix) on the cluster — it may contain stale mixed runs.

### Sync scope (agent / user intent)

| User intent | Remote directories to pull |
|-------------|----------------------------|
| Explicit **local** log sync (e.g. "拉取 local 日志", "local test logs", "only local nightly") | **Only** the latest **`nightly_jobs_local_*`** |
| Explicit **stability** log sync (e.g. "拉取 stability 日志", "stability test logs", "only stability nightly") | **Only** the latest **`nightly_stability_jobs_*`** |
| **Default** (unspecified, or "拉取 nightly 日志", general nightly report) | **Only** the latest **`nightly_jobs_YYYYMMDD-*`** (general nightly, **not** local/stability) |

If one family is missing on the remote, sync what exists and note the skip in chat.

**Implicit scope (auto-detected after a test run):** when the user starts the session by running test cases first (rather than calling the log-fetch step alone), pull **all logs related to that run** instead of asking for the keyword:

- Ran with `--test-type local` (only) → scope = `local` → pull latest `nightly_jobs_local_*`.
- Ran with `--test-type stability` (only) → scope = `stability` → pull latest `nightly_stability_jobs_*`.
- Ran full nightly (no `--test-type`) → scope = `default` → pull latest `nightly_jobs_YYYYMMDD-*`.
- Combined `--test-type local,stability` (or `--test-type all,local,stability`) → scope = `all` → pull **all three** latest runs (local + stability + general nightly).

This is automatic — the agent records the run's effective `--test-type` and applies the matching scope once the run finishes. The user can still override by stating a different keyword (e.g. "只拉取 stability 日志").

### Pick latest run dirs (remote shell)

Shared helpers — set **`SYNC_SCOPE=local|stability|default|all`** (default = `default`; `all` = local + stability + general nightly):

```bash
ROOT="${CLUSTER_REPO_ROOT:-/rebase/vllm-omni}"
LOGS_ROOT="${ROOT}/logs"
SYNC_SCOPE="${SYNC_SCOPE:-default}"   # local | stability | default | all

_latest_matching() {
  local glob_pat="$1"
  shopt -s nullglob
  local _c=( "${LOGS_ROOT}"/${glob_pat} )
  shopt -u nullglob
  if ((${#_c[@]} == 0)); then
    return 1
  fi
  ls -dt "${_c[@]}" | head -1
}

LATEST_LOCAL_RUN=""
LATEST_STABILITY_RUN=""
LATEST_NIGHTLY_RUN=""

if [[ "$SYNC_SCOPE" == "local" || "$SYNC_SCOPE" == "all" ]]; then
  if _latest_matching "nightly_jobs_local_*" >/dev/null; then
    LATEST_LOCAL_RUN="$(_latest_matching "nightly_jobs_local_*")"
  fi
fi

if [[ "$SYNC_SCOPE" == "stability" || "$SYNC_SCOPE" == "all" ]]; then
  if _latest_matching "nightly_stability_jobs_*" >/dev/null; then
    LATEST_STABILITY_RUN="$(_latest_matching "nightly_stability_jobs_*")"
  fi
fi

if [[ "$SYNC_SCOPE" == "default" || "$SYNC_SCOPE" == "all" ]]; then
  shopt -s nullglob
  _nightly=()
  for _d in "${LOGS_ROOT}"/nightly_jobs_*; do
    _base="$(basename "$_d")"
    [[ "$_base" == nightly_jobs_local_* ]] && continue
    [[ "$_base" == nightly_stability_jobs_* ]] && continue
    [[ "$_base" =~ ^nightly_jobs_[0-9]{8}- ]] && _nightly+=( "$_d" )
  done
  shopt -u nullglob
  if ((${#_nightly[@]} > 0)); then
    LATEST_NIGHTLY_RUN="$(ls -dt "${_nightly[@]}" | head -1)"
  fi
fi

PACK_DIRS=()
[[ -n "$LATEST_LOCAL_RUN" ]] && PACK_DIRS+=( "$(basename "$LATEST_LOCAL_RUN")" )
[[ -n "$LATEST_STABILITY_RUN" ]] && PACK_DIRS+=( "$(basename "$LATEST_STABILITY_RUN")" )
if [[ ( "$SYNC_SCOPE" == "default" || "$SYNC_SCOPE" == "all" ) && -n "$LATEST_NIGHTLY_RUN" ]]; then
  PACK_DIRS+=( "$(basename "$LATEST_NIGHTLY_RUN")" )
fi
if ((${#PACK_DIRS[@]} == 0)); then
  echo "No matching nightly_jobs_* directories under ${LOGS_ROOT} (scope=${SYNC_SCOPE})" >&2
  exit 1
fi
echo "Sync scope: ${SYNC_SCOPE}; packing: ${PACK_DIRS[*]}" >&2
```

Optional: ask the user to confirm **`PACK_DIRS`** before tarball when multiple runs exist the same day.

## What to copy

| Source | Required | Local destination |
|--------|----------|-------------------|
| Latest **`nightly_jobs_local_*`** (when scope = local or all) | Per sync scope above | Merged into **`$REPO_ROOT/logs/nightly_jobs/`** |
| Latest **`nightly_stability_jobs_*`** (when scope = stability or all) | Per sync scope above | Same merge target |
| Latest **`nightly_jobs_YYYYMMDD-*`** (when scope = default or all) | When scope is **default** or **all** | Same merge target |

Layout after merge: [../../vllm-omni-test-report/references/nightly-local-log-layout.md](../../vllm-omni-test-report/references/nightly-local-log-layout.md).

**Remote pack inner logic** (tarball may contain one or two top-level run dirs):

```bash
cd "${LOGS_ROOT}" || exit 1
tar czf - --ignore-failed-read "${PACK_DIRS[@]}"
```

<a id="log-sync-workflow"></a>

## Log sync workflow (H200 and H800)

Run these steps **in order** on your laptop **after the cluster run finishes** and **before** generating the report. **H200** and **H800** share steps **1**, **3**, **4**, and **5**; step **2** picks the machine-specific remote command.

**Set scope before step 2:**

```bash
# User asked for local logs only:
export SYNC_SCOPE=local

# User asked for stability logs only:
export SYNC_SCOPE=stability

# Default (general nightly only — NOT local/stability):
export SYNC_SCOPE=default

# All three (local + stability + general nightly):
export SYNC_SCOPE=all
```

<a id="clear-local-trees"></a>

### 1. Clear local `logs/` (required)

**Always delete the existing *local* `logs` directory before each sync.** Otherwise old job folders or stale perf JSON can **merge with the new pull** and skew the nightly report.

```bash
REPO_ROOT="${REPO_ROOT:-~/vllm-omni}"
rm -rf "$REPO_ROOT/logs"
```

- **Scope:** your **laptop** checkout only; nothing on the cluster/container is removed.
- **Archive instead:** if you need a backup, rename before delete, e.g. `mv "$REPO_ROOT/logs" "$REPO_ROOT/logs.bak.$(date +%Y%m%d%H%M)"`.

Run this **before** remote tarball download (step 2) and again **before** extract (step 3) if step 1 was skipped earlier.

### 2. Remote tarball (machine-specific)

Pack **`PACK_DIRS`** from [Pick latest run dirs](#pick-latest-run-dirs-remote-shell) in one archive.

**H800** — first **find your own jobid**, then wrap the pick + pack block in `srun --overlap docker exec <CONTAINER_NAME>`:

```bash
# Step 1: find YOUR OWN jobid (not "any" job on the node — those will deny access)
SLURM_USER="<username>"   # the user gave this, e.g. fq9hpsacuser07
JOBID=$(ssh -o BatchMode=yes "<SSH_CONNECTION_NAME>" \
  "bash -lc 'type module >/dev/null 2>&1 && module load slurm 2>/dev/null; squeue -u $SLURM_USER -t RUNNING -h -o %i | head -1'")
echo "Using JOBID=$JOBID"

# Step 2: enter the WORKLOAD CONTAINER (logs are only inside the container, not the slurm allocation)
CONTAINER_NAME="<container_name>"   # the user gave this, e.g. omni_wy_24g

ssh -o BatchMode=yes "<SSH_CONNECTION_NAME>" \
  "bash -lc 'type module >/dev/null 2>&1 && module load slurm 2>/dev/null; srun --jobid=\\\"$JOBID\\\" --overlap docker exec \\\"$CONTAINER_NAME\\\" bash -lc \\"
    SYNC_SCOPE=\\\\\"${SYNC_SCOPE:-default}\\\\\"
    # ... insert Pick latest run dirs block from above ...
    cd \\\\\\"\\$LOGS_ROOT\\\\\\" || exit 1
    tar czf - --ignore-failed-read \\\\\\"\\${PACK_DIRS[@]}\\\\\\"
  \\""' \
  > nightly_logs.tgz
```

**H200** — direct **`ssh`** ([nightly-local-h200.md](nightly-local-h200.md)):

```bash
ssh -o BatchMode=yes "<SSH_CONNECTION_NAME>" \
  "bash -lc '
    SYNC_SCOPE=\"${SYNC_SCOPE:-default}\"
    # ... insert Pick latest run dirs block ...
    cd \"\$LOGS_ROOT\" || exit 1
    tar czf - --ignore-failed-read \"\${PACK_DIRS[@]}\"
  '" \
  > nightly_logs.tgz
```

Use the **confirmed cluster repo root** in **`ROOT`** when the user overrode the default **`/rebase/vllm-omni`**.

### 3. Extract on laptop (same for H200 and H800)

```bash
REPO_ROOT="${REPO_ROOT:-~/vllm-omni}"
rm -rf "$REPO_ROOT/logs"
mkdir -p "$REPO_ROOT/logs"
tar xzf nightly_logs.tgz -C "$REPO_ROOT/logs"
```

The tarball contains one, two, or three top-level **`nightly_jobs_*`** folders under **`$REPO_ROOT/logs/`** (depending on scope: local / stability / default / all).

### 4. Merge into `logs/nightly_jobs` (same for H200 and H800)

Merge extracted run dir(s) into **`logs/nightly_jobs`**. **Kanban `manual_*` uses only `nightly_jobs_local_*` perf** — when a local run is in the merged set, copy it into **`logs/.kanban_perf_source`** first. Stability and general nightly runs never feed kanban `manual_*`:

```bash
REPO_ROOT="${REPO_ROOT:-~/vllm-omni}"
SYNC_SCOPE="${SYNC_SCOPE:-default}"
DEST="$REPO_ROOT/logs/nightly_jobs"
KANBAN_PERF_SRC="$REPO_ROOT/logs/.kanban_perf_source"
mkdir -p "$DEST"
rm -rf "$KANBAN_PERF_SRC"
: > "$DEST/.nightly_jobs_source"

merge_run_dir() {
  local src="$1"
  [[ -d "$src" ]] || return 0
  cp -a "$src"/. "$DEST"/
  echo "$(basename "$src")" >> "$DEST/.nightly_jobs_source"
}

merge_local_for_kanban() {
  local src="$1"
  [[ -d "$src" ]] || return 0
  rm -rf "$KANBAN_PERF_SRC"
  mkdir -p "$KANBAN_PERF_SRC"
  cp -a "$src"/. "$KANBAN_PERF_SRC"/
  merge_run_dir "$src"
}

shopt -s nullglob
_local_dirs=( "$REPO_ROOT/logs"/nightly_jobs_local_* )
_stability_dirs=( "$REPO_ROOT/logs"/nightly_stability_jobs_* )
_general_dirs=()
for _d in "$REPO_ROOT/logs"/nightly_jobs_*; do
  _base="$(basename "$_d")"
  [[ "$_base" == nightly_jobs_local_* ]] && continue
  [[ "$_base" == nightly_stability_jobs_* ]] && continue
  [[ "$_base" =~ ^nightly_jobs_[0-9]{8}- ]] && _general_dirs+=( "$_d" )
done
shopt -u nullglob

# Local run: merged AND copied to kanban perf source (scope local | stability | default | all when local is in scope)
if [[ "$SYNC_SCOPE" == "local" || "$SYNC_SCOPE" == "all" ]] && ((${#_local_dirs[@]} > 0)); then
  merge_local_for_kanban "$(ls -dt "${_local_dirs[@]}" | head -1)"
fi

# Stability run: merged only (no kanban perf source)
if [[ "$SYNC_SCOPE" == "stability" || "$SYNC_SCOPE" == "all" ]] && ((${#_stability_dirs[@]} > 0)); then
  merge_run_dir "$(ls -dt "${_stability_dirs[@]}" | head -1)"
fi

# General nightly run: merged only when scope = default or all
if [[ "$SYNC_SCOPE" == "default" || "$SYNC_SCOPE" == "all" ]] && ((${#_general_dirs[@]} > 0)); then
  merge_run_dir "$(ls -dt "${_general_dirs[@]}" | head -1)"
fi

if [[ ! -s "$DEST/.nightly_jobs_source" ]]; then
  echo "No run dirs merged into $DEST (scope=${SYNC_SCOPE})" >&2
  exit 1
fi

# Remove extracted suffix dirs after merge (optional cleanup)
for _d in "${_local_dirs[@]}" "${_stability_dirs[@]}" "${_general_dirs[@]}"; do
  [[ "$_d" != "$DEST" && -d "$_d" ]] && rm -rf "$_d"
done

echo "Synced to: $DEST (sources: $(tr '\n' ' ' < "$DEST/.nightly_jobs_source"))"
if [[ -d "$KANBAN_PERF_SRC" ]]; then
  echo "Kanban perf source: $KANBAN_PERF_SRC (nightly_jobs_local_* only)"
fi
```

- **`logs/nightly_jobs`**: merged job logs + perf for **HTML report** (local + stability + general when scope is `all`; only the matching family for `local` / `stability` / `default`).
- **`logs/.kanban_perf_source`**: **only** latest **`nightly_jobs_local_*`** — **`prepare_kanban_before_report.py`** copies perf JSON from here into kanban **`manual_*`**. Stability and general **`nightly_jobs_YYYYMMDD-*`** perf never go to kanban.

### 5. Verify, prepare kanban, and generate report (same for H200 and H800)

1. Confirm **`logs/nightly_jobs`** under your local checkout.
2. **Kanban prep (before HTML report)** — [../../vllm-omni-test-report/references/kanban-pre-report-prep.md](../../vllm-omni-test-report/references/kanban-pre-report-prep.md):

```bash
export KANBAN_REPO_ROOT="${KANBAN_REPO_ROOT:-~/vllm-omni-kanban}"
export REPO_ROOT="${REPO_ROOT:-~/vllm-omni}"
cd ~/vllm-omni-skills/skills/vllm-omni-test-report   # or your skills checkout path
python scripts/prepare_kanban_before_report.py
```

This pulls latest kanban, copies perf JSON + job logs from **`logs/nightly_jobs`** into **`data/local_nightly_raw/manual_*`** when present, and runs **`mkdocs build`** to refresh **`docs/assets/charts/*_history.json`**.

3. HTML nightly report — from **`skills/vllm-omni-test-report/`** ([../../vllm-omni-test-report/SKILL.md](../../vllm-omni-test-report/SKILL.md), report kind **nightly**). Output filename uses **generation date (UTC today)**, not the remote run suffix:

```bash
export REPO_ROOT="${REPO_ROOT:-~/vllm-omni}"
export KANBAN_REPO_ROOT="${KANBAN_REPO_ROOT:-~/vllm-omni-kanban}"
python scripts/nightly_local_log_report.py \
  --kanban-repo-root "$KANBAN_REPO_ROOT"
# default output: ./nightly-report-buildkite-latest-YYYY-MM-DD.html (generation date)
# log-dir default: $REPO_ROOT/logs/nightly_jobs
```

4. Release / combined report: **`--log-dir-h200`** or **`--log-dir-h800`** on **`compose_full_report.py`** as appropriate.

<a id="optional-scp--rsync"></a>

## Optional: scp / rsync

When the repo tree is visible on a **host bind-mount** (not only inside the container), sync without tarball. Apply **[step 1](#clear-local-trees)** first, resolve run dirs on the host per **sync scope**, then merge per **[step 4](#4-merge-into-logsnightly_jobs-same-for-h200-and-h800)**.

**Remote repo root** on the host: **`REMOTE_REPO="user@remote_host:/path/on/host/vllm-omni"`**.

### scp (recursive)

```bash
REPO_ROOT="${REPO_ROOT:-~/vllm-omni}"
SYNC_SCOPE="${SYNC_SCOPE:-default}"   # local | stability | default | all
REMOTE_REPO="user@remote_host:/path/on/host/vllm-omni"
rm -rf "$REPO_ROOT/logs"
mkdir -p "$REPO_ROOT/logs"

if [[ "$SYNC_SCOPE" == "local" || "$SYNC_SCOPE" == "all" ]]; then
  REMOTE_LOCAL="$(ssh "$REMOTE_REPO" 'ls -dt logs/nightly_jobs_local_* 2>/dev/null | head -1')"
  [[ -n "$REMOTE_LOCAL" ]] && scp -r "${REMOTE_REPO}/${REMOTE_LOCAL}" "$REPO_ROOT/logs/"
fi
if [[ "$SYNC_SCOPE" == "stability" || "$SYNC_SCOPE" == "all" ]]; then
  REMOTE_STABILITY="$(ssh "$REMOTE_REPO" 'ls -dt logs/nightly_stability_jobs_* 2>/dev/null | head -1')"
  [[ -n "$REMOTE_STABILITY" ]] && scp -r "${REMOTE_REPO}/${REMOTE_STABILITY}" "$REPO_ROOT/logs/"
fi
if [[ "$SYNC_SCOPE" == "default" || "$SYNC_SCOPE" == "all" ]]; then
  REMOTE_NIGHTLY="$(ssh "$REMOTE_REPO" 'ls -dt logs/nightly_jobs_[0-9]* 2>/dev/null | head -1')"
  [[ -n "$REMOTE_NIGHTLY" ]] && scp -r "${REMOTE_REPO}/${REMOTE_NIGHTLY}" "$REPO_ROOT/logs/"
fi
# Then run step 4 merge commands on laptop
```

### rsync

Same pattern: pull **`nightly_jobs_local_*`** when scope = `local`/`all`; **`nightly_stability_jobs_*`** when scope = `stability`/`all`; **`nightly_jobs_YYYYMMDD-*`** only when scope = `default`/`all`; then run step 4.

Then continue with [Verify and generate report](#5-verify-prepare-kanban-and-generate-report-same-for-h200-and-h800) (step 5 above).
