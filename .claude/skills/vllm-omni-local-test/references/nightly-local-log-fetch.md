# Fetch nightly logs to your laptop

Part of **vllm-omni-local-test**. Pull run directories from the cluster/container **before** running **`vllm-omni-test-report`** `scripts/nightly_local_log_report.py` on your machine.

**Before sync:** confirm laptop path defaults with the user — **`REPO_ROOT=~/vllm-omni`**, **`KANBAN_REPO_ROOT=~/vllm-omni-kanban`** — see [confirm-laptop-path-defaults.md](../../vllm-omni-test-report/references/confirm-laptop-path-defaults.md).

**H200 and H800 use the same laptop-side workflow** (clear local **`logs/`** → resolve remote run dir(s) → tarball → extract → merge into **`logs/nightly_jobs`** → report). Only the **remote pack command** differs: **H200** = direct **`ssh`** (already in container); **H800** = **`ssh` + `srun --overlap docker exec`**.

**Default destination on your laptop:** everything lands under **`$REPO_ROOT/logs/nightly_jobs/`** — job tee logs, optional perf JSON (`result_test_*.json`, …), and generated **`jobs/`** scripts. Report scripts and kanban prep read perf JSON recursively from that tree.

## Which remote directories to pull

Cluster/container paths are relative to **`CLUSTER_REPO_ROOT`** (default **`/rebase/vllm-omni`**).

Two **suffix families** exist under **`$CLUSTER_REPO_ROOT/logs/`**:

| Pattern | Meaning |
|---------|---------|
| **`nightly_jobs_local_*`** | Local-test nightly runs (`--test-type local`, etc.) |
| **`nightly_jobs_YYYYMMDD-HHMMSS`** | General / full nightly runs (**not** `nightly_jobs_local_*`) |

```text
$CLUSTER_REPO_ROOT/logs/
  nightly_jobs_local_20250628-143022/   ← local-test run (example)
  nightly_jobs_20250628-091530/         ← general nightly run (example)
  nightly_jobs_20250627-091530/
    …
```

**Do not** sync the legacy fixed path **`logs/nightly_jobs`** (no suffix) on the cluster — it may contain stale mixed runs.

### Sync scope (agent / user intent)

| User intent | Remote directories to pull |
|-------------|----------------------------|
| Explicit **local** log sync (e.g. “拉取 local 日志”, “local test logs”, “only local nightly”) | **Only** the latest **`nightly_jobs_local_*`** |
| **Default** (unspecified, or “拉取 nightly 日志”, general nightly report) | Latest **`nightly_jobs_local_*`** **and** latest **`nightly_jobs_YYYYMMDD-*`** (non-local) |

If one family is missing on the remote, sync what exists and note the skip in chat.

### Pick latest run dirs (remote shell)

Shared helpers — set **`SYNC_SCOPE=local`** or **`SYNC_SCOPE=default`**:

```bash
ROOT="${CLUSTER_REPO_ROOT:-/rebase/vllm-omni}"
LOGS_ROOT="${ROOT}/logs"
SYNC_SCOPE="${SYNC_SCOPE:-default}"   # local | default

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
LATEST_NIGHTLY_RUN=""

if _latest_matching "nightly_jobs_local_*" >/dev/null; then
  LATEST_LOCAL_RUN="$(_latest_matching "nightly_jobs_local_*")"
fi

if [[ "$SYNC_SCOPE" == "default" ]]; then
  shopt -s nullglob
  _nightly=()
  for _d in "${LOGS_ROOT}"/nightly_jobs_*; do
    _base="$(basename "$_d")"
    [[ "$_base" == nightly_jobs_local_* ]] && continue
    [[ "$_base" =~ ^nightly_jobs_[0-9]{8}- ]] && _nightly+=( "$_d" )
  done
  shopt -u nullglob
  if ((${#_nightly[@]} > 0)); then
    LATEST_NIGHTLY_RUN="$(ls -dt "${_nightly[@]}" | head -1)"
  fi
fi

PACK_DIRS=()
[[ -n "$LATEST_LOCAL_RUN" ]] && PACK_DIRS+=( "$(basename "$LATEST_LOCAL_RUN")" )
if [[ "$SYNC_SCOPE" == "default" && -n "$LATEST_NIGHTLY_RUN" ]]; then
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
| Latest **`nightly_jobs_local_*`** (always when present) | Per sync scope above | Merged into **`$REPO_ROOT/logs/nightly_jobs/`** |
| Latest **`nightly_jobs_YYYYMMDD-*`** (default scope only) | When scope is **default** | Same merge target |

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

# Default (local + general nightly):
export SYNC_SCOPE=default
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

**H800** — **`ssh` + Slurm + `docker exec`** ([nightly-local-h800.md](nightly-local-h800.md)) — embed the pick + pack block from above, then:

```bash
ssh -o BatchMode=yes "<SSH_CONNECTION_NAME>" \
  "bash -lc 'type module >/dev/null 2>&1 && module load slurm 2>/dev/null; srun --jobid=\"<JOBID>\" --overlap docker exec \"<CONTAINER_NAME>\" bash -lc \"
    SYNC_SCOPE=\\\"${SYNC_SCOPE:-default}\\\"
    # ... insert Pick latest run dirs block ...
    cd \\\"\\\$LOGS_ROOT\\\" || exit 1
    tar czf - --ignore-failed-read \\\"\\\${PACK_DIRS[@]}\\\"
  \"'" \
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

The tarball contains one or two top-level **`nightly_jobs_*`** folders under **`$REPO_ROOT/logs/`**.

### 4. Merge into `logs/nightly_jobs` (same for H200 and H800)

Merge extracted run dir(s) into **`logs/nightly_jobs`**. **Kanban `manual_*` uses only `nightly_jobs_local_*` perf** — copy that run into **`logs/.kanban_perf_source`** before merging general nightly (if any):

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
_general_dirs=()
for _d in "$REPO_ROOT/logs"/nightly_jobs_*; do
  _base="$(basename "$_d")"
  [[ "$_base" == nightly_jobs_local_* ]] && continue
  [[ "$_base" =~ ^nightly_jobs_[0-9]{8}- ]] && _general_dirs+=( "$_d" )
done
shopt -u nullglob

if ((${#_local_dirs[@]} > 0)); then
  merge_local_for_kanban "$(ls -dt "${_local_dirs[@]}" | head -1)"
fi
if [[ "$SYNC_SCOPE" == "default" && ${#_general_dirs[@]} -gt 0 ]]; then
  merge_run_dir "$(ls -dt "${_general_dirs[@]}" | head -1)"
fi

if [[ ! -s "$DEST/.nightly_jobs_source" ]]; then
  echo "No run dirs merged into $DEST (scope=${SYNC_SCOPE})" >&2
  exit 1
fi

# Remove extracted suffix dirs after merge (optional cleanup)
for _d in "${_local_dirs[@]}" "${_general_dirs[@]}"; do
  [[ "$_d" != "$DEST" && -d "$_d" ]] && rm -rf "$_d"
done

echo "Synced to: $DEST (sources: $(tr '\n' ' ' < "$DEST/.nightly_jobs_source"))"
[[ -d "$KANBAN_PERF_SRC" ]] && echo "Kanban perf source: $KANBAN_PERF_SRC (nightly_jobs_local_* only)"
```

- **`logs/nightly_jobs`**: merged job logs + perf for **HTML report** (local + general when scope is default).
- **`logs/.kanban_perf_source`**: **only** latest **`nightly_jobs_local_*`** — **`prepare_kanban_before_report.py`** copies perf JSON from here into kanban **`manual_*`**. General **`nightly_jobs_YYYYMMDD-*`** perf never goes to kanban.

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
SYNC_SCOPE="${SYNC_SCOPE:-default}"
REMOTE_REPO="user@remote_host:/path/on/host/vllm-omni"
REMOTE_LOCAL="$(ssh "$REMOTE_REPO" 'ls -dt logs/nightly_jobs_local_* 2>/dev/null | head -1')"
rm -rf "$REPO_ROOT/logs"
mkdir -p "$REPO_ROOT/logs"
[[ -n "$REMOTE_LOCAL" ]] && scp -r "${REMOTE_REPO}/${REMOTE_LOCAL}" "$REPO_ROOT/logs/"
if [[ "$SYNC_SCOPE" == "default" ]]; then
  REMOTE_NIGHTLY="$(ssh "$REMOTE_REPO" 'ls -dt logs/nightly_jobs_[0-9]* 2>/dev/null | head -1')"
  [[ -n "$REMOTE_NIGHTLY" ]] && scp -r "${REMOTE_REPO}/${REMOTE_NIGHTLY}" "$REPO_ROOT/logs/"
fi
# Then run step 4 merge commands on laptop
```

### rsync

Same pattern: pull **`nightly_jobs_local_*`** always; add **`nightly_jobs_YYYYMMDD-*`** only when **`SYNC_SCOPE=default`**, then run step 4.

Then continue with [Verify and generate report](#5-verify-prepare-kanban-and-generate-report-same-for-h200-and-h800) (step 5 above).
