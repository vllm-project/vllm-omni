# Fetch nightly logs to your laptop

Part of **vllm-omni-local-test**. Pull the **latest** **`logs/nightly_jobs_*`** run directory from the cluster/container **before** running **`vllm-omni-test-report`** `scripts/nightly_local_log_report.py` on your machine.

**Before sync:** confirm laptop path defaults with the user — **`REPO_ROOT=~/vllm-omni`**, **`KANBAN_REPO_ROOT=~/vllm-omni-kanban`** — see [confirm-laptop-path-defaults.md](../../vllm-omni-test-report/references/confirm-laptop-path-defaults.md).

**H200 and H800 use the same laptop-side workflow** (clear local **`logs/`** → resolve latest remote **`nightly_jobs_*`** → tarball → extract → rename to **`logs/nightly_jobs`** → report). Only the **remote pack command** differs: **H200** = direct **`ssh`** (already in container); **H800** = **`ssh` + `srun --overlap docker exec`**.

**Default destination on your laptop:** everything lands under **`$REPO_ROOT/logs/nightly_jobs/`** — job tee logs, optional perf JSON (`result_test_*.json`, …), and generated **`jobs/`** scripts. Report scripts and kanban prep read perf JSON recursively from that tree.

## Remote source layout

Cluster/container paths are relative to the **confirmed cluster repo root** (**`CLUSTER_REPO_ROOT`**, default **`/rebase/vllm-omni`**).

Each nightly run writes under a **timestamped or suffixed** directory:

```text
$CLUSTER_REPO_ROOT/logs/
  nightly_jobs_20250628-143022/     ← latest (example)
    *.log                           job tee logs
    jobs/                           generated wrapper scripts
    results/                        optional DFX perf JSON (may also sit at run root)
  nightly_jobs_20250627-091530/
    …
```

**Do not** sync the legacy fixed path **`logs/nightly_jobs`** (no suffix) — it may contain stale mixed runs. Always pick the **newest** directory matching **`logs/nightly_jobs_*`**.

### Pick latest `nightly_jobs_*` (remote)

Shared inner logic (runs with **`cd "$ROOT/logs"`** or equivalent):

```bash
ROOT="${CLUSTER_REPO_ROOT:-/rebase/vllm-omni}"
LOGS_ROOT="${ROOT}/logs"
shopt -s nullglob
_candidates=( "${LOGS_ROOT}"/nightly_jobs_* )
shopt -u nullglob
if ((${#_candidates[@]} == 0)); then
  echo "No nightly_jobs_* directory under ${LOGS_ROOT}" >&2
  exit 1
fi
LATEST_RUN="$(ls -dt "${_candidates[@]}" | head -1)"
LATEST_NAME="$(basename "${LATEST_RUN}")"
echo "Using latest run dir: ${LATEST_RUN}" >&2
```

Optional: ask the user to confirm **`LATEST_NAME`** before tarball when multiple runs exist the same day.

## What to copy

| Source (latest **`logs/nightly_jobs_*`**) | Required | Local destination |
|-------------------------------------------|----------|-------------------|
| Entire run directory (logs, **`jobs/`**, perf JSON) | Yes | **`$REPO_ROOT/logs/nightly_jobs/`** — layout: [../../vllm-omni-test-report/references/nightly-local-log-layout.md](../../vllm-omni-test-report/references/nightly-local-log-layout.md) |

**Remote pack inner logic** (shared by H200/H800 tarball — packs only the latest run dir):

```bash
ROOT="${CLUSTER_REPO_ROOT:-/rebase/vllm-omni}"
LOGS_ROOT="${ROOT}/logs"
shopt -s nullglob
_candidates=( "${LOGS_ROOT}"/nightly_jobs_* )
shopt -u nullglob
LATEST_RUN="$(ls -dt "${_candidates[@]}" | head -1)"
LATEST_NAME="$(basename "${LATEST_RUN}")"
cd "${LOGS_ROOT}" || exit 1
tar czf - --ignore-failed-read "${LATEST_NAME}"
```

<a id="log-sync-workflow"></a>

## Log sync workflow (H200 and H800)

Run these steps **in order** on your laptop **after the cluster run finishes** and **before** generating the report. **H200** and **H800** share steps **1**, **3**, **4**, and **5**; step **2** picks the machine-specific remote command.

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

Pack the **latest** **`logs/nightly_jobs_*`** directory in one archive.

**H800** — **`ssh` + Slurm + `docker exec`** ([nightly-local-h800.md](nightly-local-h800.md)):

```bash
ssh -o BatchMode=yes "<SSH_CONNECTION_NAME>" \
  "bash -lc 'type module >/dev/null 2>&1 && module load slurm 2>/dev/null; srun --jobid=\"<JOBID>\" --overlap docker exec \"<CONTAINER_NAME>\" bash -lc \"
    ROOT=\\\${CLUSTER_REPO_ROOT:-/rebase/vllm-omni}
    LOGS_ROOT=\\\"\\\$ROOT/logs\\\"
    shopt -s nullglob
    _c=( \\\"\\\$LOGS_ROOT\\\"/nightly_jobs_* )
    shopt -u nullglob
    LATEST=\\\$(ls -dt \\\"\\\${_c[@]}\\\" | head -1)
    NAME=\\\$(basename \\\"\\\$LATEST\\\")
    echo Using latest: \\\"\\\$LATEST\\\" >&2
    cd \\\"\\\$LOGS_ROOT\\\" || exit 1
    tar czf - --ignore-failed-read \\\"\\\$NAME\\\"
  \"'" \
  > nightly_logs.tgz
```

**H200** — direct **`ssh`** (session already in container; no Slurm, no **`docker exec`**) ([nightly-local-h200.md](nightly-local-h200.md)):

```bash
ssh -o BatchMode=yes "<SSH_CONNECTION_NAME>" \
  "bash -lc '
    ROOT=\${CLUSTER_REPO_ROOT:-/rebase/vllm-omni}
    LOGS_ROOT=\"\$ROOT/logs\"
    shopt -s nullglob
    _c=( \"\$LOGS_ROOT\"/nightly_jobs_* )
    shopt -u nullglob
    LATEST=\$(ls -dt \"\${_c[@]}\" | head -1)
    NAME=\$(basename \"\$LATEST\")
    echo Using latest: \"\$LATEST\" >&2
    cd \"\$LOGS_ROOT\" || exit 1
    tar czf - --ignore-failed-read \"\$NAME\"
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

The tarball contains a single top-level **`nightly_jobs_*`** folder under **`$REPO_ROOT/logs/`**.

### 4. Rename to `logs/nightly_jobs` (same for H200 and H800)

```bash
REPO_ROOT="${REPO_ROOT:-~/vllm-omni}"
shopt -s nullglob
_run_dirs=( "$REPO_ROOT/logs"/nightly_jobs_* )
shopt -u nullglob
if ((${#_run_dirs[@]} == 0)); then
  echo "No nightly_jobs_* under $REPO_ROOT/logs after extract" >&2
  exit 1
fi
LATEST_LOCAL="$(ls -dt "${_run_dirs[@]}" | head -1)"
if [[ -d "$REPO_ROOT/logs/nightly_jobs" ]]; then
  rm -rf "$REPO_ROOT/logs/nightly_jobs"
fi
mv "$LATEST_LOCAL" "$REPO_ROOT/logs/nightly_jobs"
echo "$(basename "$LATEST_LOCAL")" > "$REPO_ROOT/logs/nightly_jobs/.nightly_jobs_source"
echo "Synced to: $REPO_ROOT/logs/nightly_jobs (source: $(cat "$REPO_ROOT/logs/nightly_jobs/.nightly_jobs_source"))"
```

Job logs and perf JSON (wherever they sit under the run dir) remain under **`$REPO_ROOT/logs/nightly_jobs`**. Report scripts scan that tree recursively.

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

3. HTML nightly report — from **`skills/vllm-omni-test-report/`** ([../../vllm-omni-test-report/SKILL.md](../../vllm-omni-test-report/SKILL.md), report kind **nightly**). Output filename uses **UTC today**, not the remote `nightly_jobs_*` suffix:

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

When the repo tree is visible on a **host bind-mount** (not only inside the container), sync without tarball. Apply **[step 1](#clear-local-trees)** first, resolve **`LATEST_RUN`** on the host, then rename per **[step 4](#4-rename-to-logsnightly_jobs-same-for-h200-and-h800)**.

**Remote repo root** on the host: **`REMOTE_REPO="user@remote_host:/path/on/host/vllm-omni"`**.

### scp (recursive)

```bash
REPO_ROOT="${REPO_ROOT:-~/vllm-omni}"
REMOTE_REPO="user@remote_host:/path/on/host/vllm-omni"
REMOTE_LATEST="$(ssh "$REMOTE_REPO" 'ls -dt logs/nightly_jobs_* 2>/dev/null | head -1')"
rm -rf "$REPO_ROOT/logs"
mkdir -p "$REPO_ROOT/logs"
scp -r "${REMOTE_REPO}/${REMOTE_LATEST}" "$REPO_ROOT/logs/"
# Then run step 4 rename commands on laptop
```

### rsync

```bash
REPO_ROOT="${REPO_ROOT:-~/vllm-omni}"
REMOTE_REPO="user@remote_host:/path/on/host/vllm-omni"
REMOTE_LATEST="$(ssh "${REMOTE_REPO%%:*}" "ls -dt ${REMOTE_REPO#*:}/logs/nightly_jobs_* 2>/dev/null | head -1")"
REMOTE_NAME="$(basename "$REMOTE_LATEST")"
rm -rf "$REPO_ROOT/logs"
mkdir -p "$REPO_ROOT/logs"
rsync -avz -e ssh "${REMOTE_REPO}/logs/${REMOTE_NAME}/" "$REPO_ROOT/logs/${REMOTE_NAME}/"
# Then run step 4 rename commands on laptop
```

Then continue with [Verify and generate report](#5-verify-prepare-kanban-and-generate-report-same-for-h200-and-h800) (step 5 above).
