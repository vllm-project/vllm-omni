---
name: vllm-omni-local-test
description: **H200** or **H800** cluster nightly runs. **WARNING on H800:** `/rebase/vllm-omni` is NFS storage only visible inside the slurm allocation (or H200 container shell) — never on the bare SSH login node. Always `srun --overlap` (or `salloc`) before any log-listing or tarball; see SKILL §1. — confirm default **`REPO_ROOT`**, **`HF_HOME`**, **`CUDA_VISIBLE_DEVICES`** with user before run; after connect, **`cd "$REPO_ROOT"`** and **ask whether to `git pull`** before cases; then **`source /rebase/.venv/bin/activate`**, `run_nightly_jobs.sh` (optional **`--test-type local|stability|local,stability`**, **`--label-substr`**, **`--log-dir logs/nightly_jobs_*`**). **Log sync scope:** user says **local** → latest **`nightly_jobs_local_*` only**; **stability** → latest **`nightly_stability_jobs_*` only**; **default** → latest **`nightly_jobs_YYYYMMDD-*` only** (general nightly, not local/stability); when the user **starts the session by running test cases first**, sync scope is auto-detected from the run's effective `--test-type` so all related logs are pulled. Defaults: **`REPO_ROOT=/rebase/vllm-omni`**; H200 **`HF_HOME=/models/`**, **`CUDA_VISIBLE_DEVICES=0,1,2,3`**; H800 **`HF_HOME=/home/models/`**, GPU via **`nvidia-smi`** or explicit list. Use when user specifies H200/H800, local/stability nightly jobs, or fetching nightly logs.
---

# vLLM-Omni Local Test (cluster run & log sync)

## Overview

1. **Login** — **H200:** **`ssh`** → run in container shell (**no `docker exec`**). **H800:** **`ssh`** → Slurm → **`srun --overlap docker exec`**. **⚠️ The default `CLUSTER_REPO_ROOT` (`/rebase/vllm-omni`) is on shared NFS storage that is only mounted on the compute nodes, NOT on the SSH login node.** A plain `ssh <host> "ls /rebase/vllm-omni"` will return `No such file or directory`. **Before any log-listing, `cd "$REPO_ROOT"`, or tarball step, you MUST be inside the slurm allocation (H800) or the H200 container shell.** Concretely:
   - **H800:**
   ```bash
   # Step 1: SSH to the login node and find YOUR OWN jobid
   ssh <host> "bash -lc 'type module >/dev/null 2>&1 && module load slurm 2>/dev/null; squeue -u $SLURM_USER -t RUNNING -h -o %i'"

   # Step 2: srun --overlap into your allocation to check the storage
   ssh <host> "srun --jobid=<JOBID> --overlap --gres=gpu:0 bash -lc 'ls /rebase/vllm-omni/logs/'"

   # Step 3: enter the WORKLOAD CONTAINER to pull logs
   ssh <host> "srun --jobid=<JOBID> --overlap docker exec <CONTAINER_NAME> bash -lc 'cd /rebase/vllm-omni/logs && tar czf - nightly_stability_jobs_*' > nightly_logs.tgz"
   ```
   The remote pack step in [references/nightly-local-log-fetch.md](references/nightly-local-log-fetch.md) wraps `srun --overlap docker exec <CONTAINER_NAME> ...` for you. **Logs are only inside the workload container, not in the bare slurm allocation.**sh -lc 'type module >/dev/null 2>&1 && module load slurm 2>/dev/null; squeue -u $SLURM_USER -t RUNNING -h -o %i'"

   # Step 2: srun --overlap into your allocation to check the storage
   ssh <host> "srun --jobid=<JOBID> --overlap --gres=gpu:0 bash -lc 'ls /rebase/vllm-omni/logs/'"

   # Step 3: enter the WORKLOAD CONTAINER to pull logs
   ssh <host> "srun --jobid=<JOBID> --overlap docker exec <CONTAINER_NAME> bash -lc 'cd /rebase/vllm-omni/logs && tar czf - nightly_stability_jobs_*' > nightly_logs.tgz"
   ```
   The remote pack step in [references/nightly-local-log-fetch.md](references/nightly-local-log-fetch.md) wraps `srun --overlap docker exec <CONTAINER_NAME> ...` for you. **Logs are only inside the workload container, not in the bare slurm allocation.**
   - **H200:** `ssh <host> bash -lc '…'` — the login IS the compute node, no slurm needed.
   - **Test:** if `ssh <host> "ls -d /rebase/vllm-omni"` returns "No such file or directory", you are on the wrong node and must slurm-attach first.
2. **Run cases** — venv, HF / vLLM, **`CUDA_VISIBLE_DEVICES`**, **`cd "$REPO_ROOT"`**, **ask user → optional `git pull`**, then **`run_nightly_jobs.sh`** with the right **`--test-type` / `--label-substr`** — [Test run mode](#test-run-mode) and [references/nightly-local-environment.md](references/nightly-local-environment.md).
3. **Sync logs** — [references/nightly-local-log-fetch.md](references/nightly-local-log-fetch.md). The remote pack step (`tar czf - nightly_stability_jobs_* …`) MUST run inside the slurm allocation (H800) — see **§1** above.

**HTML report:** [vllm-omni-test-report](../vllm-omni-test-report/SKILL.md) **`nightly_local_log_report.py --html-report …`**.

## Machine type

| Type | Trigger | Login | Reference |
|------|---------|-------|-----------|
| **H200** | **H200**, **H200 machine**, **use H200**, **on H200** | **`ssh "<SSH_CONNECTION_NAME>"`** — **already in container**; **`bash -lc '…'`** on remote | [references/nightly-local-h200.md](references/nightly-local-h200.md) |
| **H800** | **H800**, **H800 machine**, **use H800**, **on H800** | **`ssh`** → **`squeue`** → **`srun --jobid=… --overlap docker exec …`** | [references/nightly-local-h800.md](references/nightly-local-h800.md) |

If neither **H200** nor **H800** is specified, **ask** which machine type before running commands.

### Confirm run defaults (required before run)

Show defaults and **ask the user** before connecting or executing — full rules: [references/nightly-local-environment.md](references/nightly-local-environment.md#confirm-run-defaults-with-user).

| Variable | H200 default | H800 default |
|----------|--------------|--------------|
| **`REPO_ROOT`** | `/rebase/vllm-omni` | `/rebase/vllm-omni` |
| **`HF_HOME`** | `/models/` | `/home/models/` |
| **`CUDA_VISIBLE_DEVICES`** | `0,1,2,3` | User **`X`** empty GPUs → **`nvidia-smi`** pick; or explicit list; or Slurm allocation (no `export`) — confirm which |

Do **not** run **`ssh`** / **`run_nightly_jobs.sh`** until the user confirms defaults or provides custom values (unless they already said **use defaults** in this thread).

### H200 quick path (no docker)

Collect **SSH connection name**. **Confirm run defaults** with the user first (see table above). **Do not** ask for Docker container name.

```bash
ssh -o BatchMode=yes -o ConnectTimeout=120 "<SSH_CONNECTION_NAME>" \
  "bash -lc 'source /rebase/.venv/bin/activate && export REPO_ROOT=\"\${REPO_ROOT:-/rebase/vllm-omni}\" && export HF_HOME=\"/models/\" && unset HF_HUB_CACHE && unset TRANSFORMERS_CACHE && export VLLM_ALLOW_LONG_MAX_MODEL_LEN=\"1\" && export CUDA_VISIBLE_DEVICES=0,1,2,3 && cd \"\$REPO_ROOT\" && bash tools/nightly/run_nightly_jobs.sh'"
```

Details: [references/nightly-local-h200.md](references/nightly-local-h200.md).

### H800 quick path (Slurm + docker)

Collect **SSH connection name**, **Slurm username**, **Docker container name**, optional **`X`** GPUs. **Confirm run defaults** with the user first (see table above), including **`CUDA_VISIBLE_DEVICES`** strategy.

```bash
ssh -o BatchMode=yes -o ConnectTimeout=120 "<SSH_CONNECTION_NAME>" \
  "bash -lc 'type module >/dev/null 2>&1 && module load slurm 2>/dev/null; srun --jobid=\"<JOBID>\" --overlap docker exec \"<CONTAINER_NAME>\" bash -lc \"source /rebase/.venv/bin/activate && export REPO_ROOT=\\\${REPO_ROOT:-/rebase/vllm-omni} && export HF_HOME=/home/models/ && unset HF_HUB_CACHE && unset TRANSFORMERS_CACHE && export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 && cd \\\$REPO_ROOT && bash tools/nightly/run_nightly_jobs.sh\"'"
```

Details: [references/nightly-local-h800.md](references/nightly-local-h800.md).

## Test run mode

Pick the **`run_nightly_jobs.sh`** invocation from user intent (after **`cd "$REPO_ROOT"`**). Details: [references/nightly-local-environment.md](references/nightly-local-environment.md#run-nightly-jobs-test-type).

| User says | Command |
|-----------|---------|
| Full / default nightly (no **local** / **stability** intent) | `bash tools/nightly/run_nightly_jobs.sh` |
| **local** / **local test cases** / **run local** / **run local cases** | `bash tools/nightly/run_nightly_jobs.sh --test-type local` |
| **local for `<model>`** / **run local for `<model>`** | `bash tools/nightly/run_nightly_jobs.sh --test-type local --label-substr <model>` |
| **stability** / **run stability** / **stability tests** | `bash tools/nightly/run_nightly_jobs.sh --test-type stability` |
| **stability for `<model>`** | `bash tools/nightly/run_nightly_jobs.sh --test-type stability --label-substr <model>` |
| **local + stability** / **run local and stability** | `bash tools/nightly/run_nightly_jobs.sh --test-type local,stability` |

Examples: **`--label-substr Qwen`**, **`--label-substr Wan`**, **`--label-substr FLUX`** — use the substring the user gives for **`xxxx`**.

> **Log-dir convention (so the sync scope picks the run up):**
> - Default full nightly → `--log-dir "$REPO_ROOT/logs/nightly_jobs_$(date -u +%Y%m%d-%H%M%S)"`
> - `--test-type local` → `--log-dir "$REPO_ROOT/logs/nightly_jobs_local_$(date -u +%Y%m%d-%H%M%S)"`
> - `--test-type stability` → `--log-dir "$REPO_ROOT/logs/nightly_stability_jobs_$(date -u +%Y%m%d-%H%M%S)"`
> - Combined runs may use either a single timestamped dir or one per kind — keep the `_local` / `_stability` suffix so the auto-detected sync scope matches.

## Required user inputs

| Input | H200 | H800 |
|-------|------|------|
| **SSH connection name** | Yes | Yes |
| **Docker container name** | **No** (SSH = container) | Yes |
| **Slurm username** | No | Yes |
| **Test run mode** | See [Test run mode](#test-run-mode) — default full nightly; **local** adds **`--test-type local`**; model name adds **`--label-substr`** | Same |
| **`REPO_ROOT`** | Default **`/rebase/vllm-omni`** — **confirm with user** | Same — **confirm with user** |
| **`HF_HOME`** | Default **`/models/`** + unset caches — **confirm with user** | Default **`/home/models/`** + unset caches — **confirm with user** |
| **`CUDA_VISIBLE_DEVICES`** | Default **`0,1,2,3`** — **confirm with user** | **`X`** → **`nvidia-smi`** pick, explicit list, or Slurm — **confirm with user** |
| **Git pull before run** | **Ask after connect + `cd "$REPO_ROOT"`** — pull only if user confirms | Same |

---

## 1. Login environment (H800 only — skip for H200)

See [references/nightly-local-h800.md](references/nightly-local-h800.md) for SSH, **`squeue`**, **`srun --overlap docker exec`**, BatchMode, and new-allocation fallback.

---

## 2. Run test cases

Applies to **H200** (remote **`bash -lc`**) and **H800** (inside **`docker exec … bash -lc '…'`**).

**Prerequisite:** [Confirm run defaults](#confirm-run-defaults-required-before-run) with the user **before** connecting or exporting env below.

**Before** the test script:

0. **`source /rebase/.venv/bin/activate`** — [references/nightly-local-environment.md](references/nightly-local-environment.md).

   ```bash
   export REPO_ROOT="${REPO_ROOT:-/rebase/vllm-omni}"
   ```

1. **HF / vLLM** (same shell) — **always** **`unset HF_HUB_CACHE`** and **`unset TRANSFORMERS_CACHE`**; **`HF_HOME`** by machine:

   **H200:**
   ```bash
   export HF_HOME="/models/"
   unset HF_HUB_CACHE
   unset TRANSFORMERS_CACHE
   export VLLM_ALLOW_LONG_MAX_MODEL_LEN="1"
   ```

   **H800:**
   ```bash
   export HF_HOME="/home/models/"
   unset HF_HUB_CACHE
   unset TRANSFORMERS_CACHE
   export VLLM_ALLOW_LONG_MAX_MODEL_LEN="1"
   ```

   Details: [references/nightly-local-environment.md](references/nightly-local-environment.md).

2. **H200:** **`export CUDA_VISIBLE_DEVICES=…`** (confirmed value; default **`0,1,2,3`**) in the same shell (after venv + HF / vLLM). **H800:** apply confirmed strategy — explicit list, **`nvidia-smi`** pick for **`X`** GPUs, or omit export to use Slurm — [CUDA_VISIBLE_DEVICES](references/nightly-local-environment.md#cuda_visible_devices-empty-gpus).

3. **`cd "$REPO_ROOT"`** (cluster checkout, default **`/rebase/vllm-omni`**).

4. **Git pull (confirm first)** — after connect and **`cd`**, **ask the user** whether this run needs latest code. **Do not** run **`git pull`** until they confirm **yes** / **pull** / equivalent (unless they already said so in this thread). If yes:

   ```bash
   git pull
   ```

   If **`git pull`** fails (conflicts, auth), stop and resolve with the user before **`run_nightly_jobs.sh`**. If the user declines → skip pull and continue. Details: [references/nightly-local-environment.md](references/nightly-local-environment.md#git-pull-before-run-confirm-with-user).

5. Run **`run_nightly_jobs.sh`** per [Test run mode](#test-run-mode), e.g.:

   ```bash
   # local test cases (user asked to run local)
   bash tools/nightly/run_nightly_jobs.sh --test-type local

   # local test cases for model xxxx
   bash tools/nightly/run_nightly_jobs.sh --test-type local --label-substr xxxx
   ```

**H800** example (inside docker):

```bash
srun --jobid="$JOBID" --overlap docker exec "$CONTAINER_NAME" bash -lc '
  source /rebase/.venv/bin/activate
  export REPO_ROOT="${REPO_ROOT:-/rebase/vllm-omni}"
  export HF_HOME="/home/models/"
  unset HF_HUB_CACHE
  unset TRANSFORMERS_CACHE
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN="1"
  export CUDA_VISIBLE_DEVICES="0,1"
  cd "$REPO_ROOT"
  # Ask user: git pull? If yes:
  # git pull
  bash tools/nightly/run_nightly_jobs.sh
'
```

### 2.1 Long runs

- **H200:** **`ssh`** → **`tmux`** → run §2 commands directly.
- **H800:** **`ssh`** → **`tmux`** → **`srun … docker exec …`** inside tmux.
- **`nohup`** — [references/nightly-local-environment.md](references/nightly-local-environment.md).

## 3. Sync logs off-cluster

**H200 and H800 share the same workflow** — [references/nightly-local-log-fetch.md](references/nightly-local-log-fetch.md) **[Log sync workflow](references/nightly-local-log-fetch.md#log-sync-workflow)** (**required:** clear local **`logs/`** → resolve remote run dir(s) per **sync scope** → tarball → extract → **merge** into **`logs/nightly_jobs`** → report).

**Sync scope** is chosen by the agent from user intent — set **`SYNC_SCOPE`** to one of `local` / `stability` / `default` / `all` before step 2 of the log sync workflow:

| User intent (or auto-detected from the run) | **`SYNC_SCOPE`** | Remote directories pulled |
|---|---|---|
| User says **pull local logs** / *local test logs* / *only local nightly* | `local` | latest **`nightly_jobs_local_*`** |
| User says **pull stability logs** / *stability test logs* / *only stability nightly* | `stability` | latest **`nightly_stability_jobs_*`** |
| User says **pull nightly logs** / general nightly report / no keyword specified | `default` | latest **`nightly_jobs_YYYYMMDD-*`** (general nightly, **not** local/stability) |
| Combined `--test-type local,stability` / `--test-type all,local,stability` | `all` | all three: local + stability + general nightly |

**Auto-detect when running tests first:** if the user starts the session by running test cases (rather than calling the log-fetch step alone), **do not ask for the keyword** — record the run's effective `--test-type` and apply the matching scope after the run finishes (mapped per the table above). The user can still override by stating a different keyword (e.g. "pull stability logs only").

---

## Agent workflow

1. Detect **H200** vs **H800**; if unclear, ask.
2. Detect **test run mode**: **local** → **`--test-type local`**; **`<model>` local** → add **`--label-substr <model>`**; **stability** → **`--test-type stability`**; **local + stability** → **`--test-type local,stability`**; else default script with no extra flags. Pick the matching **`--log-dir`** suffix (`_local` / `_stability` / plain) so the sync scope can find the run.
3. **Show and confirm run defaults** — display **`REPO_ROOT`**, **`HF_HOME`**, and **`CUDA_VISIBLE_DEVICES`** (or H800 GPU strategy) for the machine type (see [Confirm run defaults](references/nightly-local-environment.md#confirm-run-defaults-with-user)); wait for user **confirm / use defaults** or custom values **before** **`ssh`** or **`run_nightly_jobs.sh`**.
4. **Connect**, apply env (**`source /rebase/.venv/bin/activate`**, confirmed **`REPO_ROOT`**, **`HF_HOME`**, **`CUDA_VISIBLE_DEVICES`**, **`unset HF_HUB_CACHE`** / **`unset TRANSFORMERS_CACHE`**), **`cd "$REPO_ROOT"`**.
5. **Ask whether to `git pull`** in **`$REPO_ROOT`** for this run ([git pull before run](references/nightly-local-environment.md#git-pull-before-run-confirm-with-user)); run **`git pull`** only after user confirms.
6. Run **`run_nightly_jobs.sh`** per [Test run mode](#test-run-mode).
7. **Confirm laptop path defaults** — show **`REPO_ROOT=~/vllm-omni`** and **`KANBAN_REPO_ROOT=~/vllm-omni-kanban`** ([confirm-laptop-path-defaults](../vllm-omni-test-report/references/confirm-laptop-path-defaults.md)); wait for user **confirm / use defaults** or custom paths **before** sync / kanban prep / report.
8. After the run finishes: **clear local `$REPO_ROOT/logs`**, then sync per [Log sync workflow](references/nightly-local-log-fetch.md#log-sync-workflow). **Pick `SYNC_SCOPE` from user intent:**
   - User explicitly said **local** / **stability** / **default (no keyword)** → set `SYNC_SCOPE` to that keyword.
   - **User started this session by running test cases (no explicit log-fetch keyword)** → set `SYNC_SCOPE` automatically from the run's effective `--test-type`: `--test-type local` → `local`; `--test-type stability` → `stability`; `--test-type local,stability` (or `all,local,stability`) → `all`; default full nightly → `default`. State the chosen scope in chat before tarball.
   Then sync into **`logs/nightly_jobs`**; run [kanban prep](../vllm-omni-test-report/references/kanban-pre-report-prep.md) **`prepare_kanban_before_report.py`**, then report via **vllm-omni-test-report**.

## References

- **H200** (SSH = container, no docker): [references/nightly-local-h200.md](references/nightly-local-h200.md)
- **H800** (Slurm + docker exec): [references/nightly-local-h800.md](references/nightly-local-h800.md)
- Fetch logs: [references/nightly-local-log-fetch.md](references/nightly-local-log-fetch.md)
- Laptop path defaults (before sync/prep/report): [../vllm-omni-test-report/references/confirm-laptop-path-defaults.md](../vllm-omni-test-report/references/confirm-laptop-path-defaults.md)
- Environment: [references/nightly-local-environment.md](references/nightly-local-environment.md)
