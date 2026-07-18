# Nightly local — H800 (Slurm + docker exec)

Use when the user specifies **H800** (e.g. **H800**, **H800 machine**, **use H800**, **on H800**). This is the **Slurm cluster** path: SSH to the login/node host → **`squeue`** → **`srun --jobid=… --overlap docker exec …`**.

**⚠️ `/rebase/vllm-omni` is on shared NFS storage visible only on compute nodes, not on the SSH login node.** A plain `ssh <host> "ls /rebase/vllm-omni"` returns `No such file or directory`. Every step below that touches `$REPO_ROOT` (or its subdirs) MUST run **inside the slurm allocation** (`srun --overlap` on an existing job, or `salloc -w <node>` to get a fresh one). The "New allocation" snippet at the bottom is the fallback when no `JOBID` is available.

Shared env inside the container (venv, HF, CUDA, long runs): [nightly-local-environment.md](nightly-local-environment.md). Run steps match SKILL.md **§2**.

## Hugging Face (H800 default)

Inside **`docker exec … bash -lc '…'`**:

```bash
export HF_HOME="/home/models/"
unset HF_HUB_CACHE
unset TRANSFORMERS_CACHE
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="1"
```

(H200 uses **`HF_HOME="/models/"`** — [nightly-local-h200.md](nightly-local-h200.md).)

## Required inputs (H800)

| Input | Required | Meaning |
|-------|----------|---------|
| **SSH connection name** | Yes | `Host` alias or `user@host` for **`ssh`**. |
| **Slurm username** | Yes | For **`squeue -u …`**. |
| **Docker container name** | Yes | For **`docker exec`** on the allocation. |
| **`REPO_ROOT`** | No | Default **`/rebase/vllm-omni`** — **confirm with user** before export |
| **Test run** | No | [run-nightly-jobs-test-type](nightly-local-environment.md#run-nightly-jobs-test-type) — **local** / **`--label-substr`** |
| **Empty GPU count (`X`)** or device list | No | **`CUDA_VISIBLE_DEVICES`** — **confirm with user** before connect/export — [CUDA_VISIBLE_DEVICES](nightly-local-environment.md#cuda_visible_devices-empty-gpus) |

**Not used on H800:** direct shell-only path (that is **H200** — [nightly-local-h200.md](nightly-local-h200.md)).

## Agent routing

1. User mentions **H800** → **this file** + SKILL.md **§1–2**.
2. User mentions **H200** → [nightly-local-h200.md](nightly-local-h200.md) only.
3. Neither → ask which machine type (**H200** vs **H800**).
4. **Before run:** show default **`REPO_ROOT`**, **`HF_HOME`**, and **`CUDA_VISIBLE_DEVICES`** strategy ( **`X`** + **`nvidia-smi`**, explicit list, or Slurm) and **ask user to confirm** — [Confirm run defaults](nightly-local-environment.md#confirm-run-defaults-with-user).
5. **After connect + `cd "$REPO_ROOT"`:** ask whether to **`git pull`** — [Git pull before run](nightly-local-environment.md#git-pull-before-run-confirm-with-user).

## 1. SSH and Slurm

**Step 1: SSH to the login node** and find **YOUR OWN** running jobid:

```bash
ssh -v "<SSH_CONNECTION_NAME>"

# Load slurm and search for your own running jobs (NOT generic "any" jobs on the node)
SLURM_USER="<username>"
squeue -u "$SLURM_USER" -t RUNNING -h -o "%i"
```

Confirm **JOBID** when multiple rows exist (use the one on the same NODE you want, e.g. `hk01dgx012`).

**Step 2: enter the workload container** — log dirs are only inside `docker exec`, not in the bare slurm allocation:

```bash
JOBID="<chosen_jobid>"

# Test the storage is visible inside the slurm allocation
srun --jobid="$JOBID" --overlap docker exec "<CONTAINER_NAME>" bash -lc "ls /rebase/vllm-omni/logs/"

# Pack a stability run (long-stability log dir = nightly_stability_jobs_*)
srun --jobid="$JOBID" --overlap docker exec "<CONTAINER_NAME>" bash -lc \
  "cd /rebase/vllm-omni/logs && tar czf - nightly_stability_jobs_*" > nightly_logs.tgz
```

**Important:** the container name (`<CONTAINER_NAME>`) is what the user gave, e.g. `omni_wy_24g`. Do NOT skip `docker exec` — `/rebase/vllm-omni/logs/` is not visible from the slurm allocation alone on this H800 setup.

## 2. Run in container (no TTY)

```bash
JOBID="<chosen_jobid>"
srun --jobid="$JOBID" --overlap docker exec "<CONTAINER_NAME>" bash -lc '<INNER_CMD>'
```

Nightly one-liner (adjust **`run_nightly_jobs.sh`** flags per [Test type and model filter](nightly-local-environment.md#run-nightly-jobs-test-type); insert **`git pull`** only after user confirms — [Git pull before run](nightly-local-environment.md#git-pull-before-run-confirm-with-user)):

```bash
srun --jobid="$JOBID" --overlap docker exec "<CONTAINER_NAME>" bash -lc 'source /rebase/.venv/bin/activate && export REPO_ROOT="${REPO_ROOT:-/rebase/vllm-omni}" && export HF_HOME="/home/models/" && unset HF_HUB_CACHE && unset TRANSFORMERS_CACHE && export VLLM_ALLOW_LONG_MAX_MODEL_LEN="1" && cd "$REPO_ROOT" && bash tools/nightly/run_nightly_jobs.sh --test-type local'
```

### New allocation if no JOBID

```bash
srun -p q-fq9hpsac -w hk01dgx006 --gres=gpu:0 --mem-per-cpu=8G --pty --job-name=ci_local_test
```

Then **`docker exec "<CONTAINER_NAME>" bash -lc '<commands>'`**.

### Agent: BatchMode SSH

```bash
ssh -o BatchMode=yes -o ConnectTimeout=30 "<SSH_CONNECTION_NAME>" \
  "bash -lc 'type module >/dev/null 2>&1 && module load slurm 2>/dev/null; squeue -u \"<SLURM_USER>\" -t RUNNING -h -o \"%i\"'"

ssh -o BatchMode=yes -o ConnectTimeout=120 "<SSH_CONNECTION_NAME>" \
  "bash -lc 'type module >/dev/null 2>&1 && module load slurm 2>/dev/null; srun --jobid=\"<JOBID>\" --overlap docker exec \"<CONTAINER_NAME>\" bash -lc \"<INNER_CMD>\"'"
```

## 3. Long runs

- **`ssh`** → **`tmux new -s nightly-h800`** → **`srun … docker exec …`** inside tmux → detach.
- Or **`nohup`** inside **`docker exec bash -lc`** — [Long runs / SSH disconnect](nightly-local-environment.md#long-runs--ssh-disconnect).

## 4. Sync logs (H800)

Same workflow as **H200** — follow [Log sync workflow](nightly-local-log-fetch.md#log-sync-workflow) in [nightly-local-log-fetch.md](nightly-local-log-fetch.md) (steps 1–5). For step **3** (remote tarball), use the **H800** command: **`ssh` + `srun --overlap docker exec`**.

Release report: **`--log-dir-h800`** on **`compose_full_report.py`**.
