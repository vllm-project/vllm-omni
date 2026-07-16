# Nightly local — H200 (SSH lands in container)

Use when the user specifies **H200** (e.g. **H200**, **H200 machine**, **use H200**, **run on H200**).

**Important:** the **SSH connection already drops into the container** (configured in **`~/.ssh/config`** or site jump setup). **Do not** run **`docker exec`** — run commands directly on the remote shell after **`ssh`**.

**Do not** use Slurm **`squeue`** / **`srun`** on H200.

Shared env (venv, HF, CUDA, long runs): [nightly-local-environment.md](nightly-local-environment.md).

## GPU selection (H200 default)

**Default:** **`CUDA_VISIBLE_DEVICES=0,1,2,3`** — show in [Confirm run defaults](nightly-local-environment.md#confirm-run-defaults-with-user) and **ask the user** before export. After confirmed **`ssh`** (already in the container):

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3   # or user-confirmed list
```

Do **not** set this silently; use the value from the confirmation step.

## Hugging Face (H200 default)

After connect, in the same shell as the test run:

```bash
export HF_HOME="/models/"
unset HF_HUB_CACHE
unset TRANSFORMERS_CACHE
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="1"
```

(H800 uses **`HF_HOME="/home/models/"`** — [nightly-local-environment.md](nightly-local-environment.md).)

## Required inputs (H200)

| Input | Required | Meaning |
|-------|----------|---------|
| **SSH connection name** | Yes | `Host` alias or `user@host` — session is **already inside the container**. |
| **Test run** | No | [run-nightly-jobs-test-type](nightly-local-environment.md#run-nightly-jobs-test-type) — **local** → **`--test-type local`**; **`<model>` local** → **`--label-substr <model>`** |
| **`REPO_ROOT`** | No | Default **`/rebase/vllm-omni`** — **confirm with user** before export |
| **`CUDA_VISIBLE_DEVICES`** | No | Default **`0,1,2,3`** — **confirm with user** before export — [GPU selection](#gpu-selection-h200-default) |

**Not required on H200:** Docker container name, Slurm username, **JOBID**, **`module load slurm`**, **`nvidia-smi`** GPU picking (unless user overrides devices).

## Agent routing

1. User mentions **H200** → **this file** + SKILL.md **§2** (no **`docker exec`**, no §1 Slurm).
2. User mentions **H800** → [nightly-local-h800.md](nightly-local-h800.md).
3. Neither → ask **H200** vs **H800**.
4. **Before run:** show default **`REPO_ROOT`**, **`HF_HOME`**, **`CUDA_VISIBLE_DEVICES`** (`0,1,2,3`) and **ask user to confirm** — [Confirm run defaults](nightly-local-environment.md#confirm-run-defaults-with-user).
5. **After connect + `cd "$REPO_ROOT"`:** ask whether to **`git pull`** — [Git pull before run](nightly-local-environment.md#git-pull-before-run-confirm-with-user).

## 1. Connect and run (interactive)

```bash
ssh -v "<SSH_CONNECTION_NAME>"
# You are already in the container — run INNER_CMD directly:
```

Example commands (use **confirmed** **`REPO_ROOT`**, **`HF_HOME`**, **`CUDA_VISIBLE_DEVICES`**; values below are defaults):

```bash
source /rebase/.venv/bin/activate
export REPO_ROOT="${REPO_ROOT:-/rebase/vllm-omni}"
export HF_HOME="/models/"
unset HF_HUB_CACHE
unset TRANSFORMERS_CACHE
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="1"
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd "$REPO_ROOT"
# Ask user: git pull? If yes:
# git pull
bash tools/nightly/run_nightly_jobs.sh --test-type local
```

Model-filtered local (user says **local for model `xxxx`**):

```bash
cd "$REPO_ROOT" && bash tools/nightly/run_nightly_jobs.sh --test-type local --label-substr xxxx
```

Full nightly (no **local** intent):

```bash
cd "$REPO_ROOT" && bash tools/nightly/run_nightly_jobs.sh
```

Custom script (**`SCRIPT_REL`** relative to **`$REPO_ROOT`**) only when the user names a path other than **`run_nightly_jobs.sh`**:

```bash
cd "$REPO_ROOT" && bash "$SCRIPT_REL"
```

## 2. Agent: non-interactive one-shot

Remote command runs **in the container shell** — no **`docker exec`**:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=120 "<SSH_CONNECTION_NAME>" \
  "bash -lc 'source /rebase/.venv/bin/activate && export REPO_ROOT=\"\${REPO_ROOT:-/rebase/vllm-omni}\" && export HF_HOME=\"/models/\" && unset HF_HUB_CACHE && unset TRANSFORMERS_CACHE && export VLLM_ALLOW_LONG_MAX_MODEL_LEN=\"1\" && export CUDA_VISIBLE_DEVICES=0,1,2,3 && cd \"\$REPO_ROOT\" && bash tools/nightly/run_nightly_jobs.sh'"
```

Same inner steps as SKILL.md **§2**: venv → HF / vLLM → **`CUDA_VISIBLE_DEVICES=0,1,2,3`** → test script.

## 3. Long runs

- **`ssh`** → **`tmux new -s nightly-h200`** → run **§2** commands inside tmux → detach. See [Long runs / SSH disconnect](nightly-local-environment.md#long-runs--ssh-disconnect).
- **Automation:** **`nohup bash -lc '…'`** over **`ssh`** (no docker wrapper).

## 4. Sync logs (H200)

Same workflow as **H800** — follow [Log sync workflow](nightly-local-log-fetch.md#log-sync-workflow) in [nightly-local-log-fetch.md](nightly-local-log-fetch.md) (steps 1–5). For step **3** (remote tarball), use the **H200** command: direct **`ssh`**, no Slurm or **`docker exec`**.

Release report: **`--log-dir-h200`** on **`compose_full_report.py`**.
