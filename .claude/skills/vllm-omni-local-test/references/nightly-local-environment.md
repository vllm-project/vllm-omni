# Nightly local - cluster & container environment

## `REPO_ROOT`

- **`REPO_ROOT`** is the vLLM-Omni repository root **on the cluster / inside the container** (H200 SSH session or H800 **`docker exec`** shell).
- **Default:** **`/rebase/vllm-omni`** — show this value and **confirm with the user** before export (see [Confirm run defaults](#confirm-run-defaults-with-user)).

  ```bash
  export REPO_ROOT="${REPO_ROOT:-/rebase/vllm-omni}"
  ```

- Use **`cd "$REPO_ROOT"`** before **`git pull`** (optional) and **`run_nightly_jobs.sh`**. Override only after the user declines the default or names another path (see [Confirm run defaults](#confirm-run-defaults-with-user)).
- **Laptop sync / HTML report** use a **separate** local checkout path — see [nightly-local-log-fetch.md](nightly-local-log-fetch.md) (not **`/rebase/vllm-omni`** on your machine unless you mirror that layout).

<a id="git-pull-before-run-confirm-with-user"></a>

## Git pull before run (confirm with user)

**After** connecting to the cluster (H200 SSH session or H800 **`docker exec`** shell) and **`cd "$REPO_ROOT"`**, **ask the user** whether this test run should pull the latest code **before** starting **`run_nightly_jobs.sh`**.

**Do not** run **`git pull`** silently. Wait for **yes** / **pull** / **拉取** / equivalent (unless the user already said so in this thread).

### Agent prompt template

> Connected to the cluster and in **`$REPO_ROOT`** (`/rebase/vllm-omni` by default).
> Should I run **`git pull`** to update the repo before starting the test cases? (yes / no)

### If user confirms pull

Run in the **same shell** (after **`source /rebase/.venv/bin/activate`** and env exports if already applied):

```bash
cd "$REPO_ROOT"
git pull
```

- Pulls the **current branch** tracking remote (site default is usually **`main`**).
- If **`git pull`** fails (merge conflicts, dirty tree, auth), **stop** — show the error and resolve with the user before **`run_nightly_jobs.sh`**.

### If user declines

Skip **`git pull`** and proceed to **`run_nightly_jobs.sh`** with the checkout as-is.

### Order relative to other steps

1. **`source /rebase/.venv/bin/activate`**
2. Export confirmed **`REPO_ROOT`**, **`HF_HOME`**, **`CUDA_VISIBLE_DEVICES`** (if applicable), **`unset`** cache vars
3. **`cd "$REPO_ROOT"`**
4. **Ask → optional `git pull`** (this section)
5. **`bash tools/nightly/run_nightly_jobs.sh`** …

## Python venv inside the container

On cluster setups, activate the project virtualenv **before** any repo commands (**`run_nightly_jobs.sh`**, pytest, or other tooling):

- **H200:** in the shell after **`ssh`** (session is already inside the container).
- **H800:** inside **`docker exec … bash -lc '…'`** (or an interactive shell in the same container).

```bash
source /rebase/.venv/bin/activate
```

Use the **same** inner shell as **`cd "$REPO_ROOT"`** and the Hugging Face / vLLM exports in the next section.

## Hugging Face cache and vLLM

Before **`run_nightly_jobs.sh`** in the **same** shell (after **`source /rebase/.venv/bin/activate`**):

| Machine | Default **`HF_HOME`** |
|---------|----------------------|
| **H200** | **`/models/`** |
| **H800** | **`/home/models/`** |

**Both H200 and H800** — always unset stale cache vars:

```bash
unset HF_HUB_CACHE
unset TRANSFORMERS_CACHE
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="1"
```

**H200** (after **`ssh`**, already in container):

```bash
export HF_HOME="/models/"
unset HF_HUB_CACHE
unset TRANSFORMERS_CACHE
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="1"
```

**H800** (inside **`docker exec … bash -lc '…'`**):

```bash
export HF_HOME="/home/models/"
unset HF_HUB_CACHE
unset TRANSFORMERS_CACHE
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="1"
```

- Override **`HF_HOME`** only after the user declines the default or names a different mount (see [Confirm run defaults](#confirm-run-defaults-with-user)).

<a id="confirm-laptop-path-defaults-with-user"></a>

## Confirm laptop path defaults with user

**Before** log sync, kanban prep, or nightly HTML on the **laptop**, show and confirm local paths. Full rules and agent prompt: [vllm-omni-test-report confirm-laptop-path-defaults](../../vllm-omni-test-report/references/confirm-laptop-path-defaults.md).

| Variable | Laptop default |
|----------|----------------|
| **`REPO_ROOT`** | `~/vllm-omni` |
| **`KANBAN_REPO_ROOT`** | `~/vllm-omni-kanban` |

This is **in addition to** [Confirm run defaults with user](#confirm-run-defaults-with-user) for cluster **`REPO_ROOT=/rebase/vllm-omni`**, **`HF_HOME`**, and **`CUDA_VISIBLE_DEVICES`**.

<a id="confirm-run-defaults-with-user"></a>
<a id="confirm-path-defaults-with-user"></a>

## Confirm run defaults with user

**Before** **`ssh`** / **`docker exec`** / running **`run_nightly_jobs.sh`**, show the planned defaults (paths **and** GPU selection) and **ask whether to use them**. Do not silently assume defaults when the user has not already confirmed or overridden values in the same conversation.

| Variable | H200 default | H800 default |
|----------|--------------|--------------|
| **`REPO_ROOT`** | `/rebase/vllm-omni` | `/rebase/vllm-omni` |
| **`HF_HOME`** | `/models/` | `/home/models/` |
| **`CUDA_VISIBLE_DEVICES`** | `0,1,2,3` | No fixed index list — confirm **one** of: **`X`** empty GPUs → [nvidia-smi pick](#cuda_visible_devices-empty-gpus); explicit list (e.g. `0,1`); or **Slurm allocation** (skip `export`) |

Also apply on both machines (always run after confirmed **`HF_HOME`**):

```bash
unset HF_HUB_CACHE
unset TRANSFORMERS_CACHE
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="1"
```

### Agent prompt template (show then ask)

**H200 example:**

> Planned run defaults:
> - **`REPO_ROOT`**: `/rebase/vllm-omni`
> - **`HF_HOME`**: `/models/`
> - **`CUDA_VISIBLE_DEVICES`**: `0,1,2,3`
>
> Use these defaults? To change them, provide paths or a GPU list; I will connect and run after you confirm.

**H800 example** (user has not yet given **`X`** or a device list):

> Planned run defaults:
> - **`REPO_ROOT`**: `/rebase/vllm-omni`
> - **`HF_HOME`**: `/home/models/`
> - **`CUDA_VISIBLE_DEVICES`**: after connect, pick **X** idle GPUs inside the container via **`nvidia-smi`** (please confirm **X**, or give an explicit device list such as `0,1`; if using Slurm allocation without `export`, say so)
>
> Proceed with this plan? Confirm **`REPO_ROOT` / `HF_HOME`** defaults and specify the GPU strategy; I will connect and run after you confirm.

**H800 example** (user already said **2 GPUs** / **`CUDA_VISIBLE_DEVICES=0,1`**):

> Planned run defaults:
> - **`REPO_ROOT`**: `/rebase/vllm-omni`
> - **`HF_HOME`**: `/home/models/`
> - **`CUDA_VISIBLE_DEVICES`**: `0,1` (per your specified 2 GPUs / device list)
>
> Use these defaults? Say if you want changes; I will connect and run after you confirm.

### Rules

1. **Always display** **`REPO_ROOT`**, **`HF_HOME`**, and **`CUDA_VISIBLE_DEVICES`** (or H800 GPU strategy) **before** the first cluster run in a session (unless the user already said **use defaults** / gave explicit values).
2. **Wait for confirmation** or explicit overrides before running remote commands.
3. If the user confirms defaults → use the table values: **`export REPO_ROOT="${REPO_ROOT:-…}"`**, **`export HF_HOME="…"`**, **`export CUDA_VISIBLE_DEVICES="…"`** (H200 default **`0,1,2,3`**; H800 per confirmed strategy).
4. If the user gives custom paths or GPU list → use those for the rest of the session.
5. **H800 + `nvidia-smi` pick:** confirm **`X`** in the same prompt when not already known; run the pick script **after** connect inside **`docker exec`**, then **`export`** the chosen indices before **`run_nightly_jobs.sh`**.
6. **H800 + Slurm only:** if the user confirms using the allocation without override, **omit** **`export CUDA_VISIBLE_DEVICES`** and note that in the confirmation reply.

**Always after confirmed `HF_HOME`:** **`unset HF_HUB_CACHE`** / **`unset TRANSFORMERS_CACHE`** — so **`HF_HOME`** is not overridden by inherited cache paths.

Apply confirmed env **before** **`cd "$REPO_ROOT"`**, optional **`git pull`** ([Git pull before run](#git-pull-before-run-confirm-with-user)), and **`bash tools/nightly/run_nightly_jobs.sh`**. (Ensure **`source /rebase/.venv/bin/activate`** has already run in that shell; see [Python venv inside the container](#python-venv-inside-the-container).)

## Default log location

- Each cluster run should write to a **timestamped/suffixed** **`LOG_DIR`** that matches the run kind, e.g.:
  - Default full nightly → **`$REPO_ROOT/logs/nightly_jobs_$(date -u +%Y%m%d-%H%M%S)`**
  - `--test-type local` → **`$REPO_ROOT/logs/nightly_jobs_local_$(date -u +%Y%m%d-%H%M%S)`**
  - `--test-type stability` → **`$REPO_ROOT/logs/nightly_jobs_stability_$(date -u +%Y%m%d-%H%M%S)`**

  Pass the chosen path via **`run_nightly_jobs.sh --log-dir …`**. DFX perf JSON for that run stays under the same **`LOG_DIR`** tree (e.g. **`results/`** subdir or alongside **`*.log`**).

- **Laptop sync:** pull per [nightly-local-log-fetch.md](../vllm-omni-local-test/references/nightly-local-log-fetch.md) **sync scope** (`local` = **`nightly_jobs_local_*` only**; `stability` = **`nightly_jobs_stability_*` only**; `default` = **`nightly_jobs_YYYYMMDD-*` only**; `all` = local + stability + general nightly) → merge into **`$REPO_ROOT/logs/nightly_jobs`** — see [nightly-local-log-fetch.md](../vllm-omni-local-test/references/nightly-local-log-fetch.md). When the user started the session by running tests, the scope is **auto-detected** from the run's `--test-type` so all related logs are pulled. Job logs and perf JSON live under that path for **`nightly_local_log_report.py`** and kanban **`manual_*`** → **`docs/assets/charts/*_history.json`** ([kanban prep](../../vllm-omni-test-report/references/kanban-pre-report-prep.md)).
- **Before each laptop sync:** delete local **`$REPO_ROOT/logs`** so old artifacts are not merged with the new pull ([clear local trees](nightly-local-log-fetch.md#clear-local-trees)).
- If the run used a custom **`LOG_DIR`** outside **`logs/nightly_jobs_*`**, sync that directory explicitly or symlink into **`logs/nightly_jobs`**, then pass **`--log-dir`** when running **`vllm-omni-test-report`** `nightly_local_log_report.py`.

## SSH connection name

- The **connection name** is whatever you pass to `ssh`: a **`Host`** entry in `~/.ssh/config` (e.g. `my-hk`) or `user@login.example.com`.
- Use `ssh -v` when diagnosing host key or proxy issues.
- **H200:** SSH lands in the container — [nightly-local-h200.md](nightly-local-h200.md) (no **`docker exec`**). Default **`CUDA_VISIBLE_DEVICES=0,1,2,3`** — **confirm with user** before export.
- **H800:** Slurm + **`docker exec`** — [nightly-local-h800.md](nightly-local-h800.md).

## Finding `JOBID` for `--overlap`

- List running jobs: `squeue -u YOUR_USER -t RUNNING` or `squeue -u YOUR_USER -t R`.
- **`srun --jobid=ID --overlap docker exec ...`** attaches a step and runs a container command **without** needing **`docker -it`**.
- **`srun --jobid=ID --overlap --pty bash`** is optional for debugging only when you need an interactive shell on the allocation.
- If multiple jobs are running, pick the correct **JOBID**; ask the operator when ambiguous.

## Fallback `srun` (no attach)

When no job id is used:

`srun -p q-fq9hpsac -w hk01dgx006 --gres=gpu:0 --mem-per-cpu=8G --pty  --job-name=ci_local_test`

- Adjust **partition**, **node** (`-w`), **gres**, and **mem** per site policy.
- After this step, use **`docker exec "<container>" bash -lc '...'`** (no **`-it`**) for normal commands.

## Docker

- **Default:** **`docker exec <container> bash -lc '<command>'`** or **`docker exec <container> <argv0> ...`** - no TTY required.
- **`docker exec -it`** only for debugging when you need an interactive shell.
- Requires access to the Docker socket (or rootless equivalent) on the host where **`srun`** / your shell runs.

## Non-interactive SSH

- Prefer **`ssh -o BatchMode=yes`** for automation so missing keys fail fast instead of hanging on a password prompt.
- Remote commands often need **`bash -lc '...'`** so **`module load slurm`** runs before **`squeue`** / **`srun`**.

## Long runs / SSH disconnect

Interactive **`ssh`** sessions send **SIGHUP** on disconnect; anything tied to that shell (including a foreground **`docker exec` → `run_nightly_jobs.sh`**) can exit. For **long local/cluster tests**, use one of the following.

### tmux or screen (recommended)

1. **`ssh`** to the cluster, then **`tmux new -s nightly`** (or **`screen -S nightly`**).
2. Run **`module load slurm`** (if needed), **`srun --jobid=… --overlap docker exec … bash -lc '…'`**, and **`run_nightly_jobs.sh`** **inside** that session.
3. **Detach** (tmux: `Ctrl-b` then `d`). The workload keeps running even if **`ssh`** closes.
4. Later: **`tmux attach -t nightly`** to monitor; **`tmux ls`** lists sessions.

`screen` equivalents: detach `Ctrl-a` `d`; reattach **`screen -r nightly`**.

### nohup inside the container

When you cannot use tmux (e.g. automation-only **`ssh` `BatchMode`** one-shot), start the script detached from the **inner** shell and capture output:

```bash
# Inside docker exec bash -lc ' ... ', after source /rebase/.venv/bin/activate, HF / vLLM / CUDA exports and cd "$REPO_ROOT":
# Default full nightly:
LOG_DIR="$REPO_ROOT/logs/nightly_jobs_$(date -u +%Y%m%d-%H%M%S)"
# --test-type local:    LOG_DIR="$REPO_ROOT/logs/nightly_jobs_local_$(date -u +%Y%m%d-%H%M%S)"
# --test-type stability: LOG_DIR="$REPO_ROOT/logs/nightly_jobs_stability_$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/nightly_runner.nohup.log"
nohup bash tools/nightly/run_nightly_jobs.sh --log-dir "$LOG_DIR" >>"$LOG" 2>&1 &
echo "nightly jobs PID=$! LOG_DIR=$LOG_DIR log=$LOG"
```

- **PID** and **`tail -f "$LOG"`** (from a new **`docker exec`** or after re-**`ssh`**) are how you monitor progress.
- Ensure **`LOG_DIR`** is under **`logs/nightly_jobs_*`** so laptop sync can pick the latest run (see [Default log location](#default-log-location)).

### Caveats

- **Slurm / Kubernetes:** job lifetime is still bounded by the allocation — background only helps with **SSH/client** drops, not **`scancel`** or walltime expiry.
- **Docker:** avoid **`docker exec -it`** for long unattended runs; it expects a TTY.

## `run_nightly_jobs.sh`

- Path: **`tools/nightly/run_nightly_jobs.sh`** relative to the repo root. Forks or branches may differ.

<a id="run-nightly-jobs-test-type"></a>

### Test type and model filter

After **`cd "$REPO_ROOT"`**, choose flags from user intent (run **`git pull`** first only if the user confirmed — [Git pull before run](#git-pull-before-run-confirm-with-user)):

| User intent | Command |
|-------------|---------|
| Default (full nightly, user did **not** ask for **local** / **stability**) | `bash tools/nightly/run_nightly_jobs.sh` |
| Run **local** cases — e.g. **local**, **local test cases**, **run local**, **run local cases** | `bash tools/nightly/run_nightly_jobs.sh --test-type local` |
| Run **local** for a specific model **`xxxx`** — e.g. **Qwen local**, **run local for Wan** | `bash tools/nightly/run_nightly_jobs.sh --test-type local --label-substr xxxx` |
| Run **stability** cases — e.g. **stability**, **run stability**, **stability tests** | `bash tools/nightly/run_nightly_jobs.sh --test-type stability` |
| Run **stability** for a specific model **`xxxx`** | `bash tools/nightly/run_nightly_jobs.sh --test-type stability --label-substr xxxx` |
| Run **local + stability** together | `bash tools/nightly/run_nightly_jobs.sh --test-type local,stability` |

- **`--label-substr`** is a substring match on job labels; pass the model keyword the user gives (**`Qwen`**, **`Wan`**, **`FLUX`**, etc.).
- Apply the same command in **H200** (direct **`ssh`**) and **H800** (**`docker exec`** inner shell), and in **`nohup`** / **`tmux`** sessions.
- **Pick `--log-dir` to match the run kind** so the sync scope can find the run — see [Default log location](#default-log-location).

```bash
# All local jobs
bash tools/nightly/run_nightly_jobs.sh \
  --test-type local \
  --log-dir "$REPO_ROOT/logs/nightly_jobs_local_$(date -u +%Y%m%d-%H%M%S)"

# Local jobs whose labels contain "Qwen"
bash tools/nightly/run_nightly_jobs.sh \
  --test-type local --label-substr Qwen \
  --log-dir "$REPO_ROOT/logs/nightly_jobs_local_$(date -u +%Y%m%d-%H%M%S)"

# Stability jobs
bash tools/nightly/run_nightly_jobs.sh \
  --test-type stability \
  --log-dir "$REPO_ROOT/logs/nightly_jobs_stability_$(date -u +%Y%m%d-%H%M%S)"
```

<a id="cuda_visible_devices-empty-gpus"></a>

## `CUDA_VISIBLE_DEVICES` — empty GPUs via `nvidia-smi`

**H200:** default **`0,1,2,3`** — confirm in [Confirm run defaults](#confirm-run-defaults-with-user) before **`export`**.

**H800:** confirm GPU strategy in the same step (**`X`** + this script, explicit list, or Slurm-only). Run the pick script only **after** the user confirms **`X`** or **`nvidia-smi`** mode.

Run this on the **same machine** that will execute the tests (**inside `docker exec`**, after **`source /rebase/.venv/bin/activate`**, if the job runs in the container). **`X`** = how many cards should be **free**.

**Idle (“empty”)** here means **`memory.used == 0`** and **`utilization.gpu == 0`** in `nvidia-smi`’s CSV query (some sites reserve a few MiB on “idle” GPUs — adjust the checks if needed).

Pick **`X`** indices, then export (comma‑separated, **no spaces** unless you intend multiple vars):

```bash
NUM_GPUS=2   # X — set from user input

empty=()
while IFS=',' read -r idx mem util; do
  idx="${idx// /}"
  mem="${mem// /}"
  util="${util// /}"
  m="${mem%%.*}"
  u="${util%%.*}"
  [[ "$m" =~ ^[0-9]+$ ]] || continue
  [[ "$u" =~ ^[0-9]+$ ]] || continue
  if (( m == 0 && u == 0 )); then
    empty+=("$idx")
  fi
done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)

if (( ${#empty[@]} >= NUM_GPUS )); then
  chosen=("${empty[@]:0:NUM_GPUS}")
else
  echo "WARN: only ${#empty[@]} strictly empty GPU(s); falling back to lowest memory.use / util" >&2
  mapfile -t chosen < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | sed 's/[[:space:]]*//g' | sort -t, -k2,2n -k3,3n | head -n "$NUM_GPUS" | cut -d, -f1)
fi

CUDA_VISIBLE_DEVICES=$(IFS=','; echo "${chosen[*]}")
export CUDA_VISIBLE_DEVICES
echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
```

Then run the workload in **that shell** (use machine-appropriate **`HF_HOME`** from [Hugging Face cache and vLLM](#hugging-face-cache-and-vllm)):

```bash
# H200
export HF_HOME="/models/"
unset HF_HUB_CACHE
unset TRANSFORMERS_CACHE
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="1"
cd "$REPO_ROOT" && bash tools/nightly/run_nightly_jobs.sh
```

(H800: **`HF_HOME="/home/models/"`** instead.)

(If **`CUDA_VISIBLE_DEVICES`** was set in the steps above, it remains in effect; if you already exported **HF / vLLM** vars in the same shell, omit duplicates.)

**Notes**

- If **Slurm** binds **`CUDA_VISIBLE_DEVICES`** for the step, merge site policy: either rely on the allocation or override only when allowed.
- **`nvidia-smi` inside the container** shows the GPUs **visible to that container** (may already be filtered).
- **MIG**, **multi-tenant** nodes, or **persistence mode** can show non‑zero `memory.used` on “empty” cards — prefer your cluster’s convention or raise thresholds after checking `nvidia-smi` once.
