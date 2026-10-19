---
name: run-coverage
description: Run the vllm-omni unit-test coverage sequence inside a remote container over SSH and send the final coverage result back. The user provides a 链接名 (SSH alias, e.g. my_l20) and a 容器名 (docker container name, e.g. vllm_omni); the skill runs source .venv/bin/activate → cd vllm-omni → git pull → bash run_coverage.sh via docker exec, waits for the long-running pytest coverage pass to finish, and delivers the coverage table (TOTAL row) plus artifacts. Use when the user says "覆盖率", "运行覆盖率", "跑覆盖率", "run coverage", "coverage 报告", or provides a 链接名 + 容器名 pair asking for coverage.
---

# Run Coverage (容器内跑 vllm-omni 覆盖率)

Run the vllm-omni coverage sequence inside a **docker container that lives on
an SSH-reachable host**, then send the final coverage result to the user.

The user supplies two inputs:

| Input | Meaning | Example |
|-------|---------|---------|
| **链接名** | SSH alias on this bridge (`~/.ssh/config`) | `my_l20` |
| **容器名** | docker container name on that host | `vllm_omni` |

The host reached via the 链接名 runs docker; the 容器名 is the container that
holds the vllm-omni checkout and its venv.

## Inside-container layout (verified 2026-10-16 on my_l20 / vllm_omni)

The four user steps are **relative to a base dir inside the container**,
default **`/rebase`** (the cluster-standard root):

```
/rebase/.venv/bin/activate        # has pytest + coverage (python3.12)
/rebase/vllm-omni/                # git checkout of vllm-project/vllm-omni
/rebase/vllm-omni/run_coverage.sh # untracked helper (safe for git pull)
/rebase/vllm-omni/.coveragerc     # untracked coverage config
```

`run_coverage.sh` runs `coverage erase`, a
`pytest -m "core_model and cpu" ... --cov=vllm_omni
--cov-report=term-missing:skip-covered` pass, then `coverage html` +
`coverage xml`.

> ⚠️ Do **not** substitute `/home/wy` as the base dir on this container:
> `/home/wy/.venv` is the mkdocs venv (no pytest/coverage). If a future
> 链接名/容器名 pair uses a different base, pass it as the script's 4th
> argument instead of editing the default.

## Quick Path

1. **Preflight** (fast, read-only — run these first and report failures
   instead of proceeding):

   ```bash
   ssh -o BatchMode=yes <链接名> 'hostname'                                   # alias reachable?
   ssh -o BatchMode=yes <链接名> "docker ps --format '{{.Names}}'"           # container exists?
   ```

   - Alias not in `~/.ssh/config` → ask the user for host/port/user and add
     the entry (mirror the `my_l20` block: HostName/User/Port/IdentityFile
     `~/.ssh/id_ed25519`).
   - `Permission denied (publickey)` → the bridge key is not authorized in
     the target; print `~/.ssh/id_ed25519.pub` and give the user the
     one-time command to run on the target:
     `mkdir -p ~/.ssh && echo '<pubkey line>' >> ~/.ssh/authorized_keys`.
   - Container missing from `docker ps` → check
     `docker ps -a --format '{{.Names}} {{.Status}}'` for a stopped match
     and report; do not start containers on your own initiative.

2. **Run the coverage sequence** — this is **long-running** (tens of
   minutes; CPU-marked pytest with coverage instrumentation). Always launch
   in the background and let the task notification wake you:

   ```bash
   /root/.claude/skills/run-coverage/scripts/run_coverage_in_container.sh \
       <链接名> <容器名> \
       /home/wy/coverage_runs/coverage-<UTC-date>.log
   ```

   Run with `run_in_background: true`. The script tees everything to the
   log and ends with a `COVERAGE_EXIT=<rc>` marker line. It hard-fails with
   tagged markers (`BASE_DIR_MISSING` / `VENV_MISSING` / `REPO_MISSING` /
   `GIT_PULL_FAILED` / `SCRIPT_MISSING`) so failures are self-describing;
   `git pull` is retried 3× with `http.lowSpeedLimit=500` (GitHub TLS from
   these hosts stalls regularly).

3. **Extract the result** from the log:

   - The pytest-cov table: the block starting at the `Name   Stmts   Miss
     Cover` header through the `TOTAL` row (with
     `--cov-report=term-missing:skip-covered`, only files below 100% are
     listed). Grab the **TOTAL row** — that is "最终的覆盖率结果".
   - The `=== artifacts ===` block lists `coverage.xml`, `.coverage`,
     `htmlcov/index.html` presence.
   - pytest tail (`N passed, M failed, ...`) for run health.

4. **Send the result to the user**:

   - Normal reply: TOTAL row (+ pass/fail counts, HEAD commit from the
     `=== run_coverage.sh (HEAD: …) ===` line, notable low-coverage files).
   - Attach `coverage.xml` when present:

     ```bash
     ssh -o BatchMode=yes <链接名> \
       "docker exec <容器名> cat /rebase/vllm-omni/coverage.xml" \
       > /home/wy/coverage_runs/coverage-<UTC-date>.xml
     cc-connect send --file /home/wy/coverage_runs/coverage-<UTC-date>.xml
     ```

   - Only attach `htmlcov/` (tar.gz via
     `docker exec <容器名> tar czf - -C /rebase/vllm-omni htmlcov`) when the
     user asks for the HTML report.

## Error handling

| Symptom | Action |
|---------|--------|
| `COVERAGE_EXIT=10` (`GIT_PULL_FAILED`) | Report the git error verbatim (often TLS stall / auth); suggest rerun |
| `COVERAGE_EXIT=1x` tagged markers | Report which path is missing; do not guess alternative bases silently |
| pytest failures inside the log | Coverage table may still print — send BOTH the TOTAL row and the failure summary (`FAILED test...` lines, last ~40 lines) |
| Log ends without `COVERAGE_EXIT` | SSH dropped mid-run; rerun the script (coverage erase makes reruns clean) |

## Constraints

- **One coverage run per container at a time** — `run_coverage.sh` starts
  with `coverage erase`; a concurrent run clobbers the other's data. Check
  the log for a still-running instance before relaunching.
- The untracked `run_coverage.sh` / `.coveragerc` inside the repo are the
  user's live files — never commit, delete, or "fix" them.
- No pushes, no issue/PR writes — this skill only runs tests and reports.
