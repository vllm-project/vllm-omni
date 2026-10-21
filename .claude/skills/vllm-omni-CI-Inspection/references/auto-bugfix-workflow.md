# Auto-bugfix Workflow — `修复 <TYPE>#<build>`

Triggered when a **human sends the auto-fix command to the bot** in an
interactive session. Scheduled/cron runs only *render* the command in alerts;
they never execute it (see the write-ban section in SKILL.md).

```text
修复 READY#17435     # READY CI build #17435 (vllm-omni pipeline, non-main branch)
修复 MERGE#17523     # MERGE CI build #17523 (vllm-omni pipeline, main branch)
```

Equivalent spellings accepted: `bugfix READY#17435`, `修复:READY#17435`.
`TYPE` must be `READY` or `MERGE` (vllm-omni pipeline only).

**Scope** — failed jobs of the Buildkite `:card_index_dividers: Simple Test`
group, i.e. job names starting with `Simple · ` (`Simple · Model Executor
Test`, `Simple · Diffusion Models Test`, `Simple · Engine&Entrypoints&Diffusion
Test`, `Simple · Other Test`, `Simple · CPU Coverage Test`, plus any future
job with the same prefix). These jobs run CPU-marked tests
(`pytest -sv <paths> -m 'core_model and cpu'`) on `l4_1`, so they are
reproducible on the my_l20 box. Out of scope: NPU READY (no NPU on the fix
box), AFD / ROUTER (different repos).

**Fix environment** (verified 2026-10-20, first live run of build #17604):

| Item | Value |
|---|---|
| SSH host | `my_l20` (entry in `~/.ssh/config`; BatchMode works) |
| Container | **`vllm_omni`** — must be `Up`; if stopped, ask the user before starting it. (⚠️ NOT `vllm_omni_yenuo26`: that container's repo tracks the yenuo26 fork whose `main` is ~9 months stale (2026-01) and its venv has vllm 0.31.0 — tests fail collection with `ImportError` there.) |
| Repo | `/rebase/vllm-omni` inside `vllm_omni`; `origin = https://github.com/vllm-project/vllm-omni.git` — **the upstream repo, read-only for us; never push to it** |
| venv | `/rebase/.venv` → `source /rebase/.venv/bin/activate` (there is NO repo-local `.venv`) |
| vllm version | Must match main's alignment — **0.32.0** as of commit f7f40063d "[Core] Align vLLM-Omni with vLLM v0.32.0". Before running, `pip show vllm`; a mismatch (e.g. 0.31.0) fails at collection with `ImportError: cannot import name 'RequestError' from 'vllm.outputs'`. If stale, tell the user (they maintain this venv) — direct PyPI from the box can be very slow; tsinghua mirror `-i https://pypi.tuna.tsinghua.edu.cn/simple` completes in <1 min. |
| `gh` CLI | **not installed in the container** — PR creation happens from the bridge box |
| Note | The container may sit on any working branch; this workflow always creates a fresh branch from `origin/main` (genuine upstream main in this container). Verify the base is fresh: `git fetch origin main && git log -1 --format=%ci origin/main` should be ≤ a few days old. |

## Phase 0 — Rebuild context (bridge box, read-only)

1. `TYPE=READY` → the build ran a non-main (PR) branch; `TYPE=MERGE` → a
   main-branch build. Both are pipeline `vllm-omni`. (Alerted READY failures
   are by definition not PR-introduced, so the failure exists on main too —
   which is what we fix.)
2. Fetch the build (Bearer `$BUILDKITE_TOKEN`):

   ```bash
   curl -s -H "Authorization: Bearer $BUILDKITE_TOKEN" \
     "https://api.buildkite.com/v2/organizations/vllm/pipelines/vllm-omni/builds/<n>"
   ```

   Collect failed jobs whose `name` starts with `Simple · `.
   **None found** → reply "build #<n> 没有 Simple · 失败 job，无需修复" and stop.
3. For each such job, fetch the job log (the job-log endpoint needs
   `Accept: application/json` to return `{content}`; strip `_bk;t=<epoch_ms>`
   timestamps) and extract: failed pytest node ids (`FAILED <node>` /
   `ERROR <node>` from the short test summary), the first-error signature
   (earliest fatal traceback, ≤40 lines), and the pytest command the job ran.
4. Multiple distinct Simple jobs failed in the same build → fix them all on
   **one** branch.

## Phase 1 — Fresh branch (my_l20 container)

```bash
ssh -o BatchMode=yes my_l20
docker exec vllm_omni bash -lc '
  source /rebase/.venv/bin/activate
  cd /rebase/vllm-omni
  git fetch origin main
  git log -1 --format="%h %ci" origin/main   # sanity: must be recent, not months old
  git checkout -B fix/simple-b<n>-<yyyymmdd> origin/main
  pip show vllm | head -2                     # sanity: must match main's vllm alignment
'
```

Branch naming: `fix/simple-b<build#>-<yyyymmdd>` (UTC date).

## Phase 2 — Reproduce, judge, fix

1. **Reproduce first.** Run the failing node unchanged on the fresh branch.
   If it **passes** on the main-based branch, the failure was branch/CI-env
   specific → report that finding, change nothing, no PR.
2. **Judge 服务问题 vs 用例问题** (root cause, not symptom):

   | Verdict | Meaning | What you modify |
   |---|---|---|
   | **服务问题** | Runtime/product code broke the test — regression, API/signature change rippling to callers, crash or wrong output from `vllm_omni` code | **Service code only** (`vllm_omni/…`) — **NEVER touch the test** |
   | **用例问题** | The test is wrong — stale assertion after an intended behavior change, outdated API usage, bad fixture/mock, wrong expected value | **The test only** (`tests/…`) — **NEVER touch service code** |

   Evidence: first-error signature; `git log` / `git blame` of the involved
   test and service files; whether the assertion matches documented or
   intended behavior.
   **Genuinely ambiguous after weighing the evidence → STOP.** Report both
   hypotheses with the evidence and ask the user. Do not guess-edit.
3. Keep the fix minimal — smallest diff that makes the test pass while
   preserving intent. Match surrounding code style.

## Phase 3 — Verify (container)

```bash
docker exec vllm_omni bash -lc '
  source /rebase/.venv/bin/activate
  cd /rebase/vllm-omni
  unset HF_HUB_CACHE TRANSFORMERS_CACHE
  # HF_HOME discovery: ls -d /models /home/models /rebase/models — export the
  # one that exists, and only if the test actually needs weights (Simple
  # tests are CPU-marked; most do not download).
  pytest -sv "<failing node id>"
'
```

- All previously-failing nodes pass → Phase 4.
- Still failing → iterate (re-judge each round; **max 3 rounds total**).
- Still failing after 3 rounds → report honestly; keep the branch; **NO PR**.

## Phase 4 — Commit, push, PR

**Signature rule (user-set 2026-10-21): the fix PR carries ONLY the user's
account — `yenuo26`.** No `Co-Authored-By: Claude Code` trailer on commits,
no `🤖 Generated with Claude Code` footer in the PR body. The PR must be
created **following the repo's PR template**
(`.github/PULL_REQUEST_TEMPLATE.md` on `vllm-project/vllm-omni`).

1. Commit in the container — the `vllm_omni` container has **no git user
   configured**, so pass the identity explicitly; no co-author trailer:

   ```bash
   git -c user.name="yenuo26" -c user.email="410167048@qq.com" \
     commit -m "fix: <one-line what and why> (build #<n> <job name>)"
   ```

   (`git format-patch` → bridge `git am` preserves this author; the bridge
   clone `/home/wy/vllm-omni-yn` is also already configured
   `yenuo26 <410167048@qq.com>`.)

2. Push to the fork. The fix container's `origin` is **upstream** (read-only —
   never push there), so the patch travels via the bridge box:
   `git format-patch origin/main -o /tmp` in the container → `scp`/`ssh cat`
   the patches to the bridge box → apply on a fresh branch in
   `/home/wy/vllm-omni-yn` → `git push origin <branch>` from there (that clone
   pushes to the fork). If the container happens to have a writable fork
   remote/credentials, `git push -u <fork-remote> <branch>` is fine too (never
   print credential contents).

3. Create the PR **from the bridge box** (the container has no `gh`): the
   bridge box's `gh` is logged in as **yenuo26** (verified 2026-10-21), so
   `gh pr create -R vllm-project/vllm-omni --head yenuo26:<branch> --base
   main` authors the PR as the user. (Equivalent fallback: `POST …/pulls`
   with `GITHUB_TOKEN` — same account.) If the PR cannot be created, report
   the exact error plus a ready-to-run manual command — never silently skip.
   **The PR body follows the repo's PR template:**

   - **Fetch the current template first** — read
     `.github/PULL_REQUEST_TEMPLATE.md` from the local clone
     (`/home/wy/vllm-omni`, after a pull) or the GitHub API; never trust a
     remembered shape (the template evolves).
   - Fill it section by section:
     - **Contributor intent (4 required human answers)** — the template
       forbids the agent from inventing, inferring, or prefilling these.
       Q1 (outcome) / Q3 (how this moves it) can be *reused* from the human's
       修复 command + the fix summary (public-safe); **Q2 (highest-leverage
       variable) and Q4 (whose user needs this) must be asked** — show the
       user the draft body and ask; reuse their explicit answers verbatim.
       No answers → stop after pushing the branch and hand the user the
       ready-to-run `gh pr create` command. Never publish private
       conversation content without authorization.
     - **Purpose** — behavior before vs after the fix; link the affected
       module/feature contract.
     - **Test Plan** — **vLLM Version** (`pip show vllm` in the container) +
       **vLLM-Omni Commit** (the fix branch's base commit on origin/main).
     - **Test Result** — failing test nodes, first-error excerpt, and the
       Phase 3 verification command + passing output; explicitly list
       relevant checks NOT run (e.g. GPU tests, full suite).
   - Everything below the template's "(anything written below this line will
     be removed by GitHub Actions)" line is stripped — never put content
     there.
   - **Pre-flight — 5-open-PR cap**: contributors without write access are
     capped at 5 open PRs. Check
     `gh pr list -R vllm-project/vllm-omni --author yenuo26 --state open`;
     if ≥ 5, warn the user instead of creating a PR that will be blocked.
   - **Pre-flight — precheck**: the template's BEFORE-SUBMITTING note asks
     for the repo's `precheck-pr` self-check (`.claude/skills/precheck-pr`
     in the repo); run it on the fix diff and fix what it flags.
   - PR body carries **no** Claude attribution footer.

## Phase 5 — Report back

PR link (or exact blocker), branch name, verdict + one-line evidence, files
changed, verification result. If any phase failed, say exactly which and why —
never claim a PR that was not created.

## Hard rules

- **Never modify both service code and the test in the same fix** — one or
  the other (服务问题 → 只改服务代码；用例问题 → 只改用例).
- **Never create a PR when verification did not pass.**
- **Never run this workflow from a scheduled/cron run** — interactive,
  human-sent command only.
- No GitHub issue/comment writes anywhere in this workflow.
- Leave the container worktree on the fix branch; do not delete or reset
  other branches; do not touch unrelated files (untracked `run_coverage.sh`
  / `.coveragerc` in `/rebase/vllm-omni` are user files — never modify,
  commit, or delete them).
