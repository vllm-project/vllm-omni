# Archive test reports to [vllm-omni-kanban](https://github.com/hsliuustc0106/vllm-omni-kanban)

Target repo: [vllm-omni-kanban](https://github.com/hsliuustc0106/vllm-omni-kanban)

## When to run

Only after the user prompt includes archive/push intent, e.g. **archive**, **commit**, **push**, **kanban**, **upload report**.

## Prerequisites

- **[GitHub CLI (`gh`)](https://cli.github.com/)** installed and authenticated (`gh auth login`, or `GH_TOKEN` / `GITHUB_TOKEN` with `repo` scope). If `gh` is missing, stop and ask the user to install it first (do not fall back to raw `git push` without `gh`).
- Local git clone of [vllm-omni-kanban](https://github.com/hsliuustc0106/vllm-omni-kanban) with push access to `origin` (typically `main`).
- Default **`KANBAN_REPO_ROOT=~/vllm-omni-kanban`** ([confirm with user](confirm-laptop-path-defaults.md); clone: `git clone https://github.com/hsliuustc0106/vllm-omni-kanban`) or pass `--kanban-repo-root`.
- Generated report must be **HTML** (MkDocs site reads HTML snapshots).

### Install `gh` (if not present)

```bash
gh --version   # must succeed before archive push

# Windows
winget install --id GitHub.cli

# macOS
brew install gh

# After install
gh auth login
```

Agents: if `gh --version` fails, tell the user to install `gh` and run `gh auth login` before continuing.

## What to commit (important)

**Commit HTML under `data/*_test_report/` and, for nightly workflows, the new `data/local_nightly_raw/manual_*` directory when present. Never add files under `docs/assets/test_reports/`.**

In [vllm-omni-kanban `.gitignore`](https://github.com/hsliuustc0106/vllm-omni-kanban/blob/main/.gitignore):

```gitignore
# Generated test report copies for MkDocs (do not commit)
docs/assets/test_reports/
```

| Report kind | **Commit these paths** | Do **not** commit |
|-------------|------------------------|-------------------|
| **nightly** | `data/nightly_test_report/nightly-report-buildkite-latest-YYYY-MM-DD.html` **+** optional `data/local_nightly_raw/manual_YYYYMMDD/` (or `manual_YYYYMMDD_HHMMSS/`) when created by [prepare_kanban_before_report.py](../scripts/prepare_kanban_before_report.py) | `docs/assets/test_reports/**`, `data/local_nightly_raw/.last_manual_dir` |
| **release** | `data/release_test_report/vllm-omni-release-test-report-YYYY-MM-DD.html` (+ same `manual_*` if you ran nightly prep and want raw archived) | `docs/assets/test_reports/**` |

Kanban **MkDocs** (`mkdocs serve` / `mkdocs build`) runs `scripts/mkdocs_hooks.py` → `scripts/sync_test_reports.py`, which **locally** copies `data/*_test_report/` into `docs/assets/test_reports/` for the Reports page ([docs/reports.md](https://github.com/hsliuustc0106/vllm-omni-kanban/blob/main/docs/reports.md)). That output is gitignored and regenerated at build time — **do not** `git add` it during archive push.

## Automated archive and push (two steps)

From **`skills/vllm-omni-test-report/`**:

Archive and push are **separate commands** (report generators do not push):

```bash
# Step 1: copy report, pull, git add, print push preview (no commit/push)
python scripts/push_report_to_kanban.py \
  --report ./nightly-report-buildkite-latest-YYYY-MM-DD.html \
  --kanban-repo-root "$KANBAN_REPO_ROOT" \
  --kind nightly

# Step 2: confirm, commit, push (only this script runs git push)
python scripts/push_kanban_report.py \
  --kanban-repo-root "$KANBAN_REPO_ROOT"
```

## Step 1 — archive + stage (`push_report_to_kanban.py`)

Uses **`gh`** for GitHub authentication on `git pull`:

1. Verify **`gh`** is installed; else exit with install instructions.
2. Verify **`gh auth status`** (or valid `GH_TOKEN` / `GITHUB_TOKEN`).
3. Copy report HTML into `data/nightly_test_report/` or `data/release_test_report/` (canonical filename).
4. `git pull --rebase origin <branch>` with `gh auth git-credential` (unless `--skip-pull`).
5. `git add` the new/updated report HTML under `data/*_test_report/` **and** the **`data/local_nightly_raw/manual_*`** recorded in `.last_manual_dir` (written by `prepare_kanban_before_report.py`; override with `--local-nightly-manual-dir`; skip with `--skip-local-nightly-raw`). **Not** `docs/assets/test_reports/`.
6. **Print push preview** inline in stdout (repository, remote, branch, commit message, staged file list with sizes, `name-status`, `diff --stat`). Agents must **paste this entire block** to the user — not a one-line summary.
7. **Stop** — no commit or push. Tell the user to run `push_kanban_report.py` after review.

Use **`--dry-run`** to preview planned steps only; **`--archive-only`** to copy into `data/` without staging.

## Step 2 — confirm + push (`push_kanban_report.py`)

**Only this script** runs `git commit` and `git push`.

1. Build preview from **already staged** files under `data/*_test_report/`.
2. **Print full push preview** (same fields as step 1). Use **`--preview-only`** to show the preview and exit without push.
3. **Confirm before push:**
   - Interactive terminal: prompt `Proceed with git commit and push to kanban? [y/N]`
   - Non-interactive / agent: exit with code 3 and print the **full preview inline**; **ask the user in chat**, then re-run with **`--yes`**
   - User declines (interactive): unstage and exit without push
4. Commit: `chore(reports): archive {nightly|release} test report YYYY-MM-DD` (appends `+ manual_YYYYMMDD` when local raw is included)
5. `git push origin <branch>` with `gh auth git-credential`

Use **`--dry-run`** to preview commit/push commands; **`--yes`** to push immediately after preview (**only** after explicit user confirmation in chat).

## Troubleshooting

- **`gh` not found** — install from [cli.github.com](https://cli.github.com/); Windows: `winget install --id GitHub.cli`.
- **`gh auth status` failed** — run `gh auth login` or export `GH_TOKEN` with `repo` scope.
- **Not a git repository** — clone [vllm-omni-kanban](https://github.com/hsliuustc0106/vllm-omni-kanban) and set `KANBAN_REPO_ROOT`.
- **push failed** — resolve rebase conflicts manually in the kanban checkout; re-run after `gh auth status` succeeds.
- **Wrong filename** — pass `--date YYYY-MM-DD` and `--kind nightly|release` explicitly.
- **Reports page empty locally** — run `mkdocs serve` in kanban; hooks sync `data/` → `docs/assets/test_reports/` at build time (no git commit needed).
