# vLLM-Omni Issue Monitor

Generate a **standalone HTML report** for the **Top 20 DI contributors
(Outstanding DI)** sub-section of the vllm-omni-test-report nightly. This
skill emits one self-contained HTML file with no Buildkite, local log, or
kanban dependencies — only the GitHub REST/GraphQL calls needed to
recompute Outstanding DI.

> **Renamed 2026-08-27** from `vllm-omni-di-top20` to
> `vllm-omni-issue-monitor`. Output filenames changed from
> `di-top20-report-YYYY-MM-DD.html` to
> `issue-monitor-report-YYYY-MM-DD.html`. The skill's scope (Outstanding DI
> + Top 20 contributors) is unchanged.

| | |
|---|---|
| **Output** | `issue-monitor-report-YYYY-MM-DD.html` (one file, default dated UTC) |
| **Audience** | Engineers triaging outstanding bug backlog between nightly runs |
| **Refresh cadence** | On-demand, after any DI-relevant issue creation/closure |
| **Source skill** | Sibling of [vllm-omni-test-report](../vllm-omni-test-report/SKILL.md) |
| **Archive** | On explicit `归档报告` → `vllm-omni-kanban/data/issue_monitor/issue-monitor-report.html` (no date, overwrites same-name, pushes to repo). See [Report archival (`归档报告`)](#report-archival-归档报告). |

## What it renders

1. **Outstanding DI** metric card — total SLO-escalating DI for all open
   `label:bug` issues on `vllm-project/vllm-omni`; red when total > 30
   (matches the development-snapshot red-alert rule).
2. **Open bugs** counter — total number of `label:bug` issues counted.
3. **Top 20 DI contributors** sub-card — sortable-by-DI table:
   `#` · `Title` · `Priority` · `Days` · `DI` · `Assignee` ·
   `Maintainer` · `Bugfix`. Rows 21–40 hide behind an **Expand to show
   Top 40** toggle when more issues exist.
4. **Stale Bugs (open > 2 months)** sub-card — open `label:bug` issues
   still unresolved past the typical resolution window
   (`days_open > STALE_BUG_DAYS = 60`), excluding `wontfix` / `invalid`,
   sorted by age (oldest first). Same 8 columns as the Top 20 table so
   Assignee / Maintainer / Bugfix remain editable and persisted via
   `localStorage`; Days column carries a red `--stale` accent and the
   header strip uses a warning tint so the section reads as
   "old / needs attention" alongside the clean Top 20 sub-card. Hidden
   when there are no open bugs at all (the parent panel already shows
   the empty-state hint); renders a positive indicator when open bugs
   exist but none are stale.
5. **Data source footer** — repo URL + token state + generation timestamp.

## Math / data sources

Identical to the parent skill's Daily-focus DI card
(`vllm-omni-test-report/SKILL.md` → "Daily focus DI card (SLO-escalating
model)"). Constants are inlined verbatim into
`scripts/issue_monitor_report.py` so behavior stays aligned without an
import edge against the parent.

- **Issues**: `GET /repos/vllm-project/vllm-omni/issues?state=open&labels=bug&per_page=100&page=…` (paginated, PRs excluded).
- **Bugfix PRs**: `Issue.closedByPullRequestsReferences` via one GraphQL
  alias query — the canonical mirror of GitHub's Development panel.
- **DI formula**: `DI = base × ⌈days_open / slo_days⌉` where `base` /
  `slo_days` come from the `BUG_DI_TABLE` priority ladder.
- **Stale threshold**: `STALE_BUG_DAYS = 60` (module constant in
  `scripts/issue_monitor_report.py`). Issues with `days_open > 60`
  surface in the Stale Bugs sub-card, regardless of DI ranking.

## Quick Path

Ask for or infer:
- **GitHub token**: `GITHUB_TOKEN` / `GH_TOKEN` in the environment (recommended
  — unauthenticated REST is rate-limited). Do **not** paste the token into chat.
- Optional output path: `--out ./issue-monitor-report-2026-08-27.html`.
- Optional UTC date: `--date YYYY-MM-DD` (default: today UTC).
- Optional explicit token: `--gh-token TOKEN` (overrides env; not recommended).
- Optional tokenless run: `--no-token` (degraded REST; degrade note shown in footer).
- Optional smoke test: `--preview` (no GitHub calls; writes `issue-monitor-report-preview-YYYY-MM-DD.html`).

```bash
export GITHUB_TOKEN=...   # or GH_TOKEN; do not paste the token in chat
python scripts/issue_monitor_report.py                       # writes issue-monitor-report-YYYY-MM-DD.html
python scripts/issue_monitor_report.py --preview             # empty dataset, offline smoke test
python scripts/issue_monitor_report.py --date 2026-08-26     # explicit UTC date
python scripts/issue_monitor_report.py --out ./my.html       # override output path
python scripts/issue_monitor_report.py --no-token            # degraded REST (rate-limited)
```

Default output: `<skill_dir>/issue-monitor-report-YYYY-MM-DD.html` (today UTC).
Override with `--out`.

## Authentication (environment only)

- Reads `GITHUB_TOKEN` / `GH_TOKEN` from the **process environment** (either
  name works). Use one consistently per run.
- **Do not** paste the secret into chat, on the command line, or in files
  committed to git. If unset, ask the user to **export it in their local
  shell** (or in CI) and retry.
- `--no-token` skips auth and produces a degraded report (card footer
  notes the unauthenticated REST fallback).

## Editable cells

Both **Assignee** and **Maintainer** cells are inline editable
`<input>` fields:

- Edit on the page → value is mirrored to `data-*-value` attribute
  (so a `Ctrl+S` "Save Page As" captures the value into the saved HTML)
  **and** to `localStorage` (reload-same-origin persistence).
- Reload same origin → `data-*-value` is preferred over `localStorage` so
  a copy-saved-as-file retains edits across origins.
- In-memory fallback (`mem = {}`) covers Chrome `file://` where
  `localStorage` throws.

## When to Apply

- User asks for an Outstanding DI / Top DI contributors / DI Top 20 view
  without rerunning the full nightly.
- The Daily-focus DI card in the parent nightly needs refreshing
  between scheduled nightly builds.
- A user pastes a GitHub issue number and wants to triage its position in
  the DI ladder without opening GitHub's bug dashboard.

## Definitions

| Term | Meaning |
|------|---------|
| **Outstanding DI** | Sum of per-issue SLO-escalating DI across **all** open `label:bug` issues, including `wontfix`/`invalid` rows (they count toward the total but never appear in the ranking). |
| **Top DI contributors** | Open `label:bug` issues **excluding** `wontfix`/`invalid`, ranked by DI descending. |
| **Bugfix column** | PR numbers surfaced via `Issue.closedByPullRequestsReferences` (Development panel mirror). Empty when no GraphQL link exists or the token is missing. |
| **Red threshold** | Outstanding DI > 30 ⇒ metric card renders red (`focus-card--fail`). |
| **Stale bug** | An open `label:bug` issue with `days_open > 60` (= 2 months), excluding `wontfix` / `invalid`. Surfaced in the **Stale Bugs (open > 2 months)** sub-card, sorted by age (oldest first). |

## Constraints

1. Do not invent counts; the script only renders what GitHub returns. If
   the API fails (rate limit / network / missing token), the card shows
   `0 open bug(s); no open bug in snapshot` and the data-source footer
   explains the failure mode.
2. If the GraphQL pass for `Bugfix` fails, every row shows `—` in the
   Bugfix column (graceful degradation; same as the parent skill).
3. The script never runs Buildkite, never reads local logs, and never
   touches the kanban repo — completely independent of the parent
   nightly pipeline **unless** `--archive` (or the `归档报告` intent) is
   used. See [Report archival (`归档报告`)](#report-archival-归档报告).

## Report archival (`归档报告`)

Only activate this behavior when the user explicitly says **`归档报告`**
(or an explicit equivalent such as "归档", "archive to kanban", "提交并推送报告"):

1. Generate the report in the requested format (HTML by default) as usual.
2. Locate the `vllm-omni-kanban` repository. Default **`/home/wy/vllm-omni-kanban`**
   (override via `--kanban-repo-root PATH` or `$KANBAN_REPO_ROOT`).
3. Copy the report to `vllm-omni-kanban/data/issue_monitor/issue-monitor-report.html`
   — **no date in the filename**, **overwrite** any existing file with the
   same name. Create the destination directory if it does not exist.
4. Inside the kanban repo: verify `gh` is installed and authenticated,
   then run `git pull --rebase`, `git add` only the archived file,
   confirm the staged diff is exactly the intended file, commit with a
   descriptive message (`data: archive issue monitor report (UTC <ISO>)`),
   and push the current branch to its configured remote (defaults:
   `origin` on the branch returned by `git rev-parse --abbrev-ref HEAD`).
5. Report the archived path, commit hash, branch, and push result. If any
   step (generation, copy, commit, push) fails, surface the exact error
   and do **not** claim archival completed.

The user's explicit `归档报告` request authorizes this report-specific
commit and push; no separate approval step is needed (unlike
[vllm-omni-test-report](../vllm-omni-test-report/SKILL.md) which uses a
two-step stage/confirm flow because its archive may include multi-file
`manual_*` payloads). The `--archive` flag drives the same workflow from
the CLI:

```bash
# Single-shot: generate, copy (overwrite), commit, push
python scripts/issue_monitor_report.py --archive
python scripts/issue_monitor_report.py --archive --date 2026-08-26
python scripts/issue_monitor_report.py --archive --kanban-repo-root /custom/path/to/vllm-omni-kanban
```

Use `--archive-dry-run` to print the planned archive + commit + push
without touching git. The default filename is always
`issue-monitor-report.html` — there is no dated archive variant because
the file always reflects the most recent Outstanding DI snapshot.

Without the exact `归档报告` intent (and without `--archive`), preserve
the normal behavior: generate the dated report in the skill directory and
do **not** touch the kanban repo.

## Related

- Parent nightly that embeds the same sub-card:
  [vllm-omni-test-report](../vllm-omni-test-report/SKILL.md)
- Sibling skill for stale PRs: [vllm-omni-pr-monitor](../vllm-omni-pr-monitor/SKILL.md)
- GitHub REST API: <https://docs.github.com/en/rest/issues/issues>
- GitHub GraphQL: <https://docs.github.com/en/graphql>