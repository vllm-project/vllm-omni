# vLLM-Omni PR Monitor

Generate a **standalone HTML report** for the **Top 20 Stale PRs (by
Bot-Mentioned Reviewers)** sub-section of the vllm-omni-test-report nightly.
This skill emits one self-contained HTML file with no Buildkite, local
log, or kanban dependencies — only the GitHub calls needed to enumerate
open PRs, search for the triage bot's comments, and bucket each PR by the
reviewers the bot mentioned.

> **Renamed 2026-08-27** from `vllm-omni-stale-prs` to
> `vllm-omni-pr-monitor`. Output filenames changed from
> `stale-prs-report-YYYY-MM-DD.html` to
> `pr-monitor-report-YYYY-MM-DD.html`. The skill's scope (Top 20
> bot-mentioned-reviewer buckets + high-priority PRs surface) is unchanged.

| | |
|---|---|
| **Output** | `pr-monitor-report-YYYY-MM-DD.html` (one file, default dated UTC) |
| **Audience** | Reviewers triaging their pending reviews between nightly runs |
| **Refresh cadence** | On-demand, after any bot-triage comment or new PR |
| **Source skill** | Sibling of [vllm-omni-test-report](../vllm-omni-test-report/SKILL.md) |
| **Archive** | On explicit `归档报告` → omni-reviewbot `kanban/data/vllm_omni/pr_monitor/pr-monitor-report.html` (no date, overwrites same-name, pushes to repo; sole target since 2026-10-21). See [Report archival (`归档报告`)](#report-archival-归档报告). |

## What it renders

1. **Pending reviewers** / **Pending PRs** / **Oldest pending** metric cards
   (the "Oldest pending" card flips red when any bucket has a PR ≥ 30
   days open).
2. **Top 20 Stale PRs (by Bot-Mentioned Reviewers)** sub-card — one
   collapsible **per-reviewer bucket** (`@alice`, `@bob`, …) ordered by
   the bucket's oldest-pending PR. The first bucket opens by default.
3. Within each bucket:
   - High-priority PRs (`critical` / `p0` / `p1` / `high priority`)
     surface first.
   - First 5 rows visible by default; remaining rows (up to 20 per
     bucket) hide behind a **Show all N** button.
4. **Data source footer** — repo URL + bot login + token state + generation
   timestamp.

## Math / data sources

Identical to the parent skill's Daily-focus Stale PRs sub-card
(`vllm-omni-test-report/SKILL.md` → "Top 20 Stale PRs"). Constants and
regexes are inlined verbatim into `scripts/pr_monitor_report.py`:

- **Open PRs**: `GET /repos/vllm-project/vllm-omni/pulls?state=open&per_page=100&page=…` (paginated).
- **Bot PR search** (bounds the per-PR fetch to PRs the bot commented on):
  `GET /search/issues?q=commenter:<bot>+is:open+is:pr+repo:vllm-project/vllm-omni` (3 pages × 100).
- **Per-PR triage**: `GET /issues/{n}/comments` for each PR the bot commented on, in parallel (8 workers).
- **Triage parsing**: regex for `Module owners: @alice @bob` (and `Model` /
  `Issue` / `Component` / `File` variants). Self-review mentions in the
  next paragraph are intentionally excluded.

## Quick Path

Ask for or infer:
- **GitHub token**: `GITHUB_TOKEN` / `GH_TOKEN` in the environment (recommended).
  Do **not** paste the token into chat.
- Optional output path: `--out ./pr-monitor-report-2026-08-27.html`.
- Optional UTC date: `--date YYYY-MM-DD` (default: today UTC).
- Optional explicit token: `--gh-token TOKEN` (overrides env; not recommended).
- Optional tokenless run: `--no-token` (degraded REST).
- Optional smoke test: `--preview` (no GitHub calls; writes `pr-monitor-report-preview-YYYY-MM-DD.html`).
- Optional bot login override: `--bot-login vllm-omni-review-bot` (defaults to the production bot).

```bash
export GITHUB_TOKEN=...   # or GH_TOKEN; do not paste the token in chat
python scripts/pr_monitor_report.py                       # writes pr-monitor-report-YYYY-MM-DD.html
python scripts/pr_monitor_report.py --preview             # empty dataset, offline smoke test
python scripts/pr_monitor_report.py --date 2026-08-26     # explicit UTC date
python scripts/pr_monitor_report.py --out ./my.html       # override output path
python scripts/pr_monitor_report.py --no-token            # degraded REST (rate-limited)
python scripts/pr_monitor_report.py --bot-login my-bot    # override the triage bot login
```

Default output: `<skill_dir>/pr-monitor-report-YYYY-MM-DD.html` (today UTC).
Override with `--out`.

## Authentication (environment only)

- Reads `GITHUB_TOKEN` / `GH_TOKEN` from the **process environment** (either
  name works). Use one consistently per run.
- **Do not** paste the secret into chat, on the command line, or in files
  committed to git. If unset, ask the user to **export it in their local
  shell** (or in CI) and retry.
- `--no-token` skips auth and produces a degraded report (the per-PR
  comment fetch requires auth — without it the section renders empty
  with a "set `GITHUB_TOKEN`" hint instead of raising).

## When to Apply

- User asks for a stale-PR / pending-review / reviewer-bucket view
  without rerunning the full nightly.
- A reviewer wants to see only their own bucket without scrolling the
  parent nightly.
- The bot's triage logic changed (new regex variant / new exclusion
  list) and the per-reviewer buckets need regenerating between nightly
  builds.

## Definitions

- **Bot-mentioned reviewer** — A GitHub login that the
  `vllm-omni-review-bot` account mentions in its triage comment on a PR
  (e.g. `Module owners: @alice @bob`). The PR is bucketed under each
  reviewer separately.
- **High priority** — PRs whose labels include any of `critical`, `p0`,
  `p1`, or `high priority` (case-insensitive). Within a bucket,
  high-priority PRs sort first regardless of age.
- **Oldest pending** — the largest `days_open` across all PRs in all
  buckets. Drives the third metric card and its red-alert rule
  (`≥ 30 days ⇒ focus-card--fail`).
- **Bucket cap** — 20 PRs per reviewer (`_LEGACY_PR_PER_REVIEWER_LIMIT`).

## Constraints

- **Token required** for the per-PR comment fetch. Without
  `GITHUB_TOKEN` the script reports 0 buckets with a clear footer note
  instead of silently misbucketing.
- Search API is capped at 300 results (3 pages of 100). In practice the
  bot has commented on ~160 open PRs; if the repo grows past 300, the
  cap should be revisited.
- **No PR-payload mutations** — the script is strictly read-only.
- The script never touches the kanban repo unless `--archive` (or the
  `归档报告` intent) is used. See [Report archival (`归档报告`)](#report-archival-归档报告).

## Report archival (`归档报告`)

Only activate this behavior when the user explicitly says **`归档报告`**
(or an explicit equivalent such as "归档", "archive", "提交并推送报告"):

> **Target switch (user-set 2026-10-21): report archives go ONLY to
> [JiusiServe/omni-reviewbot](https://github.com/JiusiServe/omni-reviewbot) —
> STOP pushing to vllm-omni-kanban.** The script's `--archive` /
> `--archive-dry-run` CLI flags (and `--kanban-repo-root`) target the kanban
> repo and are therefore **retired** — do not use them.

1. Generate the report (HTML by default, dated filename) as usual and sanity-check
   the bucket/reviewer data before archiving.
2. On the bridge box run:

   ```bash
   ../vllm-omni-test-report/scripts/archive_to_omni_review.sh \
     <pr-monitor-report-YYYY-MM-DD.html> data/pr_monitor/pr-monitor-report.html
   ```

   — **no date in the destination filename**, overwrite semantics. The shared
   archive script lives in the sibling **vllm-omni-test-report** skill
   (`scripts/archive_to_omni_review.sh`; the path above is relative to this
   skill's directory). It copies the report into omni-reviewbot at
   `kanban/data/vllm_omni/pr_monitor/`,
   commits as `yenuo26`, and pushes; the push auto-triggers the repo's
   `Build kanban site` workflow. It never touches the kanban repo.
3. Report the archived path, commit hash, and push result. If generation or
   the archive script fails (non-zero exit), surface the exact error and do
   **not** claim archival completed. In non-bridge environments (script or
   `/home/wy/omni-reviewbot` clone absent), say so and leave the report on
   disk — do **not** fall back to pushing to vllm-omni-kanban.

The user's explicit `归档报告` request authorizes this report-specific
commit and push; no separate approval step is needed. The archived filename
is always `pr-monitor-report.html` — there is no dated archive variant
because the file always reflects the most recent stale-PR snapshot.

Without the exact `归档报告` intent (and without `--archive`), preserve
the normal behavior: generate the dated report in the skill directory and
do **not** touch the kanban repo.

## Related

- Parent nightly that embeds the same sub-card:
  [vllm-omni-test-report](../vllm-omni-test-report/SKILL.md)
- Sibling skill for outstanding DI: [vllm-omni-issue-monitor](../vllm-omni-issue-monitor/SKILL.md)
- GitHub Search API: <https://docs.github.com/en/rest/search>
- GitHub PR API: <https://docs.github.com/en/rest/pulls/pulls>