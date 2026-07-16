# CI testing: GitHub CI failure issues (`label:bug` + `label:ci-failure`)

Data rules for the **### Analysis (CI Failure)** table under **## CI testing** in the report.

## Data sources (manual cross-check / API)

- **Labeled list (open + closed):** [issues · `label:bug` + `label:ci-failure`](https://github.com/vllm-project/vllm-omni/issues?q=is%3Aissue+label%3Abug+label%3Aci-failure)
- **Open `label:bug` only:** [open bugs](https://github.com/vllm-project/vllm-omni/issues/?q=is%3Aissue%20state%3Aopen%20label%3Abug)

For a full enumeration aligned with **`compose_full_report.py`**, use **GitHub Search API** `GET /search/issues` with:

- `repo:vllm-project/vllm-omni is:issue label:bug label:ci-failure created:{from}..{to}`

where `{from}` and `{to}` are **`YYYY-MM-DD` (UTC)** and match **`--stats-from`** and **`--stats-to`** (the same window as **Metrics overview** Buildkite `created_at` stats).

## Filter rules

| Criterion | Description |
|-----------|-------------|
| **Labels** | **`bug`** **and** **`ci-failure`** (exact label name on the repo; see GitHub API `labels[].name`). |
| **Time** | Issue **`created`** date (**UTC**) in the **inclusive** range **`--stats-from`** … **`--stats-to`** from `compose_full_report.py` (same window as **Open issues (stats window)**). |
| **Exclude** | Pull requests (`pull_request` non-null in API). |

**Note:** Older reports filtered by **`[CI Failure]` title prefix**; that missed titles like **`[Bug][CI Failure]:`** without a colon. Label-based matching is authoritative.

## Table columns

| Column | Content |
|--------|---------|
| **Issue #** | Link to `https://github.com/vllm-project/vllm-omni/issues/{n}` |
| **Title** | Issue `title` (escape `\|` in Markdown if needed) |
| **Status** | **Open** or **Closed** |

If nothing matches the date range: state that explicitly—do not leave an ambiguous empty table.
