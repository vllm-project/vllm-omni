# Buildkite API Notes (vllm-omni)

## Endpoints

| Purpose | Method | Path |
|---------|--------|------|
| List builds | GET | `/v2/organizations/vllm/pipelines/vllm-omni/builds` |
| Single build | GET | `/v2/organizations/vllm/pipelines/vllm-omni/builds/{number}` |

Query params for list: `branch=main`, `per_page=30` (max 100 per Buildkite docs).

## Authentication

Put a read-only Buildkite API token in the environment as **`BUILDKITE_TOKEN`** or **`BUILDKITE_API_TOKEN`**. HTTP header: `Authorization: Bearer` + that value (e.g. in bash: `-H "Authorization: Bearer $BUILDKITE_TOKEN"` when `BUILDKITE_TOKEN` is set).

**Do not** commit tokens or pass them on the command line as literal arguments; keep them in shell env or CI secrets only.

Without a credential, the REST API typically returns **401**. The public web UI at [buildkite.com/vllm/vllm-omni/builds?branch=main](https://buildkite.com/vllm/vllm-omni/builds?branch=main) is for humans; automated report generation should use the API with an env-sourced token.

### GitHub API (bug metrics / `compose_full_report.py`)

Optional **`GITHUB_TOKEN`** or **`GH_TOKEN`** for GitHub REST calls in `buildkite_build_stats.py` (e.g. **Bug avg first response**) and in `compose_full_report.py` (open bugs, CI Failure search).

If Python fails TLS verification against **`api.github.com`** (`SSLCertVerificationError`, common behind some corporate proxies), you can set **`GITHUB_INSECURE_SSL=1`** (or `true` / `yes` / `on`) so **only GitHub** HTTPS requests skip certificate verification. **Buildkite requests stay verified.** Prefer fixing the trust store or `REQUESTS_CA_BUNDLE` when possible; disabling verification is weaker security.

## Identifying scheduled nightly builds

Scheduled builds usually have `message` like:

- `Scheduled nightly build`

Filter client-side after fetching `builds[]`:

```text
builds | map(select(.message | test("Scheduled nightly"; "i")))
```

Take index `0` for the latest if the API returns builds newest-first (default).

## Useful JSON fields (build object)

- `number`, `state`, `branch`, `commit`, `message`, `created_at`, `finished_at`
- `jobs[]`: each job has `name`, `state`, `id` (for deep links), `log_url`, `raw_log_url`

## Job logs (pytest detail)

Each job may include:

- `raw_log_url` - plain-text log (preferred for parsing)
- `log_url` - formatted log API URL

Fetch with the same `Authorization: Bearer` header as other REST calls (value from `BUILDKITE_TOKEN` or `BUILDKITE_API_TOKEN`).

Very large logs: keep a **tail window** (pytest prints the session summary at the end). The helper script
[`scripts/nightly_job_pytest_table.py`](../scripts/nightly_job_pytest_table.py) downloads the full log but,
if it grows beyond an internal cap, retains only the last ~10 MiB so the session footer remains available.

## Jobs excluded from test reporting

**Do not** include the following in job counts, failure lists, or the per-job pytest table:

- Any job whose **name** matches `(?i)^Upload .+ Pipeline$`
  (e.g. `Upload Ready Pipeline`, `Upload Nightly Pipeline`, `Upload Merge Pipeline`)

These steps are **artifact/metadata uploads**, not test executors. The overall Buildkite build `state` may still
reflect them; prose should clarify that **test health** is evaluated on the remaining jobs only.

## Per-job test case table (Markdown)

When filling the report's **Per-job test execution (pytest)** section:

1. Iterate **reportable jobs** only (after the Upload filter above).
2. For each job, pull `raw_log_url` and parse pytest output:
   - **Aggregate row**: **Result** from parsed pytest outcome / Buildkite step state (open the step log for the `=== ... ===` session line).
   - **Failure / error rows**: one row per `FAILED ...` / `ERROR ...` node id; put **`Job — node id`** in the **Job** column (same **Step link** as the job’s aggregate row).
3. **Do not invent** node ids or counts: if the log was truncated and no `FAILED` lines appear but the
   aggregate shows failures, note in prose or rely on the step link; do not fabricate log excerpts in the table.
4. Non-pytest jobs (Docker build, email, init): a single row with **Result** ending in `non-pytest or log truncated` when
   no pytest session line was found (omit long explanations from the table).

## CI aggregate stats (`buildkite_build_stats.py`)

The script [scripts/buildkite_build_stats.py](../scripts/buildkite_build_stats.py) pages through **all builds** in a
`created_from` / `created_to` window (UTC, Buildkite query params) and buckets each finished build into:

| Bucket | Rule |
|--------|------|
| ready | `branch != "main"` |
| merge | `branch == "main"` and **not** the scheduled-nightly case below (typical merged-PR `main` builds) |
| nightly | `branch == "main"` and (`source == "schedule"` or message suggests scheduled/nightly) |

**Success rate** (per bucket): `passed / (passed + failed)` only. Counts of `canceled`, `blocked`, etc. are reported separately.

**Average wall time** (per bucket): among builds counted as `passed` or `failed`, those with both `created_at` and
`finished_at`, compute `finished_at - created_at` in seconds and take the arithmetic mean.

Run with `BUILDKITE_TOKEN` or `BUILDKITE_API_TOKEN` and optional `--markdown`. **`--from` / `--to`** (`YYYY-MM-DD`, UTC, inclusive) are optional: if both are omitted, the script uses the **current UTC month through today**; if you set one, set both.

**`ut` / `ut (exclude models)`** (Metrics table): **not** limited by `--from` / `--to`. The script selects the **newest [`main` build](https://buildkite.com/vllm/vllm-omni/builds?branch=main) that is not** a scheduled nightly (same heuristic as the **merge** bucket: `classify_build` → `main_non_nightly` in `buildkite_build_stats.py`), then parses the **Simple Unit Test** step log from that build.

**`ut (exclude models)`** recomputes line coverage from coverage.py **per-file** `Stmts` / `Miss` rows after dropping any path whose slash-separated segments include a directory named **`models`** (case-insensitive; e.g. `foo/models/bar.py` is excluded; `models.py` alone is not). Per-file rows use the **Cover** ``NN%`` column even when a **Missing** column follows (the line does not end with ``%``). The script fetches the step log up to **`UT_LOG_MAX_BYTES`** (default **200MiB**, rolling tail if the log is larger; see `buildkite_build_stats.py`), then parses only the text report from the **last** pytest **`tests coverage`** banner through EOF (`extract_pytest_coverage_text_report`). Summed per-file **Stmts** (including `models/` paths) should match the **TOTAL** row when the log is complete; if **ut** or **ut (exclude models)** look inconsistent, raise **`UT_LOG_MAX_BYTES`** or check for truncation. The **TOTAL** row is matched only when the line **starts** with `TOTAL` (after stripping a leading `[timestamp]` and ANSI codes), so timestamps or paths that merely contain the substring `TOTAL` are not mistaken for the summary row.

## Example: jq one-liner for latest scheduled nightly number

```bash
curl -s -H "Authorization: Bearer $BUILDKITE_TOKEN" \
  "https://api.buildkite.com/v2/organizations/vllm/pipelines/vllm-omni/builds?branch=main&per_page=50" \
  | jq '[.[] | select(.message | test("Scheduled nightly"; "i"))] | .[0].number'
```
