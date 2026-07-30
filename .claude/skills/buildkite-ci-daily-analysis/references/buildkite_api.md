# Buildkite REST API Reference — Daily Job Analysis

## Endpoints used

### List builds for a pipeline

```
GET /v2/organizations/{org_slug}/pipelines/{pipeline_slug}/builds
```

**Parameters**:
- `created_from` (ISO 8601 datetime, e.g. `2026-07-23T00:00:00Z`)
- `created_to` (ISO 8601 datetime, e.g. `2026-07-23T23:59:59Z`)
- `per_page` (integer, default 100, max 100)
- Pagination via the `Link` header (`rel="next"`)

The list endpoint may omit `jobs[]` in each build object. When `jobs` is
missing or empty on a returned build, the script refetches the build
individually.

### Fetch a single build with jobs

```
GET /v2/organizations/{org_slug}/pipelines/{pipeline_slug}/builds/{build_number}
```

Returns the full build object including `jobs[]` with every job field.

### Authentication

All requests require a Bearer token:

```
Authorization: Bearer {token}
```

Set via `BUILDKITE_API_TOKEN` or `BUILDKITE_TOKEN` environment variable.

## Build object fields used

| Field | Type | Description |
|-------|------|-------------|
| `number` | integer | Build number (e.g. `4521`). |
| `state` | string | Build state: `passed`, `failed`, `canceled`, `running`, `scheduled`, `blocked`, `not_run`, `skipped`. |
| `branch` | string | Source branch that triggered the build (e.g. `main`, `yenuo26/some-feature`). |
| `commit` | string | Git commit SHA. |
| `created_at` | string (ISO 8601) | Timestamp the build was created. Used for date filtering. |
| `started_at` | string (ISO 8601) | Timestamp the first job started (nullable). |
| `finished_at` | string (ISO 8601) | Timestamp the build finished (nullable). |
| `web_url` | string | URL to the build on Buildkite (e.g. `https://buildkite.com/vllm/vllm-omni/builds/4521`). |
| `jobs` | array | Embedded jobs (may be missing — refetch if so). |

## Job object fields used

| Field | Type | Description |
|-------|------|-------------|
| `id` | string (UUID) | Unique job identifier. |
| `type` | string | Job type: `script`, `wait`, `trigger`, `block`, `input`. Only `script` jobs are analyzed — they run on agents. |
| `name` | string | Job label (e.g. `unit-tests`, `lint`, `pytest-mcore-models`). |
| `state` | string | Job state. See classification below. |
| `started_at` | string (ISO 8601) | Timestamp when an agent began executing the job (nullable). |
| `finished_at` | string (ISO 8601) | Timestamp when the job completed (nullable). |
| `created_at` | string (ISO 8601) | Timestamp the job was queued. |
| `scheduled_at` | string (ISO 8601) | Timestamp the job entered the agent queue (often identical to `created_at`). |
| `exit_status` | integer | Numeric exit code from the script (0 = success). |
| `web_url` | string | URL to the job on Buildkite. |
| `agent_query_rules` | array | Agent tag rules. Often includes `queue=<pool-name>` — not needed for this skill, but kept in case future analyses want pool-level grouping. |

## Job state classification

For "success" vs "failure" analytics, the Buildkite job state is collapsed:

| API state   | Bucket       | Notes |
|-------------|--------------|-------|
| `passed`    | **passed**   | Job succeeded. |
| `failed`    | **failed**   | Job exited with a non-zero status. |
| `canceled`  | **canceled** | Manually canceled or preempted. Reported separately but excluded from success-rate denominator. |
| `running`   | **running**  | Still executing at fetch time. Excluded from success-rate denominator. |
| `scheduled` | **other**    | Not yet picked up. Excluded from success-rate denominator. |
| `blocked`   | **other**    | Waiting on a `block` step. Excluded from success-rate denominator. |
| `skipped`   | **other**    | Skipped due to a `if` condition. Excluded from success-rate denominator. |
| `not_run`   | **other**    | Build was canceled before this job started. |
| `broken`    | **other**    | Internal Buildkite error. |

**Success rate** is computed as `passed / (passed + failed)` — canceled,
running, and other states are excluded so they don't drag the number down.

## Duration calculation

**Job duration** = `finished_at − started_at`

Only jobs with both timestamps and a positive difference are counted.
Jobs still running or missing timestamps are excluded.

`format_duration` produces human-readable strings:

- `< 60s` → `Ns` (e.g. `42.3s`)
- `< 1h` → `MmSs` (e.g. `12m07s`)
- `≥ 1h` → `Hh Mm Ss` (e.g. `1h 23m 04s`)

## Branch extraction

Each build carries its own `branch`. The script flattens this to a per-job
record so jobs from the same build share the same branch. Branches like
`main` and short-lived feature branches (`alice/add-thing`) appear as
distinct values in the filter dropdown.

## Pipelines covered

| Pipeline slug          | Buildkite URL                                                    | Hardware |
|------------------------|------------------------------------------------------------------|----------|
| `vllm-omni`            | https://buildkite.com/vllm/vllm-omni/                            | GPU CI (H200, H800, H100, A100, etc.) |
| `vllm-omni-npu-ci`     | https://buildkite.com/vllm/vllm-omni-npu-ci/                     | NPU CI (Ascend 910B, etc.)            |

## Rate limiting

The Buildkite API may return HTTP 429 (rate limit exceeded). The script:

1. Reads the `Retry-After` header and sleeps for that duration (+1s).
2. Retries up to 10 times per request.
3. Sleeps 0.12s between paginated requests (tunable via the
   `BUILDKITE_BUILDS_PAGE_SLEEP` environment variable).
