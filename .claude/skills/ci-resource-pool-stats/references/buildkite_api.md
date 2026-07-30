# Buildkite REST API Reference — Resource Pool Stats

## Endpoints used

### List builds for a pipeline

```
GET /v2/organizations/{org_slug}/pipelines/{pipeline_slug}/builds
```

**Parameters**:
- `created_from` (ISO 8601 datetime, e.g. `2026-07-22T00:00:00Z`)
- `created_to` (ISO 8601 datetime, e.g. `2026-07-22T23:59:59Z`)
- `per_page` (integer, default 100, max 100)
- `branch` (optional, filter by branch name)
- Pagination via `Link` header (`rel="next"`)

**Note**: The list endpoint may omit `jobs[]` in each build object. When `jobs` is missing or empty, the script refetches the build individually.

### Fetch a single build with jobs

```
GET /v2/organizations/{org_slug}/pipelines/{pipeline_slug}/builds/{build_number}
```

Returns the full build object including `jobs[]` with all job details.

### Authentication

All requests require a Bearer token:
```
Authorization: Bearer {token}
```

Set via `BUILDKITE_API_TOKEN` or `BUILDKITE_TOKEN` environment variable.

## Job object fields

The key fields from each job object used by `resource_pool_stats.py`:

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Job type: `"script"`, `"wait"`, `"trigger"`, `"block"`, `"input"`. Only `"script"` jobs are analyzed (they run on agents). |
| `state` | string | Job state: `"scheduled"`, `"assigned"`, `"running"`, `"passed"`, `"failed"`, `"canceled"`, `"blocked"`, `"skipped"`, `"not_run"`, `"broken"` |
| `scheduled_at` | string (ISO 8601) | Timestamp when the job was created/queued |
| `started_at` | string (ISO 8601) | Timestamp when an agent began executing the job |
| `finished_at` | string (ISO 8601) | Timestamp when the job completed (pass/fail/etc.) |
| `agent_query_rules` | array | Rules that determine which agent picks up this job. Can be a list of dicts (`{"rule": "include", "query": "queue=gpu-h200"}`) or a list of strings (`"queue=gpu-h200"`). |
| `queue` | string | Convenience field — the primary queue name derived from `agent_query_rules`. May be absent in some API versions. |

## Queue wait time calculation

**Queue wait time** = `started_at` − `scheduled_at`

This measures how long a job waited in the queue before an agent was available to run it. Only computed for jobs that have both `scheduled_at` and `started_at` timestamps.

## Job duration calculation

**Job duration** = `finished_at` − `started_at`

This measures how long the job spent executing on the agent. Only computed for jobs that have both `started_at` and `finished_at` timestamps.

## Resource pool identification

Resource pools (queues) in Buildkite are identified by the `agent_query_rules` on each job:

- **Standard convention**: `queue=<pool-name>` entries, e.g. `queue=gpu-h200`, `queue=npu-910b`
- **Include rules**: `{"rule": "include", "query": "queue=gpu-h200"}` means the job should run on an agent tagged with `queue=gpu-h200`
- **Default pool**: Jobs without any explicit queue rule are assigned to the `"default"` pool
- **Fallback**: If `agent_query_rules` is absent or empty, the script checks the convenience `queue` field, then falls back to `"default"`

### Agent metadata

Agents are configured with metadata tags that match the `agent_query_rules`. An agent tagged with `queue=gpu-h200` will pick up jobs that specify `queue=gpu-h200` in their `agent_query_rules`.

## Pipelines covered

| Pipeline | Buildkite URL | Description |
|----------|---------------|-------------|
| `vllm-omni` | https://buildkite.com/vllm/vllm-omni/ | GPU CI pipeline (H200, H800, H100, A100, etc.) |
| `vllm-omni-npu-ci` | https://buildkite.com/vllm/vllm-omni-npu-ci/ | NPU CI pipeline (Ascend NPU 910B, etc.) |

## Rate limiting

The Buildkite API may return HTTP 429 (rate limit exceeded). The script handles this by:
1. Reading the `Retry-After` header and sleeping for that duration (+1s)
2. Retrying up to 10 times per request
3. Sleeping 0.12s between paginated requests (configurable via `BUILDKITE_BUILDS_PAGE_SLEEP`)
