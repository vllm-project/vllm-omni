# CUDA and AMD case coverage

Job counts do not measure test-case parity. CUDA test-pipeline rendering and the
AMD test template enable a pytest recorder and upload
`artifacts/ci-case-coverage/*.jsonl`. It records the selected node IDs **after**
marker filtering and sharding, setup/call/teardown results, skip reasons, the
Buildkite commit/job, CI tier, and blocking status. Collection-only invocations
are identified separately and never counted as execution. Reporting does not
change test assertions, skip policy, or exit codes.

Each invocation writes its own file, including xdist workers. Events are flushed
as they arrive; a killed process leaves incomplete evidence rather than a false
pass. Worker records are combined by job, and xdist scheduling group suffixes
are removed without changing parameter IDs. No command arguments or environment
snapshots are recorded.

## Compare an explicit scope

Download the case-report artifacts from CUDA and AMD builds of the **same full
commit SHA** into one directory. Keep each job's artifact directory separate
when downloading. Create a scope JSON listing every actual Buildkite job ID
being compared, including every shard. For example:

```json
{
  "schema_version": 1,
  "commit": "0123456789abcdef0123456789abcdef01234567",
  "jobs": [
    {"platform": "cuda", "tier": "ready", "job_id": "CUDA-JOB-ID"},
    {"platform": "amd", "tier": "ready", "job_id": "AMD-JOB-ID"}
  ],
  "unsupported": [],
  "allowed_skips": []
}
```

Replace the example SHA/job IDs with the source builds' values. Use only jobs
that invoke pytest; image builds, shell-only checks, and email jobs are outside
this report's scope. Buildkite conditional uploads and source filtering can
omit jobs. Do not interpret a deliberately narrowed scope as the entire CI
inventory, and do not omit a failed or missing shard to improve the result.

```bash
python tools/ci_case_coverage.py --scope scope.json \
  --artifacts downloaded-case-reports --output case-coverage.json
```

The command exits nonzero for missing execution artifacts, incomplete sessions,
nonzero pytest exits, empty selections, collection failures/skips, duplicate
selected IDs, unexpected execution skips, and incomplete node execution.
Revision or metadata mismatches reject the comparison. Inspect `valid` and
`problems` before using any case totals. Conditional `soft_fail` exit-code rules
are recorded as `conditional` and rejected rather than guessed to be blocking.

## Interpret the rows

Each row is an exact `(tier, nodeid)` pair, with independent CUDA/AMD evidence:

| Status | Meaning within the supplied job scope |
| --- | --- |
| `already-covered` | A blocking job completed setup, call, and teardown successfully. |
| `nonblocking` | Only nonblocking jobs selected the node; inspect results separately. |
| `unverified` | A blocking job selected the node, but complete passing execution is missing. |
| `absent` | No scoped job selected the node on this platform. |
| `unsupported` | An explicit reviewed support exclusion explains the absent selection. |

`selected_jobs`, `blocking_jobs`, `passed_jobs`, phase outcome counts, and skip
details remain visible. A skip is never treated as a pass. `unsupported` and
`allowed_skips` entries require `platform`, `tier`, an **exact** `nodeid`, and a
nonempty `reason`. Review and version these explanations with the scope. Do not
infer unsupported hardware from missing marks or from a runtime failure. A
support exclusion that contradicts actual selection is an evidence problem.

These are pytest outcomes, not Buildkite job conclusions. Review job status
separately for shell cleanup failures, canceled jobs, and other non-pytest checks.

Exact node IDs avoid conflating test methods, parameter cases, and job counts.
Different node IDs testing similar behavior still need a human equivalence
review. AMD's combined uploads deduplicate identical jobs; their artifacts are
attributed to the retained suite group, not counted again for omitted copies.

## Local verification and limitations

The recorder is inactive unless `VLLM_OMNI_CI_CASE_DIR` is set. For an isolated
pytest invocation without the repository conftest, load it explicitly with
`-p tests.helpers.fixtures.ci_case_coverage`. CI commands that disable conftest
and do not load this plugin produce no report; the scoped comparison flags
missing evidence instead of assuming those tests passed.

CPU regression checks (pytest and pytest-xdist installed; no model/GPU needed):

```bash
python -m pytest --noconftest -q -o addopts='' \
  tests/helpers/tests/test_ci_case_coverage.py
```

The recorder uses the Buildkite job's commit metadata. This is evidence about
those jobs, not proof that arbitrary downloaded artifacts are trustworthy or
that the runtime image matches the declared commit. Keep build links and image
provenance with any published report.
