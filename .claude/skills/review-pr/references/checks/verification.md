# Active Verification

Read this reference when a runnable affected path and suitable local or remote
hardware are available. Never claim device evidence from static analysis or a
simulated backend.

Official docs: [installation](https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/installation/)
and [test execution guide](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/ci/test_execution_guide/).

## Choose the narrowest level

| Available environment | Verification |
| --- | --- |
| Matching accelerator, model/assets, and server | Targeted unit/integration test plus one representative E2E request. |
| Matching accelerator without serving setup | Targeted model/runtime test and output or metric inspection. |
| CPU/static environment only | Import/version preflight, focused CPU tests, config/registry resolution, and static contract checks. |
| Missing dependencies or inaccessible hardware | Record the exact gap and evaluate contributor/CI evidence. |

## Execute safely

1. Bind the run to the frozen head SHA and record Python, platform, dependency,
   model, device, and relevant environment fingerprints.
2. Run the smallest affected unit test before an E2E workload.
3. Exercise the changed public or production-dispatch path with bounded inputs.
4. Compare actual output, metrics, cleanup, and exit behavior with the PR claim.
5. For base/head claims, use the equivalent A/B protocol in
   [perf-verification.md](perf-verification.md).

Avoid disturbing shared servers, ports, models, or accelerators without user
authorization. Do not install large dependencies, download gated assets, or
start remote workloads merely because hardware might exist.

Report the command, environment, result, and any skipped path. Classify failures
as change-induced, pre-existing, infrastructure, environment, or flaky before
making a finding. A hardware gap lowers confidence; it is not automatic proof
of a defect.
