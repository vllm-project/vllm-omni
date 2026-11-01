# Tests and Documentation Checklist

Use this reference to connect changed behavior to coverage, CI selection,
examples, documentation, and reproducible PR evidence.

Official docs: [test writing guide](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/ci/test_writing_guide/),
[test execution guide](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/ci/test_execution_guide/),
and [documentation guide](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/DOCS_GUIDE/).

## Coverage packet

For each high-risk change, record a compact internal packet:

```text
change and failure risk -> existing unit/E2E coverage -> uncovered boundary
  -> smallest test and stable assertion that closes the gap
```

Prefer a regression test that fails on the frozen base for bug fixes. Check
numeric tolerances, deterministic inputs, realistic mocks, hardware/domain
markers, and Buildkite/CI wiring using the documentation present on the reviewed
branch. A skipped test is not a pass.

## Documentation sync

Require user-facing updates when the diff changes a model, feature, CLI/API,
config key, default, compatibility behavior, or supported platform. Verify:

- the actual docs navigation and support tables used by the branch;
- exact identifiers, defaults, constraints, known limitations, and migration;
- a minimal runnable example for newly exposed behavior;
- examples and docs agree on commands, parameters, output schema, and supported
  modes;
- links and the bounded live contract build or resolve correctly.

Do not request docs for internal behavior with no user-visible contract.

## PR evidence

For contributor-run tests or benchmarks, require enough provenance to evaluate
the claim: hardware, relevant software versions, exact commands, inputs/model,
result, and approximate runtime. Performance or accuracy evidence must compare
base and head under equivalent conditions; use
[perf-verification.md](perf-verification.md).

Turn a missing test or document into a finding only when it protects or explains
changed behavior. Name the smallest useful addition rather than requesting a
broad coverage expansion.
