# Test Quality Evaluation

Read this reference when tests change, a risky production path has no test, or
a test-only PR may not prove the intended behavior.

Official docs: [test system overview](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/ci/test_system_overview/)
and [test writing guide](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/ci/test_writing_guide/).

## Static proof check

For each changed semantic path, ask whether the test would fail if that behavior
were reverted or broken. Check:

- assertions pin the contract, not only non-null output, logs, or “no crash”;
- the production dispatcher, registry, connector, or scheduler is reached
  instead of the changed behavior being mocked away;
- fakes preserve relevant types, MRO, shapes, devices, async behavior, and
  lifecycle transitions;
- seeds, ordering, timing, synchronization, external services, and numeric
  tolerances are deterministic or explicitly controlled;
- normal, invalid, boundary, feature-off, failure/cancellation, and regression
  paths are covered where the diff changes them;
- run-level and domain markers place hardware/model tests in the intended CI
  lane without silently skipping the contract.

Map source symbols to tests with bounded `rg` searches; do not assume the test
tree mirrors production paths.

## Runtime check

Run the narrowest affected tests supported by the environment. Record skipped
hardware/model cases as gaps. Classify failures as code, test, infrastructure,
or flaky before turning them into findings; one passing rerun does not erase a
flakiness signal.

Keep grades or matrices internal. Report only concrete code bugs or the one or
two test defects that leave changed behavior materially unprotected. Use
[tests-docs-checklist.md](tests-docs-checklist.md) for CI and documentation
coverage.
