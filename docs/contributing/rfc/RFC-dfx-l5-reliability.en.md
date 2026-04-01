# RFC: L5(b) Reliability Test Framework under tests/dfx

> **Status**: Draft (locally archived; can be synced later as a GitHub Issue discussion)  
> **Template**: Aligned with [.cursor/skills/RFC-generate/SKILL.md](../../../../.cursor/skills/RFC-generate/SKILL.md); density can reference [Issue #1313](https://github.com/vllm-project/vllm-omni/issues/1313).

### Motivation

In [CI_5levels.md](../ci/CI_5levels.md), L5 is split into long-term **Stability** and **Reliability (fault/recovery)**. This RFC fills in reliability scenario test cases. Reliability scenarios should include at least three categories: **abnormal input**, **OOM (boundary)**, and **process-level faults (process kill)**.

### Proposed Change

- Add a new directory `tests/dfx/reliability/`, and use `tests/scenarios.json` (or `test.json`, choose one naming style consistent with stability and fix it) to describe **server_params + scenario_type + fault + expect**.
- Reuse `load_configs`, `create_unique_server_params`, and `omni_server` (indirect) from `[tests/dfx/conftest.py](../../../tests/dfx/conftest.py)`. On the test side, explicitly use `openai_client` (consistent with `tests/conftest.py`), and do not invent a `client` fixture.
- Define structured `RecoveryResult` (fields in Detailed Design), and keep docs and code aligned; when optional dependencies are missing, use `pytest.skip` or fail explicitly, and do not return 0 to mask failures.
- **OOM scenario**: first focus on boundary OOM (`oom_boundary`) with the flow "oversized request -> expected error + process alive + subsequent success".
- **Process kill**: as an independent `scenario.type` (see table below), enabled by default only in exclusive environments plus explicit switches.
- Update Chapter 4 in [CI_5levels.md](../ci/CI_5levels.md): change L5(b) path and run examples to `tests/dfx/reliability/`.

#### Adopted Reliability Scenarios (Three Categories)

The **product scope** aligned with this RFC includes the following three categories, all distinguished in `scenarios.json` via `scenario.type` and optional fields.

| Scenario | `scenario.type` (recommended enum) | Validation points |
|------|------------------------------|----------|
| **Abnormal input** | `abnormal_input` | Invalid input / oversized payload / wrong modality; assert `expect.error_expected=true` (optional `error_contains`), `expect.process_alive=true`, `expect.min_post_success>=1`, and record `post_fault_error_count` |
| **OOM** | `oom_boundary` | Construct oversized requests to trigger rejection or allocation failure; assert `expect.error_expected=true`, `expect.process_alive=true`, `expect.min_post_success>=1`, and `recovery_time_sec` is within threshold (optional `expect.max_recovery_time_sec`) |
| **Process kill** | `process_kill` | Send `SIGTERM/SIGKILL` only to safe-to-kill targets (e.g., worker); assert `expect.recovered=true`, `expect.health_check_ok=true`, `expect.min_post_success>=1`, and gate via `requires` / env vars |


### Design

**Overview**: the three dfx layers (`perf` / `stability` / `reliability`) share the same server parameter generation flow; once server is ready, reliability runs per scenario as **fault_inject -> health/post-fault probe -> fill RecoveryResult**.

```text
tests/dfx/reliability/
  README.md
  conftest.py              # optional: report directory and hooks aligned with stability
  stage_configs/           # same model yaml as stability, or docs describing reused paths
  tests/
    scenarios.json
  scripts/
    fault_inject.py        # pure functions, unit-testable
    test_reliability.py    # @pytest.mark.slow; parametrize + omni_server + openai_client
```

**Module responsibilities**:

- `scenarios.json`: single source of scenarios.
- `fault_inject.py`: dispatch by `scenario_type` (**`abnormal_input`**, **`oom_boundary`**, **`process_kill`**); client timeout can be a subtype of `abnormal_input` or an independent type; `process_kill` should be in a dedicated branch and gated.
- `test_reliability.py`: assemble fixtures, assert RecoveryResult, and write logs/optional JSON artifacts.

#### Detailed Design

**Config entry (illustrative)**: each entry contains `test_name`, `server_params` (same fields as stability: `model`, `stage_config_name`, `update`/`delete`), `scenario` (`type`, `fault_params`, `expect`), optional `requires` (e.g., `baremetal`), and `oom_tier` (`none` | `boundary`). In `process_kill` scenarios, `fault_params` may specify **target role** (e.g., `worker`) and **signal** (e.g., `SIGTERM`); exact behavior depends on `OmniServer` process model investigation results (open question).

**RecoveryResult (TypedDict or dataclass recommended)**:

- `recovered: bool`
- `recovery_time_sec: float | None`
- `health_check_ok: bool` (if no independent health API exists, use "minimal valid request succeeds" as equivalent and document it)
- `post_fault_success_count: int` / `post_fault_error_count: int`
- `notes: str`

**Concurrency and lifecycle**: same as `[test_benchmark_stability.py](../../../tests/dfx/stability/scripts/test_benchmark_stability.py)` using module-scope `omni_server`; when locking, ensure **multiple scenarios in one process run serially**; fault steps must not run in parallel with sub-benchmarks (serial execution in a single test file is sufficient for v1).

**Errors and timeout**: client timeout and connection errors must be explicitly recorded in `notes`; do not swallow exceptions with bare `except: return False`.

### Use Case

**How to extend**: add one more `test_name` in `scenarios.json`; add yaml under `stage_configs/` when needed.

**Run commands**:

```bash
pytest --collect-only tests/dfx/reliability
pytest -s -v tests/dfx/reliability/scripts/test_reliability.py -m slow
```

**Artifacts**: console logs; optional `reliability_result_<test_name>.json` (aligned with perf/stability result file conventions; open question: must naming be in the same directory/style as stability).

**Single scenario example (snippet)**:

```json
{
  "test_name": "qwen3_omni_abnormal_then_ok",
  "server_params": {
    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "stage_config_name": "qwen3_omni.yaml"
  },
  "scenario": {
    "type": "abnormal_input",
    "fault_params": { "max_tokens": 999999 },
    "expect": { "process_alive": true, "min_post_success": 1 }
  }
}
```

### CC List



### Any Other Things

- **Risk**: `process_kill` on shared GPUs may impact other jobs on the same host -> disabled by default and enabled only in exclusive environments.
- **Open question**: whether `OmniServer` exposes an independent health endpoint; if not, the equivalent criterion for `health_check_ok` should be fixed in the first PR.



## Rollout / Migration

1. Merge `tests/dfx/reliability/` + minimal scenarios; **run locally for now**.
2. Update CI_5levels.md; if old path `tests/e2e/reliability` no longer exists, remove references or mark it Deprecated.

## Testing & CI

- **Integration**: run 1-2 default available scenarios on GPU; `process_kill` runs locally for now.
