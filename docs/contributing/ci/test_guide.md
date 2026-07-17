# Test Guide
## Setting Up the Test Environment
### Creating a Container
vLLM-Omni provides an official Docker image for deployment. These images are built upon vLLM Docker images and are available on [Docker Hub](https://hub.docker.com/r/vllm/vllm-omni/tags). The version of vLLM-Omni indicates which vLLM release it is based on.
For a local test environment, you can follow the steps below to create a container:
## Installing Dependencies
### vLLM & vLLM-Omni
vLLM-Omni is built based on vLLM. You can follow [install guide](../../getting_started/installation/README.md) to build your local environment.

### Test Case Dependencies
When running test cases, you may need to install the following dependencies:

```bash
uv pip install ".[dev]"
apt-get install -y espeak-ng jq
```

## Running Tests
Our test scripts use the pytest framework. First, please use `git clone https://github.com/vllm-project/vllm-omni.git` to download the vllm-omni source code. Then, in the root directory of vllm-omni, you can run the following commands in your local test environment to execute the corresponding test cases.

### CI job runners (L2–L4): logs, timing, and timeouts

[`tools/run_ready_jobs.sh`](https://github.com/vllm-project/vllm-omni/blob/main/tools/run_ready_jobs.sh), [`tools/run_merge_jobs.sh`](https://github.com/vllm-project/vllm-omni/blob/main/tools/run_merge_jobs.sh), and [`tools/nightly/run_nightly_jobs.sh`](https://github.com/vllm-project/vllm-omni/blob/main/tools/nightly/run_nightly_jobs.sh) share the same run-time behavior (via [`tools/run_jobs_common.sh`](https://github.com/vllm-project/vllm-omni/blob/main/tools/run_jobs_common.sh)):

**Log layout** (default under `logs/ready_jobs`, `logs/merge_jobs`, or `logs/nightly_jobs`; override with `--log-dir`):

| Path | Description |
|------|-------------|
| `jobs/*.sh` | Generated per-step bash wrappers |
| `jobs/.job_timeouts` | Job key → `timeout_in_minutes` from Buildkite YAML (when set) |
| `<job_key>.log` | Tee output for each job |
| `timing_summary.log` | Per-job duration and total wall time after the run completes |

**Timing:** while each job runs, stderr prints `finished in …` or `failed after …`. When all jobs finish, a summary is printed and written to `timing_summary.log`, for example:

```
=== Job timing summary ===
  TTS_VoxCPM2_Test  25m 40s  OK
  Omni_Qwen3-Omni_Test  8m 05s  TIMED OUT
Total wall time: 33m 45s (2 jobs)
Failed jobs: 1/2
```

**Timeouts:** if a Buildkite step defines `timeout_in_minutes`, the runner wraps that job with `timeout ${N}m` (aligned with CI). Steps without `timeout_in_minutes` rely on any inline `timeout …` already present in the extracted pytest command. On timeout the process exits with code `124` and the summary marks the job as `TIMED OUT`.

=== "L1 level"

    ```bash
    cd tests
    pytest -s -v -m "core_model and cpu"
    ```
    The latest test command is available in the "Simple Unit Test" step of this [pipeline](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/test-ready.yml).

=== "L2 level"

    **Recommended:** run CI-aligned jobs from the repo root via [`tools/run_ready_jobs.sh`](https://github.com/vllm-project/vllm-omni/blob/main/tools/run_ready_jobs.sh). The script reads [`.buildkite/test-ready.yml`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/test-ready.yml), generates per-step bash wrappers, runs pytest, and tees logs under `logs/ready_jobs/` (see [CI job runners](#ci-job-runners-l2l4-logs-timing-and-timeouts) for timing and timeout behavior). Requires `bash`, `python3`, and PyYAML (`pip install pyyaml`).

    ```bash
    # All L2 and L1 jobs (default: every pytest step in test-ready.yml)
    bash tools/run_ready_jobs.sh

    # Skip L1-style "Simple Test" steps in test-ready.yml (group or "Simple ·" labels)
    bash tools/run_ready_jobs.sh --skip-simple
    bash tools/run_ready_jobs.sh --skip-simple --model-type diffusion

    # Preview extracted commands without running
    bash tools/run_ready_jobs.sh --dry-run

    # Filter by model area (OR semantics): omni | tts | diffusion | all
    bash tools/run_ready_jobs.sh --model-type omni
    bash tools/run_ready_jobs.sh --model-type tts,diffusion

    # Match a Buildkite label substring (e.g. one model suite)
    bash tools/run_ready_jobs.sh --model-type tts --label-substr "VoxCPM2"

    ```

    **Ad hoc:** run a single test file or marker expression manually:

    ```bash
    cd tests
    pytest -s -v test_xxxx.py --run-level=core_model
    pytest -s -v -m "core_model and distributed_cuda and L4" --run-level=core_model
    ```

=== "L3 level"

    **Recommended:** run CI-aligned jobs from the repo root via [`tools/run_merge_jobs.sh`](https://github.com/vllm-project/vllm-omni/blob/main/tools/run_merge_jobs.sh). The script reads [`.buildkite/test-merge.yml`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/test-merge.yml), generates per-step bash wrappers, runs pytest, and tees logs under `logs/merge_jobs/` (see [CI job runners](#ci-job-runners-l2l4-logs-timing-and-timeouts)). Requires `bash`, `python3`, and PyYAML (`pip install pyyaml`).

    ```bash
    # All L3 and L1 jobs (default: every pytest step in test-merge.yml)
    bash tools/run_merge_jobs.sh

    # Skip L1-style "Simple Test" steps in test-merge.yml (group or "Simple ·" labels)
    bash tools/run_merge_jobs.sh --skip-simple
    bash tools/run_merge_jobs.sh --skip-simple --model-type omni

    # Preview extracted commands without running
    bash tools/run_merge_jobs.sh --dry-run

    # Filter by model area (OR semantics): omni | tts | diffusion | all
    bash tools/run_merge_jobs.sh --model-type diffusion
    bash tools/run_merge_jobs.sh --model-type omni,tts

    # Match a Buildkite label substring
    bash tools/run_merge_jobs.sh --model-type diffusion --label-substr "Wan22"

    ```

    **Ad hoc:** run a single test file or marker expression manually:

    ```bash
    pytest -s -v test_xxxx.py --run-level=advanced_model
    pytest -s -v -m "advanced_model and distributed_cuda and L4" --run-level=advanced_model
    ```


=== "L4 level"

    **Recommended:** run CI-aligned nightly jobs from the repo root via [`tools/nightly/run_nightly_jobs.sh`](https://github.com/vllm-project/vllm-omni/blob/main/tools/nightly/run_nightly_jobs.sh). The script reads [`.buildkite/test-nightly.yml`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/test-nightly.yml), generates per-step bash wrappers, runs pytest (perf jobs first, then aggregates perf JSON into Excel), and tees logs under `logs/nightly_jobs/` (see [CI job runners](#ci-job-runners-l2l4-logs-timing-and-timeouts); `generate_nightly_perf_excel` is included in the timing summary). Requires `bash`, `python3`, and PyYAML (`pip install pyyaml`).

    ```bash
    # All nightly jobs (default: test-type all, model-type all)
    bash tools/nightly/run_nightly_jobs.sh

    # Preview extracted commands without running
    bash tools/nightly/run_nightly_jobs.sh --dry-run

    # Test kind (--test-type, OR semantics): perf | acc | function | stability | local | all
    bash tools/nightly/run_nightly_jobs.sh --test-type function
    bash tools/nightly/run_nightly_jobs.sh --test-type perf,acc

    # Model area (--model-type, OR semantics): omni | tts | diffusion | all
    bash tools/nightly/run_nightly_jobs.sh --test-type function --model-type omni
    bash tools/nightly/run_nightly_jobs.sh --test-type perf --model-type diffusion

    # Stability scripts (tests/dfx/stability/scripts/) and local marker tests
    bash tools/nightly/run_nightly_jobs.sh --test-type stability --model-type omni
    bash tools/nightly/run_nightly_jobs.sh --test-type local --model-type tts
    ```

    Perf steps (label contains ``Perf Test``) run before other jobs; Excel output is written under ``logs/``. Diffusion nightly steps may also appear in [test-nightly-diffusion.yml](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/test-nightly-diffusion.yml) on CI.

    **Ad hoc:** run a single test file or marker expression manually:

    ```bash
    cd tests
    pytest -s -v test_xxxx.py --run-level=full_model
    pytest -s -v -m "full_model and distributed_cuda and L4" --run-level=full_model
    pytest -s -v -m "full_model and (omni or tts) and H100" --run-level=full_model
    ```
    If you only want to run specific test cases on a particular platform, you can use:
    ```bash
    pytest -s -v -m "full_model and distributed_cuda and L4"  --run-level=full_model
    ```
    Note: ``run_benchmark.py`` and ``run_diffusion_benchmark.py`` accept an optional ``--test-config-file``. If omitted, each loads every ``*.json`` under ``tests/dfx/perf/tests/`` (omni/tts vs diffusion split by ``is_diffusion_perf_config``) and pytest ``-m`` filters by each case's JSON ``mark``:
    ```bash
    pytest -sv tests/dfx/perf/scripts/run_benchmark.py -m "full_model and tts and H100"
    pytest -sv tests/dfx/perf/scripts/run_diffusion_benchmark.py -m "full_model and diffusion and H100"
    pytest -sv tests/dfx/perf/scripts/run_benchmark.py --test-config-file tests/dfx/perf/tests/test_tts.json
    pytest -sv tests/dfx/perf/scripts/run_diffusion_benchmark.py --test-config-file tests/dfx/perf/tests/test_cosmos3_vllm_omni.json
    ```
    Nightly **Perf Test** jobs in ``test-nightly.yml`` use ``--test-config-file`` only (no ``-m``). E2e L4 function tests still use ``full_model`` + ``--run-level full_model`` (see [test-nightly.yml](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/test-nightly.yml)). Example:

=== "L5 level"

    L5 includes stability and reliability testing. Typical commands:

    ```bash
    cd tests

    # Stability: Qwen3-Omni
    pytest -s -v dfx/stability/scripts/test_stability_qwen3_omni.py -m slow

    # Stability: Wan2.2 (v1/videos diffusion benchmark loop)
    pytest -s -v dfx/stability/scripts/test_stability_wan22.py -m slow

    # Reliability: Qwen3-Omni (H100 × 2)
    pytest -s -v dfx/reliability/test_reliability_qwen3_omni.py -m slow

    # Reliability: Wan2.2 (H100)
    pytest -s -v dfx/reliability/test_reliability_wan22.py -m slow

    # Reliability: HunyuanImage DiT (H100 × 4)
    pytest -s -v dfx/reliability/test_reliability_hunyuan_image.py -m slow

    # Reliability: VoxCPM2 (L4)
    pytest -s -v dfx/reliability/test_reliability_voxcpm2.py -m slow

    ```

    The latest L5 CI jobs (reliability + invalid-parameter weekly steps) are in [test-weekly.yml](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/test-weekly.yml).

You can find more information about markers in the documentation: [marker doc](./tests_markers.md)

## Adding New Test Cases
Please refer to the [L5 Layering Specification document](./CI_5levels.md).
