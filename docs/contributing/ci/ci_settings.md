# CI Settings

This document describes **where** Buildkite YAML lives in the repo, **how each platform organizes CI**, and **how to add a new job**. It does not document agent queues, GPU types, or container plugin details—those belong in infra / preset files (for example `.buildkite/common/ci_mirror_hardwares.yml`).

For CI levels (L1–L5) and triggers, see [Test System Overview](./test_system_overview.md). For test authoring, see [Test Writing Guide](./test_writing_guide.md). For running tests locally or replaying CI jobs, see [Test Execution Guide](./test_execution_guide.md).

## Directory layout

Canonical layout (prefer these paths for new changes):

```text
.buildkite/
├── common/                          # Shared across platforms
│   ├── scripts/
│   │   ├── ci_conditions.py         # Shared gates, YAML whitelists, uploader constants
│   │   ├── skip_ci.py               # skip-ci decision (docs / skip-mark / CI YAML paths)
│   │   ├── upload_pipeline.py       # Bootstrap + test-pipeline uploader (CUDA/NPU)
│   │   └── resolve_skip_ci.sh       # Shell helpers for AMD/Intel bootstrap
│   ├── ci_mirror_hardwares.yml      # CUDA/NPU hardware presets (referenced by name only)
│   └── ci_source_file_dependencies.yml  # L2–L5 path-prefix presets (referenced by name only)
├── cuda/                            # Primary NVIDIA CUDA CI
│   ├── pipeline.yml                 # Bootstrap entry (hook upload)
│   ├── bootstrap-upload-steps.yml   # Bootstrap child steps (upload_pipeline --upload)
│   ├── test-ready.yml               # L2
│   ├── test-merge.yml               # L3
│   ├── test-nightly.yml             # L4
│   ├── test-weekly.yml              # L5
│   └── rebase-pipeline.yml
├── npu/
│   ├── pipeline-npu.yml             # Bootstrap entry (hook upload)
│   ├── bootstrap-upload-steps.yml   # Bootstrap child steps (upload_pipeline --upload)
│   ├── pipeline-npu-a3.yml          # A3 variant (when used)
│   ├── test-npu-ready.yml           # L2
│   ├── test-npu-nightly.yml         # L4
│   └── scripts/
├── amd/
│   ├── test-amd-ready.yml           # L2 job definitions (template input)
│   ├── test-amd-merge.yml           # L3 job definitions
│   ├── test-amd-nightly.yml         # Nightly full-model jobs (experimental)
│   ├── test-template-amd-omni.j2    # Renders final pipeline.yaml
│   └── scripts/
│       ├── bootstrap-amd-omni.sh    # Entry: skip-ci → Jinja → upload
│       └── run-amd-test.sh          # Wraps pytest inside ROCm docker
├── intel/
│   ├── pipeline-intel.yml           # Static Intel XPU pipeline
│   └── scripts/
│       ├── bootstrap-intel-omni.sh
│       └── run-xpu-test.sh
└── release/
    ├── release-pipeline.yml
    └── scripts/
```

### Placement rules

| Rule | Detail |
| ---- | ------ |
| Platform code under platform dir | New CUDA jobs go in `.buildkite/cuda/`; do not add new top-level `.buildkite/test-*.yml` files. |
| Shared logic in `common/` | Skip-ci and CUDA upload rendering stay in `.buildkite/common/scripts/`. |
| Bootstrap vs test YAML | **Bootstrap** (`pipeline*.yml`) builds images and uploads **child** test pipelines. **Test** YAML (`test-*.yml`) lists pytest steps only. |
| Register CI YAML in `ci_conditions.py` | If you add a new whitelisted test pipeline file, update `L2_YAML_FILES`, `L3_YAML_FILES`, or `L45_YAML_FILES` in `.buildkite/common/scripts/ci_conditions.py` so skip-ci paths stay correct. |

## Platform comparison

| Platform | Bootstrap entry | Test job files | Upload mechanism | Job hardware in YAML |
| -------- | ---------------- | -------------- | ---------------- | -------------------- |
| **CUDA** | `cuda/pipeline.yml` | `test-ready.yml`, `test-merge.yml`, `test-nightly.yml`, `test-weekly.yml` | `upload_pipeline.py --upload` (expands uploader-only keys) | omit `mirror_hardwares` (from `-m`) / preset string + `MIRROR_HW` |
| **NPU** | `npu/pipeline-npu.yml` | `test-npu-ready.yml`, `test-npu-nightly.yml` | `upload_pipeline.py --upload` | `mirror_hardwares: a2b3_npu_1` / `a2b3_npu_4` / `a3_npu_2` |
| **AMD** | `amd/scripts/bootstrap-amd-omni.sh` | `test-amd-ready.yml`, `test-amd-merge.yml`, `test-amd-nightly.yml`, `test-amd-weekly.yml` | Jinja (`test-template-amd-omni.j2`) → `pipeline upload` | `agent_pool` + `mirror_hardwares: [amdproduction]` (array, template filter) |
| **Intel** | `intel/scripts/bootstrap-intel-omni.sh` | `intel/pipeline-intel.yml` (steps inline) | Direct `pipeline upload` | Inline `agents.queue` on each step |

## Platform configuration style

=== "CUDA"

    **Bootstrap:** [`cuda/pipeline.yml`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/cuda/pipeline.yml) (hook upload) + [`cuda/bootstrap-upload-steps.yml`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/cuda/bootstrap-upload-steps.yml) (child steps). Document 1 runs [`upload_pipeline.py --upload .buildkite/cuda/bootstrap-upload-steps.yml`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/common/scripts/upload_pipeline.py), which injects `if` by step `key` from skip-ci and uploads image build plus L2–L5 child pipeline upload steps (see [Diff-aware CI — Bootstrap skip](#bootstrap-skip)).

    **Test YAML:** `cuda/test-ready.yml` (L2), `test-merge.yml` (L3), `test-nightly.yml` (L4), `test-weekly.yml` (L5). Each file starts with shared `env:` then `steps:`.

    | CI level | File | Typical trigger |
    | -------- | ---- | ----------------- |
    | L2 | `cuda/test-ready.yml` | `ready` label |
    | L3 | `cuda/test-merge.yml` | `merge-test` label / direct release-line push (`main`, or non-PR `release/vX.Y.Z`) |
    | L4 | `cuda/test-nightly.yml` | `nightly-test` label or `NIGHTLY=1` |
    | L5 | `cuda/test-weekly.yml` | `weekly-test` label, `WEEKLY=1`, or `NON_CRITICAL=1` (see step `if`s below) |

    A pull request whose source branch is `release/vX.Y.Z` is not a release-line push: `build.branch` is the PR head, so that build still needs a label. Direct pushes to `release/vX.Y.Z` keep the release-line gates. `main` is still matched by branch name alone.

    On scheduled release-line builds (`main` or non-PR `release/vX.Y.Z`), `test-weekly.yml` is uploaded when **`WEEKLY=1` or `NON_CRITICAL=1`**. Inside that file:

    - **Reliability** / **Perf Test** → `WEEKLY=1` (or PR label `weekly-test`). Pipeline upload for scheduled env vars still requires a release line: `main` or a branch named `release/vX.Y.Z` (for example `release/v0.30.0`).
    - **E2E Tests** group (slow Omni/TTS/Diffusion sweeps) → `NON_CRITICAL=1`

    **Simple · CPU Coverage Test** is in `test-nightly.yml` and runs when `NIGHTLY=1`.

    **`--e2e`:** when uploading ready/merge on a release line (`main` or `release/vX.Y.Z`) with **`WEEKLY=1`**, bootstrap passes `--e2e` so only the **E2E Test** group is kept (same flag `run_cov_split.sh` uses to enable per-model coverage).

    **Upload:** `upload_pipeline.py --upload` expands uploader-only keys before Buildkite upload.

    **Hardware in YAML:** omit `mirror_hardwares` to compose `{chip}_{n}` from pytest `-m` (SKU + `cards_n`). Several positive `cards_*` take the **max** n (`cards_2 and cards_3` → `{chip}_3`). `not cards_1` with no positive `cards_*` uses that chip's **highest existing** preset in `ci_mirror_hardwares.yml` (L4 → `l4_4`, H100 → `h100_4`, B200 → `b200_4`). Set a **preset string** to force a pool (marks are ignored). Do **not** set `agents` / `plugins` on the same step as `mirror_hardwares`. `MIRROR_HW=b200` remaps inferred jobs whose `-m` includes `B200`, and omits CUDA `h100_*` / `l4_*` strings. Jobs whose `-m` does not match (unset: no H100/L4; set: no B200) are **skipped**.

    ```yaml
    # inferred: H100+B200 + cards_2 → h100_2, or b200_2 when MIRROR_HW=b200
    commands:
      - pytest -sv tests/e2e -m "full_model and H100 and B200 and omni and cards_2"
    # no mirror_hardwares

    # inferred: not cards_1 (no positive cards_*) → that chip's highest pool (l4_4 / h100_4 / b200_4)
    commands:
      - pytest -sv tests/e2e -m "full_model and diffusion and L4 and not cards_1"

    # forced preset (ignores -m SKU/cards for pool selection)
    mirror_hardwares: h100_1
    ```

    Inferred form: unset/`empty` `MIRROR_HW` matches `H100` or `L4` in `-m` (both present → H100); no match skips the step. `MIRROR_HW=b200` must appear in `-m` or the step is skipped. The uploader does **not** rewrite pytest `-m`; B200 collection comes from the YAML expression and from tests that declare `B200` in `hardware_test` / `hardware_marks` (for example `res={"cuda": ["H100", "B200"]}`). `MIRROR_HW` must be empty or `b200` (case-insensitive); unknown values (for example `b20o`) **fail the upload**. A CUDA preset string such as `mirror_hardwares: h100_4` is skipped when `MIRROR_HW=b200`. Unset `MIRROR_HW` keeps string presets. NPU presets ignore `MIRROR_HW`. A `mirror_hardwares` name that is not a key in `ci_mirror_hardwares.yml` **fails the upload**.

    **Path filter in YAML:** `source_file_dependencies: <preset>` (string, or a list of preset names)—preset names in [`common/ci_source_file_dependencies.yml`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/common/ci_source_file_dependencies.yml). Required on **E2E Test** leaf jobs in L2/L3, and on L4/L5 leaf jobs that should be diff-gated on PR labels; see [Step filtering](#step-filtering).

    **Conventions**

    - **`depends_on`:** leaf jobs depend on `upload-ready-pipeline`, `upload-merge-pipeline`, etc.
    - **`group` / `label`:** `:card_index_dividers:` groups; labels like `Diffusion · Qwen Image Test`.
    - **`commands`:** pytest with markers and `--run-level` for the pipeline level. Bound the step with `timeout_in_minutes`. A CUDA ready or merge directory sweep that should split from measured durations sets `automatic_shard: true` and does not use a command-level `timeout`.
    - **`source_file_dependencies`:** required on **E2E Test** leaf jobs in L2/L3, and on L4/L5 leaf jobs that should be diff-gated on PR labels; name a job key from `ci_source_file_dependencies.yml` (see [Step filtering](#step-filtering)). The key must list every script the step runs; `commands` is not scanned.

    **Adding a job**

    1. Pick the level file (`test-ready.yml` / `test-merge.yml` / `test-nightly.yml` / `test-weekly.yml`).
    2. Add a step under the right **group** (usually **E2E Test** for model pytest).
    3. Set `label`, `commands`, `mirror_hardwares`, `depends_on: upload-<level>-pipeline`.
    4. For L2–L5 jobs that should be diff-gated, add a job key in `common/ci_source_file_dependencies.yml` and set `source_file_dependencies: <key>` on the step. YAML anchors hold model and shared runtime dependencies. The job key (`{model}_function` / `_perf` / `_accuracy` / `_reliability` / `_doc` / `_cov`) aliases the relevant anchors and lists the pytest / example / benchmark scripts the step actually runs. The uploader does **not** parse `commands` for path filters—omitting a script from the key means editing that script will not select the job.
    5. Dry-run:

    ```bash
    python3 .buildkite/common/scripts/upload_pipeline.py .buildkite/cuda/test-ready.yml
    ```

=== "NPU"

    **Bootstrap:** [`npu/pipeline-npu.yml`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/npu/pipeline-npu.yml) + [`npu/bootstrap-upload-steps.yml`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/npu/bootstrap-upload-steps.yml)—same split as CUDA; builds A2/B3, A3, A5, and 310P CI images, then uploads child test pipelines.

    **Test YAML:** `npu/test-npu-ready.yml` (L2), `test-npu-nightly.yml` (L4).

    **L2 trigger:** PR label `ready` (CUDA/AMD L2). PR labels `ready` + `npu-test` trigger NPU CI only.

    **Upload:** same as CUDA—`upload_pipeline.py --upload`.

    **Hardware in YAML:** `mirror_hardwares` preset (string), expanded to `agents`, top-level `image`, and `plugins`. NPU presets in `common/ci_mirror_hardwares.yml` are:

    - A2: `a2b3_npu_1`, `a2b3_npu_4`, `a2b3_npu_8`
    - A3: `a3_npu_2`, `a3_npu_4`, `a3_npu_8`, `a3_npu_16`
    - A5: `a5_npu_2`, `a5_npu_4`, `a5_npu_8`, `a5_npu_16`
    - 310P: `310p_npu_1`, `310p_npu_2`, `310p_npu_4`

    **Path filter in YAML:** same `source_file_dependencies` presets as CUDA, on **L4** leaf jobs in `test-npu-nightly.yml`. Filtering applies on the PR `nightly-test` label; a release line (`main` or `release/vX.Y.Z`) + `NIGHTLY=1` keeps every job. If no job-key prefix matches, a change to the pipeline YAML being uploaded, or to any path under `source_filter_fallback`, keeps every job in that upload. See [Step filtering](#step-filtering).

    **Conventions**

    - **`depends_on: upload-ready-pipeline`** (or `upload-nightly-pipeline`) ties jobs to bootstrap upload keys.
    - Do not duplicate `agents` / `image` / `plugins` when using `mirror_hardwares`.

    **Adding a job**

    1. Edit `test-npu-ready.yml` (L2) or `test-npu-nightly.yml` (L4).
    2. Add a step with `mirror_hardwares` (add a new preset in `ci_mirror_hardwares.yml` first if needed).
    3. Set `commands` to your pytest file and markers. For L4, set `source_file_dependencies` to a job key from `ci_source_file_dependencies.yml` that aliases the model anchor and lists that pytest file.
    4. Dry-run:

    ```bash
    python3 .buildkite/common/scripts/upload_pipeline.py .buildkite/npu/test-npu-ready.yml
    ```

    5. Register new pipeline paths in `ci_conditions.py` (`L2_YAML_FILES` or `L45_YAML_FILES`) when applicable.

=== "AMD"

    **Bootstrap:** [`amd/scripts/bootstrap-amd-omni.sh`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/amd/scripts/bootstrap-amd-omni.sh)—skip-ci, diff filtering, Jinja render, then `buildkite-agent pipeline upload`.

    **Test YAML (data):** `amd/test-amd-ready.yml` (L2 / `ready`), `test-amd-merge.yml` (L3 / `merge-test` and an ordinary release line, `main` or `release/vX.Y.Z`), experimental `test-amd-nightly.yml` (`full_model` / `nightly-test` or a release line with `NIGHTLY=1`), and `test-amd-weekly.yml` (`reliability` / `weekly-test` or a release line with `WEEKLY=1`). Multiple PR tier labels combine their suites behind one image build. A later suite's step is dropped only when its complete parsed definition equals a step in an earlier suite; distinct definitions and repeated steps within one suite are preserved. The bootstrap logs how many duplicate step definitions and physical jobs were dropped. PR builds that reach the bootstrap without a tier label retain the legacy ready-suite fallback. `DEBUG_TEST_YAML` accepts `ready`, `merge`, `nightly`, and `weekly` as an explicit override. During nightly burn-in every AMD nightly leaf is non-blocking.

    **Trigger boundary:** AMD label handling has two layers. First, the external `vllm-omni-amd-ci` Buildkite pipeline condition decides whether a GitHub label event creates a build. Only after that build starts does this repository's bootstrap inspect all current PR labels and select the ready, merge, nightly, weekly, or combined suites. Repository-side selection therefore cannot start AMD CI by itself. AMD-only nightly validation uses both `amd-test` and `nightly-test`: the external condition should accept an event for either label only when the complete PR label set contains both. `amd-test` admits the AMD pipeline, while `nightly-test` makes the bootstrap select `NIGHTLY_TESTS:test-amd-nightly.yml`. Existing `ready` / `merge-test` behavior and non-PR release-line (`main` or `release/vX.Y.Z`) / scheduled builds remain separate. For PR builds without a `DEBUG_TEST_YAML` override, an unreadable label set fails the bootstrap instead of silently selecting the wrong tier. Debug overrides skip label lookup and are validated directly; invalid values, including whitespace-only values, fail the bootstrap. Until the paired-label condition is configured, use an authorized manual Buildkite build with `DEBUG_TEST_YAML=nightly` and verify that the bootstrap reports only the nightly suite.

    **Rendering:** [`test-template-amd-omni.j2`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/amd/test-template-amd-omni.j2) wraps data steps with `amd-build` image build and `amd_<agent_pool>` queues. Do **not** hand-edit generated `pipeline.yaml`.

    **Hardware in YAML:** `agent_pool` (for example `mi325_1`) plus `mirror_hardwares: [amdproduction]` (array—Buildkite template filter, not the CUDA/NPU uploader preset mechanism).

    **READY/MERGE CPU timing:** the three **Model Executor** shards and **Other Test** use
    `.buildkite/amd/scripts/run_cpu_suite.py`. It invokes pytest once with the existing
    selectors and shard arguments, adding `--durations=50` and JUnit output. Test
    grades, timeouts, and shard counts are unchanged. READY and MERGE retain identical
    definitions for these jobs so combined builds can still deduplicate them.

    Each invocation uploads `artifacts/amd-cpu-suites/<suite>-<build>-<job>-<shard>-<unique>/`:

    - `invocation.json`: pytest arguments, marker/run level, shard parameters, commit,
      and Buildkite identifiers. The configured image tag is recorded; the resolved
      digest is `null` because it is not exposed to this runner. Use the Buildkite
      group/build metadata to identify READY, MERGE, or a combined tier.
    - `selection.json`: node IDs collected before filtering, selected after filtering
      and sharding, and `not_selected` (their difference). `deselected` records
      pytest notifications only: pytest-shard 0.1.2 removes items silently.
      No extra collection pass is run.
    - `events.jsonl`: elapsed timestamps for collection, test starts, and each
      setup/call/teardown report, including duration and outcome.
    - `pytest.xml`: pytest's JUnit execution results.
    - `cpu-before.json` / `cpu-after.json`: CPU affinity, cgroup v1/v2 limits,
      throttling counters, CPU pressure, load, and thread environment settings.
      Unavailable Linux fields are identified explicitly.
    - `summary.json`: pytest exit code, collection status, node counts, and phase
      totals. `executed` counts nodes with a call report; setup skips/errors are
      separate outcomes. A teardown failure makes the node failed even if its call
      passed. `skipped` includes expected failures (see `wasxfail` in events).

    `runner_wall` spans the measured invocation, including artifact overhead.
    Phase totals describe this pytest process: setup/teardown include fixture costs,
    while collection is wall time around pytest's collection hook. They exclude
    image-build, dependency-wait, and queue time and are not per-fixture profiles.
    Match job IDs to Buildkite timestamps when analyzing end-to-end latency.

    Events are closed after every write so completed records survive process
    termination. SIGTERM/SIGKILL can leave no final summary or JUnit file; treat
    such artifacts as partial, never as a passing run. Upload still depends on
    the agent retaining the workspace. The runner preserves nonzero pytest exit
    codes; artifact errors fail an otherwise successful invocation. This adds
    measurement only; CI overhead and build-time savings require hardware evidence.

    **Data file fields**

    | Field | Purpose |
    | ----- | ------- |
    | `agent_pool` | ROCm pool; template maps to `queue: amd_<pool>`. |
    | `mirror_hardwares` | Which mirror HW runs the step (for example `[amdproduction]`). |
    | `commands` | Passed into `run-amd-test.sh` via `TEST_COMMAND`. |

    **Adding a job**

    1. Edit `test-amd-ready.yml`, `test-amd-merge.yml`, or `test-amd-nightly.yml`.
    2. Copy a neighboring block: `label`, `agent_pool`, `mirror_hardwares`, `commands`, optional `grade`.
    3. Regenerate via bootstrap / Jinja; update `ci_conditions.py` if you add a new YAML path.

=== "Intel"

    **Bootstrap:** [`intel/scripts/bootstrap-intel-omni.sh`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/intel/scripts/bootstrap-intel-omni.sh)—skip-ci, then direct `pipeline upload`.

    **Test YAML:** steps live inline in [`intel/pipeline-intel.yml`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/intel/pipeline-intel.yml); runners under `intel/scripts/` (for example `run-xpu-test.sh`).

    **Upload:** `buildkite-agent pipeline upload` (no `upload_pipeline.py` mirror expansion).

    **Hardware in YAML:** inline `agents.queue` (for example `intel-gpu-omni`) on each step.

    **Adding a job**

    1. Add a step to `pipeline-intel.yml`, or extend an existing runner script.
    2. Match `agents`, `env`, `timeout_in_minutes`, and `command`/`commands` with sibling steps.
    3. Keep `pipeline-intel.yml` listed in `L2_YAML_FILES` in `ci_conditions.py`.

## AMD CPU duration partitions and qualification

READY/MERGE configure five duration-weighted Model Executor shards using
`.buildkite/amd/cpu-timings/model-executor-v1.json`. Selectors, exclusions, blocking
grades and timeouts are unchanged. The checked-in profile contains all 9,145
selected nodes from the successful six-way target READY trial in
[build #15651](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/15651), merge
commit `79b424021514b082757f60fbacf2fe010cdadd7c`. The trial used the same
four-thread limit as production, selected every live node exactly once, and had
no test, collection, or artifact failures. Repacking those measured phase totals
into five module-preserving shards estimates 152.91 seconds per shard **before**
collection and job overhead. This is a planning estimate until the five-way
candidate completes target CI; it is not a measured p95.
The Model Executor jobs also set OMP/MKL/OpenBLAS/NumExpr to four threads, matching
their CPU quota. In the source CI image on an AMD EPYC 9965 host, with a four-CPU
quota and 16 GiB memory, the four reference cases below all passed at 1/2/4 threads.
Three measured fresh-process runs after one warmup averaged 31.32s, 22.73s and
19.38s respectively (ranges 30.31–32.67s, 22.30–23.15s and 19.30–19.44s).
This is a thread-count comparison, not a measured full-build speedup.
Software: PyTorch `2.13.0+rocm10.0.0`, vLLM `0.33.0+rocm100`, Python 3.12.15,
pytest 9.1.1; image digest
`sha256:921f117d536431ea91bd2fbf08a74821af5738fb909aa48d557dd0bc5a564e14`.

### Diffusion CPU: five-shard rebalance and bounded threads

The five regular Diffusion shards use `diffusion-v1.json`, with 11,609 node
timings from all five successful instrumented artifacts in
[build #15658](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/15658), merge
commit `fb25dd4336fb2098e5a4de1d3fc90cf1b72ad4f4`. The first bounded-thread
trial selected every live node exactly once, with no failures, incomplete tests,
collection errors or artifact errors. Its legacy profile had 204 unknown and
116 stale nodes; the regular job walls ranged from 4m42s to 7m42s. Repacking
the measured setup + call + teardown totals estimates 255.19 seconds per shard
before collection/overhead, with zero unknown or stale nodes. Exact refreshed-
profile CI remains the qualification gate. Keep five shards; trial six only if
future measured p95 rises above 20 minutes and queue capacity allows it.

Regular and Slow CPU jobs bound OMP/MKL/OpenBLAS/NumExpr to two threads per
process, allowing two gloo ranks to fit the observed four-CPU quota. Four unchanged
reference cases (Wan fused/exact decode, long LingBot metadata rollout, and real
causal-VAE continuation) passed at 1/2/4 threads. In the same image and host
configuration documented above, three measured runs after a warmup averaged
46.30s, 46.34s and 47.14s respectively (ranges 44.58–48.50s, 45.64–46.99s and
46.03–48.20s). These results do not establish a meaningful advantage of two over
one thread; two is a quota-bounded trial setting, including for spawned ranks.

No network shape, rollout length, tolerance or assertion is reduced. Slow CPU
retains exactly its six reference cases, blocking grade, 40-minute timeout and
exit-134 retry rule. Regular shards retain their marker expression, both ignored
files, six deselections, 60-minute timeout and retry rule. Whole modules retain
fixture locality. Both jobs now emit phase, selection, CPU pressure and cgroup
memory current/peak/limit/events artifacts under `artifacts/amd-cpu-suites/`.
JUnit lives in each fresh run directory as `pytest.xml`.

The π0 remapping fixture constructs the real full model hierarchy on `meta`,
then allocates only the original vision/embedding target parameters on CPU,
preserving tied aliases. The three remapping tests retain their original tensor
shapes, payloads and assertions. They do not execute the unused backbone; the
separate checkpoint/inference parity coverage is unchanged. Their shared mutable
targets remain order-independent because each test loads a distinct constant.
This avoids initializing unrelated full-network weights without shrinking the
property under test. Reproduce the fixture check with the same dependencies as
the reference benchmark:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  python -m pytest -q tests/diffusion/models/pi0/test_pi0_units.py \
  -k 'remap' --durations=10
```

In paired fresh containers on the reference host/image above, with four CPUs,
two threads and a 32 GiB memory limit, one warmup plus three measured runs per
variant passed all three original tests. Mean process wall time fell from
61.62s (61.11–62.18s) to 31.62s (31.52–31.83s). Peak pytest process RSS was
26.33 GiB before and 15.24 GiB after. This measures the fixture change only;
it does not establish the full shard p95 or a target-agent memory limit.

Reproduce the initial profile from the original artifact filenames:

```bash
python .buildkite/amd/scripts/cpu_shard_plan.py --suite diffusion \
  --junit downloaded/pytest-*-of-5.xml \
  --source-build-url https://buildkite.com/vllm/vllm-omni-amd-ci/builds/15582 \
  --source-commit 438155621484248827a30256e679cb4c84f544fb \
  --output diffusion-v2.json
```

The importer rejects missing/duplicate shards, failures, unresolved modules and
overlapping node IDs. Zero-duration collection skips are not executable node IDs.
Only use legacy JUnit produced with pytest's total-duration setting. For new
profiles, use the instrumented artifact builder below and provide the five
regular job directories separately from Slow CPU.

```bash
python tests/helpers/benchmark_cpu_threads.py --output /tmp/diffusion-thread-trial -- \
  -q 'tests/diffusion/distributed/test_wan_decoder_fast_path.py::test_sharded_fast_path_on_every_rank[True-fused]' \
  'tests/diffusion/ar_diffusion/test_kv_orchestration.py::test_rollout_keeps_metadata_bounded[2-False-lingbot-two-minutes]' \
  'tests/diffusion/distributed/test_wan_decoder_fast_path.py::test_sharded_fast_path_on_every_rank[True-exact]' \
  'tests/diffusion/models/lingbot_world/test_pipeline_lingbot_world.py::test_condition_continues_real_causal_vae_beyond_initial_horizon[realtime]'
```

The refreshed profile must pass exact-head target CI with the bounded thread
setting. Compare the exact executed node union and skip outcomes, and retain the
original numerical checks. Measure concurrent five-shard plus Slow CPU memory
peaks, CPU throttling, queue latency, summed agent-minutes and paired READY/MERGE
wall time. The planning target remains p95 below 18 minutes for every regular
shard and Slow CPU. Revert the pipeline thread export and timing-profile option
to roll back; do not move Slow CPU cases or weaken their assertions to meet the
target.

### Prepare a shared profile

Download the artifact directories from successful instrumented CPU jobs.
Supply exactly one successful attempt of every shard from each source build:

```bash
python .buildkite/amd/scripts/cpu_shard_plan.py \
  downloaded/build-1/shard-0 downloaded/build-1/shard-1 downloaded/build-1/shard-2 \
  --suite model-executor --output model-executor-timings-v2.json
```

The tool consumes `invocation.json`, `selection.json`, `summary.json`, and
`events.jsonl`. Failed/incomplete runs, missing or duplicate shards, overlapping
selected nodes, and incomplete phase reports are rejected. It sums setup, call,
and teardown per node, then takes the median across supplied builds. The output
records `schema_version: 1`, the suite, `node_seconds`, and source build URLs/commits.
The same builder accepts `--suite other` and `--suite diffusion` with their
existing marker and directory contracts. Do not combine different suites in one profile.
Use a new output path; existing profiles are never overwritten.

### Run a candidate shard

Run from the repository root in the AMD CI image with the normal test dependencies.
Use the **same profile bytes and source revision** on all five workers. The runner
reads the count/index from Buildkite and creates a fresh artifact directory:

```bash
BUILDKITE_PARALLEL_JOB_COUNT=5 BUILDKITE_PARALLEL_JOB=0 \
python .buildkite/amd/scripts/run_cpu_suite.py \
  --timing-profile=.buildkite/amd/cpu-timings/model-executor-v1.json \
  model-executor -sv tests/model_executor \
  -m 'core_model and cpu and not omni' \
  --deselect=tests/model_executor/models/personaplex/test_temporal_streaming_hoist.py::test_mimi_streaming_step_matches_legacy_end_to_end
```

These options replace `--num-shards` / `--shard-id` for the qualification run.
Combining the two partitioners or enabling xdist workers fails before execution.
The Model Executor scope is `core_model` + `cpu`, excluding `omni`, under
`tests/model_executor/`. Marker, keyword, and explicit deselection filters run
before partitioning. Without `--cpu-shard-profile`, collection behavior is unchanged.

The current collected node set is authoritative. Nodes in a module stay together
to preserve module/class fixture locality. Modules are assigned longest estimated
cost first to the least-loaded shard. Module ties use path order; shard ties use
node count, then shard index. New
nodes use the median positive timing among currently collected measured nodes,
or a one-second planning weight when none exists. Stale timing entries never
select tests. Empty/unknown timing history preserves coverage but cannot establish
a useful performance balance. Output preserves the original collection order
inside each shard; an empty selected shard retains pytest's exit code 5.

`plan.json` records all node assignments, estimated shard costs, matched/unknown/
stale timing counts, source builds, the profile SHA-256, an eligible-node SHA-256,
and algorithm version `module-lpt-v1`. The file is written before execution and
is **not** an execution result. Invalid profiles or plan-write errors fail closed.
All workers must report identical profile/eligible-node hashes and assignments.
Compare one complete successful baseline build against one complete candidate
build using the downloaded artifact directories:

```bash
python .buildkite/amd/scripts/verify_cpu_shards.py --suite model-executor \
  --baseline downloaded/baseline/shard-* --candidate downloaded/candidate/shard-*
```

This rejects missing/duplicate attempts, failed or incomplete execution, changed
node unions, changed skips, and execution that disagrees with the shared plan.
Run baseline and candidate on the same source revision, image and resource limits.

Shared fixture cost can be counted repeatedly in historical hash shards; collection,
imports, and CPU contention are not modeled by the weights. Estimated loads and
synthetic tests are not hardware speedup evidence. Before merging the five-worker
trial, verify available `amd_mi300_1` capacity and queue latency at that concurrency.
The source jobs expose 192 logical CPUs but have a four-CPU cgroup quota.
An agent inventory alone cannot establish spare cluster capacity. Measure paired
READY/MERGE runs, whole-build latency, queue latency, CPU pressure, peak memory,
and summed agent-minutes. The planning target is each shard p95 below 18 minutes;
also check a median slowest/fastest ratio at most 1.20 and no more than 10% growth
in aggregate job minutes. Require unchanged coverage and numerical assertions.
Rollback restores `parallelism: 3`, removes `--timing-profile`, and restores the
original `--num-shards` / `--shard-id` arguments; timing instrumentation remains.

Benchmark CPU thread counts without changing reference inputs or tolerances:

```bash
python tests/helpers/benchmark_cpu_threads.py --output /tmp/cpu-thread-trial -- \
  -q tests/model_executor/models/ming3_tts/test_ming3_batched_campplus.py::test_batched_campplus_matches_per_clip_pipeline \
  tests/model_executor/models/kokoro/test_kokoro.py::test_speed_shortens_durations \
  tests/model_executor/models/cosyvoice3/test_speaker_embedding_capture_failure.py::test_first_replay_failure_leaves_no_bucket \
  tests/model_executor/models/kokoro/test_kokoro.py::test_network_is_deterministic_per_generator
```

Use the CI image and its quota; the script interleaves 1/2/4 threads, with one
warmup and three measured fresh processes per count. Logs, JUnit results and
software versions are retained. A failure or skip invalidates the benchmark.

### Other CPU: two-shard trial

READY/MERGE split Other into two duration-weighted shards using
`.buildkite/amd/cpu-timings/other-v1.json`, generated from all 8,954 selected
nodes in the successful two-way target READY trial in
[build #15655](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/15655), merge
commit `ae8a431e3e53b102b77b2e7ebdd2ad3dea31fe5f`. The first trial selected
every live node exactly once with no test, collection, or artifact failures;
both workers launched with seven-second queue delays and finished in 12m04s and
14m50s. Repacking those measured setup + call + teardown totals estimates
12m17s per shard before collection and job overhead. This remains a planning
estimate until the refreshed profile completes target CI. A third shard should
be considered only after measuring the two-shard p95 and queue/agent cost.

The existing `core_model and cpu` expression and all five `--ignore` arguments
are retained. Whole modules stay together, including expensive subprocess and
import fixtures. Fixture scopes, mutation isolation and pass criteria do not
change. New tests are assigned from live collection even without timing history.

The historical artifact contains three zero-duration `unittest.subTest` call
reports before one parent call. The importer accepts this passed zero-duration
prefix only while requiring exactly one test start per selected node; repeated
nonzero phase reports still fail. New artifacts record `subtest_report` separately,
preserve subtest failures, and avoid double-counting parent call duration.

```bash
python .buildkite/amd/scripts/cpu_shard_plan.py downloaded/other-job \
  --suite other --output other-v2.json
python .buildkite/amd/scripts/verify_cpu_shards.py --suite other \
  --baseline downloaded/other-baseline --candidate downloaded/other-candidate-*
```

Before rollout, require the unchanged executed node union and skip outcomes,
per-shard p95 below 18 minutes, and measured summed agent-minutes, build wall
time and queue latency. Other retains its existing thread environment. Rollback
removes `parallelism: 2` and `--timing-profile`, and restores the unsharded label.

## Cross-cutting conventions

### Diff-aware CI {#diff-aware-ci}

PR diffs drive **two independent skip layers**. Both read changed files from git, but at different pipeline stages and with different granularity:

| Layer | Script | When | What is skipped | Where you configure |
| ----- | ------ | ---- | --------------- | ------------------- |
| **Bootstrap** | [`skip_ci.py`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/common/scripts/skip_ci.py) + [`upload_pipeline.py`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/common/scripts/upload_pipeline.py) | Before child test pipelines upload (`cuda/pipeline.yml`, `npu/pipeline-npu.yml`, AMD/Intel bootstraps) | Entire default CI, or whole L2/L3 upload for a platform | Whitelists in `ci_conditions.py`; bootstrap `if` injected by step `key` in `upload_pipeline.py` |
| **Step filter** | [`upload_pipeline.py`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/common/scripts/upload_pipeline.py) | While uploading CUDA L2–L5 YAML | Individual Buildkite steps inside `test-ready.yml` / `test-merge.yml` / `test-nightly.yml` / `test-weekly.yml` | `source_file_dependencies` key on each step (presets in `ci_source_file_dependencies.yml`) |

**Changed files** (both layers):

| Build context | Diff command |
| --- | --- |
| Pull request | `git diff --name-only origin/<base>...<BUILDKITE_COMMIT>` |
| Release-line push (`main`, or non-PR `release/vX.Y.Z`) | `git diff --name-only <commit>^..<commit>` |
| Local dry-run / non-PR | Diff unavailable → bootstrap and step filter both keep all steps; uploader-only keys are still stripped |

Label triggers (`ready`, `merge-test`) are unchanged—diff-aware logic only reduces what runs **after** a pipeline is already scheduled.

#### Bootstrap skip {#bootstrap-skip}

[`skip_ci.py`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/common/scripts/skip_ci.py) classifies the git diff into buckets (docs / qualifying skip-mark / whitelisted L2·L3·L4/L5 YAML / other), then picks one decision path. CUDA/NPU apply that decision by injecting Buildkite `if` on `bootstrap-upload-steps.yml` step keys; AMD/Intel call `gate_bootstrap_ci <platform> <l2|l3>`.

##### Decision overview

| Diff shape | Path | Default L2/L3 |
| --- | --- | --- |
| Product code / non-whitelisted paths | normal CI | all on |
| Docs and/or qualifying skip-mark only | `skip_all` | all off (scheduled release-line NIGHTLY/WEEKLY/`NON_CRITICAL` exceptions below; `main` or `release/vX.Y.Z`) |
| Whitelisted CI YAML only | yaml-gated (`skip_l2_l3`) | per platform/level matrix |
| Docs/skip-mark **+** whitelisted CI YAML | yaml-gated (`skip_l2_l3`) | same as CI-YAML-only (does **not** widen to normal CI) |
| Non-qualifying skip-mark and no CI YAML | normal CI | all on |
| Diff unavailable | normal CI | all on |

**`skip_all` exceptions:** For CUDA/NPU, PR labels (`nightly-test`, `merge-test`, `weekly-test`, `ready`, …) do **not** revive jobs under `skip_all`; on a release line (`main` or `release/vX.Y.Z`), scheduled `NIGHTLY=1` still builds the image and uploads L4, while `WEEKLY=1` or `NON_CRITICAL=1` still builds the image and uploads L5 (CUDA). `WEEKLY=1` on a release line also uploads L2/L3 with `--e2e`. AMD applies `skip_ci.py gate` only to selected ready/merge specs. An explicitly selected AMD nightly spec (`nightly-test`, `NIGHTLY=1`, or a nightly debug override) remains runnable even for a docs-only diff. AMD weekly remains runnable for a scheduled release line (`main` or `release/vX.Y.Z`) + `WEEKLY=1`; a PR `weekly-test` selection still honors `skip_all`.

**Yaml-gated nightly/weekly:** L4/L5 upload steps keep their normal label / `NIGHTLY` / `WEEKLY` / `NON_CRITICAL` conditions (for example PR `nightly-test` still uploads nightly). Only L2/L3 upload steps are matrix-gated.

##### Whitelisted CI YAML → platform

Register new files in `L2_YAML_FILES`, `L3_YAML_FILES`, or `L45_YAML_FILES` in `ci_conditions.py`.

| Level | File | Platform |
| --- | --- | --- |
| L2 | `.buildkite/cuda/test-ready.yml` | cuda |
| L2 | `.buildkite/npu/test-npu-ready.yml` | npu |
| L2 | `.buildkite/amd/test-amd-ready.yml` | amd |
| L2 | `.buildkite/intel/pipeline-intel.yml` | intel |
| L3 | `.buildkite/cuda/test-merge.yml` | cuda |
| L3 | `.buildkite/amd/test-amd-merge.yml` | amd |
| L4/L5 | `.buildkite/cuda/test-nightly.yml`, `test-weekly.yml` | cuda |
| L4/L5 | `.buildkite/amd/test-amd-nightly.yml`, `test-amd-weekly.yml` | amd |
| L4/L5 | `.buildkite/npu/test-npu-nightly.yml` | npu |

##### Yaml-gated category branches

`L2?` / `L3?` / `L45?` = whether any file in that whitelist changed. Untouched platforms/levels end up off.

| ID | L2? | L3? | L45? | Enabled L2/L3 |
| --- | --- | --- | --- | --- |
| G1 | | | ✓ | none (all L2/L3 off) |
| G2 | ✓ | | | L2 only on platforms whose L2 YAML changed |
| G3 | | ✓ | | L3 only on platforms whose L3 YAML changed |
| G4 | ✓ | ✓ | | union of G2 and G3 (same platform ready+merge → both on) |
| G5 | ✓ | | ✓ | same as G2 (L45 rescued by L2 YAML) |
| G6 | | ✓ | ✓ | same as G3 (L45 rescued by L3 YAML) |
| G7 | ✓ | ✓ | ✓ | same as G4 |

##### Yaml-gated examples (platform-level)

Docs/skip-mark mixed with any row below follows the same matrix.

| Diff (whitelist) | Branch | Enabled L2/L3 |
| --- | --- | --- |
| Only `cuda/test-ready.yml` | G2 | `cuda/l2` |
| Only `amd/test-amd-ready.yml` | G2 | `amd/l2` |
| Only `npu/test-npu-ready.yml` | G2 | `npu/l2` |
| Only `intel/pipeline-intel.yml` | G2 | `intel/l2` |
| Only `cuda/test-merge.yml` | G3 | `cuda/l3` |
| Only `amd/test-amd-merge.yml` | G3 | `amd/l3` |
| CUDA ready + CUDA merge | G4 | `cuda/l2` + `cuda/l3` |
| CUDA ready + AMD merge | G4 | `cuda/l2` + `amd/l3` |
| CUDA ready + AMD ready | G2 | `cuda/l2` + `amd/l2` |
| CUDA merge + AMD merge | G3 | `cuda/l3` + `amd/l3` |
| Only nightly / weekly / amd-nightly / amd-weekly / npu-nightly YAML | G1 | (none) |
| CUDA ready + nightly | G5 | `cuda/l2` |
| CUDA merge + nightly | G6 | `cuda/l3` |
| CUDA ready + merge + nightly | G7 | `cuda/l2` + `cuda/l3` |

##### How platforms consume the decision

| Platform | Mechanism |
| --- | --- |
| **CUDA** | `upload_pipeline.py` injects `if` by step `key` (`image-build`, `upload-ready-pipeline`, `upload-merge-pipeline`, …) from `is_run(cuda, l2/l3)` |
| **NPU** | Same injection; no merge upload step (L3 always off in bootstrap) |
| **AMD / Intel** | `gate_bootstrap_ci <platform> <l2\|l3>` exits 0 when `skip_all` or that target is off |

Unit coverage: `tests/buildkite/test_skip_ci.py`.

#### Step filtering {#step-filtering}

CUDA **L2** (`.buildkite/cuda/test-ready.yml`), **L3** (`.buildkite/cuda/test-merge.yml`), **L4** (`.buildkite/cuda/test-nightly.yml`), **L5** (`.buildkite/cuda/test-weekly.yml`), and NPU **L4** (`.buildkite/npu/test-npu-nightly.yml`). Bootstrap upload entry: `upload_pipeline.py --upload .buildkite/cuda/bootstrap-upload-steps.yml` (or `npu/bootstrap-upload-steps.yml`).

**When filtering applies:** **PR label** uploads (`ready`, `merge-test`, `nightly-test`, `weekly-test`) and **post-merge release-line L3** (`main` or `release/vX.Y.Z`, `test-merge.yml`). Scheduled **release-line + env** uploads that pass `--all` (`NIGHTLY=1`, `WEEKLY=1` / `NON_CRITICAL=1`) or `--e2e` (`WEEKLY=1` on L2/L3) keep every selected job and still strip the key. If **no** job-key prefix matches the diff, a change to the pipeline YAML being uploaded, or any path under the `source_filter_fallback` key in `ci_source_file_dependencies.yml`, keeps every job in that upload. `source_filter_fallback` is not a job key — do not attach it to a step. When any job-key prefix already matches, normal filtering wins (the fallback does not expand the selection).

**Uploader-only keys** — removed before Buildkite sees the YAML; never used at runtime on agents:

| Key | Purpose |
| --- | ------- |
| `source_file_dependencies` | Named preset (or list of presets) from [`ci_source_file_dependencies.yml`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/common/ci_source_file_dependencies.yml). The uploader expands only that key; it does **not** scan `commands`. If any changed file equals a listed prefix or starts with `prefix/`, keep the step (or group); otherwise omit it. Inline path-prefix lists are still accepted. |
| `mirror_hardwares` | Expand to `agents` + `plugins` (+ optional `image`) from `ci_mirror_hardwares.yml`. |
| `automatic_shard` | CUDA ready/merge directory sweep. The uploader strips the key and may set `parallelism` from that step's timing profile. With no usable timings it uses the smaller of `MAX_AUTOMATIC_SHARDS` and the discovered file count. See [Automatic shard planning](#automatic-shard-planning). |

#### Automatic shard planning {#automatic-shard-planning}

`automatic_shard: true` does not reserve a pool when the YAML is uploaded.

- **Fallback:** no usable timings means `min(MAX_AUTOMATIC_SHARDS, discovered files)` jobs, spread evenly across those files. `MAX_AUTOMATIC_SHARDS` is the same cap the measured packing uses. One file stays one job. A timed-out build is not a passed build, so it never becomes the next profile; this cap is what keeps that suite from staying one job. The step still writes `timings/jobs/<job>-<pid>.json`. Two pytest processes in one job keep both files; durations for the same file are summed. Shards of one build sum their slices of a file, and a retry of the same shard index keeps the max.
- **Profile:** planning reads raw `timings/jobs` from up to 20 `passed` or `finished` builds. Ready history is every non-`main` passed build of this pipeline. Those builds are other pull requests, so a step's durations include whatever that pull request changed. Merge history stays on `main`. The builds API is requested at `per_page=100` (its cap) and the next page comes from the `Link` header. The artifact lookup stops after 20 builds that contain job files, after 10 consecutive builds with none, or after 90 seconds. Each job records its agent queue; a step ignores a profile recorded on a different queue. Only raw `timings/jobs` files are observations. Failed and running builds are excluded. The label key drops a `· Shard N/M` suffix, so one step does not use another step's time for the same path.
- **Plan:** a greedy longest-processing-time packing of measured files into at most `MAX_AUTOMATIC_SHARDS` jobs. A bin must fit 80% of `timeout_in_minutes`; the other 20% is left for checkout, collection, and process startup, which are not included in the file durations. A faster suite still uses fewer jobs when its file time fits that smaller budget. Each bin is uploaded as `timings/file-shards/<step>/<index>.txt`. Shard N keeps the original directory targets so each directory `conftest.py` loads, then runs only the files in that list. One file already over the budget does not add shards. A file with no measurement adds no weight, so it does not by itself open another job; when the step does split, that file is still placed in the lightest bin and runs.
- **Overrides:** `VLLM_OMNI_SHARD_TIMINGS` selects a local profile. `VLLM_OMNI_SKIP_SHARD_TIMINGS_DOWNLOAD=1` skips the history download. `VLLM_OMNI_SKIP_SHARD_FILE_WRITE=1` still plans the jobs but does not write `timings/file-shards`. The Buildkite schema hook sets both.

E2E model jobs stay one job. A hung command is bounded only by `timeout_in_minutes`.

### Policy

- **Always uploaded** (no key): L2 **Simple Test** except **Simple · ROCm R2-01 Collection Count**, plus L2/L3 groups outside **E2E Test** (Diffusion unit tests, Engine/Model Executor, Distributed, Custom Pipeline, Entrypoints on L2, LoRA / Entrypoints on L3).
- **Diff-gated**: every **E2E Test** leaf job in L2/L3, the L2 **Simple · ROCm R2-01 Collection Count** check (it runs the exact collection guard of the non-blocking AMD READY R2-01 job), L4 nightly leaf jobs (CUDA and NPU), and L5 weekly jobs whose `if` includes the `weekly-test` label (Reliability / Perf). Jobs gated only by `WEEKLY=1`, `NON_CRITICAL=1`, or the nightly **Simple · CPU Coverage Test** (`NIGHTLY=1`) have no `source_file_dependencies`. Registry layout: YAML anchors group model code and shared runtime dependencies. Job keys alias the relevant anchors and **also** list the scripts that job runs (pytest files, `run_cov_split.sh --offline` / `--online` targets, example README runners). Feature keys stay cross-cutting. A **group** may define the key instead; the whole group drops if no prefix matches.

Qwen3-Omni nightly accuracy and performance jobs compose their evaluation key
with the reusable platform key `ci_cuda`, `ci_npu`, or `ci_rocm`. CUDA jobs also
compose `omni_v2_runtime` for their V2 and prefix-cache dependencies.
Other models can opt into a platform key without inheriting these runtime choices.
Shared scheduling, engine, SHM transport, data contracts, serving, configuration,
benchmark infrastructure, and test fixtures select both accuracy and performance
on all three platforms, including the existing CUDA Duplex performance jobs.
Platform workers, CI images, requirements, build scripts,
and nightly YAML select the corresponding platform. Accuracy datasets/evaluators
and performance runners also have their own dependencies; Duplex helpers select
the existing CUDA Duplex jobs.

These are direct dependencies, so a diff containing both `tests/helpers/runtime.py`
and MiniCPM model code retains Qwen3 accuracy/performance even when another job
matches. CUDA V2 and prefix-cache dependencies select the existing CUDA jobs;
selection alone does not add a cache-hit workload or exercise every deploy
override. Scheduled full uploads still keep every job. Routing regressions live
in `tests/buildkite/test_qwen3_omni_source_dependencies.py`.

### YAML examples

```yaml
      - label: "Diffusion · Qwen Image Test"
        source_file_dependencies: diffusion_qwen_image_function
        # diffusion_qwen_image_function: *qwen_image + tests/e2e/online_serving/test_qwen_image.py
        commands:
          - pytest -s -v tests/e2e/online_serving/test_qwen_image.py -m 'core_model' ...
        mirror_hardwares: h100_1

      - label: "TTS · Qwen3-TTS CustomVoice Test"
        source_file_dependencies: tts_qwen3_tts_function
        # tts_qwen3_tts_function: *qwen3_tts + tests/e2e/online_serving/test_qwen3_tts_customvoice.py
        commands:
          - pytest -s -v tests/e2e/online_serving/test_qwen3_tts_customvoice.py ...
        mirror_hardwares: l4_1
```

### Local dry-run

```bash
python3 .buildkite/common/scripts/upload_pipeline.py .buildkite/cuda/test-ready.yml
python3 .buildkite/common/scripts/upload_pipeline.py .buildkite/cuda/test-merge.yml | grep source_file_dependencies
# (no output expected)
```

On PR builds, `upload_pipeline.py` logs `skip '…' (no changes under …)` for omitted steps.

### Labels and grouping

- Use **groups** for dashboard readability; keep **E2E Test** as the group name CUDA diff filtering expects for `--e2e` weekly runs on a release line (`main` or `release/vX.Y.Z`, `WEEKLY=1`).
- Prefix labels by model domain: **Omni ·**, **TTS ·**, **Diffusion ·**, **Simple ·**, etc., matching existing steps.

### Per-model coverage

Pilot (v1): one Merge-tier (L3) job in `.buildkite/cuda/test-merge.yml` uploads
per-model, per-entry-mode coverage as Buildkite artifacts — "TTS · Qwen3-TTS Base
Test". Coverage + artifact upload run only when bootstrap uploaded that job with
`--e2e` (`WEEKLY=1` and a release-line branch, `main` or `release/vX.Y.Z`, via `run_cov_split.sh`); otherwise
the same job runs plain pytest without `--cov`. It runs on a single GPU so the
pilot is cheap to reproduce.

**L1 package coverage (CPU):** ready Simple Test jobs run without `--cov`. Scheduled nightly on a release line (`main` or `release/vX.Y.Z`) with `NIGHTLY=1` runs **Simple · CPU Coverage Test** in `.buildkite/cuda/test-nightly.yml` (`pytest -m 'core_model and cpu'` + Cobertura XML artifact).

**Naming convention**: `coverage-<model_id>-<mode>-<step_id>.xml.gz`, where
`<model_id>` is the model's directory name under
`vllm_omni/diffusion/models/<model_id>/` or
`vllm_omni/model_executor/models/<model_id>/` (e.g. `z_image`, `qwen3_omni`),
`<mode>` is `online` or `offline`, and `<step_id>` is `BUILDKITE_STEP_ID`
(`local` outside Buildkite). `gunzip` before feeding a report to a Cobertura
consumer.

Reports are gzipped because `--cov=vllm_omni` sets coverage's `source`, which
makes it walk the package and emit one `<line>` element per statement for every
file — including the ones a single model's tests never import. That inventory,
not the coverage, is the size: a Bagel report is 7.1 MB, of which 96.5% is
`<line>` elements and 77% belongs to the 840 of 1169 files that run never
imported. A run covering zero lines produces the same 7 MB. gzip takes it to
roughly 410 KB and changes nothing about the data.

`<step_id>` is what keeps two steps that cover the same model apart. Artifacts are
scoped to the build, and a weekly build can carry ready/merge (with `--e2e`) and
weekly tiers at once, so several same-model steps land in one namespace — `qwen3_tts` already has
both a Base and a CustomVoice job, and `minicpmo_4_5` a base and a duplex one. Two
uploads on one path do not fail loudly: an exact-name download can report the path
as ambiguous, and a glob download fetches every match concurrently and renames each
onto the same destination, so whichever finishes last wins. The step id remains the
logical step identifier across retries, while Buildkite's default artifact lookup
selects only the latest attempt — `BUILDKITE_JOB_ID` would instead give every
attempt its own filename, leaving a failed attempt's partial report to be picked up
alongside the real one. A step using `parallelism`/matrix would need
`BUILDKITE_PARALLEL_JOB` added, since its jobs share a step id; neither pilot is
one.

**Opting in a new model**: replace the job's combined `pytest` command with
[`run_cov_split.sh`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/common/scripts/run_cov_split.sh),
which runs each mode, names the reports, uploads them, and fails the step if
either half or the upload failed:

```yaml
commands:
  - |
    .buildkite/common/scripts/run_cov_split.sh \
      --model-id <model_id> \
      --offline <offline test path> \
      --online <online test path> \
      -- <the job's existing pytest flags, e.g. -m '...' --run-level '...'>
```

Everything after `--` is forwarded verbatim to both runs with quoting preserved,
so an expression like `-m 'advanced_model and cuda'` survives intact and a job can
pass anything else it needs. `--offline`/`--online` are repeatable and only one is
required, so a job with tests in a single mode runs just that half. Runtime is
bounded solely by the step's `timeout_in_minutes` — raise it, since the split
loads the model once per mode.

Before splitting, confirm each half actually collects at least one test under the job's forwarded filters — pytest exits with code 5 ("no tests collected") if a half is empty, which the script reports as a failure, red-ing a job that used to pass as one combined invocation.

Also check the tests' `num_cards` against the job's `mirror_hardwares` preset.
The hardware-mark helpers turn `num_cards > 1` into a device-count `skipif` on
the active CUDA, ROCm, XPU, NPU or MUSA platform, so a test asking for more
devices than the preset provides is silently skipped and its
coverage artifact comes back empty. Splitting such a job produces a file that
looks fine and measures nothing — "Diffusion · Z Image Test" is the current
example (online tests want 4 cards, the job runs on `l4_1`), which is why it is
not a coverage pilot.

Online-mode coverage depends on the `[tool.coverage.run]` `patch`/`sigterm`
settings in `pyproject.toml`: online tests launch the server with
`subprocess.Popen` and stop it with SIGTERM, and without those settings the XML
reflects only the pytest parent process, not the server's code paths.

The upload depends on `buildkite-agent` being callable inside the container.
CUDA (`l4_*`, `h100_*`) and NPU (`*_npu_*`) presets all use the `kubernetes`
plugin, which provides the agent. `l4_*` jobs run on the EKS `l4-k8s` queue
(agent-stack-k8s); they no longer use the docker plugin or
`mount-buildkite-agent`. A new **docker** preset that runs a coverage job
still needs `mount-buildkite-agent: true`.

List both `run_cov_split.sh` and `pyproject.toml` in every opted-in job's
`source_file_dependencies` preset — both change what the job measures, so without them a
change there is filtered out of normal PR builds and only surfaces in a later
nightly. `tests/buildkite/test_upload_pipeline.py` covers the filter behavior with
a synthetic job (it does not pin real merge labels).
If no job-key prefix matches, editing the pipeline YAML being uploaded, or any
path under the `source_filter_fallback` registry key, keeps every job in that
upload (including coverage jobs) so the wiring can be validated before merge.
When a job-key prefix already matches, bypass does not expand the selection. A
change to a different pipeline YAML still does not schedule them. When checking a new model's artifacts, compare `lines-covered`
between the online and offline XML rather than just confirming both files exist.

### Validation checklist

| Check | Command / location |
| ----- | ------------------ |
| CUDA render | `python3 .buildkite/common/scripts/upload_pipeline.py .buildkite/cuda/test-<level>.yml` |
| No leaked uploader keys | Pipe the render into `grep -E 'mirror_hardwares\|source_file_dependencies'` — expect empty |
| Skip-ci / upload unit tests | `pytest tests/buildkite/` |
| Local job replay (CUDA L2+) | `tools/run_ready_jobs.sh`, `tools/run_merge_jobs.sh`, `tools/nightly/run_nightly_jobs.sh` (read YAML from `cuda/`) |

## Related documentation

- [Test System Overview](./test_system_overview.md) — L1–L5 scope, triggers, test pyramid
- [Test Writing Guide](./test_writing_guide.md) — markers, directories, L1–L5 examples
- [Test Execution Guide](./test_execution_guide.md) — running CI-aligned jobs locally
- Implementation: `.buildkite/common/scripts/upload_pipeline.py`, `.buildkite/common/scripts/skip_ci.py`

## Docker Hub release credentials

Docker publishing and nightly cleanup use the Buildkite secret
`OMNI_DOCKERHUB_TOKEN`, mapped to `RELEASE_DOCKERHUB_TOKEN` only on Docker jobs.
Agents must support step-level secrets (Buildkite agent 3.106.0 or newer).
Configure an organization access token for `vllm/vllm-omni` and use `vllm`
as the Docker Hub username. Image Push grants image pull/push; Tag Admin
grants tag listing/deletion for nightly cleanup. Do not grant access to
other repositories or organization management.

Restrict the secret's access policy to the immutable `omni-release` pipeline
ID, main, and the small CPU release queue. A job's `secrets` entry
controls injection; the access policy controls whether other jobs may
retrieve the secret. Protect release refs and pipeline settings as well.

The cleanup script uses Docker Hub's `/v2/auth/token` exchange and
namespace-scoped tag API. Shell tracing stays disabled to avoid exposing
credentials or bearer tokens. The legacy S3 `DOCKERHUB_TOKEN` variable is
not used by these Docker jobs. After validating the replacement, remove
obsolete S3 copies and revoke shared tokens only after all consumers have
migrated. Never commit token values to this repository.
