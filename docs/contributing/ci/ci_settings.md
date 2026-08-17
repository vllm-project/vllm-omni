# CI Settings

This document describes **where** Buildkite YAML lives in the repo, **how each platform organizes CI**, and **how to add a new job**. It does not document agent queues, GPU types, or container plugin details—those belong in infra / preset files (for example `.buildkite/common/ci_mirror_hardwares.yml`).

For CI levels (L1–L5) and triggers, see [Test System Overview](./test_system_overview.md). For test authoring, see [Test Writing Guide](./test_writing_guide.md). For running tests locally or replaying CI jobs, see [Test Execution Guide](./test_execution_guide.md).

## Directory layout

Canonical layout (prefer these paths for new changes):

```
.buildkite/
├── common/                          # Shared across platforms
│   ├── scripts/
│   │   ├── skip_ci.py               # skip-ci decision (docs / skip-mark / CI YAML paths)
│   │   ├── upload_pipeline.py       # Bootstrap + test-pipeline uploader (CUDA/NPU)
│   │   └── resolve_skip_ci.sh       # Shell helpers for AMD/Intel bootstrap
│   └── ci_mirror_hardwares.yml      # CUDA uploader presets (referenced by name only)
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

**Placement rules**

| Rule | Detail |
| ---- | ------ |
| Platform code under platform dir | New CUDA jobs go in `.buildkite/cuda/`; do not add new top-level `.buildkite/test-*.yml` files. |
| Shared logic in `common/` | Skip-ci and CUDA upload rendering stay in `.buildkite/common/scripts/`. |
| Bootstrap vs test YAML | **Bootstrap** (`pipeline*.yml`) builds images and uploads **child** test pipelines. **Test** YAML (`test-*.yml`) lists pytest steps only. |
| Register CI YAML in `skip_ci.py` | If you add a new whitelisted test pipeline file, update `L2_YAML_FILES`, `L3_YAML_FILES`, or `L45_YAML_FILES` in `.buildkite/common/scripts/skip_ci.py` so skip-ci paths stay correct. |

There are still **legacy copies** at `.buildkite/*.yaml` (without the `cuda/` prefix). Treat `.buildkite/cuda/*` as source of truth.

## Platform comparison

| Platform | Bootstrap entry | Test job files | Upload mechanism | Job hardware in YAML |
| -------- | ---------------- | -------------- | ---------------- | -------------------- |
| **CUDA** | `cuda/pipeline.yml` | `test-ready.yml`, `test-merge.yml`, `test-nightly.yml`, `test-weekly.yml` | `upload_pipeline.py --upload` (expands uploader-only keys) | `mirror_hardwares: <preset>` (string) |
| **NPU** | `npu/pipeline-npu.yml` | `test-npu-ready.yml`, `test-npu-nightly.yml` | `upload_pipeline.py --upload` | `mirror_hardwares: a2b3_npu_1` / `a2b3_npu_4` / `a3_npu_2` |
| **AMD** | `amd/scripts/bootstrap-amd-omni.sh` | `test-amd-ready.yml`, `test-amd-merge.yml` | Jinja (`test-template-amd-omni.j2`) → `pipeline upload` | `agent_pool` + `mirror_hardwares: [amdproduction]` (array, template filter) |
| **Intel** | `intel/scripts/bootstrap-intel-omni.sh` | `intel/pipeline-intel.yml` (steps inline) | Direct `pipeline upload` | Inline `agents.queue` on each step |

## Platform configuration style

=== "CUDA"

    **Bootstrap:** [`cuda/pipeline.yml`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/cuda/pipeline.yml) (hook upload) + [`cuda/bootstrap-upload-steps.yml`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/cuda/bootstrap-upload-steps.yml) (child steps). Document 1 runs [`upload_pipeline.py --upload .buildkite/cuda/bootstrap-upload-steps.yml`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/common/scripts/upload_pipeline.py), which injects `if` by step `key` from skip-ci and uploads image build plus L2–L5 child pipeline upload steps (see [Diff-aware CI — Bootstrap skip](#bootstrap-skip)).

    **Test YAML:** `cuda/test-ready.yml` (L2), `test-merge.yml` (L3), `test-nightly.yml` (L4), `test-weekly.yml` (L5). Each file starts with shared `env:` then `steps:`.

    | CI level | File | Typical trigger |
    | -------- | ---- | ----------------- |
    | L2 | `cuda/test-ready.yml` | `ready` label |
    | L3 | `cuda/test-merge.yml` | `merge-test` label / main merge |
    | L4 | `cuda/test-nightly.yml` | `nightly-test` label or `NIGHTLY=1` |
    | L5 | `cuda/test-weekly.yml` | `weekly-test` label or `WEEKLY=1` |

    **Upload:** `upload_pipeline.py --upload` expands uploader-only keys before Buildkite upload.

    **Hardware in YAML:** `mirror_hardwares: <preset>` (string)—preset names in [`common/ci_mirror_hardwares.yml`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/common/ci_mirror_hardwares.yml). Do **not** set `agents` / `plugins` on the same step.

    **Conventions**

    - **`depends_on`:** leaf jobs depend on `upload-ready-pipeline`, `upload-merge-pipeline`, etc.
    - **`group` / `label`:** `:card_index_dividers:` groups; labels like `Diffusion · Qwen Image Test`.
    - **`commands`:** `timeout … pytest …` with markers and `--run-level` for the pipeline level.
    - **`source_file_dependencies`:** required on **E2E Test** leaf jobs in L2/L3; see [Step filtering](#step-filtering).

    **Adding a job**

    1. Pick the level file (ready / merge / nightly / weekly).
    2. Add a step under the right **group** (usually **E2E Test** for model pytest).
    3. Set `label`, `commands`, `mirror_hardwares`, `depends_on: upload-<level>-pipeline`.
    4. For L2/L3 E2E, add `source_file_dependencies` (pytest + model + deploy YAML prefixes).
    5. Dry-run:

    ```bash
    python3 .buildkite/common/scripts/upload_pipeline.py .buildkite/cuda/test-ready.yml
    ```

=== "NPU"

    **Bootstrap:** [`npu/pipeline-npu.yml`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/npu/pipeline-npu.yml) + [`npu/bootstrap-upload-steps.yml`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/npu/bootstrap-upload-steps.yml)—same split as CUDA; builds A2/B3 and A3 CI images, then uploads child test pipelines.

    **Test YAML:** `npu/test-npu-ready.yml` (L2), `test-npu-nightly.yml` (L4).

    **Upload:** same as CUDA—`upload_pipeline.py --upload`.

    **Hardware in YAML:** `mirror_hardwares` preset (string), expanded to `agents`, top-level `image`, and `plugins`. Presets: `a2b3_npu_1`, `a2b3_npu_4`, `a3_npu_2` in `common/ci_mirror_hardwares.yml`.

    **Conventions**

    - **`depends_on: upload-ready-pipeline`** (or `upload-nightly-pipeline`) ties jobs to bootstrap upload keys.
    - Do not duplicate `agents` / `image` / `plugins` when using `mirror_hardwares`.

    **Adding a job**

    1. Edit `test-npu-ready.yml` (L2) or `test-npu-nightly.yml` (L4).
    2. Add a step with `mirror_hardwares` (add a new preset in `ci_mirror_hardwares.yml` first if needed).
    3. Set `commands` to your pytest file and markers.
    4. Dry-run:

    ```bash
    python3 .buildkite/common/scripts/upload_pipeline.py .buildkite/npu/test-npu-ready.yml
    ```

    5. Register new pipeline paths in `skip_ci.py` (`L2_YAML_FILES` or `L45_YAML_FILES`) when applicable.

=== "AMD"

    **Bootstrap:** [`amd/scripts/bootstrap-amd-omni.sh`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/amd/scripts/bootstrap-amd-omni.sh)—skip-ci, diff filtering, Jinja render, then `buildkite-agent pipeline upload`.

    **Test YAML (data):** `amd/test-amd-ready.yml` (L2 / PR), `test-amd-merge.yml` (L3 / main).

    **Rendering:** [`test-template-amd-omni.j2`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/amd/test-template-amd-omni.j2) wraps data steps with `amd-build` image build and `amd_<agent_pool>` queues. Do **not** hand-edit generated `pipeline.yaml`.

    **Hardware in YAML:** `agent_pool` (for example `mi325_1`) plus `mirror_hardwares: [amdproduction]` (array—Buildkite template filter, not the CUDA/NPU uploader preset mechanism).

    **Data file fields**

    | Field | Purpose |
    | ----- | ------- |
    | `agent_pool` | ROCm pool; template maps to `queue: amd_<pool>`. |
    | `mirror_hardwares` | Which mirror HW runs the step (for example `[amdproduction]`). |
    | `commands` | Passed into `run-amd-test.sh` via `TEST_COMMAND`. |

    **Adding a job**

    1. Edit `test-amd-ready.yml` or `test-amd-merge.yml`.
    2. Copy a neighboring block: `label`, `agent_pool`, `mirror_hardwares`, `commands`, optional `grade`.
    3. Regenerate via bootstrap / Jinja; update `skip_ci.py` if you add a new YAML path.

=== "Intel"

    **Bootstrap:** [`intel/scripts/bootstrap-intel-omni.sh`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/intel/scripts/bootstrap-intel-omni.sh)—skip-ci, then direct `pipeline upload`.

    **Test YAML:** steps live inline in [`intel/pipeline-intel.yml`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/intel/pipeline-intel.yml); runners under `intel/scripts/` (for example `run-xpu-test.sh`).

    **Upload:** `buildkite-agent pipeline upload` (no `upload_pipeline.py` mirror expansion).

    **Hardware in YAML:** inline `agents.queue` (for example `intel-gpu-omni`) on each step.

    **Adding a job**

    1. Add a step to `pipeline-intel.yml`, or extend an existing runner script.
    2. Match `agents`, `env`, `timeout_in_minutes`, and `command`/`commands` with sibling steps.
    3. Keep `pipeline-intel.yml` listed in `L2_YAML_FILES` in `skip_ci.py`.

## Cross-cutting conventions

### Diff-aware CI {#diff-aware-ci}

PR diffs drive **two independent skip layers**. Both read changed files from git, but at different pipeline stages and with different granularity:

| Layer | Script | When | What is skipped | Where you configure |
| ----- | ------ | ---- | --------------- | ------------------- |
| **Bootstrap** | [`skip_ci.py`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/common/scripts/skip_ci.py) + [`upload_pipeline.py`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/common/scripts/upload_pipeline.py) | Before child test pipelines upload (`cuda/pipeline.yml`, `npu/pipeline-npu.yml`, AMD/Intel bootstraps) | Entire default CI, or whole L2/L3 upload for a platform | Whitelists in `skip_ci.py`; bootstrap `if` injected by step `key` in `upload_pipeline.py` |
| **Step filter** | [`upload_pipeline.py`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/common/scripts/upload_pipeline.py) | While uploading CUDA L2/L3 YAML | Individual Buildkite steps inside `test-ready.yml` / `test-merge.yml` | `source_file_dependencies` on each step or group |

**Changed files** (both layers):

| Build context | Diff command |
| --- | --- |
| Pull request | `git diff --name-only origin/<base>...<BUILDKITE_COMMIT>` |
| `main` push | `git diff --name-only <commit>^..<commit>` |
| Local dry-run / non-PR | Diff unavailable → bootstrap and step filter both keep all steps; uploader-only keys are still stripped |

Label triggers (`ready`, `merge-test`) are unchanged—diff-aware logic only reduces what runs **after** a pipeline is already scheduled.

#### Bootstrap skip {#bootstrap-skip}

[`skip_ci.py`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/common/scripts/skip_ci.py) classifies the git diff into buckets (docs / qualifying skip-mark / whitelisted L2·L3·L4/L5 YAML / other), then picks one decision path. CUDA/NPU apply that decision by injecting Buildkite `if` on `bootstrap-upload-steps.yml` step keys; AMD/Intel call `gate_bootstrap_ci <platform> <l2|l3>`.

##### Decision overview

| Diff shape | Path | Default L2/L3 |
| --- | --- | --- |
| Product code / non-whitelisted paths | normal CI | all on |
| Docs and/or qualifying skip-mark only | `skip_all` | all off (scheduled `main` NIGHTLY/WEEKLY exceptions below) |
| Whitelisted CI YAML only | yaml-gated (`skip_l2_l3`) | per platform/level matrix |
| Docs/skip-mark **+** whitelisted CI YAML | yaml-gated (`skip_l2_l3`) | same as CI-YAML-only (does **not** widen to normal CI) |
| Non-qualifying skip-mark and no CI YAML | normal CI | all on |
| Diff unavailable | normal CI | all on |

**`skip_all` exceptions (CUDA/NPU bootstrap only):** PR labels (`nightly-test`, `merge-test`, `npu-test`, `weekly-test`, …) do **not** revive jobs. On `main`, scheduled `NIGHTLY=1` still builds the image and uploads L4 plus L2/L3 (with `--e2e`); `WEEKLY=1` still builds the image and uploads L5 (CUDA).

**Yaml-gated nightly/weekly:** L4/L5 upload steps keep their normal label / `NIGHTLY` / `WEEKLY` conditions (for example PR `nightly-test` still uploads nightly). Only L2/L3 upload steps are matrix-gated.

##### Whitelisted CI YAML → platform

Register new files in `L2_YAML_FILES`, `L3_YAML_FILES`, or `L45_YAML_FILES` in `skip_ci.py`.

| Level | File | Platform |
| --- | --- | --- |
| L2 | `.buildkite/cuda/test-ready.yml` | cuda |
| L2 | `.buildkite/npu/test-npu-ready.yml` | npu |
| L2 | `.buildkite/amd/test-amd-ready.yml` | amd |
| L2 | `.buildkite/intel/pipeline-intel.yml` | intel |
| L3 | `.buildkite/cuda/test-merge.yml` | cuda |
| L3 | `.buildkite/amd/test-amd-merge.yml` | amd |
| L4/L5 | `.buildkite/cuda/test-nightly.yml`, `test-weekly.yml` | cuda |
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
| Only nightly / weekly / npu-nightly YAML | G1 | (none) |
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

CUDA **L2** (`.buildkite/cuda/test-ready.yml`) and **L3** (`.buildkite/cuda/test-merge.yml`) only. Bootstrap upload entry: `upload_pipeline.py --upload .buildkite/cuda/bootstrap-upload-steps.yml`.

**Uploader-only keys** — removed before Buildkite sees the YAML; never used at runtime on agents:

| Key | Purpose |
| --- | ------- |
| `source_file_dependencies` | List of path **prefixes**. If any changed file equals a prefix or starts with `prefix/`, keep the step (or group); otherwise omit it. |
| `mirror_hardwares` | Expand to `agents` + `plugins` (+ optional `image`) from `ci_mirror_hardwares.yml`. |

**Policy**

- **Always uploaded** (no key): groups outside **E2E Test**—Simple Test, Diffusion unit tests, Engine/Model Executor, Distributed, Custom Pipeline, Entrypoints (L2), LoRA / Entrypoints (L3).
- **Diff-gated**: every **E2E Test** leaf job. List the smallest prefix set per step—pytest file(s), model code under `vllm_omni/model_executor/models/` or `vllm_omni/diffusion/models/`, plus `stage_input_processors/` and `vllm_omni/deploy/*.yaml` when applicable. A **group** may define the key instead; the whole group drops if no prefix matches.

**YAML examples**

```yaml
      - label: "Diffusion · Qwen Image Test"
        source_file_dependencies:
          - tests/e2e/online_serving/test_qwen_image.py
          - vllm_omni/diffusion/models/qwen_image/
        commands:
          - pytest -s -v tests/e2e/online_serving/test_qwen_image.py -m 'core_model' ...
        mirror_hardwares: h100_1

      - label: "TTS · Qwen3-TTS CustomVoice Test"
        source_file_dependencies:
          - tests/e2e/online_serving/test_qwen3_tts_customvoice.py
          - vllm_omni/model_executor/models/qwen3_tts/
          - vllm_omni/model_executor/stage_input_processors/qwen3_tts.py
          - vllm_omni/deploy/qwen3_tts.yaml
        commands:
          - pytest -s -v tests/e2e/online_serving/test_qwen3_tts_customvoice.py ...
        mirror_hardwares: l4_4
```

**Local dry-run**

```bash
python3 .buildkite/common/scripts/upload_pipeline.py .buildkite/cuda/test-ready.yml
python3 .buildkite/common/scripts/upload_pipeline.py .buildkite/cuda/test-merge.yml | grep source_file_dependencies
# (no output expected)
```

On PR builds, `upload_pipeline.py` logs `skip '…' (no changes under …)` for omitted steps.

### Labels and grouping

- Use **groups** for dashboard readability; keep **E2E Test** as the group name CUDA diff filtering expects for `--e2e` nightly runs on main.
- Prefix labels by model domain: **Omni ·**, **TTS ·**, **Diffusion ·**, **Simple ·**, etc., matching existing steps.

### Per-model coverage

Pilot (v1): one Merge-tier (L3) job in `.buildkite/cuda/test-merge.yml` uploads
per-model, per-entry-mode coverage as Buildkite artifacts — "TTS · Qwen3-TTS Base
Test". It runs on a single GPU so the pilot is cheap to reproduce. (A second
diffusion pilot, "Diffusion · Bagel Test", was retired from merge; reintroduce
another diffusion coverage job if a second pilot is needed again.) This is a
pilot, not a rollout: no dashboard/visualization lives in this repo, and nothing
is gated on coverage.

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
scoped to the build, and a nightly build carries the ready, merge and nightly tiers
at once, so several same-model steps land in one namespace — `qwen3_tts` already has
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
`cuda_marks` turns `num_cards > 1` into `skipif(device_count() < num_cards)`, so a
test asking for more GPUs than the preset provides is silently skipped and its
coverage artifact comes back empty. Splitting such a job produces a file that
looks fine and measures nothing — "Diffusion · Z Image Test" is the current
example (online tests want 4 cards, the job runs on `l4_1`), which is why it is
not a coverage pilot.

Online-mode coverage depends on the `[tool.coverage.run]` `patch`/`sigterm`
settings in `pyproject.toml`: online tests launch the server with
`subprocess.Popen` and stop it with SIGTERM, and without those settings the XML
reflects only the pytest parent process, not the server's code paths.

The upload depends on `buildkite-agent` being callable inside the container. The
`kubernetes` presets (`h100_*`, `*_npu_*`) provide it; the `docker` ones (`l4_*`)
only do because they set `mount-buildkite-agent: true`. A new docker preset that
runs a coverage job needs the same.

List both `run_cov_split.sh` and `pyproject.toml` in every opted-in job's
`source_file_dependencies` — both change what the job measures, so without them a
change there is filtered out of normal PR builds and only surfaces in a later
nightly. `tests/buildkite/test_upload_pipeline.py` covers the filter behavior with
a synthetic job (it does not pin real merge labels).
Editing only the surrounding CI YAML still does not schedule them, so a PR that
touches just the wiring needs a full E2E run (or the commands run on a GPU host)
to produce artifacts. When checking a new model's
artifacts, compare `lines-covered` between the online and offline XML rather than
just confirming both files exist.

### Validation checklist

| Check | Command / location |
| ----- | ------------------ |
| CUDA render | `python3 .buildkite/common/scripts/upload_pipeline.py .buildkite/cuda/test-<level>.yml` |
| No leaked uploader keys | `… \| grep -E 'mirror_hardwares|source_file_dependencies'` → empty |
| Skip-ci / upload unit tests | `pytest tests/buildkite/` |
| Local job replay (CUDA L2+) | `tools/run_ready_jobs.sh`, `tools/run_merge_jobs.sh`, `tools/nightly/run_nightly_jobs.sh` (read YAML from `cuda/`) |

## Related documentation

- [Test System Overview](./test_system_overview.md) — L1–L5 scope, triggers, test pyramid
- [Test Writing Guide](./test_writing_guide.md) — markers, directories, L1–L5 examples
- [Test Execution Guide](./test_execution_guide.md) — running CI-aligned jobs locally
- Implementation: `.buildkite/common/scripts/upload_pipeline.py`, `.buildkite/common/scripts/skip_ci.py`
