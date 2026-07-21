# CI Settings (Buildkite Layout & Conventions)

This document describes **where** Buildkite YAML lives in the repo, **how each platform organizes CI**, and **how to add a new job**. It does not document agent queues, GPU types, or container plugin details—those belong in infra / preset files (for example `.buildkite/common/ci_mirror_hardwares.yml`).

For CI levels (L1–L5), triggers, and diff-aware skipping behavior, see [Test System Overview](./test_system_overview.md). For test authoring, see [Test Writing Guide](./test_writing_guide.md). For running tests locally or replaying CI jobs, see [Test Execution Guide](./test_execution_guide.md).

## Directory layout

Canonical layout (prefer these paths for new changes):

```
.buildkite/
├── common/                          # Shared across platforms
│   ├── scripts/
│   │   ├── skip_ci.py               # Docs/skip-mark / CI-YAML-only level logic
│   │   ├── upload_pipeline.py       # CUDA bootstrap + test-pipeline uploader
│   │   └── resolve_skip_ci.sh       # Shell helpers for AMD/Intel bootstrap
│   └── ci_mirror_hardwares.yml      # CUDA uploader presets (referenced by name only)
├── cuda/                            # Primary NVIDIA CUDA CI
│   ├── pipeline.yml                 # Bootstrap (document 1 + `---` + document 2)
│   ├── test-ready.yml               # L2
│   ├── test-merge.yml               # L3
│   ├── test-nightly.yml             # L4
│   ├── test-weekly.yml              # L5
│   └── rebase-pipeline.yml
├── npu/
│   ├── pipeline-npu.yml             # Bootstrap + image build + upload child pipelines
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
| Register CI-YAML in `skip_ci.py` | If you add a new whitelisted test pipeline file, update `L2_YAML_FILES`, `L3_YAML_FILES`, or `L45_YAML_FILES` in `.buildkite/common/scripts/skip_ci.py` so docs-only / CI-YAML-only PRs behave correctly. |

There are still **legacy copies** at `.buildkite/*.yaml` (without the `cuda/` prefix). Treat `.buildkite/cuda/*` as source of truth.

## Platform comparison

| Platform | Bootstrap entry | Test job files | Upload mechanism | Job hardware in YAML |
| -------- | ---------------- | -------------- | ---------------- | -------------------- |
| **CUDA** | `cuda/pipeline.yml` | `test-ready.yml`, `test-merge.yml`, `test-nightly.yml`, `test-weekly.yml` | `upload_pipeline.py --upload` (expands uploader-only keys) | `mirror_hardwares: <preset>` (string) |
| **NPU** | `npu/pipeline-npu.yml` | `test-npu-ready.yml`, `test-npu-nightly.yml` | `upload_pipeline.py --upload` | `mirror_hardwares: a2b3_npu_1` / `a2b3_npu_4` / `a3_npu_2` |
| **AMD** | `amd/scripts/bootstrap-amd-omni.sh` | `test-amd-ready.yml`, `test-amd-merge.yml` | Jinja (`test-template-amd-omni.j2`) → `pipeline upload` | `agent_pool` + `mirror_hardwares: [amdproduction]` (array, template filter) |
| **Intel** | `intel/scripts/bootstrap-intel-omni.sh` | `intel/pipeline-intel.yml` (steps inline) | Direct `pipeline upload` | Inline `agents.queue` on each step |

## CUDA configuration style

### Two-document bootstrap

`cuda/pipeline.yml` is parsed twice:

1. **Document 1** (before `---`): one CPU step runs `upload_pipeline.py --upload cuda/pipeline.yml`.
2. **Document 2** (after `---`): image build + upload steps for L2–L5 child pipelines.

Placeholders like `__UPLOAD_READY_IF__` are replaced by `upload_pipeline.py` from [skip-ci decision](./test_system_overview.md#diff-aware-buildkite-uploads-source_file_dependencies) logic.

### Test pipeline files

Each file starts with a shared `env:` block, then `steps:`.

**Conventions**

- **`depends_on`:** leaf jobs depend on the matching upload step key (`upload-ready-pipeline`, `upload-merge-pipeline`, etc.) so they run only after the child pipeline is uploaded and the CI image exists.
- **`group`:** use `:card_index_dividers:` groups for related jobs (Simple Test, Diffusion Test, E2E Test, …).
- **`label`:** `"<Area> · <Short name>"` (for example `Diffusion · Qwen Image Test`, `Omni · Qwen3-Omni Test`).
- **`commands`:** prefer `timeout … pytest …` with markers and `--run-level` aligned to the pipeline level (see Test Writing Guide).
- **`mirror_hardwares`:** uploader-only preset name (string). Do **not** set `agents` / `plugins` on the same step. Preset names are defined in `common/ci_mirror_hardwares.yml`.
- **`source_file_dependencies`:** uploader-only list of path prefixes; required for **E2E Test** leaf jobs in `test-ready.yml` and `test-merge.yml`. Stripped before upload. See [Test System Overview](./test_system_overview.md#diff-aware-buildkite-uploads-source_file_dependencies).

**Which file to edit**

| CI level | File | Typical PR trigger |
| -------- | ---- | ------------------ |
| L2 | `cuda/test-ready.yml` | `ready` label |
| L3 | `cuda/test-merge.yml` | `merge-test` label / main merge |
| L4 | `cuda/test-nightly.yml` | `nightly-test` label or `NIGHTLY=1` |
| L5 | `cuda/test-weekly.yml` | `weekly-test` label or `WEEKLY=1` |

### Adding a CUDA job

1. Pick the **level file** (ready / merge / nightly / weekly) from the table above.
2. Add a step (or nested step under the right **group**), usually under **E2E Test** for model-specific pytest.
3. Set `label`, `commands`, `mirror_hardwares`, and `depends_on: upload-<level>-pipeline`.
4. For **L2/L3 E2E**, add `source_file_dependencies` with pytest file + model code + deploy YAML prefixes.
5. Dry-run locally:

```bash
python3 .buildkite/common/scripts/upload_pipeline.py .buildkite/cuda/test-ready.yml
```

6. Run `pytest tests/buildkite/` if you changed uploader behavior or skip-ci mappings.

## NPU configuration style

NPU test pipelines use the same **`mirror_hardwares` preset** mechanism as CUDA. Presets live in `common/ci_mirror_hardwares.yml` (`a2b3_npu_1`, `a2b3_npu_4`, `a3_npu_2`) and expand to `agents`, top-level `image`, and `plugins` at upload time.

- **Bootstrap:** `npu/pipeline-npu.yml` builds NPU CI images, then runs `upload_pipeline.py --upload` for `test-npu-ready.yml` / `test-npu-nightly.yml`.
- **Child steps:** set `mirror_hardwares` only—do not duplicate `agents` / `image` / `plugins` on the same step.
- **`depends_on: upload-ready-pipeline`** (or nightly equivalent) ties jobs to bootstrap upload keys.

### Adding an NPU job

1. Edit `npu/test-npu-ready.yml` (L2) or `npu/test-npu-nightly.yml` (L4).
2. Add a step with `mirror_hardwares` pointing at an existing preset, or add a new preset under `common/ci_mirror_hardwares.yml` first.
3. Point `commands` at your pytest file and markers.
4. Dry-run: `python3 .buildkite/common/scripts/upload_pipeline.py .buildkite/npu/test-npu-ready.yml`
5. If you add a **new pipeline file**, register it in `skip_ci.py` under `L2_YAML_FILES` or `L45_YAML_FILES`.

## AMD configuration style

AMD separates **data** from **rendering**:

1. **`test-amd-ready.yml` / `test-amd-merge.yml`** — list of job definitions (label, `agent_pool`, `commands`, optional `mirror_hardwares: [amdproduction]`, `grade`, env).
2. **`test-template-amd-omni.j2`** — wraps those steps with docker image build (`amd-build`) and agent queue naming (`amd_<agent_pool>`).
3. **`bootstrap-amd-omni.sh`** — skip-ci, diff filtering, selects ready vs merge YAML, runs `minijinja-cli`, uploads generated `pipeline.yaml`.

Step fields in the YAML **data** files:

| Field | Purpose |
| ----- | ------- |
| `agent_pool` | Selects ROCm pool (for example `mi325_1`); template maps to `queue: amd_<pool>`. |
| `mirror_hardwares` | List tag for which mirror HW runs the step (for example `[amdproduction]`). |
| `depends_on` | Usually implicit via template → `amd-build`. |
| `commands` | Passed into `run-amd-test.sh` via `TEST_COMMAND`. |

### Adding an AMD job

1. Edit `amd/test-amd-ready.yml` (PR / L2) or `amd/test-amd-merge.yml` (main / L3).
2. Add a block matching neighbors: `label`, `agent_pool`, `mirror_hardwares`, `commands`, `grade` if needed.
3. Do **not** hand-edit generated `pipeline.yaml`; regenerate via bootstrap / Jinja.
4. Ensure `skip_ci.py` still lists the file if you split into a new YAML path.

## Intel configuration style

Intel is a **small static pipeline**:

- **`intel/pipeline-intel.yml`** — steps call shell scripts under `intel/scripts/`.
- **`bootstrap-intel-omni.sh`** — skip-ci then `pipeline upload pipeline-intel.yml`.

### Adding an Intel job

1. Add a step to `intel/pipeline-intel.yml` (or extend `run-xpu-test.sh` if the work fits an existing runner).
2. Set `agents.queue`, `env`, `timeout_in_minutes`, and `command`/`commands` consistently with existing steps.
3. Register `pipeline-intel.yml` in `L2_YAML_FILES` in `skip_ci.py` if the bootstrap path changes.

## Cross-cutting conventions

### Skip-ci and CI-YAML-only PRs

`.buildkite/common/scripts/skip_ci.py` classifies changed files. Whitelisted test pipeline paths are listed in `L2_YAML_FILES`, `L3_YAML_FILES`, and `L45_YAML_FILES`. When a PR touches **only** those files (plus docs/skip marks), bootstrap may skip uploading L2/L3 for unaffected platforms. Any new pipeline file must be added to the correct dict.

### Uploader-only keys (CUDA)

These keys are **removed** before Buildkite sees the YAML:

| Key | Purpose |
| --- | ------- |
| `mirror_hardwares` | Expand to `agents` + `plugins` from `ci_mirror_hardwares.yml` |
| `source_file_dependencies` | Omit step when PR diff does not touch listed prefixes |

Never rely on them at runtime inside the agent—they exist only for `upload_pipeline.py`.

### Labels and grouping

- Use **groups** for dashboard readability; keep **E2E Test** as the group name CUDA diff filtering expects for `--e2e` nightly runs on main.
- Prefix labels by model domain: **Omni ·**, **TTS ·**, **Diffusion ·**, **Simple ·**, etc., matching existing steps.

### Validation checklist

| Check | Command / location |
| ----- | ------------------ |
| CUDA render | `python3 .buildkite/common/scripts/upload_pipeline.py .buildkite/cuda/test-<level>.yml` |
| No leaked uploader keys | `… \| grep -E 'mirror_hardwares|source_file_dependencies'` → empty |
| Skip-ci / upload unit tests | `pytest tests/buildkite/` |
| Local job replay (CUDA L2+) | `tools/run_ready_jobs.sh`, `tools/run_merge_jobs.sh`, `tools/nightly/run_nightly_jobs.sh` (read YAML from `cuda/`) |

## Related documentation

- [Test System Overview — Diff-aware uploads](./test_system_overview.md#diff-aware-buildkite-uploads-source_file_dependencies)
- [Test Writing Guide](./test_writing_guide.md) — markers, directories, L1–L5 examples
- [Test Execution Guide](./test_execution_guide.md) — running CI-aligned jobs locally
- Implementation: `.buildkite/common/scripts/upload_pipeline.py`, `.buildkite/common/scripts/skip_ci.py`
