# Wire diffusion accuracy cases into CI

Use this guide when adding or extending accuracy coverage for an already
supported image or video model. Reuse the existing pytest and Buildkite paths;
an accuracy case does not require another model port or a new benchmark runner.
Run the command examples from the repository root in an installed test
environment. Examples containing `<...>` are templates: replace every placeholder
with the actual file, node, hardware, and artifact path before running them.

## Files and responsibilities

| Location | Change when needed |
| --- | --- |
| [tests/e2e/accuracy](../../../../tests/e2e/accuracy/) | Add a model-specific test module or extend its existing suite. Keep model parameters and reference runners near that suite. |
| [accuracy helpers](../../../../tests/e2e/accuracy/helpers.py) and [accuracy fixtures](../../../../tests/e2e/accuracy/conftest.py) | Reuse metrics, video probing, assets, and output directories. Extend these only for a shared capability or fixture. |
| [CUDA nightly](../../../../.buildkite/cuda/test-nightly.yml) or [CUDA weekly](../../../../.buildkite/cuda/test-weekly.yml) | Add the exact pytest files/nodes to an appropriate step, or add a scoped step with resources, timeout, prerequisites, and artifacts. |
| [source dependency registry](../../../../.buildkite/common/ci_source_file_dependencies.yml) | Extend the job's alias to include the model code, tests, reference scripts, and relevant assets/configuration. |
| [hardware presets](../../../../.buildkite/common/ci_mirror_hardwares.yml) | Reuse an existing preset; edit only if the required resource configuration is genuinely absent. |

The [vllm-omni-test skill](../../vllm-omni-test/SKILL.md) supplies general test
conventions. Accuracy is a separate CI pillar from functional expansion tests
and performance configurations. Adding a file under `tests/e2e/accuracy/` does
not automatically schedule it.

## Model directory and migration

For new accuracy suites, use `tests/e2e/accuracy/<slug>/test_<slug>.py` by default;
task/variant suffixes are allowed when multiple test modules are useful. Once a
model directory exists, keep that model's accuracy tests, reference runners,
README/contract, and dedicated helpers together there. Shared `helpers.py` and
`conftest.py` stay in their public locations; model-only fixtures may live in a
local `conftest.py`. This rule covers the accuracy suite, not the model's separate
functional tests. Existing examples below retain their current repository paths.

When moving an existing case into its model directory:

- Update runner lookup, imports/fixtures, `__file__`-relative repository roots,
  `PYTHONPATH`, input/artifact paths, README commands, and CI file/node selectors.
- Replace scattered test/reference dependencies with the model directory prefix;
  retain model source, shared helpers, configuration, and external asset dependencies.
- Check reference imports/path resolution and pytest collection with the exact CI
  selector, preserving intended node coverage/marks; check rendering, dependency
  routing, and artifact retention at the new paths.

A pure move can reuse recorded numerical evidence only when effective weights,
inputs, environments, and numerical settings are unchanged. Do not migrate
unrelated models or rerun GPU calibration solely because paths moved.

## Select tests and real checkpoints deliberately

- Nightly accuracy cases normally carry `pytest.mark.full_model` and
  `pytest.mark.diffusion`. Existing expensive weekly cases may use
  `pytest.mark.slow` instead of `full_model`; match their weekly selector rather
  than adding unrelated smoke marks.
- Apply SKU and card-count marks through
  [hardware_test/hardware_marks](../../../../tests/helpers/mark.py), for example
  `@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=1)`. Do not write
  `pytest.mark.H100` directly. The helper adds both allowed SKU marks and
  `cards_1`, which explains the existing `H100 and B200` selectors.
- `-m` selects marked items; `--run-level` controls fixture/runtime behavior.
  The [run-level option](../../../../tests/helpers/fixtures/pytest_run_args.py)
  defaults to `core_model`. The shared
  [runtime fixtures](../../../../tests/helpers/runtime.py) can substitute tiny
  diffusion weights at that level while preserving the served model name.
  Use `--run-level full_model` for real-model accuracy and inspect the actual
  model path and loader logs. A `full_model` mark alone does not select real
  weights, and direct `OmniServer` or reference runners have their own loading
  contracts.
- Hardware marks are routing metadata, not complete prerequisite guards.
  Verify the actual device, card count, backend, and memory. Do not infer
  availability or support from the selector.
- Prefer explicit files/nodes over a broad accuracy-directory sweep. Read
  prerequisite generation cases before selecting a comparison node. Retain
  their marks in the selector, or make a new comparison own its inputs.

For example, [the Wan2.2 I2V suite](../../../../tests/e2e/accuracy/wan22_i2v/test_wan22_i2v_video_similarity.py)
has a one-card Diffusers generation case and two-card online/comparison cases.
Its comparison reads artifacts from the generation cases and skips if they are
missing. Selecting only the comparison node, or filtering out `cards_1`, does
not establish fresh reference accuracy. Prevent reuse of outputs from an older
run; preserve old artifacts before refreshing the suite's output directory.

## Choose nightly or weekly

Use the matching maintained lane and its resource budget:

- Nightly image accuracy runs under **Diffusion X2I(&A&T)**. Current examples
  include Qwen-Image, JoyAI-Image-Edit, and HunyuanImage3-DIT.
- Nightly video accuracy runs under **Diffusion X2V**. Current examples include
  Wan2.2 I2V, MiniMax H3 I2VA/Ref2VA, and HunyuanVideo-1.5.
- Weekly carries expensive or non-critical cases. Current examples include
  Qwen-Image-Edit/Layered, LTX/LTX-2.5, and SANA-Video. Follow their `slow`
  selection and explicit hardware preset. These accuracy examples are in the
  weekly **E2E Test** group, gated by `NON_CRITICAL=1`; `weekly-test` or `WEEKLY=1`
  alone does not activate that group. Not every accuracy case belongs in nightly,
  and the output modality does not imply a new pytest marker.

Read the current step rather than copying its label alone. Align card count,
timeout, checkpoint access, reference dependencies, and scheduled frequency.
The [bootstrap upload steps](../../../../.buildkite/cuda/bootstrap-upload-steps.yml)
control lane activation: scheduled nightly runs use `NIGHTLY=1`, while PR
selection uses the corresponding CI labels and diff-aware upload. Check bootstrap,
group, and step conditions separately for the intended build context; uploading
a YAML file or retaining a rendered step does not prove its `if` is satisfied.

## Wire the dependency alias and step together

The [dependency registry](../../../../.buildkite/common/ci_source_file_dependencies.yml)
uses YAML anchors for model business-code paths and aliases such as
`diffusion_qwen_image_accuracy` or `diffusion_wan22_accuracy` for individual
jobs. Extend the existing alias when extending that job. If adding a new job,
create its corresponding alias and reference it with `source_file_dependencies`.
Include every test/reference script the step runs and the inputs whose changes
should retrigger it. The uploader does not inspect `commands` to infer these
dependencies. Include relevant shared accuracy helpers/fixtures and used
`tests/helpers/` paths in the job's alias, directly or through an anchor.
`source_filter_fallback` only keeps all jobs when no job dependency matches the
diff; it does not guarantee a particular job runs on a shared-helper change
combined with another model's change. Do not attach that fallback as a job alias.
For a model directory, use `tests/e2e/accuracy/<slug>/` as the prefix
covering its tests, runners, and dedicated support files. Keep the other source
and shared dependencies alongside it; that directory is not the entire alias.

This is an adaptable fragment for a new **one-card nightly image** step inside
the existing X2I group. Replace the label, alias, file, artifact subdirectory,
hardware selector, and timeout; create/extend the alias separately. For video
or a prerequisite chain, use the appropriate file list and card selector.

```yaml
- label: ":full_moon: Diffusion X2I(&A&T) · <Model> Accuracy Test"
  source_file_dependencies: diffusion_<slug>_accuracy
  timeout_in_minutes: 180
  artifact_paths:
    - tests/e2e/accuracy/artifacts/<artifact-subdirectory>/**/*
  commands:
    - >-
      pytest -s -v tests/e2e/accuracy/<slug>/test_<slug>.py
      -m "full_model and diffusion and H100 and B200 and cards_1"
      --run-level full_model
      --junitxml=tests/e2e/accuracy/artifacts/<artifact-subdirectory>/pytest.xml
```

Keep the containing group's activation and `depends_on` contract. For a new
top-level arrangement, follow the existing lane and
[CI settings](../../../../docs/contributing/ci/ci_settings.md).
Choose artifact globs from the test's actual output locations; video suites
often write under a model-local `result/`, rather than the shared `artifacts/`.

The [uploader](../../../../.buildkite/common/scripts/upload_pipeline.py) infers
hardware from positive SKU/card marks when `mirror_hardwares` is omitted.
Several positive card counts select the largest matching resource preset.
An explicit preset such as `h100_2` bypasses pytest-mark inference. During
inference, B200 mirroring requires a matching `B200` selector. An explicit
`b200_*` preset selects resources without that inference; explicit H100-only
presets are omitted from a B200 mirror. Keep the pytest selector aligned with
the test's marks and intended hardware. Do not advertise a B200 row without
its scoped accuracy evidence.

## Check collection, routing, and execution separately

Before model loading, complete or reuse a still-matching
[reference/environment preflight](reference-contract.md#preflight-before-model-loading)
and use the same launch environments in local and CI commands. Collection-only
checks need the test stack's imports/path setup and any collection-time probes;
they do not require loading weights or running the full comparison preflight.

The following existing image example runs the original Qwen case, rather than
both checkpoints in its file. The environment override must name a complete,
frozen local snapshot accessible to both runtimes; replace the path.

```bash
export QWEN_IMAGE_MODEL=/path/to/frozen/qwen-image-snapshot
pytest -sv tests/e2e/accuracy/test_qwen_image.py::test_qwen_image_matches_diffusers \
  -m "full_model and diffusion and H100 and B200 and cards_1" \
  --run-level full_model
```

This existing video example retains the Wan suite's reference and online
generation prerequisites. Use a machine with the required two-card deployment;
the selector also includes its one-card reference case.

```bash
pytest -sv tests/e2e/accuracy/wan22_i2v/test_wan22_i2v_video_similarity.py \
  -m "full_model and diffusion and H100 and B200 and (cards_1 or cards_2)" \
  --run-level full_model
```

For a new case, replace the file, node, and selector in this template. Collection
imports the test stack and can probe hardware; it still requires its import
dependencies, even though it does not generate media.

```bash
pytest --collect-only -q tests/e2e/accuracy/<slug>/test_<slug>.py::<test_node> \
  -m "<the exact CI marker expression>" --run-level full_model
python tools/pre_commit/check_test_marks.py tests/e2e/accuracy/<slug>/test_<slug>.py
```

Inspect the expected collected node IDs, including parametrized deployments.
The mark check validates marker conventions; it does not validate the CI file
list, dependency alias, or model accuracy.

The existing uploader supports rendering without uploading. In a no-install or
no-network environment, first ensure PyYAML is installed and, for PR context,
the base ref `origin/<base-branch>` exists locally. The script can auto-install
PyYAML and its Git-context resolver can fetch a missing PR base, even with
`--all`. If those prerequisites are absent, use static inspection and report
rendering/filter execution unvalidated rather than triggering those operations.

```bash
python .buildkite/common/scripts/upload_pipeline.py --all .buildkite/cuda/test-nightly.yml
python .buildkite/common/scripts/upload_pipeline.py .buildkite/cuda/test-nightly.yml
```

The first command checks the full rendering; the second applies the current
Git/CI context's diff filter. Inspect the retained step, resource preset, file
list, and artifacts; the renderer preserves `if` expressions without evaluating
them, so check activation separately. Also resolve the step's dependency alias
and check that
the changed model/test/helper/input paths match its prefixes. An `--all` render
does not exercise that filter; when diff context is unavailable, record a static
alias/path check and mark filter execution unvalidated. For weekly work, replace
the YAML path with `.buildkite/cuda/test-weekly.yml`. These commands omit `--upload`; rendering is
neither a Buildkite run nor proof that GPU tests passed.

## Completion and rerun scope

Default to one representative real-weight comparison. Accuracy CI validation
is complete when all three conditions hold for the requested scope:

- The intended node actually ran and passed its numerical assertions against
  a user-selected calibrated/reused in-scope gate; skips, deselection and
  collection are not passes. Meeting a proposed gate is provisional until selected.
- Reference/candidate media, effective request and provenance, metrics/thresholds,
  commands, pytest results and useful logs are saved. Configure actual CI artifact
  paths for success and failure; preserve pytest's exit status when uploading.
- The exact CI selector collects the intended node and its prerequisites;
  rendering confirms hardware/file/artifact settings, the bootstrap/group/step
  activation conditions match the intended build context, and the source
  dependency alias/filter has been checked against relevant changed paths.

Deliver the [threshold recommendation and headroom evidence](reference-contract.md#threshold-recommendation-and-user-selection)
with the results, then stop expanding or repeating validation. Pending user
selection, hand off the completed implementation/evidence and mark that decision
outstanding. If hardware, calibration or CI context is unavailable, deliver
feasible changes and commands, explicitly list unvalidated conditions, and do
not claim complete validation.

For a later change, rerun only the affected checks:

| Change | Minimum follow-up |
|---|---|
| Skill/documentation only | Format and link checks; no accuracy rerun |
| Model-directory move only | Migration checks above; reuse numerical evidence if the effective execution contract is unchanged |
| CI routing/markers/artifact paths only | Collection/rendering, dependency selection and retention checks; reuse numerical evidence if the execution contract is unchanged |
| Threshold-only user selection | Re-evaluate saved complete metrics against the selected bound; rerun if required measurements are missing or the contract changed |
| Reference environment, weights, inputs or numerical settings | Affected preflight and comparison; repeatability/calibration if the gate's scope changed |
| Model, test or metric implementation | Smallest affected test nodes; diagnose failures in the reference guide's order |

New failures or evidence can justify broader checks. Preserve completed evidence
instead of restarting the full workflow after every edit.
