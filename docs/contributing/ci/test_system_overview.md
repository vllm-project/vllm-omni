# Multi-Level Automated Testing System Overview

## Document Overview

This testing system aims to build a complete, efficient, and well-structured quality assurance framework for the development, integration, and release of model services. It draws on the concept of the test pyramid from modern software engineering, progressively expanding testing activities from basic code logic verification to complex end-to-end (E2E) functionality, performance, accuracy, and even long-term stability validation.

Through five levels (L1-L5) and common (Common) specifications, the system clarifies the testing objectives, scope, execution frequency, and required resources for different development stages (e.g., each commit, PR merge, daily build, pre-release). This ensures that models meet high standards for functionality, performance, and reliability across various deployment scenarios (online serving and offline inference).

<table>
  <thead>
    <tr>
      <th>Level</th>
      <th>Scope & Focus</th>
      <th>Model Coverage Strategy</th>
      <th>Feature Coverage Strategy</th>
      <th>Interface Coverage Strategy</th>
      <th>Tags</th>
      <th>Time Cost</th>
      <th>Test Dir</th>
      <th>Doc</th>
      <th>Frequency</th>
      <th>Hardware</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><strong>Common</strong></td>
      <td>Contribution Guideline & PR checklist</td>
      <td>/</td>
      <td>/</td>
      <td>/</td>
      <td>/</td>
      <td>/</td>
      <td>/</td>
      <td>.github/PULL_REQUEST_TEMPLATE.md <a href="../../.github/PULL_REQUEST_TEMPLATE.md"> PR Checklist</a></td>
      <td>/</td>
      <td>/</td>
    </tr>
    <tr>
      <td>CI Failure Description</td>
      <td>/</td>
      <td>/</td>
      <td>/</td>
      <td>/</td>
      <td>/</td>
      <td>/</td>
      <td><a href="../failures/"> CI Failures</a></td>
      <td>/</td>
      <td>/</td>
    </tr>
    <tr>
      <td><strong>L1</strong><br>(Unit & Logic)</td>
      <td>Unit tests for components like entrypoints, models</td>
      <td>/</td>
      <td>/</td>
      <td>/</td>
      <td><code>core_model and cpu</code></td>
      <td rowspan="2">&lt;15min</td>
      <td>/tests/{component_name}/test_xxx</td>
      <td>
        <a href="test_writing_guide.md#l1--l2-level-testing-unit-testing-and-basic-end-to-end-verification">L1 &amp; L2</a><br>
        Section 1 L1&amp;L2: Purpose, Test Content, Directory Location, Example
      </td>
      <td>PR with ready label (also can run locally)</td>
      <td>CPU</td>
    </tr>
    <tr>
      <td><strong>L2</strong><br>(E2E across models & GPU-required UT)</td>
      <td>Online (basic deployment scenarios):<br>dummy, normal inference function (output format, stream), some instance startup UT</td>
      <td>High-priority models + online basic scenarios; request success, non-empty output, format match (no Whisper/accuracy)</td>
      <td>High-priority features (using random lightweight models)</td>
      <td>High-priority interfaces (using random lightweight models)</td>
      <td><code>core_model and hardware_test(H100, L4, etc.) and omni/tts/diffusion</code></td>
      <td>
        <strong>Model tests:</strong><br>
        /tests/e2e/online_serving/test_{model_name}.py<br>
        <strong>Feature tests:</strong><br>
        /tests/{component_name}/test_xxx<br>
        <strong>Interface tests:</strong><br>
        /tests/entrypoints/test_xxx
      </td>
      <td>
        <a href="test_writing_guide.md#l1--l2-level-testing-unit-testing-and-basic-end-to-end-verification">L1 &amp; L2</a><br>
        L1&amp;L2: Purpose, Test Content, Directory Location, Example
      </td>
      <td>PR with ready label</td>
      <td>GPU</td>
    </tr>
    <tr>
      <td><strong>L3</strong><br>(Important Perf & Integration & Accuracy)</td>
      <td>Online & Offline (multiple deployment scenarios):<br>real model, normal inference function, normal accuracy</td>
      <td>High/medium-priority models + key online/offline scenarios; real weights, Whisper/similarity, preset voice gender, basic accuracy</td>
      <td>Medium-priority features (using random lightweight models)</td>
      <td>Medium-priority interfaces (using random lightweight models)</td>
      <td><code>advanced_model and hardware_test(H100, L4, etc.) and omni/tts/diffusion</code></td>
      <td>&lt;30min</td>
      <td>
        <strong>Model tests:</strong><br>
        /tests/e2e/online_serving/test_{model_name}.py<br>
        /tests/e2e/offline_inference/test_{model_name}.py<br>
        <strong>Feature tests:</strong><br>
        /tests/{component_name}/test_xxx<br>
        <strong>Interface tests:</strong><br>
        /tests/entrypoints/test_xxx
      </td>
      <td>
        <a href="test_writing_guide.md#l3-level-testing-core-integration-performance-and-accuracy-verification">L3</a><br>
        L3: Purpose, Test Content, Directory Location, Example
      </td>
      <td>PR Merged (Also run L1&L2 Tests)</td>
      <td>GPU</td>
    </tr>
    <tr>
      <td><strong>L4</strong><br>(Perf & Integration & Accuracy)</td>
      <td>Online: full functional scenarios + performance test + doc test + accuracy test</td>
      <td>High-priority models: function, performance, accuracy, and doc testing<br>Medium-priority models: function and doc testing</td>
      <td>Low-priority features (using real weights)</td>
      <td>Low-priority interfaces (using real weights)</td>
      <td><code>full_model and hardware_test(H100, L4, etc.) and omni/tts/diffusion</code></td>
      <td>&lt;3 hour</td>
      <td>
        <strong>Model tests:</strong><br>
        /tests/e2e/online_serving/test_{model_name}_expansion.py<br>
        <strong>Feature tests:</strong><br>
        /tests/{component_name}/test_xxx<br>
        <strong>Interface tests:</strong><br>
        /tests/entrypoints/test_xxx<br>
        <strong>Performance:</strong><br>
        /tests/dfx/perf/tests/test_qwen3_omni_*.json (Omni), test_tts.json (TTS),<br>
        test_voxcpm2.json, test_higgs_audio_v3.json, and<br>
        /tests/dfx/perf/tests/test_{diffusion_model}_vllm_omni.json (Diffusion)<br>
        <strong>Doc Test:</strong><br>
        tests/examples/online_serving/test_{model_name}.py<br>
        tests/examples/offline_inference/test_{model_name}.py<br>
        <strong>Accuracy Test:</strong><br>
        /tests/e2e/accuracy/test_{model_name}.py
      </td>
      <td>
        <a href="test_writing_guide.md#l4-level-testing-full-functionality-performance-and-documentation-testing">L4</a><br>
        L4: Purpose, Test Content, Directory Location, Example
      </td>
      <td>Nightly</td>
      <td>GPU</td>
    </tr>
    <tr>
      <td><strong>L5</strong><br>(Stability & Reliability)</td>
      <td>Online: long-term stability test + reliability test</td>
      <td>Long-term stability and reliability testing for high-priority models<br>Low-priority models: function and doc testing</td>
      <td>/</td>
      <td>Invalid-parameter validation for high-priority interfaces</td>
      <td><code>slow and hardware_test(H100, L4, etc.) and omni/tts/diffusion</code></td>
      <td> Depends on reality </td>
      <td>
        <strong>Stability:</strong><br>
        /tests/dfx/stability/tests/test_qwen3_omni.json<br>
        /tests/dfx/stability/tests/test_wan22.json<br>
        <strong>Reliability:</strong><br>
        tests/dfx/reliability/test_reliability_{model_key}.py<br>
        (e.g. <code>test_reliability_qwen3_omni.py</code>, <code>test_reliability_wan22.py</code>, <code>test_reliability_hunyuan_image.py</code>, <code>test_reliability_voxcpm2.py</code>)
      </td>
      <td>
        <a href="test_writing_guide.md#l5-level-testing-stability-and-reliability-testing">L5</a><br>
        L5: Purpose, Test Content, Directory Location, Example
      </td>
      <td>Weekly / Days before Release</td>
      <td>GPU</td>
    </tr>
  </tbody>
</table>

For per-level test authoring (directories, markers, examples), see [Test Writing Guide](./test_writing_guide.md).

## Common Specifications

Before entering specific testing levels, the project establishes two common specifications aimed at standardizing the development process and quickly locating issues.

1.  ***PR Checklist ([.github/PULL_REQUEST_TEMPLATE.md](../../.github/PULL_REQUEST_TEMPLATE.md))***: This template defines the self-check items that must be completed before submitting a code review (Pull Request). It ensures that each code change meets basic requirements such as code style, dependency updates, and documentation synchronization before entering the automated testing pipeline, serving as the first manual line of defense for quality assurance.
2.  ***CI Failure Explanation ([CI Failures](./failures.md))***: This document archives and explains common failure patterns in the Continuous Integration (CI) pipeline, error log interpretation, and preliminary troubleshooting steps. It helps developers and testers quickly diagnose the causes of automated test failures, improving problem-solving efficiency.

## Notes

### Diff-aware Buildkite uploads (`source_file_dependencies`)

L2 (`.buildkite/cuda/test-ready.yml`) and L3 (`.buildkite/cuda/test-merge.yml`) pipelines can **skip unrelated GPU jobs at upload time** based on the PR diff. This is implemented by `.buildkite/common/scripts/upload_pipeline.py`, which filters steps before calling `buildkite-agent pipeline upload`.

#### What `source_file_dependencies` is

- A **uploader-only** YAML key on a Buildkite step or group. **Buildkite does not understand it**; `upload_pipeline.py` always **removes** it from the YAML that is uploaded.
- A list of path **prefixes** (directories or individual files). If **any** changed file in the diff equals a prefix or starts with `prefix/`, the step is kept; otherwise the step (or entire group) is **omitted** from the uploaded pipeline.

#### When filtering runs

| Build context | Changed files used for matching |
| --- | --- |
| Pull request | `git diff --name-only origin/<base>...<BUILDKITE_COMMIT>` |
| `main` branch push | `git diff --name-only <commit>^..<commit>` |
| Other (e.g. local dry-run, non-PR branch) | Diff cannot be resolved → **no filtering**; all steps are uploaded and `source_file_dependencies` is still stripped |

Docs-only PRs are handled earlier in bootstrap (`.buildkite/cuda/pipeline.yml`) via skip-ci logic; `source_file_dependencies` applies only to the **uploaded** L2/L3 test pipelines.

#### Where it is configured

| Level | Pipeline file | Upload entry |
| --- | --- | --- |
| L2 | `.buildkite/cuda/test-ready.yml` | `upload_pipeline.py --upload .buildkite/cuda/test-ready.yml` (from `cuda/pipeline.yml`) |
| L3 | `.buildkite/cuda/test-merge.yml` | `upload_pipeline.py --upload .buildkite/cuda/test-merge.yml` (from `cuda/pipeline.yml`) |

Steps **without** `source_file_dependencies` are always uploaded (subject to the usual label conditions: `ready` for L2, `merge-test` for L3).

#### Current skip policy (L2 / L3)

To balance CI cost and coverage:

- **Always run** (no `source_file_dependencies`): baseline groups outside E2E Test—e.g. Simple Test, Diffusion unit tests, Engine/Model Executor, Distributed, Custom Pipeline, Entrypoints (L2), LoRA / Entrypoints (L3).
- **Diff-gated** (`source_file_dependencies` set): **every leaf job under the E2E Test group** in `test-ready.yml` and `test-merge.yml`, regardless of queue (`mithril-h100-pool`, `gpu_1_queue`, or `gpu_4_queue`). Each step lists the smallest set of prefixes that should trigger it—typically:
  - pytest file(s) exercised by the job (online and/or offline);
  - model code under `vllm_omni/model_executor/models/` or `vllm_omni/diffusion/models/`;
  - related `vllm_omni/model_executor/stage_input_processors/` and `vllm_omni/deploy/*.yaml` when applicable.

Adding a new E2E step: add `source_file_dependencies` on the leaf job with those prefixes. Prefer **per-step** deps rather than a broad group-level list unless every child shares the same paths.

#### YAML examples

H100 E2E (kubernetes / `mithril-h100-pool`):

```yaml
      - label: "Diffusion · Qwen Image Test"
        source_file_dependencies:
          - tests/e2e/online_serving/test_qwen_image.py
          - vllm_omni/diffusion/models/qwen_image/
        commands:
          - pytest -s -v tests/e2e/online_serving/test_qwen_image.py -m 'core_model' ...
        agents:
          queue: "mithril-h100-pool"
```

Docker E2E (`gpu_1_queue` / `gpu_4_queue`)—same key, same upload-time filtering:

```yaml
      - label: "TTS · Qwen3-TTS CustomVoice Test"
        source_file_dependencies:
          - tests/e2e/online_serving/test_qwen3_tts_customvoice.py
          - vllm_omni/model_executor/models/qwen3_tts/
          - vllm_omni/model_executor/stage_input_processors/qwen3_tts.py
          - vllm_omni/deploy/qwen3_tts.yaml
        commands:
          - pytest -s -v tests/e2e/online_serving/test_qwen3_tts_customvoice.py ...
        agents:
          queue: "gpu_4_queue"
```

A **group** may also define `source_file_dependencies`; nested steps inherit filtering as a unit—the whole group is dropped if no prefix matches.

#### Local dry-run

```bash
# Render filtered YAML to stdout (no upload)
python3 .buildkite/common/scripts/upload_pipeline.py .buildkite/cuda/test-ready.yml

# Confirm uploader-only keys are stripped
python3 .buildkite/common/scripts/upload_pipeline.py .buildkite/cuda/test-merge.yml | grep source_file_dependencies
# (no output expected)
```

On a PR build, Buildkite logs from `upload_pipeline.py` include lines such as `skip '…' (no changes under …)` for omitted steps.

#### Related

- [CI Settings](./ci_settings.md) — `.buildkite` layout, platform CI styles, adding Buildkite jobs
- Implementation: `.buildkite/common/scripts/upload_pipeline.py`
- L2/L3 diff skip does **not** replace label-based triggers (`ready`, `merge-test`); it only reduces which steps appear **after** the pipeline is already scheduled.

### Test helper environment variables

Some shared helpers under `tests/helpers/` honor optional environment variables for local debugging. These are **not** set in CI by default.

| Variable | Accepted values | Description |
| -------- | --------------- | ----------- |
| `VLLM_OMNI_KEEP_REQUEST_MEDIA` | `1`, `true`, `yes` (case-insensitive) | When enabled, temporary WAV files created by `tests.helpers.media.convert_audio_bytes_to_text` are **not** deleted when the pytest process exits. By default, each call writes a unique file under the system temp directory via `tempfile.mkstemp` and registers `atexit` cleanup. Use this when debugging audio output validation (Whisper transcription, keyword checks, text–audio similarity). The saved path is logged as `audio data is saved: <path>`. |

Example (Linux / macOS):

```bash
export VLLM_OMNI_KEEP_REQUEST_MEDIA=1
pytest -s -v tests/e2e/online_serving/test_qwen3_omni.py -k test_mix_to_text_audio
```

Example (Windows PowerShell):

```powershell
$env:VLLM_OMNI_KEEP_REQUEST_MEDIA = "1"
pytest -s -v tests/e2e/online_serving/test_qwen3_omni.py -k test_mix_to_text_audio
```

## Summary

This multi-level testing system achieves continuous, progressive validation of model service quality by tightly integrating testing activities with the development workflow (commit, review, merge, release). From rapid unit testing to comprehensive end-to-end testing, and further to in-depth performance, stability, and reliability verification, each level has clear objectives, collectively building a robust quality protection net. By following this system, teams can deliver high-quality, highly reliable model services more efficiently.
