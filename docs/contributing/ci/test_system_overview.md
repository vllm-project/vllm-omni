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
      <td><a href="https://github.com/vllm-project/vllm-omni/blob/main/.github/PULL_REQUEST_TEMPLATE.md">PR Checklist</a></td>
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
      <td><a href="./failures.md">CI Failures</a></td>
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

1.  ***PR Checklist ([`.github/PULL_REQUEST_TEMPLATE.md`](https://github.com/vllm-project/vllm-omni/blob/main/.github/PULL_REQUEST_TEMPLATE.md))***: This template defines the self-check items that must be completed before submitting a code review (Pull Request). It ensures that each code change meets basic requirements such as code style, dependency updates, and documentation synchronization before entering the automated testing pipeline, serving as the first manual line of defense for quality assurance.
2.  ***CI Failure Explanation ([CI Failures](./failures.md))***: This document archives and explains common failure patterns in the Continuous Integration (CI) pipeline, error log interpretation, and preliminary troubleshooting steps. It helps developers and testers quickly diagnose the causes of automated test failures, improving problem-solving efficiency.

## Notes

### L2 / L3 diff-aware CI (CUDA)

On CUDA **L2** and **L3**, **E2E Test** Buildkite jobs may be omitted at pipeline upload when the PR diff does not touch their path prefixes; other groups still always upload. Full two-layer mechanics and YAML examples: [CI Settings — Diff-aware CI](./ci_settings.md#diff-aware-ci) ([step filtering](./ci_settings.md#step-filtering)).

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
