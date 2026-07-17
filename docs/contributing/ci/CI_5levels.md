# Multi-Level Automated Testing System Documentation

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
      <td>.github/PULL_REQUEST_TEMPLATE.md <a href="../tests_style/"> Test Style (PR Checklist)</a></td>
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
        <a href="#chapter-1-l1-l2-level-testing-unit-testing-and-basic-end-to-end-verification">Chapter 1</a><br>
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
        <a href="#chapter-1-l1-l2-level-testing-unit-testing-and-basic-end-to-end-verification">Chapter 1</a><br>
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
        <a href="#chapter-2-l3-level-testing-core-integration-performance-and-accuracy-verification">Chapter 2</a><br>
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
        <a href="#chapter-3-l4-level-testing-full-functionality-performance-and-documentation-testing">Chapter 3</a><br>
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
        <a href="#chapter-4-l5-level-testing-stability-and-reliability-testing">Chapter 4</a><br>
        L5: Purpose, Test Content, Directory Location, Example
      </td>
      <td>Weekly / Days before Release</td>
      <td>GPU</td>
    </tr>
  </tbody>
</table>


---
<details>
<summary> The folder structure for tests file based on the 5 levels design</summary>
Legend: `✅` = test exists, `⬜` = suggested to add.
```
vllm_omni/                                    tests/
├── config/                             →     ├── config/
│   ├── model.py                              │   └── test_model.py                    ⬜
│   └── lora.py                               │   └── test_lora.py                      ⬜
│
├── core/                               →     ├── core/
│   └── sched/                                 │   └── sched/
│       ├── omni_ar_scheduler.py               │       ├── test_omni_ar_scheduler.py    ⬜
│       ├── omni_generation_scheduler.py       │       ├── test_omni_generation_scheduler.py  ⬜
│       └── output.py                          │       └── test_output.py               ✅ currently in entrypoints/test_omni_new_request_data.py (tests output.OmniNewRequestData)
│
├── diffusion/                          →     ├── diffusion/
│   ├── diffusion_engine.py                    │   ├── test_diffusion_engine.py          ⬜
│   ├── attention/                             │   ├── attention/
│   │   ├── layer.py                            │   │   ├── test_attention_sp.py         ✅
│   │   └── backends/                           │   │   └── test_flash_attn.py           ✅
│   ├── distributed/                           │   ├── distributed/
│   │   └── ...                                 │   │   ├── test_comm.py                 ✅
│   │                                            │   │   ├── test_cfg_parallel.py        ✅
│   │                                            │   │   └── test_sp_plan_hooks.py       ✅
│   ├── lora/                                   │   ├── lora/
│   │   └── ...                                 │   │   ├── test_base_linear.py          ✅
│   │                                            │   │   └── test_lora_manager.py        ✅
│   ├── models/                                 │   ├── models/
│   │   ├── qwen_image/                         │   │   ├── qwen_image/                 (e2e coverage)
│   │   ├── ovis_image/                         │   │   ├── ovis_image/
│   │   │   └── ...                             │   │   │   └── test_ovis_image.py     ✅
│   │   ├── z_image/                            │   │   └── z_image/
│   │   └── ...                                 │   │       └── test_zimage_tp_constraints.py  ✅
│   └── worker/                                 │   └── worker/
│       ├── diffusion_worker.py                 │       └── test_diffusion_worker.py   ✅ file at diffusion/test_diffusion_worker.py
│       └── diffusion_model_runner.py            │
│
├── distributed/                         →     ├── distributed/
│   └── omni_connectors/                         │   └── omni_connectors/
│       ├── adapter.py                           │       ├── test_adapter_and_flow.py   ✅
│       ├── kv_transfer_manager.py               │       ├── test_basic_connectors.py   ✅
│       ├── connectors/                           │       ├── test_kv_flow.py             ✅
│       └── utils/                               │       └── test_omni_connector_configs.py  ✅
│
├── engine/                             →     ├── engine/
│   ├── input_processor.py                      │   ├── test_input_processor.py         ⬜  (no processor.py in source)
│   ├── output_processor.py                     │   └── test_output_processor.py         ⬜
│   └── arg_utils.py                            │   └── test_arg_utils.py               ⬜
│
├── entrypoints/                        →     ├── entrypoints/
│   ├── stage_utils.py                          │   ├── test_stage_utils.py            ✅
│   ├── cli/                                     │   ├── cli/                           (benchmarks/test_serve_cli.py covers CLI serve)
│   │   └── ...                                  │   │   └── test_*.py                  ⬜
│   └── openai/                                  │   └── openai_api/                    # maps to entrypoints/openai/
│       ├── api_server.py                        │       ├── test_api_server.py         ⬜  (e2e indirect coverage)
│       ├── serving_chat.py                       │       ├── test_serving_chat_sampling_params.py  ✅
│       ├── serving_speech.py                     │       ├── test_serving_speech.py     ✅
│       └── image_api_utils.py                   │       └── test_image_server.py      ✅
│
├── inputs/                             →     ├── inputs/
│   ├── data.py                                 │   ├── test_data.py                   ⬜
│   ├── parse.py                                │   ├── test_parse.py                 ⬜
│   └── preprocess.py                            │   └── test_preprocess.py            ✅ currently in entrypoints/test_omni_input_preprocessor.py
│
├── model_executor/                     →     ├── model_executor/
│   ├── layers/                                  │   ├── layers/
│   │   └── mrope.py                             │   │   └── test_mrope.py              ⬜
│   ├── model_loader/                            │   ├── model_loader/
│   │   └── weight_utils.py                      │   │   └── test_weight_utils.py      ⬜
│   ├── models/                                  │   ├── models/
│   │   ├── qwen2_5_omni/                         │   │   ├── qwen2_5_omni/
│   │   │   ├── qwen2_5_omni_thinker.py           │   │   │   ├── test_audio_length.py  ✅
│   │   │   ├── qwen2_5_omni_talker.py            │   │   │   ├── test_qwen2_5_omni_thinker.py  ⬜
│   │   │   └── qwen2_5_omni_token2wav.py         │   │   │   ├── test_qwen2_5_omni_talker.py  ⬜
│   │   └── qwen3_omni/                          │   │   │   └── test_qwen2_5_omni_token2wav.py  ⬜
│   │       └── ...                               │   │   └── qwen3_omni/
│   ├── stage_configs/                           │   │       └── test_*.py              ⬜
│   │   └── *.yaml                               │   └── stage_configs/                 (used by e2e, test_*.py can be added)  ⬜
│   └── stage_input_processors/                  │   └── stage_input_processors/
│       └── ...                                  │       └── test_*.py                 ⬜
│
├── sample/                             →     ├── sample/
│   └── __init__.py                             │   └── test_*.py                      ⬜
│
├── utils/                              →     ├── utils/
│   └── __init__.py                             │   └── test_*.py                       ⬜  (no platform_utils.py currently)
│
├── worker/                             →     ├── worker/
│   ├── gpu_ar_model_runner.py                  │   ├── test_gpu_ar_model_runner.py    ⬜
│   ├── gpu_ar_worker.py                        │   ├── test_gpu_ar_worker.py           ⬜
│   ├── gpu_generation_model_runner.py          │   ├── test_gpu_generation_model_runner.py  ✅
│   ├── gpu_generation_worker.py                │   ├── test_gpu_generation_worker.py  ⬜
│   ├── gpu_model_runner.py                     │   ├── test_omni_gpu_model_runner.py   ✅
│   └── mixins.py                               │   └── (npu under platforms/npu/worker/)  # not worker/npu/
│
├── platforms/                          →     (no tests/platforms/, e2e and stage_configs provide indirect coverage)
│   ├── cuda/
│   ├── npu/worker/                             # NPU worker here, not vllm_omni/worker/npu/
│   ├── rocm/
│   └── xpu/worker/
│
├── outputs.py                          →     test_outputs.py                         ✅ (at tests root)
├── (logger, patch, request, version)    →     (no corresponding unit test)
│
└── e2e (tests side only)               →     ├── e2e/
                                               ├── online_serving/                     ✅ non-empty
                                               │   ├── test_qwen2_5_omni.py
                                               │   ├── test_async_omni.py
                                               │   ├── test_qwen3_omni.py
                                               │   ├── test_qwen3_omni_expansion.py
                                               │   ├── test_mimo_audio.py
                                               │   └── test_images_generations_lora.py
                                               └── offline_inference/                  ✅
                                                   ├── test_qwen2_5_omni.py
                                                   ├── test_qwen3_omni.py
                                                   ├── test_bagel_text2img.py
                                                   ├── test_z_image.py
                                                   ├── test_wan22.py
                                                   ├── test_zimage_tensor_parallel.py
                                                   ├── test_cache_dit.py
                                                   ├── test_teacache.py
                                                   ├── test_stable_audio_expansion.py
                                                   ├── test_diffusion_cpu_offload.py
                                                   ├── test_diffusion_layerwise_offload.py
                                                   ├── test_diffusion_lora.py
                                                   ├── test_sequence_parallel.py
                                                   └── stage_configs/                  (legacy schema, still
                                                       ├── bagel_*.yaml                 present for unmigrated
                                                       └── npu/, rocm/, etc.            models)

# Migrated models (qwen3_omni_moe, qwen2_5_omni, qwen3_tts) live under
# vllm_omni/deploy/ instead — see docs/configuration/stage_configs.md.
```


</details>


## Common Specifications

Before entering specific testing levels, the project establishes two common specifications aimed at standardizing the development process and quickly locating issues.

1.  ***PR Checklist ([Tests Style](../ci/tests_style.md))***: This template defines the self-check items that must be completed before submitting a code review (Pull Request). It ensures that each code change meets basic requirements such as code style, dependency updates, and documentation synchronization before entering the automated testing pipeline, serving as the first manual line of defense for quality assurance.
2.  ***CI Failure Explanation ([CI Failures](../ci/failures.md))***: This document archives and explains common failure patterns in the Continuous Integration (CI) pipeline, error log interpretation, and preliminary troubleshooting steps. It helps developers and testers quickly diagnose the causes of automated test failures, improving problem-solving efficiency.

## Notes

### Diff-aware Buildkite uploads (`source_file_dependencies`)

L2 (`.buildkite/test-ready.yml`) and L3 (`.buildkite/test-merge.yml`) pipelines can **skip unrelated GPU jobs at upload time** based on the PR diff. This is implemented by `.buildkite/scripts/upload_pipeline.py`, which filters steps before calling `buildkite-agent pipeline upload`.

#### What `source_file_dependencies` is

- A **uploader-only** YAML key on a Buildkite step or group. **Buildkite does not understand it**; `upload_pipeline.py` always **removes** it from the YAML that is uploaded.
- A list of path **prefixes** (directories or individual files). If **any** changed file in the diff equals a prefix or starts with `prefix/`, the step is kept; otherwise the step (or entire group) is **omitted** from the uploaded pipeline.

#### When filtering runs

| Build context | Changed files used for matching |
| --- | --- |
| Pull request | `git diff --name-only origin/<base>...<BUILDKITE_COMMIT>` |
| `main` branch push | `git diff --name-only <commit>^..<commit>` |
| Other (e.g. local dry-run, non-PR branch) | Diff cannot be resolved → **no filtering**; all steps are uploaded and `source_file_dependencies` is still stripped |

Docs-only PRs are handled earlier in bootstrap (`pipeline.yml`) via skip-ci logic; `source_file_dependencies` applies only to the **uploaded** L2/L3 test pipelines.

#### Where it is configured

| Level | Pipeline file | Upload entry |
| --- | --- | --- |
| L2 | `.buildkite/test-ready.yml` | `upload_pipeline.py --upload .buildkite/test-ready.yml` (from `pipeline.yml`) |
| L3 | `.buildkite/test-merge.yml` | `upload_pipeline.py --upload .buildkite/test-merge.yml` (from `pipeline.yml`) |

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
python3 .buildkite/scripts/upload_pipeline.py .buildkite/test-ready.yml

# Confirm uploader-only keys are stripped
python3 .buildkite/scripts/upload_pipeline.py .buildkite/test-merge.yml | grep source_file_dependencies
# (no output expected)
```

On a PR build, Buildkite logs from `upload_pipeline.py` include lines such as `skip '…' (no changes under …)` for omitted steps.

#### Related

- Implementation: `.buildkite/scripts/upload_pipeline.py`
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

## Chapter 1: L1 & L2 Level Testing - Unit Testing and Basic End-to-End Verification

### 1.1 Testing Purpose

L1 and L2 level testing form the foundation of the quality assurance system. L1 level testing focuses on verifying the internal logic correctness of code units (e.g., functions, classes), ensuring each independent component behaves as designed.

L2 level testing builds upon L1 by introducing GPU resources and verifying that the end-to-end (E2E) process of the model in basic deployment scenarios is smooth. For example, it uses dummy models to confirm that core interfaces like the inference pipeline, output format, and streaming response work properly. The common goal of these two levels is to provide developers with rapid feedback, discovering and fixing issues early in the development cycle.



### 1.2 Testing Content and Scope

-   ***L1 (Unit & Logic Testing)***:
-   -   ***Scope***: Tests internal functions and methods of core components such as `entrypoints`, `models`.
    -   ***Focus***: Branch coverage, exception handling, algorithm logic correctness. Does not involve external dependencies or the complete service stack.
    -   ***Time Cost***: Execution time is controlled within ***15 minutes*** to ensure fast feedback.
-   ***L2 (Basic End-to-End Testing)***:
-   -   ***Scope***: Covers two basic deployment scenarios: `online` (serving) and `offline` (inference).
    -   ***Focus***: Uses `dummy` weights (via deploy YAML patching at `core_model`) or lightweight real models to verify that the entire chain from request input to result output works normally, including output data structure, streaming (stream) support, and **cheap payload checks** at `--run-level core_model` (see below). Also includes some unit tests that require launching independent service instances.
    -   ***L2 response validation (`core_model`)***: Implemented in `tests/helpers/assertions.py` and invoked by `OpenAIClientHandler` based on `--run-level`. At L2 we require **request success** plus minimal output sanity—not full accuracy:
        -   **Speech / TTS** (`assert_audio_speech_response`): decoded audio must be present and non-empty (or exceed `min_audio_bytes` when set in `request_config`); `response_format` must match the returned content-type (e.g. `wav`, `pcm`). Whisper transcript similarity, PCM HNR, and preset-voice gender checks run only at L3+.
        -   **Diffusion** (`assert_diffusion_response`): at least one non-empty image, video, or audio artifact. Resolution/frame-count parity and other deep checks run only at L3+.
        -   **Omni multimodal** (`assert_omni_response`): L2 asserts successful completion; keyword, transcript, and cross-modal similarity checks run only at L3+.
    -   ***Characteristic***: Requires ***GPU*** resources to perform model computations.

### 1.3 Test Directory and Execution Files

A clear directory structure is key to managing test cases efficiently.

-   ***L1 Test Directory***: `/tests/{component_name}/test_xxx.py`
-   -   Here, `{component_name}` corresponds to modules in the source code, such as `distributed`, `entrypoints`, etc., and `test_xxx.py` is the specific test file.
-   ***L2 Test Directory***:
-   -   Online Serving: `/tests/e2e/online_serving/test_{model_name}.py`
    -   Offline Inference: `/tests/e2e/offline_inference/test_{model_name}.py`

### 1.4 Execution Method and Example

-   ***Trigger Timing***: **`PR with ready label`**. That is, when a developer adds a "ready for review" or similar label to a PR on platforms like GitHub, L1 and L2 tests are automatically triggered.
-   ***Diff-aware step skipping***: On L2, **E2E Test** jobs may be omitted at pipeline upload when the PR diff does not touch their [`source_file_dependencies`](#diff-aware-buildkite-uploads-source_file_dependencies) prefixes (H100 and docker queues alike); non-E2E groups still always upload. See [Diff-aware Buildkite uploads](#diff-aware-buildkite-uploads-source_file_dependencies).
-   ***Execution Environment***: L1 uses ***CPU*** environment; L2 requires ***GPU*** environment.
-   ***Script Example***:

<details>
<summary> L1 Test Examples</summary>

Examples from `tests/model_executor/models/qwen2_5_omni/test_audio_length.py`
```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

def test_resolve_max_mel_frames_default():
    from vllm_omni.model_executor.models.qwen2_5_omni.audio_length import resolve_max_mel_frames

    assert resolve_max_mel_frames(None, default=30000) == 30000
    assert resolve_max_mel_frames(None, default=6000) == 6000


def test_resolve_max_mel_frames_explicit():
    from vllm_omni.model_executor.models.qwen2_5_omni.audio_length import resolve_max_mel_frames

    # Explicit argument always wins over default
    assert resolve_max_mel_frames(123, default=30000) == 123
    assert resolve_max_mel_frames(6000, default=30000) == 6000
    assert resolve_max_mel_frames(0, default=30000) == 0


@pytest.mark.parametrize("repeats", [2, 4])
@pytest.mark.parametrize("code_len", [0, 1, 32768])
@pytest.mark.parametrize("max_mel_frames", [None, -1, 0, 1, 6000, 30000])
def test_cap_and_align_mel_length_no_mismatch(repeats, code_len, max_mel_frames):
    """Guard that any max_mel_frames yields a mel length aligned to repeats, and
    consistent with the truncated code length (prevents concat mismatch).
    """
    from vllm_omni.model_executor.models.qwen2_5_omni.audio_length import cap_and_align_mel_length

    target_code_len, target_mel_len = cap_and_align_mel_length(
        code_len=code_len,
        repeats=repeats,
        max_mel_frames=max_mel_frames,
    )

    assert isinstance(target_code_len, int)
    assert isinstance(target_mel_len, int)

    if code_len == 0:
        assert target_code_len == 0
        assert target_mel_len == 0
        return

    assert target_code_len >= 1
    assert target_mel_len >= repeats
    assert target_mel_len % repeats == 0
    assert target_mel_len == target_code_len * repeats
    assert target_code_len <= code_len

    if max_mel_frames is not None and int(max_mel_frames) > 0 and int(max_mel_frames) >= repeats:
        assert target_mel_len <= int(max_mel_frames)
```
</details>

<details>
<summary> L2 Test Examples</summary>
You can refer to Test Examples in Chapter 2 to see example test cases that incorporate both L2 and L3 testing logic.
</details>

-   -   ***Run Command***:

    `pytest -s -v /tests/e2e/online_serving/test_{model_name}.py`
    `pytest -s -v -m 'core_model and cpu' --run-level=core_model`

## Chapter 2: L3 Level Testing - Core Integration, Performance, and Accuracy Verification

### 2.1 Testing Purpose

L3 level testing executes after code is merged into the main branch. Its core purpose is to verify the integration effect, key performance indicators, and output accuracy of ***real models*** in ***multiple deployment scenarios***

. It acts as the "quality gatekeeper" for the main branch, ensuring that no merge breaks the core capabilities of the model service. Testing needs to provide clear conclusions within a relatively short time (<30min), balancing test depth with feedback speed.



### 2.2 Testing Content and Scope

-   ***Deployment Scenarios***: Covers richer `online` and `offline` deployment configurations, which may include different hardware configurations, batch sizes, concurrency levels, etc.
-   ***Core Verification***:
-   1.  ***Inference Functionality***: Ensures real models can perform forward computation normally and return results.
    2.  ***Accuracy Compliance***: Verifies that the model's evaluation metrics (e.g., accuracy) meet the expected baseline, preventing code changes from introducing accuracy issues.
    3.  ***Important Performance***: Verifies whether performance (e.g., P99 latency, throughput) in core scenarios meets preset thresholds.
-   ***L3 response validation (`advanced_model`)***: At `--run-level advanced_model`, `tests/helpers/assertions.py` adds semantic checks on top of L2 payload gates (see Chapter 1):
    -   **Speech / TTS** (`assert_audio_speech_response`): Whisper transcript of returned audio vs `request_config["input"]` (cosine similarity &gt; 0.9 when input text is set); optional `min_audio_bytes` floor; PCM harmonic-to-noise ratio when `response_format` is `pcm`; **preset voice gender** via `_assert_preset_voice_gender_from_audio` when `voice` matches a known preset in `_PRESET_VOICE_GENDER_MAP` (pitch/F0-based classifier; skipped for unknown voices or `pcm` output).
    -   **Omni multimodal** (`assert_omni_response`): non-empty text/audio outputs; `key_words` in transcript or text; text–audio similarity / containment; preset **`speaker`** gender check (same helper as TTS).
    -   **Diffusion** (`assert_*_diffusion_response`): image/video dimension and frame-count parity with request parameters where configured.
    -   **Accuracy suites**: pixel/video similarity and metric baselines under `/tests/e2e/accuracy/` (separate from inline helper assertions).

### 2.3 Test Directory and Execution Files

-   ***Functional Testing***:
-   -   Online Serving: `/tests/e2e/online_serving/test_{model_name}_expansion.py`
    -   Offline Inference: `/tests/e2e/offline_inference/test_{model_name}_expansion.py`
    -   (Note: `_expansion.py` likely means it contains more comprehensive scenario cases compared to L2 tests).

### 2.4 Execution Method and Example

-   ***Trigger Timing***: **`PR Merged`**. Automatically triggered after code review is approved and merged into the main branch (typically via `merge-test` label on the PR before merge).
-   ***Diff-aware step skipping***: Same [`source_file_dependencies`](#diff-aware-buildkite-uploads-source_file_dependencies) mechanism as L2—**all E2E Test** leaf steps are diff-gated; other L3 groups always upload. See [Diff-aware Buildkite uploads](#diff-aware-buildkite-uploads-source_file_dependencies).
-   ***Execution Environment***: ***GPU*** servers.
-   ***Script Example***:

???+ example "Test Examples"

    **2.4.1 Mark Declaration Section**

    ```python
    @pytest.mark.advanced_model
    @pytest.mark.core_model
    @pytest.mark.parametrize("omni_server", test_params, indirect=True)
    ```

    **Explanation**:

    @pytest.mark.advanced_model: Marks the test as L3 merge level (`--run-level advanced_model`): real weights, Whisper/similarity/keyword checks, diffusion deep checks, and **preset voice/speaker gender** validation where applicable. @pytest.mark.full_model: Marks L4 nightly-only suites (e.g. `test_*_expansion.py`, doc examples).

    @pytest.mark.core_model: Marks the test as L1 or L2 level. At `--run-level core_model`, validation is limited to request success plus cheap payload checks (e.g. non-empty audio bytes, response format/content-type, non-empty diffusion outputs)—not Whisper, keyword, or accuracy gates. Deploy YAML may use `load_format: dummy` for fast PR feedback.

    @pytest.mark.parametrize: A parameterization decorator that allows abstracting test data into parameters, enabling reuse of the same test logic across different data configurations. indirect=True indicates that parameters will be passed to the fixture for processing.

    **Notes**: If you believe the test case only needs to execute basic run logic at the PR-level CI, you can mark it only with @pytest.mark.core_model. If you believe it only needs to execute deep validation at merge (L3), use @pytest.mark.advanced_model. For L4 nightly-only expansion and doc-example tests, use @pytest.mark.full_model with `--run-level full_model`. If the test case needs both basic run and deep validation, mark with @pytest.mark.core_model and the appropriate L3/L4 marker (`advanced_model` and/or `full_model`).

    **2.4.2 Test Function Definition and Documentation**

    ```python
    def test_mix_to_text_audio_001(omni_server, openai_client) -> None:
        """
        Test multi-modal input processing and text/audio output generation via OpenAI API.
        Deploy Setting: default yaml
        Input Modal: text + audio + video + image
        Output Modal: text + audio
        Input Setting: stream=True
        Datasets: single request
        """
    ```

    **Explanation**:

    **Function Naming Convention**: Uses the test_ prefix, describes the test scenario mix_to_text_audio, and the number 001 indicates the first test case for this scenario.

    **Parameter Explanation**:

    omni_server: Omni server instance obtained via fixture, containing model information and configuration.

    openai_client: Unified OpenAI client processing instance, encapsulating request sending and response validation logic.

    Docstring: Describes the test purpose, deployment settings, input/output modalities, streaming settings, and dataset type in detail, providing clear context for test maintenance.

    **2.4.3 Multimodal Data Preparation**

    ```python
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(224, 224, 300)['base64']}"
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(224, 224)['base64']}"
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(5, 1)['base64']}"
    ```

    **Explanation**:

    **Data Generation Functions**: Use the generate_synthetic_* series of functions to generate synthetic test data, avoiding reliance on external resources and ensuring test reproducibility and stability.

    **Parameter Explanation**:

    Video: width, height, duration_frames

    Image: width, height

    Audio: duration_seconds, channels

    **2.4.4 Request Configuration and Keyword Validation**

    ```python
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "key_words": {
            "audio": ["water", "cricket"],
            "video": ["sphere", "globe", "circle", "round"],
            "image": ["square", "quadrate"],
            "text": ["beijing"]
        },
    }
    ```

    **Explanation**:

    **Model Specification**: Uses omni_server.model to ensure the test aligns with the model configured on the server.

    **Keyword Validation Mechanism**: This is an innovative design of the template to address the specific needs of multimodal testing:

    Audio Keywords: Validate whether the generated text's description of audio content contains expected elements (e.g., "water" for water sounds, "cricket" for cricket sounds). If you provide multiple keywords, the validation is considered successful if at least one keyword is present.

    **Video Keywords**: Validate whether the generated text's description of video content contains expected elements. If you provide multiple keywords, the validation is considered successful if at least one keyword is present.

    Image Keywords: Validate whether the generated text's description of image content contains expected elements. If you provide multiple keywords, the validation is considered successful if at least one keyword is present.

    Text Keywords: Validate whether the generated text contains expected elements. If you provide multiple keywords, the validation is considered successful if at least one keyword is present.

    **2.4.5 Request Execution**

    ```python
    openai_client.send_omni_request(request_config, request_num=1)  # for omni-understanding models
    # or
    openai_client.send_diffusion_request(request_config, request_num=1)  # for diffusion models
    ```

    **Explanation**:

    **Unified Client**: Uses the OpenAIClientHandler instance to send requests. This client encapsulates error handling, retry mechanisms, and response validation logic.

    **Single Request**: The comment clearly states this is a single-request completion test. For concurrent testing, it can be extended to multiple requests using request_num = n.

    **Implicit Validation**: `send_omni_request`, `send_audio_speech_request`, and `send_diffusion_request` call `assert_*_response` helpers with the session `--run-level`. At `core_model`, checks are limited to success plus cheap payload sanity (non-empty audio/media, format/content-type, optional `min_audio_bytes`). At `advanced_model` and `full_model`, deep validation adds Whisper transcripts, keyword/similarity, diffusion dimension checks, PCM HNR, etc.

    **Audio output debugging**: Deep validation may transcribe returned audio via `convert_audio_bytes_to_text` (Whisper). If an audio keyword or text–audio similarity assertion fails, set `VLLM_OMNI_KEEP_REQUEST_MEDIA=1` before running pytest to keep the intermediate WAV files for inspection (see [Test helper environment variables](#test-helper-environment-variables)).

-   ***Run Command (L3 merge)***: `pytest -s -v /tests/e2e/online_serving/test_{model_name}.py -m advanced_model --run-level=advanced_model`

-   ***Run Command (L4 nightly expansion)***: `pytest -s -v /tests/e2e/online_serving/test_{model_name}_expansion.py -m full_model --run-level=full_model`

## Chapter 3: L4 Level Testing - Full Functionality, Performance, and Documentation Testing

### 3.1 Testing Purpose

L4 level testing is a comprehensive quality audit before a version release. It expands upon L3, executing ***full*** functional scenarios, conducting systematic ***performance stress tests***, and simultaneously verifying the correctness of accompanying ***example documentation***. Its purpose is to perform deep validation of the system during off-peak nighttime hours, providing quality trend reports for daytime development and data support for release decisions.



### 3.2 Testing Content and Scope

-   ***Full Functionality Testing***: Executes all test cases defined in `test_{model_name}_expansion.py`, covering all implemented features, positive flows, boundary conditions, and exception handling.
-   ***Performance Testing***: Uses `tests/dfx/perf/tests/test_qwen3_omni_*.json` (Omni), `test_tts.json` / `test_voxcpm2.json` / `test_higgs_audio_v3.json` (TTS), and diffusion configs `tests/dfx/perf/tests/test_*_vllm_omni.json` (passed to `run_benchmark.py` or `run_diffusion_benchmark.py` via `--test-config-file` in nightly **Perf Test** steps) to drive throughput, latency, and memory benchmarks. Each JSON **case** may declare an optional top-level **`mark`** array: exactly one ``hardware_marks`` object plus pytest marker name strings (`full_model`, `omni` / `tts` / `diffusion`, …). Runners attach those marks to each parametrized `(server, benchmark index)` pair so **local** bulk runs can filter with `-m` (for example `-m "full_model and H100 and diffusion"`). Nightly CI perf jobs select workloads by **`--test-config-file`**, not `-m`. Details are in the Performance Tests example below.
-   ***Documentation Testing***: Verifies whether the example code provided to users is runnable and its results match the description.

### 3.3 Test Directory and Execution Files

-   ***Functional Testing***: Same directories as L3.
-   ***Performance Test Configuration***: `tests/dfx/perf/tests/test_qwen3_omni_*.json`, `test_tts.json`, `test_voxcpm2.json`, `test_higgs_audio_v3.json`, and diffusion configs `tests/dfx/perf/tests/test_*_vllm_omni.json` (e.g. `test_qwen_image_vllm_omni.json`, `test_cosmos3_vllm_omni.json`). Optional per-case `mark` for local `-m` filtering is documented in Section 3.4 Performance Tests.
-   ***Documentation Example Tests***:
    -   `tests/examples/online_serving/test_{model_name}.py`
    -   `tests/examples/offline_inference/test_{model_name}.py`

### 3.4 Execution Method and Example

-   ***Trigger Timing***: **`Nightly`**, automatically executed every night.
-   ***Execution Environment***: ***GPU*** server clusters to meet the resource demands of performance testing.
-   ***Script Example***:

??? example "Test Examples: Documentation Example Tests"

    --8<-- "docs/contributing/ci/test_examples/l4_doc_example_tests.inc.md"

??? example "Test Examples: Performance Tests"

    --8<-- "docs/contributing/ci/test_examples/l4_performance_tests.inc.md"

??? example "Test Examples: Functionality Tests"

    --8<-- "docs/contributing/ci/test_examples/l4_functionality_tests.inc.md"

-   ***Run Command***: (Specific commands would depend on the performance testing tool and configuration defined in `nightly.json`).

## Chapter 4: L5 Level Testing - Stability and Reliability Testing

### 4.1 Testing Purpose

L5 level testing focuses on the performance of model services under ***long-running*** and ***abnormal fault*** scenarios. It aims to uncover deep-seated issues that only manifest under sustained pressure or extreme conditions, such as memory leaks, resource contention, gradual performance degradation, and lack of fault tolerance mechanisms. This is the final, yet crucial, line of defense for ensuring service high availability and production environment robustness.



### 4.2 Testing Content and Scope

-   ***Long-term Stability (Stability) Testing***: Uses JSON under `tests/dfx/stability/tests/` (for example `test_qwen3_omni.json` and `test_wan22.json`) to run the service under moderate load for an extended period (e.g., over 12 hours), monitoring whether metrics like memory/VRAM usage, response time, and throughput degrade over time, and whether the service process remains stable.
-   ***Reliability Testing***: Uses pytest suites under `tests/dfx/reliability/` to inject controlled faults against a **live** `vllm_omni serve` instance (same **`omni_server` / `omni_server_function`** fixture style as E2E). Current suites emphasize **GPU memory pressure** (CUDA sidecar “memory hog”), **worker / runtime process kill** (`SIGKILL` on `VLLM::Worker` for Qwen3-Omni, `multiprocessing.spawn` for Wan2.2 video workers, or `vLLM-Omni::` for HunyuanImage DiT workers), **large multimodal chat**, **`/v1/videos`**, **`/v1/images/generations`**, or **`/v1/audio/speech`** jobs under OOM, **`/health` → 503** and **fast-fail / non-hanging concurrent** requests after kill, and **OpenAI-style 5xx error contracts** (e.g. text vs text+audio under OOM). **Post-fault recovery** checks exist where enabled (some cases may be `skip` while issues are tracked). See the Reliability `<details>` block in Section 4.4 for file-level responsibilities and CI markers (`slow`, `hardware_test`, POSIX-only kill).

### 4.3 Test Directory and Execution Files

-   ***Stability Test Configuration***: `tests/dfx/stability/tests/test_qwen3_omni.json`, `tests/dfx/stability/tests/test_wan22.json` (one JSON per model / runner family)
-   ***Reliability Test Suite*** (`tests/dfx/reliability/`):
    -   `test_reliability_qwen3_omni.py` — Qwen3-Omni chat / multimodal reliability (GPU OOM, process kill, recovery, error contract under `--async-chunk` vs default).
    -   `test_reliability_wan22.py` — Wan2.2 I2V video API reliability (`/v1/videos` under OOM and process kill, recovery).
    -   `test_reliability_hunyuan_image.py` — HunyuanImage-3.0-Instruct DiT-only reliability (`/v1/images/generations` under OOM and process kill; deploy `hunyuan_image3_dit.yaml`, H100 × 4).
    -   `test_reliability_voxcpm2.py` — VoxCPM2 TTS reliability (`/v1/audio/speech` under OOM and process kill; deploy `voxcpm2.yaml`, L4 × 1).
    -   `helpers.py` — Shared primitives used by current suites: raw HTTP probes for `/v1/chat/completions` and `/health`, OpenAI-style error parsing, GPU OOM sidecar (`inject_gpu_oom` / `stop_gpu_oom_hogs`), and `pgrep`-based process-kill injector construction (`make_process_kill_fault_injector`).
    -   `conftest.py` — `fault_injector` and `omni_server_after_fault` / `omni_server_after_fault_function` fixtures to run a callable **after** the server is ready.
    -   `README.md` — Short local run commands for this directory.

### 4.4 Execution Method and Example

-   ***Trigger Timing***: **`Weekly`** (weekly) or **`Days before Release`** (several days before a major release). Due to long execution times, the frequency is lower.
-   ***Execution Environment***: ***GPU*** servers, requiring a stable and exclusive testing environment.
-   ***Script Example***:
<details>
<summary> Test Examples</summary>

When you want to add L5-level stability test cases, add or extend the appropriate JSON file under `tests/dfx/stability/tests/` (for example `test_qwen3_omni.json` for Omni bench traffic, or `test_wan22.json` for diffusion `/v1/videos` workloads). The following illustrates the Qwen3-Omni shape:

```json
{
    "test_name": "test_qwen3_omni_stability",
    "server_params": {
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "stage_config_name": "qwen3_omni.yaml"
    },
    "benchmark_params": [
        {
            "dataset_name": "random",
            "backend": "openai-chat-omni",
            "endpoint": "/v1/chat/completions",
            "duration_sec": 43200,
            "request_rate": 0.5,
            "num_prompts_per_batch": 20,
            "random_input_len": 2500,
            "random_output_len": 900,
            "ignore_eos": true,
            "percentile-metrics": "ttft,tpot,itl,e2el,audio_rtf,audio_ttfp,audio_duration"
        }
    ]
}
```

#### Parameter Explanation

***Overview***

| Field            | Required | Description                                                                 |
| ---------------- | -------- | --------------------------------------------------------------------------- |
| test_name        | Yes      | Unique identifier for the stability test case                               |
| server_params    | Yes      | Server-side configuration parameters (model, stage configuration, etc.)     |
| benchmark_params | Yes      | Stability benchmark running parameters (supports multiple configurations)   |

#### server_params Configuration

##### Basic Parameters

| Parameter         | Required | Example                            | Description                         |
| ----------------- | -------- | ---------------------------------- | ----------------------------------- |
| model             | Yes      | "Qwen/Qwen3-Omni-30B-A3B-Instruct" | Model name or path                  |
| stage_config_name | Yes      | "qwen3_omni.yaml"                  | Stage configuration file name       |

##### Dynamic Configuration (update/delete)

Supports incremental modifications based on the basic configuration:

| Operation | Description                          |
| --------- | ------------------------------------ |
| update    | Update or add configuration items    |
| delete    | Delete specified configuration items |

***Example***:
You can refer to Test Examples in Chapter 3.4

#### benchmark_params Configuration

For stability testing, the key parameters are:

-   **duration_sec**: Total duration (in seconds) during which benchmark traffic is sent. The stability benchmark will keep sending batches until this duration is reached.
-   **request_rate** / **max_concurrency**: Exactly one of them must be specified. They control how the traffic is generated for each batch:
    -   `request_rate`: Number of requests per second. The benchmark will send `num_prompts_per_batch` requests at the given rate.
    -   `max_concurrency`: Maximum number of concurrent requests. When this is used, `request_rate` is set to `inf` internally.
-   **num_prompts_per_batch**: Number of prompts sent in each batch. Multiple batches will be executed sequentially within `duration_sec`.

All other optional parameters follow the same rules as the in Chapter 3.4.

</details>

<details>
<summary> Reliability test suite (<code>tests/dfx/reliability</code>)</summary>

#### Purpose and relationship to stability

Reliability tests are **short fault-injection** integration runs (L5 **(b)** in `tests/dfx/reliability/README.md`). They complement **stability** JSON-driven long runs: instead of hours of steady traffic, they **perturb** the server (GPU OOM sidecar, fatal signals on selected processes) and check **failure mode** and **latency bounds** (e.g. chat or `/v1/videos` must not hang under concurrent fault-time load).

#### Directory layout

| Path | Responsibility |
| ---- | -------------- |
| `helpers.py` | Shared helpers used by current reliability suites: raw `POST`/`GET` probes (`/v1/chat/completions`, `/health`), OpenAI error parsing (`extract_openai_error_contract_from_bytes`), GPU OOM sidecar lifecycle (`inject_gpu_oom`, `stop_gpu_oom_hogs`), and process-kill injector builder (`make_process_kill_fault_injector`). |
| `conftest.py` | Pytest fixtures: indirect `fault_injector`, `omni_server_after_fault` / `omni_server_after_fault_function` (run injector after server is ready, then yield server). |
| `test_reliability_qwen3_omni.py` | Qwen3-Omni: OOM vs **text vs text+audio** error contract, large multimodal chat under OOM, concurrent pressure, **SIGKILL** on `VLLM::Worker`, `/health` → 503 + fast-fail + concurrent chat; optional OOM recovery scenario (may be skipped while tracked in issues). |
| `test_reliability_wan22.py` | Wan2.2 I2V: large `/v1/videos` under OOM, **SIGKILL** on `multiprocessing.spawn` chain, health / fast-fail / concurrent video requests; optional recovery test (may be skipped). |
| `test_reliability_hunyuan_image.py` | HunyuanImage DiT-only: large `/v1/images/generations` under OOM, **SIGKILL** on `vLLM-Omni::` workers and serve/tree targets, health / fast-fail / concurrent image requests; some OOM/recovery cases may be skipped while tracked in issues. |
| `test_reliability_voxcpm2.py` | VoxCPM2: `/v1/audio/speech` under OOM (error contract), **SIGKILL** on `VLLM::` workers and serve/tree targets, health / fast-fail / concurrent speech requests; some OOM cases may be skipped while tracked in issues. |
| `README.md` | Minimal run / collect examples. |

#### Parametrization and markers

- Each test module defines a **`RELIABILITY_SCENARIOS`** list (`test_name`, `server_params`: model, `stage_config_name` or diffusion `server_args`, etc.). **`create_reliability_omni_server_params()`** in `tests/dfx/conftest.py` resolves stage paths (including XPU substitutions where applicable) and builds **`OmniServerParams`** lists consumed by **`@pytest.mark.parametrize(..., indirect=True)`** on `omni_server` or `omni_server_function`.
- Cases are tagged **`@pytest.mark.slow`** for weekly / selective CI. GPU-heavy suites use **`@hardware_test(res={"cuda": "H100"}, num_cards=...)`** or **`@hardware_test(res={"cuda": "L4"}, num_cards=1)`** (Qwen3-Omni **2**× H100; Wan2.2 **1**× H100; HunyuanImage DiT **4**× H100; VoxCPM2 **1**× L4).
- **Process-kill** tests use **`@pytest.mark.skipif(os.name == "nt", ...)`** because injection uses POSIX **`pgrep` / `kill`**.

#### CI trigger

Weekly Buildkite (`.buildkite/test-weekly.yml`) runs one step per model suite (trigger: `WEEKLY=1` or PR label `weekly-test`), for example:

| Buildkite step | Test file | CI hardware |
| -------------- | --------- | ----------- |
| Reliability Test - qwen3-omni | `test_reliability_qwen3_omni.py` | H100 × 2 (`mithril-h100-pool`) |
| Reliability Test - wan22 | `test_reliability_wan22.py` | H100 × 2 (`mithril-h100-pool`) |
| Reliability Test - hunyuan-image | `test_reliability_hunyuan_image.py` | H100 × 4 (`mithril-h100-pool`) |
| Reliability Test - voxcpm2 | `test_reliability_voxcpm2.py` | L4 × 1 (`gpu_1_queue`) |

```bash
pytest -s -v tests/dfx/reliability/test_reliability_qwen3_omni.py -m "slow"
pytest -s -v tests/dfx/reliability/test_reliability_wan22.py -m "slow"
pytest -s -v tests/dfx/reliability/test_reliability_hunyuan_image.py -m "slow"
pytest -s -v tests/dfx/reliability/test_reliability_voxcpm2.py -m "slow"
```

#### Local commands

```bash
pytest --collect-only tests/dfx/reliability
pytest -s -v tests/dfx/reliability/test_reliability_qwen3_omni.py -m slow
pytest -s -v tests/dfx/reliability/test_reliability_wan22.py -m slow
pytest -s -v tests/dfx/reliability/test_reliability_hunyuan_image.py -m slow
pytest -s -v tests/dfx/reliability/test_reliability_voxcpm2.py -m slow
```

#### Adding a new model suite

1. Add `test_reliability_<model>.py` under `tests/dfx/reliability/`.
2. Define **`RELIABILITY_SCENARIOS`** and pass them through **`create_reliability_omni_server_params()`** with the correct deploy or e2e stage-config directory (same pattern as existing files).
3. Reuse **`helpers`** for OOM / kill / raw HTTP; prefer **`assert_fault_exception()`** and **`resolve_oom_device_spec()`** from `tests/dfx/conftest.py` for consistent device selection vs stage YAML.
4. Register **`slow`** (and **`hardware_test`** if needed); extend **`.buildkite/test-weekly.yml`** when the suite should run in weekly L5.

</details>

-   -   ***Stability***: `pytest -s -v tests/dfx/stability/scripts/test_stability_qwen3_omni.py` or `pytest -s -v tests/dfx/stability/scripts/test_stability_wan22.py` (or add `test_stability_<model>.py` alongside a matching JSON config)
    -   ***Reliability***: `pytest -s -v tests/dfx/reliability/test_reliability_<model>.py -m slow` (current suites: `qwen3_omni`, `wan22`, `hunyuan_image`, `voxcpm2`; add `test_reliability_<suite>.py` and a matching step in `.buildkite/test-weekly.yml` for new models)

## Summary

This multi-level testing system achieves continuous, progressive validation of model service quality by tightly integrating testing activities with the development workflow (commit, review, merge, release). From rapid unit testing to comprehensive end-to-end testing, and further to in-depth performance, stability, and reliability verification, each level has clear objectives, collectively building a robust quality protection net. By following this system, teams can deliver high-quality, highly reliable model services more efficiently.
