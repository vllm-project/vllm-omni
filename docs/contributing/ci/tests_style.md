# Test File Structure and Style Guide

To ensure project maintainability and sustainable development, we encourage contributors to submit test code (unit tests, system tests, or end-to-end tests) alongside their code changes. This document outlines the guidelines for organizing and naming test files.

## Checklist before submitting your test files

1. The file is saved in an appropriate place and the file name is clear.
2. The coding style follows the requirements outlined below.
3. All test functions have appropriate pytest markers.
4. For tests that need run in CI, please ensure it labeled as ``@pytest.mark.core_model` the test is configured under the `./buildkite/` folder.


## Test Types

For more details about our [Five Levels Tests design](../ci/CI_5levels.md).

### Unit Tests and System Tests
For unit tests and system tests, we strongly recommend placing test files in the same directory structure as the source code being tested, using the naming convention `test_*.py`.

### End-to-End (E2E) Tests for Models
End-to-end tests verify the complete functionality of a system or component. For our project, the E2E tests for different omni models are organized into two subdirectories:

- **`tests/e2e/offline_inference/`**: Tests for offline inference modes (e.g., Qwen3Omni offline inference)

- **`tests/e2e/online_serving/`**: Tests for online serving scenarios (e.g., API server tests)

**Example:** The test file for `vllm_omni/entrypoints/omni_llm.py` should be located at `tests/entrypoints/test_omni_llm.py`.

## Test Directory Structure

The ideal directory structure mirrors the source code organization. Legend: `✅` = test exists, `⬜` = suggested to add.

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
│   ├── omni.py                                 │   ├── test_omni.py                    ⬜  (E2E covered by e2e/offline, e2e/online)
│   ├── omni_llm.py                             │   ├── test_omni_llm.py                ✅
│   ├── omni_stage.py                            │   ├── test_omni_stage.py              ⬜  (partial in test_omni_stage_diffusion_config.py)
│   ├── omni_diffusion.py                       │   ├── test_omni_diffusion.py          ✅
│   ├── async_omni.py                            │   ├── test_async_omni.py              ✅ actually in e2e/online_serving/test_async_omni.py
│   ├── async_omni_diffusion.py                 │   ├── test_async_omni_diffusion_config.py  ✅
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
                                               │   ├── test_async_omni.py
                                               │   ├── test_qwen3_omni.py
                                               │   ├── test_qwen3_omni_expansion.py
                                               │   ├── test_image_gen_edit.py
                                               │   ├── test_images_generations_lora.py
                                               │   └── stage_configs/
                                               └── offline_inference/                  ✅
                                                   ├── test_qwen2_5_omni.py
                                                   ├── test_qwen3_omni.py
                                                   ├── test_bagel_text2img.py
                                                   ├── test_t2i_model.py
                                                   ├── test_t2v_model.py
                                                   ├── test_ovis_image.py
                                                   ├── test_zimage_tensor_parallel.py
                                                   ├── test_cache_dit.py
                                                   ├── test_teacache.py
                                                   ├── test_stable_audio_model.py
                                                   ├── test_diffusion_cpu_offload.py
                                                   ├── test_diffusion_layerwise_offload.py
                                                   ├── test_diffusion_lora.py
                                                   ├── test_sequence_parallel.py
                                                   └── stage_configs/
                                                       ├── qwen2_5_omni_ci.yaml
                                                       ├── qwen3_omni_ci.yaml
                                                       ├── bagel_*.yaml
                                                       └── npu/, rocm/, etc.
```



### Naming Conventions

- **Unit Tests**: Use `test_<module_name>.py` format. Example: `omni_llm.py` → `test_omni_llm.py`

- **E2E Tests**: Place in `tests/e2e/offline_inference/` or `tests/e2e/online_serving/` with descriptive names. Example: `tests/e2e/offline_inference/test_qwen3_omni.py`, `tests/e2e/offline_inference/test_diffusion_model.py`

- **Expansion Tests**

### Best Practices

1. **Mirror Source Structure**: Test directories should mirror the source code structure
2. **Test Type Indicators**: Use comments to indicate test types (UT for unit tests, E2E for end-to-end tests)
3. **Shared Resources**: Place shared test configurations (e.g., CI configs) in appropriate subdirectories
4. **Consistent Naming**: Follow the `test_*.py` naming convention consistently across all test files


## Test codes requirements

### Coding style

1. **File header**: Add SPDX license header to all test files
2. **Imports**: Pls don't use manual `sys.path` modifications, use standard imports instead.
3. **Test type differentiation**:

      - Unit tests: Maintain mock style
      - E2E tests for models: Consider using OmniRunner uniformly, avoid decorators

4. **Documentation**: Add docstrings to all test functions
5. **Environment variables**: Set uniformly in `conftest.py` or at the top of files
6. **Type annotations**: Add type annotations to all test function parameters
7. **Pytest Markers**: Add necessary markers like `@pytest.mark.core_model` and use `@hardware_test` to declare hardware requirements (check detailed in [Markers for Tests](../ci/tests_markers.md)).

### Template
#### E2E - Online serving

E2E Online tests for Qwen3-Omni model with mix input and audio+text output. Based on `tests/e2e/online_serving/test_qwen3_omni.py`.

```python
"""
E2E Online tests for Qwen3-Omni model with mix input and audio+text output.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

import threading
from pathlib import Path

import openai
import pytest

from tests.conftest import (
    OmniServer,
    convert_audio_to_text,
    cosine_similarity_text,
    dummy_messages_from_mix_data,
    generate_synthetic_video,
    merge_base64_and_convert_to_text,
)
from vllm_omni.platforms import current_omni_platform

# Edit: model name and stage config path
models = ["Qwen/Qwen3-Omni-30B-A3B-Instruct"]

#If you use the default configuration file, you can directly use the following address.
def get_default_config():
    return str(Path(__file__).parent.parent / "stage_configs" / "qwen3_omni_ci.yaml")

#If you need to modify the configuration file, you can use modify_stage_config.
def get_chunk_config():
    path = modify_stage_config(
        get_default_config(),
        updates={
            "async_chunk": True,
            "stage_args": {
                0: {
                    "engine_args.custom_process_next_stage_input_func": "vllm_omni.model_executor.stage_input_processors.qwen3_omni.thinker2talker_async_chunk"
                },
                1: {
                    "engine_args.custom_process_next_stage_input_func": "vllm_omni.model_executor.stage_input_processors.qwen3_omni.talker2code2wav_async_chunk"
                },
            },
        },
        deletes={"stage_args": {2: ["custom_process_input_func"]}},
    )
    return path

stage_configs = [get_default_config(), CHUNK_CONFIG_PATH]

test_params = [(model, stage_config) for model in models for stage_config in stage_configs]


#Please use this method to launch the online instance.
_omni_server_lock = threading.Lock()

@pytest.fixture(scope="module")
def omni_server(request):
    """Start vLLM-Omni server as a subprocess. Use module scope for multi-stage init (10-20+ min)."""
    with _omni_server_lock:
        model, stage_config_path = request.param
        with OmniServer(
            model,
            ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "120"],
        ) as server:
            yield server


@pytest.fixture
def client(omni_server):
    """OpenAI client for the running vLLM-Omni server."""
    return openai.OpenAI(
        base_url=f"http://{omni_server.host}:{omni_server.port}/v1",
        api_key="EMPTY",
    )

#Please use function definitions above the test function to define the prompts and other parameters you need.
def get_system_prompt():
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are Qwen, a virtual human developed by the Qwen Team, "
                    "Alibaba Group, capable of perceiving auditory and visual inputs, "
                    "as well as generating text and speech."
                ),
            }
        ],
    }

...

#Please define test case tags according to the instructions in the marker documentation.
@pytest.mark.core_model
@pytest.mark.omni
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_mix_to_text_audio_001(client: openai.OpenAI, omni_server, request) -> None:
    # PLEASE FOLLOW THESE TEMPLATE INSTRUCTIONS:
    # ============================================================================
    # TEMPLATE USAGE GUIDE:
    # 1. Copy this entire function as a starting point for multi-modal tests
    # 2. Update the test name to reflect your specific test scenario
    # 3. Modify input/output modalities as needed (see OPTIONS section below)
    # 4. Adjust assertions based on your expected outcomes
    # 5. Add custom validation logic for your specific use case
    # ============================================================================

    #Please list the relevant test points.
    """
    Test multi-modal input processing and text/audio output generation via OpenAI API.
    Deploy Setting: default yaml
    Input Modal: text + audio + video + image
    Output Modal: text + audio
    Input Setting: stream=True
    Datasets: single request
    """
    # SECTION 1: TEST SETUP AND INITIALIZATION
    # =========================================
    # INSTRUCTIONS: Initialize test variables and prepare test environment
    # MODIFY: Add any additional test setup required for your scenario
    e2e_list = list()
    # SECTION 2: TEST DATA GENERATION
    # ================================
    # INSTRUCTIONS: Generate or load test data for each input modality
    # MODIFY: Replace synthetic generators with your actual data sources
    # VIDEO DATA - Generate synthetic video for testing
    # FORMAT: data:video/mp4;base64,{base64_encoded_video}
    # PARAMETERS: width, height, duration_frames
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(224, 224, 300)['base64']}"
    # IMAGE DATA - Generate synthetic image for testing
    # FORMAT: data:image/jpeg;base64,{base64_encoded_image}
    # PARAMETERS: width, height
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(224, 224)['base64']}"
    # AUDIO DATA - Generate synthetic audio for testing
    # FORMAT: data:audio/wav;base64,{base64_encoded_audio}
    # PARAMETERS: duration_seconds, channels
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(5, 1)['base64']}"

    # SECTION 3: MESSAGE CONSTRUCTION
    # ================================
    # INSTRUCTIONS: Assemble the complete message payload for API request
    # MODIFY: Add/remove modalities or change prompt structure as needed

    # USAGE: Construct a message containing all input modalities
    # IMPORTANT: Ensure the message structure matches OpenAI API expectations
    # CUSTOMIZATION POINTS:
    #   - system_prompt: Controls the assistant's behavior
    #   - content_text: The user's text prompt/question
    #   - *_data_url: URLs for media content (video/image/audio)
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        video_data_url=video_data_url,
        image_data_url=image_data_url,
        audio_data_url=audio_data_url,
        content_text=get_prompt("mix"),
    )

    # SECTION 4: API REQUEST EXECUTION
    # =================================
    # INSTRUCTIONS: Make the API call and measure performance
    # MODIFY: Add timeout, retry logic, or additional parameters
    start_time = time.perf_counter()
    chat_completion = client.chat.completions.create(model=omni_server.model, messages=messages, stream=True)

    #Call using your preferred method and obtain the final audio and text outputs.
    ...

    # SECTION 5: OUTPUT VALIDATION
    # =============================
    # INSTRUCTIONS: Verify that outputs meet expected criteria
    # MODIFY: Adjust validation logic for your specific requirements

    # ASSERTION 1: E2E Validation
    # PURPOSE: Verify that the E2E latency is less than the baseline.
    current_e2e = time.perf_counter() - start_time
    print(f"the request e2e is: {current_e2e}")
    e2e_list.append(current_e2e)

    print(f"the avg e2e is: {sum(e2e_list) / len(e2e_list)}")



    # ASSERTION 2: Text Output Validation
    # PURPOSE: Verify that text output was generated with keyword content
    assert text_content is not None and len(text_content) >= 2, "No text output is generated"
    assert any(
        keyword in text_content.lower() for keyword in ["square", "quadrate", "sphere", "globe", "circle", "round"]
    ), "The output does not contain any of the keywords."


    # ASSERTION 3: Cross-Modal Consistency
    # PURPOSE: Verify text and audio outputs convey the same information
    # CUSTOMIZATION: Adjust similarity threshold (0.9) based on accuracy requirements
    assert audio_data is not None, "No audio output is generated"
    audio_content = merge_base64_and_convert_to_text(audio_data)
    print(f"text content is: {text_content}")
    print(f"audio content is: {audio_content}")
    similarity = cosine_similarity_text(audio_content.lower(), text_content.lower())
    print(f"similarity is: {similarity}")
    assert similarity > 0.9, "The audio content is not same as the text"
```


#### E2E - Offline inference
```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Offline E2E smoke test for an omni model (video → audio).
"""

import os
from pathlib import Path

import pytest
from vllm.assets.video import VideoAsset

from tests.utils import hardware_test
from ..multi_stages.conftest import OmniRunner

# Optional: set process start method for workers
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

models = ["{your model name}"] #Edit here to load your model
stage_configs = [str(Path(__file__).parent / "stage_configs" / {your model yaml})] #Edit here to load your model yaml

# Create parameter combinations for model and stage config
test_params = [(model, stage_config) for model in models for stage_config in stage_configs]

# function name: test_{input_modality}_to_{output_modality}
# modality candidate: text, image, audio, video, mixed_modalities
@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(
    res={"cuda": "L4", "rocm": "MI325", "npu": "A2"},
    num_cards=2,
)
@pytest.mark.parametrize("test_config", test_params)
def test_video_to_audio(omni_runner: type[OmniRunner], model: str) -> None:
    """Offline inference: video input, audio output."""
    model, stage_config_path = test_config
    with omni_runner(model, seed=42, stage_configs_path=stage_config_path) as runner:
        # Prepare inputs
        video = VideoAsset(name="sample", num_frames=4).np_ndarrays

        outputs = runner.generate_multimodal(
            prompts="Describe this video briefly.",
            videos=video,
        )

        # Minimal assertions: got outputs and at least one audio result
        assert outputs
        has_audio = any(o.final_output_type == "audio" for o in outputs)
        assert has_audio
```
