# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
E2E offline tests for Omni model with video input and audio output.

Abort / sleep-admission lives in ``test_qwen3_omni_colocate_async.py`` so it
does not overlap this module's OmniRunners.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest
from vllm import SamplingParams

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_image, generate_synthetic_video
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from vllm_omni.config.omni_config import (
    VllmOmniARStageConfig,
    VllmOmniGenerationStageConfig,
)
from vllm_omni.platforms import current_omni_platform

models = ["Qwen/Qwen3-Omni-30B-A3B-Instruct"]
thinker_only_models = ["Qwen/Qwen3-Omni-30B-A3B-Captioner"]

# Single CI deploy YAML; rocm/xpu deltas are picked automatically via the
# platforms: section. Only CUDA needs an extra enforce_eager tweak.
_CI_DEPLOY = get_deploy_config_path("ci/qwen3_omni_moe.yaml")
_PRODUCTION_DEPLOY = get_deploy_config_path("qwen3_omni_moe.yaml")


def get_cuda_graph_config():
    return modify_stage_config(
        _CI_DEPLOY,
        updates={
            "stages": {
                0: {"enforce_eager": True},
                1: {"enforce_eager": True},
            },
        },
    )


if current_omni_platform.is_xpu():
    stage_configs = [_CI_DEPLOY]
else:
    stage_configs = [get_cuda_graph_config()]

# Create parameter combinations for model and stage config
test_params = [(model, stage_config) for model in models for stage_config in stage_configs]
production_test_params = [(model, _PRODUCTION_DEPLOY) for model in models]
# we can use the same config for a model that only has thinker (i.e., does not
# enable audio output) because the resolver should figure out that it doesn't
# need the full pipeline based on the HF config.
thinker_test_params = [(model, stage_config) for model in thinker_only_models for stage_config in stage_configs]


def get_prompt(prompt_type: str):
    prompts = {
        "video": "Describe the video briefly.",
        "image": "Describe the image.",
        "audio": "Describe the audio.",
        "text": "What is the capital of the United Kingdom?",
    }
    return prompts[prompt_type]


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_text_to_text(omni_runner, omni_runner_handler) -> None:
    """Test generating text from text input"""
    request_config = {"prompts": get_prompt("text"), "modalities": ["text"]}

    # Test single completion
    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_text_to_audio(omni_runner, omni_runner_handler) -> None:
    """Test processing audio from text input"""
    request_config = {"prompts": get_prompt("text"), "modalities": ["audio"]}

    # Test single completion
    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_image_to_text(omni_runner, omni_runner_handler) -> None:
    """Test processing image, generating text output."""
    image = generate_synthetic_image(224, 224)["np_array"]

    request_config = {"prompts": get_prompt("image"), "images": image, "modalities": ["text"]}

    # Test single completion
    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_image_to_audio(omni_runner, omni_runner_handler) -> None:
    """Test processing image, generating audio output."""
    image = generate_synthetic_image(224, 224)["np_array"]

    request_config = {"prompts": get_prompt("image"), "images": image, "modalities": ["audio"]}

    # Test single completion
    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_video_to_text(omni_runner, omni_runner_handler) -> None:
    """Test processing video, generating text output."""
    video = generate_synthetic_video(224, 224, 300)["np_array"]

    request_config = {"prompts": get_prompt("video"), "videos": video, "modalities": ["text"]}

    # Test single completion
    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", production_test_params, indirect=True)
def test_structured_multistage_config_reaches_runtime(omni_runner, offline_client) -> None:
    """Deploy settings reach the materialized configs used by live stages."""
    engine = omni_runner.omni.engine

    # The resolved typed stages remain the Omni-owned source for topology and
    # placement, which are not fields of the terminal vLLM config.
    resolved_stages = engine.stage_configs
    assert len(resolved_stages) == 3
    thinker, talker, code2wav = resolved_stages
    assert isinstance(thinker, VllmOmniARStageConfig)
    assert isinstance(talker, VllmOmniARStageConfig)
    assert isinstance(code2wav, VllmOmniGenerationStageConfig)
    assert [stage.stage_id for stage in resolved_stages] == [0, 1, 2]
    assert [stage.model_stage for stage in resolved_stages] == ["thinker", "talker", "code2wav"]
    assert code2wav.final_output_type == "audio"
    assert thinker.runtime_config.devices == "0"
    assert talker.runtime_config.devices == code2wav.runtime_config.devices == "1"
    assert talker.connector_config.input_connectors == {"from_stage_0": "connector_of_shared_memory"}
    assert code2wav.connector_config.input_connectors == {"from_stage_1": "connector_of_shared_memory"}

    # Engine-owned deploy settings must survive the startup projection into
    # the actual VllmConfig objects retained by the stage pools.
    assert len(engine.stage_vllm_configs) == 3
    thinker_vllm, talker_vllm, code2wav_vllm = engine.stage_vllm_configs
    assert thinker_vllm is not None
    assert talker_vllm is not None
    assert code2wav_vllm is not None
    assert thinker_vllm.scheduler_config.max_num_seqs == 64
    assert talker_vllm.scheduler_config.max_num_seqs == 64
    assert code2wav_vllm.scheduler_config.max_num_seqs == 64
    assert thinker_vllm.scheduler_config.max_num_batched_tokens == 32768
    assert talker_vllm.scheduler_config.max_num_batched_tokens == 32768
    assert code2wav_vllm.scheduler_config.max_num_batched_tokens == 65536
    assert thinker_vllm.cache_config.gpu_memory_utilization == 0.9
    assert talker_vllm.cache_config.gpu_memory_utilization == 0.6
    assert code2wav_vllm.cache_config.gpu_memory_utilization == 0.1
    assert all(config.cache_config.enable_prefix_caching is False for config in engine.stage_vllm_configs)
    assert all(config.model_config.trust_remote_code is True for config in engine.stage_vllm_configs)
    assert code2wav_vllm.scheduler_config.enable_chunked_prefill is False
    assert code2wav_vllm.scheduler_config.async_scheduling is False

    # Sampling defaults are consumed from StageClient metadata, rather than
    # from either the resolver output or VllmConfig. These literals mirror
    # vllm_omni/deploy/qwen3_omni_moe.yaml exactly.
    expected_sampling = (
        {"temperature": 0.0, "max_tokens": 2048},
        {"temperature": 0.9, "top_k": 50, "max_tokens": 4096, "repetition_penalty": 1.05},
        {
            "temperature": 0.0,
            "top_p": 1.0,
            # Deploy YAML literal. The resolver check below compares it verbatim;
            # the runtime check normalizes it through SamplingParams, which turns
            # the greedy-sampling sentinel -1 into 0.
            "top_k": -1,
            "max_tokens": 65536,
            "repetition_penalty": 1.1,
        },
    )
    assert len(engine.default_sampling_params_list) == len(expected_sampling)
    for stage, runtime_params, expected in zip(
        resolved_stages, engine.default_sampling_params_list, expected_sampling, strict=True
    ):
        # The resolver keeps the deploy values verbatim; the pipeline only adds
        # its own ``detokenize`` / ``stop_token_ids`` constraints on top.
        assert expected.items() <= (stage.model_config.default_sampling_params or {}).items()
        # The runtime object is a SamplingParams built from those values, so it
        # carries vLLM's normalization: greedy sampling (temperature 0) stores
        # top_k=0, top_p=1.0 and min_p=0.0 even when the YAML spells the
        # disabled top_k as -1, and backend defaults such as ``detokenize=True``
        # appear. Compare against the same normalization, not the raw literals.
        normalized = SamplingParams(**expected)
        assert all(getattr(runtime_params, name) == getattr(normalized, name) for name in expected)

    if current_omni_platform.is_cuda():
        assert thinker_vllm.model_config.enforce_eager is False
        assert talker_vllm.model_config.enforce_eager is False

    # Exercise the materialized stages and their connector path with a real
    # request; the assertions above are not merely resolver-shape checks.
    offline_client.send_omni_request(
        {
            "prompts": "Answer with one short sentence: what is the capital of China?",
            "modalities": ["text"],
        }
    )


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_video_to_audio(omni_runner, offline_client) -> None:
    """Test processing video, generating audio output."""
    video = generate_synthetic_video(224, 224, 300)["np_array"]

    request_config = {"prompts": get_prompt("video"), "videos": video, "modalities": ["audio"]}

    # Test single completion
    offline_client.send_omni_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", thinker_test_params, indirect=True)
def test_thinker_only_model_request(omni_runner, offline_client) -> None:
    """Test that we can load and run a request through a model that only has the thinker stage."""
    request_config = {"prompts": "what color is the sky?", "modalities": ["text"]}

    # Test single completion
    offline_client.send_omni_request(request_config)
