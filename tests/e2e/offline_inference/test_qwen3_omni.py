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

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_video
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
# we can use the same config for a model that only has thinker (i.e., does not
# enable audio output) because the resolver should figure out that it doesn't
# need the full pipeline based on the HF config.
thinker_test_params = [(model, stage_config) for model in thinker_only_models for stage_config in stage_configs]


def get_question(prompt_type="video"):
    prompts = {
        "video": "Describe the video briefly.",
    }
    return prompts.get(prompt_type, prompts["video"])


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_structured_multistage_config_reaches_runtime(omni_runner) -> None:
    """User deploy settings survive resolution and affect the live stages."""
    stage_configs = omni_runner.omni.engine.stage_configs
    assert len(stage_configs) == 3
    thinker, talker, code2wav = stage_configs
    assert isinstance(thinker, VllmOmniARStageConfig)
    assert isinstance(talker, VllmOmniARStageConfig)
    assert isinstance(code2wav, VllmOmniGenerationStageConfig)
    assert [stage.stage_id for stage in stage_configs] == [0, 1, 2]
    assert [stage.model_stage for stage in stage_configs] == ["thinker", "talker", "code2wav"]
    assert code2wav.final_output_type == "audio"
    assert thinker.runtime_config.devices == "0"
    assert talker.runtime_config.devices == code2wav.runtime_config.devices == "1"
    assert thinker.scheduler_config.max_num_seqs == talker.scheduler_config.max_num_seqs == 64
    assert code2wav.scheduler_config.max_num_seqs == 64
    assert thinker.scheduler_config.max_num_batched_tokens == talker.scheduler_config.max_num_batched_tokens == 32768
    assert code2wav.scheduler_config.max_num_batched_tokens == 65536
    assert thinker.cache_config.gpu_memory_utilization == 0.9
    assert talker.cache_config.gpu_memory_utilization == 0.6
    assert code2wav.cache_config.gpu_memory_utilization == 0.1
    assert all(stage.cache_config.enable_prefix_caching is False for stage in stage_configs)
    assert all(stage.model_config.trust_remote_code is True for stage in stage_configs)
    assert thinker.model_config.default_sampling_params == {"temperature": 0.0, "max_tokens": 2048}
    assert talker.model_config.default_sampling_params == {
        "temperature": 0.9,
        "top_k": 50,
        "max_tokens": 4096,
        "repetition_penalty": 1.05,
    }
    assert code2wav.model_config.default_sampling_params == {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 65536,
        "repetition_penalty": 1.1,
    }
    assert talker.connector_config.input_connectors == {"from_stage_0": "connector_of_shared_memory"}
    assert code2wav.connector_config.input_connectors == {"from_stage_1": "connector_of_shared_memory"}

    if current_omni_platform.is_cuda():
        assert thinker.model_config.enforce_eager is True
        assert talker.model_config.enforce_eager is True


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_video_to_audio(omni_runner, offline_client) -> None:
    """Test processing video, generating audio output."""
    video = generate_synthetic_video(224, 224, 300)["np_array"]

    request_config = {"prompts": get_question(), "videos": video, "modalities": ["audio"]}

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
