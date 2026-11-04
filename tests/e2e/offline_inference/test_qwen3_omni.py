# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
E2E offline tests for Omni model with video input and audio output.

``test_structured_multistage_config_reaches_runtime`` and
``test_video_to_audio`` share one CI Instruct ``omni_runner``. Keep them
adjacent and on ``test_params``; a different deploy param cold-starts the
30B model again. Production YAML literals live in
``tests/config/test_qwen3_omni_moe_deploy.py``.

Abort / sleep-admission lives in ``test_qwen3_omni_colocate_async.py`` so it
does not overlap this module's OmniRunners.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest
from vllm import SamplingParams

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_video
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from vllm_omni.config.omni_config import (
    VllmOmniARStageConfig,
    VllmOmniGenerationStageConfig,
)
from vllm_omni.platforms import current_omni_platform

models = ["Qwen/Qwen3-Omni-30B-A3B-Instruct"]
thinker_only_models = ["Qwen/Qwen3-Omni-30B-A3B-Captioner", "Qwen/Qwen3-Omni-30B-A3B-Thinking"]

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

# Create parameter combinations for model and stage config.
# ``test_structured_multistage_config_reaches_runtime`` and ``test_video_to_audio``
# must keep this same param so the module-scoped runner is reused.
test_params = [(model, stage_config) for model in models for stage_config in stage_configs]
# we can use the same config for a model that only has thinker (i.e., does not
# enable audio output) because the resolver should figure out that it doesn't
# need the full pipeline based on the HF config.
thinker_test_params = [(model, stage_config) for model in thinker_only_models for stage_config in stage_configs]


# Keys compared after SamplingParams normalization. ``stop_token_ids`` is a
# pipeline constraint and is not part of that check.
_PROJECTED_SAMPLING_KEYS = (
    "temperature",
    "max_tokens",
    "top_k",
    "top_p",
    "repetition_penalty",
    "seed",
    "ignore_eos",
    "detokenize",
)


def get_question(prompt_type="video"):
    prompts = {
        "video": "Describe the video briefly.",
    }
    return prompts.get(prompt_type, prompts["video"])


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_structured_multistage_config_reaches_runtime(omni_runner, offline_client) -> None:
    """Deploy settings reach the materialized configs used by live stages.

    Shares the CI Instruct runner with ``test_video_to_audio``. Production
    literals (``max_num_seqs``, gpu memory utilization, ``enforce_eager``,
    sampling defaults) are pinned by the CPU resolver twin.
    """
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
    # the actual VllmConfig objects retained by the stage pools. Compare
    # against the resolved CI stage (the CUDA graph overlay sets
    # ``enforce_eager`` on stages 0 and 1; the CI overlay sets ``max_num_seqs``
    # and the talker utilization). Production literals are the CPU twin's job.
    assert len(engine.stage_vllm_configs) == 3
    thinker_vllm, talker_vllm, code2wav_vllm = engine.stage_vllm_configs
    assert thinker_vllm is not None
    assert talker_vllm is not None
    assert code2wav_vllm is not None
    for stage, vllm_config in zip(resolved_stages, engine.stage_vllm_configs, strict=True):
        assert vllm_config.scheduler_config.max_num_seqs == stage.scheduler_config.max_num_seqs
        assert vllm_config.scheduler_config.max_num_batched_tokens == stage.scheduler_config.max_num_batched_tokens
        assert vllm_config.cache_config.gpu_memory_utilization == stage.cache_config.gpu_memory_utilization
        assert vllm_config.model_config.enforce_eager is stage.model_config.enforce_eager
    assert thinker_vllm.scheduler_config.max_num_batched_tokens == 32768
    assert talker_vllm.scheduler_config.max_num_batched_tokens == 32768
    assert code2wav_vllm.scheduler_config.max_num_batched_tokens == 65536
    # Platform overrides are already reflected in the resolved stage config.
    # Verify that startup projection preserves that value in vLLM's runtime
    # config instead of restating each platform's policy here.
    assert thinker_vllm.cache_config.enable_prefix_caching is thinker.cache_config.enable_prefix_caching
    assert talker_vllm.cache_config.enable_prefix_caching is False
    assert code2wav_vllm.cache_config.enable_prefix_caching is False
    assert all(config.model_config.trust_remote_code is True for config in engine.stage_vllm_configs)
    assert code2wav_vllm.scheduler_config.enable_chunked_prefill is False
    assert code2wav_vllm.scheduler_config.async_scheduling is False

    # Sampling defaults are consumed from StageClient metadata. The resolver
    # keeps the deploy values (CI max_tokens overrides included); SamplingParams
    # normalizes them (greedy top_k -1 becomes 0). Production YAML literals are
    # pinned by the CPU twin.
    assert len(engine.default_sampling_params_list) == len(resolved_stages)
    for stage, runtime_params in zip(resolved_stages, engine.default_sampling_params_list, strict=True):
        raw = stage.model_config.default_sampling_params or {}
        expected = {name: raw[name] for name in _PROJECTED_SAMPLING_KEYS if name in raw}
        assert expected
        normalized = SamplingParams(**expected)
        assert all(getattr(runtime_params, name) == getattr(normalized, name) for name in expected)

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
