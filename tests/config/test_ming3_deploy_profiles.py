# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ming deployment choices survive legacy and structured config resolution."""

from pathlib import Path

import pytest

from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.pipeline_registry import resolve_pipeline_config
from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy, resolve_deploy_yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DEPLOY = Path(__file__).parents[2] / "vllm_omni" / "deploy"
_PROFILES = [
    f"ming3_audio_{size}{variant}.yaml" for size in ("flash", "tiny") for variant in ("", "_asr", "_thinker_only")
]


@pytest.mark.parametrize("profile", _PROFILES)
def test_ming_profile_context_and_topology_reach_both_config_builders(profile):
    deploy = load_deploy_config(_DEPLOY / profile)
    pipeline = resolve_pipeline_config(deploy.pipeline)
    assert pipeline is not None
    legacy = merge_pipeline_deploy(pipeline, deploy)
    structured = VllmOmniConfig.from_pipeline_config(pipeline, user_deploy_config=deploy)
    thinker = structured.stage_by_id(0)
    assert thinker.scheduler_config.max_model_len == legacy[0].yaml_engine_args["max_model_len"] == 131072
    assert thinker.scheduler_config.max_num_batched_tokens == legacy[0].yaml_engine_args["max_num_batched_tokens"]
    assert 0 < thinker.scheduler_config.max_num_batched_tokens <= thinker.scheduler_config.max_model_len
    assert thinker.model_config.default_sampling_params == legacy[0].yaml_extras["default_sampling_params"]
    expected_tp = 1 if "tiny" in profile else 4
    assert thinker.parallel_config.tensor_parallel_size == expected_tp
    assert thinker.runtime_config.devices == ("0" if expected_tp == 1 else "0,1,2,3")
    text_only = profile.endswith(("_asr.yaml", "_thinker_only.yaml"))
    expected_stages = 1 if text_only else 2
    assert len(legacy) == len(structured.stage_configs) == expected_stages
    assert thinker.cache_config.enable_prefix_caching is True
    assert thinker.cache_config.mamba_cache_mode == "align"
    assert thinker.scheduler_config.enable_chunked_prefill is True
    assert legacy[0].yaml_engine_args["enable_prefix_caching"] is True
    assert legacy[0].yaml_engine_args["mamba_cache_mode"] == "align"
    asr = profile.endswith("_asr.yaml")
    assert thinker.model_config.limit_mm_per_prompt == {"audio": 1 if asr else 20}
    if asr:
        assert thinker.model_config.default_sampling_params["extra_args"]["no_repeat_ngram_size"] == 10
    if len(structured.stage_configs) == 2:
        talker = structured.stage_by_id(1)
        assert talker.parallel_config.tensor_parallel_size == 1
        assert talker.runtime_config.devices == ("0" if expected_tp == 1 else "3")
        assert talker.model_config.default_sampling_params["max_tokens"] == 1
        assert talker.cache_config.enable_prefix_caching is False
        assert talker.model_config.hf_overrides["enable_llm_cuda_graph"] is True
        assert talker.model_config.hf_overrides["low_latency_streaming"] is False


@pytest.mark.parametrize("profile", _PROFILES)
def test_ming_profile_context_can_be_lowered_without_changing_other_stages(profile):
    deploy = load_deploy_config(_DEPLOY / profile)
    pipeline = resolve_pipeline_config(deploy.pipeline)
    assert pipeline is not None
    config = VllmOmniConfig.from_pipeline_config(
        pipeline,
        user_deploy_config=deploy,
        cli_overrides={"stage_0_max_model_len": 4096, "stage_0_max_num_batched_tokens": 512},
    )
    assert config.stage_by_id(0).scheduler_config.max_model_len == 4096
    assert config.stage_by_id(0).scheduler_config.max_num_batched_tokens == 512
    if len(config.stage_configs) == 2:
        assert config.stage_by_id(1).scheduler_config.max_num_batched_tokens == 8192


def test_ming3_public_tts_profile_resolves_without_remote_code():
    pipeline = resolve_pipeline_config("ming3_audio_tts")
    assert pipeline is not None
    deploy = load_deploy_config(_DEPLOY / "ming3_audio_tts.yaml")
    assert deploy.trust_remote_code is False
    legacy = merge_pipeline_deploy(pipeline, deploy)
    structured = VllmOmniConfig.from_pipeline_config(pipeline, user_deploy_config=deploy)
    assert len(legacy) == len(structured.stage_configs) == 1
    assert structured.stage_by_id(0).model_stage == "ming3_tts"


def test_batched_tts_profile_only_widens_the_talker_batch():
    base = resolve_deploy_yaml(_DEPLOY / "ming3_audio_tts.yaml")
    batched = resolve_deploy_yaml(_DEPLOY / "ming3_audio_tts_batched.yaml")
    stage = batched["stages"][0]
    assert stage["max_num_seqs"] == 64
    assert stage["enforce_eager"] is False
    assert stage["engine_extras"]["hf_overrides"]["low_latency_streaming"] is True
    for key in ("devices", "gpu_memory_utilization", "max_num_batched_tokens", "default_sampling_params"):
        assert stage[key] == base["stages"][0][key]
    assert batched["pipeline"] == base["pipeline"]
