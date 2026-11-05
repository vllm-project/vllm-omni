# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ming deployment choices survive legacy and structured config resolution."""

from pathlib import Path

import pytest

from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.pipeline_registry import resolve_pipeline_config
from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy, resolve_deploy_yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DEPLOY = Path(__file__).parents[2] / "vllm_omni" / "deploy"
_LONG_CONTEXT_PROFILES = [
    f"ming3_audio_{size}{variant}.yaml"
    for size in ("flash", "tiny")
    for variant in ("", "_asr", "_thinker_only")
    if size != "flash" or variant
]


@pytest.mark.parametrize("model_type", ["ming3_tts", "lingv3_tts"])
def test_ming3_tts_default_serves_audio_with_one_mrv2_stage(model_type):
    pipeline = resolve_pipeline_config(model_type)
    assert pipeline is not None
    # Exercise default loading, as used by `vllm serve MODEL --omni`.
    config = VllmOmniConfig.from_pipeline_config(pipeline)
    (stage,) = config.stage_configs
    assert stage.model_stage == "llm"
    assert stage.engine_output_type == "audio"
    assert stage.final_output is True
    assert stage.model_config.use_v2_model_runner is True
    assert stage.scheduler_config.async_scheduling is True
    assert stage.scheduler_config.max_num_seqs == 64
    assert stage.scheduler_config.max_model_len == 4096
    assert stage.model_config.default_sampling_params["max_tokens"] == 801


@pytest.mark.parametrize("profile", _LONG_CONTEXT_PROFILES)
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


@pytest.mark.parametrize("profile", ["ming3_audio_flash.yaml", *_LONG_CONTEXT_PROFILES])
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
    legacy, _ = StageConfigFactory._create_legacy_from_registry(pipeline, {})
    structured = VllmOmniConfig.from_pipeline_config(pipeline)
    assert len(legacy) == len(structured.stage_configs) == 1
    stage = structured.stage_by_id(0)
    assert stage.model_stage == "ming3_tts"
    assert stage.scheduler_config.max_num_seqs == legacy[0].yaml_engine_args["max_num_seqs"] == 64
    assert stage.model_config.enforce_eager is legacy[0].yaml_engine_args["enforce_eager"] is False
    assert stage.model_config.hf_overrides["compile_flow"] is True
    assert stage.model_config.hf_overrides["low_latency_streaming"] is True


def test_legacy_batched_tts_profile_matches_default():
    base = resolve_deploy_yaml(_DEPLOY / "ming3_audio_tts.yaml")
    batched = resolve_deploy_yaml(_DEPLOY / "ming3_audio_tts_batched.yaml")
    assert batched == base


def test_flash_default_profile_batches_both_stages_without_deploy_override():
    pipeline = resolve_pipeline_config("bailing_moe_v3_audio")
    assert pipeline is not None
    assert pipeline.default_deploy_config_name == "ming3_audio_flash.yaml"
    default = resolve_deploy_yaml(_DEPLOY / pipeline.default_deploy_config_name)
    thinker, talker = default["stages"]
    reference_thinker = resolve_deploy_yaml(_DEPLOY / "ming3_audio_flash_thinker_only.yaml")["stages"][0]
    # Batching must preserve the existing input limits and sampling defaults.
    for key in ("max_model_len", "limit_mm_per_prompt", "default_sampling_params"):
        assert thinker[key] == reference_thinker[key]
    assert thinker["max_num_seqs"] == talker["max_num_seqs"] == 32
    assert thinker["model_runner"] == talker["model_runner"] == "v2"
    assert thinker["async_scheduling"] is True
    # Audio prefills reach hundreds of tokens; they must stay within the graphs.
    assert max(thinker["compilation_config"]["cudagraph_capture_sizes"]) >= 1024
    assert thinker["compilation_config"]["cudagraph_mode"] == "FULL_AND_PIECEWISE"
    overrides = talker["engine_extras"]["hf_overrides"]
    assert overrides["compile_flow"] is True and overrides["low_latency_streaming"] is True
    assert overrides["enable_llm_cuda_graph"] is True
    assert overrides["normalize_text"] is True and overrides["tn_mode"] == "upstream"
    assert thinker["devices"] == "0,1" and thinker["tensor_parallel_size"] == 2
    assert talker["devices"] == "1"
    legacy, _ = StageConfigFactory._create_legacy_from_registry(pipeline, {})
    structured = VllmOmniConfig.from_pipeline_config(pipeline)
    stage = structured.stage_by_id(0)
    assert stage.scheduler_config.max_num_seqs == legacy[0].yaml_engine_args["max_num_seqs"] == 32
    assert structured.stage_by_id(1).scheduler_config.max_num_seqs == legacy[1].yaml_engine_args["max_num_seqs"] == 32
    assert (
        stage.scheduler_config.max_model_len == legacy[0].yaml_engine_args["max_model_len"] == thinker["max_model_len"]
    )
    assert stage.scheduler_config.max_num_batched_tokens == legacy[0].yaml_engine_args["max_num_batched_tokens"] == 2048
    assert stage.cache_config.enable_prefix_caching is legacy[0].yaml_engine_args["enable_prefix_caching"] is False
    assert stage.cache_config.mamba_cache_mode == legacy[0].yaml_engine_args["mamba_cache_mode"] == "none"
    assert (
        stage.cache_config.gpu_memory_utilization
        == legacy[0].yaml_engine_args["gpu_memory_utilization"]
        == thinker["gpu_memory_utilization"]
    )
    assert stage.model_config.limit_mm_per_prompt == thinker["limit_mm_per_prompt"]
    assert (
        stage.model_config.default_sampling_params
        == legacy[0].yaml_extras["default_sampling_params"]
        == thinker["default_sampling_params"]
    )
    assert thinker["additional_config"] == {
        "ar_prefill_admission_min_batch": 8,
        "ar_prefill_admission_max_wait_ms": 100,
        "ar_prefill_admission_min_running": 4,
    }
