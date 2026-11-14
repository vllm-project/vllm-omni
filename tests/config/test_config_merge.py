# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from copy import deepcopy

import pytest

from vllm_omni.config.stage_config import (
    DeployConfig,
    StageDeployConfig,
    _apply_platform_overrides,
    _merge_config_fields,
    resolve_deploy_yaml,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_selected_fields_merge_recursively_without_mutating_inputs():
    base = {"section": {"nested": {"a": 1, "b": 2}}, "replace": {"old": 1}}
    overlay = {"section": {"nested": {"b": 0}}, "replace": {"new": 2}}
    before = deepcopy((base, overlay))
    merged = _merge_config_fields(base, overlay, deep_merge_keys=frozenset({"section"}))
    assert merged == {"section": {"nested": {"a": 1, "b": 0}}, "replace": {"new": 2}}
    assert (base, overlay) == before


@pytest.mark.parametrize("value", [None, False, 0, [1, 2]])
def test_non_mapping_overlay_replaces_mapping(value):
    assert _merge_config_fields({"section": {"a": 1}}, {"section": value}, deep_merge_keys=frozenset({"section"})) == {
        "section": value
    }


def test_recursive_deploy_inheritance(tmp_path):
    # Test raw YAML merging independently of the current flat speech schema.
    (tmp_path / "base.yaml").write_text("speech_cache:\n  nested:\n    a: 1\n    b: 2\n")
    (tmp_path / "middle.yaml").write_text("base_config: base.yaml\nspeech_cache:\n  nested:\n    b: 0\n")
    leaf = tmp_path / "leaf.yaml"
    leaf.write_text("base_config: middle.yaml\nasync_chunk: false\n")
    merged = resolve_deploy_yaml(leaf)
    assert merged["speech_cache"] == {"nested": {"a": 1, "b": 0}}
    assert merged["async_chunk"] is False


def test_platform_stage_engine_extras_merge_recursively():
    deploy = DeployConfig(
        stages=[
            StageDeployConfig(
                stage_id=0,
                engine_extras={
                    "hf_overrides": {
                        "runtime": {
                            "preserved": 1,
                            "replaced": 1,
                        },
                    },
                },
            ),
        ],
        platforms={
            "rocm": {
                "stages": [
                    {
                        "stage_id": 0,
                        "engine_extras": {
                            "hf_overrides": {
                                "runtime": {
                                    "replaced": 2,
                                },
                            },
                        },
                    },
                ],
            },
        },
    )

    merged = _apply_platform_overrides(deploy, platform="rocm")

    assert merged.stages[0].engine_extras == {
        "hf_overrides": {
            "runtime": {
                "preserved": 1,
                "replaced": 2,
            },
        },
    }


def test_single_stage_streaming_capability_uses_central_async_resolver():
    from vllm_omni.config.stage_config import DeployConfig, _resolve_pipeline_async_chunk_enabled
    from vllm_omni.model_executor.models.qwen3_tts.pipeline import QWEN3_TTS_FUSED_PIPELINE

    assert _resolve_pipeline_async_chunk_enabled(QWEN3_TTS_FUSED_PIPELINE, DeployConfig())
    with pytest.raises(ValueError, match="requires async_chunk=True"):
        _resolve_pipeline_async_chunk_enabled(QWEN3_TTS_FUSED_PIPELINE, DeployConfig(async_chunk=False))


@pytest.mark.parametrize(
    "pipeline_key,profile,stage_count",
    [
        ("qwen3_tts", "qwen3_tts_high_concurrency_mrv2_single_gpu.yaml", 2),
        ("qwen3_tts_fused", "qwen3_tts_fused_single_gpu.yaml", 1),
    ],
)
def test_qwen3_tts_deployment_selection_preserves_output_and_transport_contract(pipeline_key, profile, stage_count):
    from tests.helpers.stage_config import get_deploy_config_path
    from vllm_omni.config.pipeline_registry import resolve_pipeline_config
    from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy

    pipeline = resolve_pipeline_config(pipeline_key)
    assert pipeline is not None
    deploy = load_deploy_config(get_deploy_config_path(profile))
    stages = merge_pipeline_deploy(pipeline, deploy)
    assert len(stages) == stage_count
    talker = stages[0]
    if stage_count == 1:
        assert talker.final_output and talker.final_output_type == "audio"
        assert talker.yaml_engine_args["engine_output_type"] == "audio"
        assert "custom_process_next_stage_input_func" not in talker.yaml_engine_args
        assert not talker.yaml_extras.get("output_connectors")
        assert talker.yaml_engine_args["supports_running_prefix_cache_reset"] is False
        assert talker.yaml_engine_args["additional_config"]["ref_code_context_frames"] == 72
    else:
        assert not talker.final_output
        assert talker.yaml_engine_args["engine_output_type"] == "latent"
        assert talker.yaml_engine_args["custom_process_next_stage_input_func"].endswith("talker2code2wav_async_chunk")
        assert talker.yaml_extras["output_connectors"]
        assert stages[1].input_sources == [0] and stages[1].final_output_type == "audio"
        assert talker.yaml_engine_args.get("supports_running_prefix_cache_reset", True)


def test_single_stage_running_reset_capability_cannot_be_overridden():
    from vllm_omni.config.omni_config import VllmOmniConfig
    from vllm_omni.config.stage_config import DeployConfig, StageDeployConfig, merge_pipeline_deploy
    from vllm_omni.model_executor.models.qwen3_tts.pipeline import QWEN3_TTS_FUSED_PIPELINE

    deploy = DeployConfig(
        stages=[StageDeployConfig(stage_id=0, engine_extras={"supports_running_prefix_cache_reset": True})]
    )
    legacy = merge_pipeline_deploy(QWEN3_TTS_FUSED_PIPELINE, deploy)[0]
    legacy.runtime_overrides["supports_running_prefix_cache_reset"] = True
    assert legacy.to_omegaconf().engine_args.supports_running_prefix_cache_reset is False
    with pytest.raises(ValueError, match="no structured config owner: supports_running_prefix_cache_reset"):
        VllmOmniConfig.from_pipeline_config(QWEN3_TTS_FUSED_PIPELINE, user_deploy_config=deploy)
    structured = VllmOmniConfig.from_pipeline_config(QWEN3_TTS_FUSED_PIPELINE)
    assert structured.stage_by_id(0).model_config.supports_running_prefix_cache_reset is False


@pytest.fixture
def single_stage_tts_config(mocker):
    from vllm.config import VllmConfig

    config = mocker.Mock(spec=VllmConfig)
    config.additional_config = {"talker_stream_decode": True}
    config.model_config = mocker.Mock(
        stage_connector_config={},
        engine_output_type="audio",
        final_output=True,
        supports_running_prefix_cache_reset=False,
        use_v2_model_runner=True,
        async_chunk=True,
    )
    config.parallel_config = mocker.Mock(
        tensor_parallel_size=1, pipeline_parallel_size=1, distributed_executor_backend="uni"
    )
    config.cache_config = mocker.Mock(enable_prefix_caching=False)
    config.device_config = mocker.Mock(device="cuda")
    mocker.patch(
        "vllm_omni.model_executor.models.qwen3_tts.stream_decode.current_omni_platform.is_cuda", return_value=True
    )
    return config


def test_single_stage_tts_rejects_unsupported_decoder_configuration(single_stage_tts_config):
    from vllm_omni.model_executor.models.qwen3_tts.stream_decode import talker_stream_decode_enabled

    config = single_stage_tts_config
    config.cache_config.enable_prefix_caching = True
    with pytest.raises(ValueError, match="enable_prefix_caching=False"):
        talker_stream_decode_enabled(config)
    config.cache_config.enable_prefix_caching = False
    assert talker_stream_decode_enabled(config)


@pytest.mark.parametrize("option_location", ["additional_config", "connector"])
@pytest.mark.parametrize("output_type", ["latent", "audio"])
def test_two_stage_talker_rejects_single_stage_pcm_option(single_stage_tts_config, option_location, output_type):
    from vllm_omni.model_executor.models.qwen3_tts.stream_decode import talker_stream_decode_enabled

    config = single_stage_tts_config
    config.model_config.engine_output_type = output_type
    config.model_config.final_output = False
    if option_location == "connector":
        config.model_config.stage_connector_config = {"extra": config.additional_config}
        config.additional_config = {}
    with pytest.raises(ValueError, match="final audio-output Talker"):
        talker_stream_decode_enabled(config)


def test_two_stage_first_audio_remains_independent(single_stage_tts_config):
    from vllm_omni.model_executor.models.qwen3_tts.first_audio import talker_first_audio_enabled
    from vllm_omni.model_executor.models.qwen3_tts.stream_decode import talker_stream_decode_enabled

    config = single_stage_tts_config
    config.model_config.engine_output_type = "latent"
    config.additional_config = {}
    config.model_config.stage_connector_config = {"extra": {"talker_first_audio": True}}
    assert not talker_stream_decode_enabled(config)
    assert talker_first_audio_enabled(config)
    config.model_config.stage_connector_config["extra"]["talker_stream_first_audio"] = True
    with pytest.raises(ValueError, match="requires the single-stage"):
        talker_stream_decode_enabled(config)


def test_single_stage_pcm_requires_cuda_device_and_own_first_audio_option(single_stage_tts_config):
    from vllm_omni.model_executor.models.qwen3_tts.stream_decode import talker_stream_decode_enabled

    config = single_stage_tts_config
    config.device_config.device = "cpu"
    with pytest.raises(ValueError, match="requires CUDA"):
        talker_stream_decode_enabled(config)
    config.device_config.device = "cuda"
    config.additional_config["talker_first_audio"] = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        talker_stream_decode_enabled(config)
    config.additional_config.pop("talker_first_audio")
    config.additional_config["talker_stream_first_audio"] = True
    assert talker_stream_decode_enabled(config)


def test_single_stage_audio_talker_requires_pcm_option(single_stage_tts_config):
    from vllm_omni.model_executor.models.qwen3_tts.stream_decode import talker_stream_decode_enabled

    single_stage_tts_config.additional_config = {}
    with pytest.raises(ValueError, match="requires talker_stream_decode=True"):
        talker_stream_decode_enabled(single_stage_tts_config)


def test_single_stage_pcm_requires_reset_capability_declaration(single_stage_tts_config):
    from vllm_omni.model_executor.models.qwen3_tts.stream_decode import talker_stream_decode_enabled

    single_stage_tts_config.model_config.supports_running_prefix_cache_reset = True
    with pytest.raises(ValueError, match="must disable running prefix-cache reset"):
        talker_stream_decode_enabled(single_stage_tts_config)


def test_qwen3_tts_high_concurrency_npu_enables_graphs():
    """The NPU override must not force eager on either stage (#7993).

    Forcing eager on NPU halved throughput and broke the 100 ms streaming
    SLO at concurrency 8+ (910B4: RTF 1.079 as shipped with 733/782 samples
    over SLO, vs RTF 0.604 with none once graphs were enabled). The NPU
    override aligns with the base qwen3_tts.yaml recipe instead; stage 1's
    CUDA-tuned eager default in its own definition stays untouched.
    """
    from tests.helpers.stage_config import get_deploy_config_path
    from vllm_omni.config.stage_config import _apply_platform_overrides, load_deploy_config

    deploy = load_deploy_config(get_deploy_config_path("qwen3_tts_high_concurrency.yaml"))
    deploy = _apply_platform_overrides(deploy, platform="npu")
    by_id = {stage.stage_id: stage for stage in deploy.stages}
    for stage_id in (0, 1):
        stage = by_id[stage_id]
        assert stage.enforce_eager is False, f"stage {stage_id} must keep graphs enabled on NPU"
        assert (stage.compilation_config or {}).get("cudagraph_mode") == "PIECEWISE"
