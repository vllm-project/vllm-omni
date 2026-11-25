# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Resolve the MiniCPM-o default duplex deployment's CUDA and non-CUDA runners."""

import os
from pathlib import Path

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.stage_config import (
    _apply_platform_overrides,
    load_deploy_config,
    merge_pipeline_deploy,
    resolve_deploy_yaml,
)
from vllm_omni.distributed.omni_connectors.utils.initialization import load_omni_transfer_config
from vllm_omni.engine.stage_init_utils import get_stage_connector_spec
from vllm_omni.model_executor.models.minicpmo_4_5.config import npu_prefill_startup_environment
from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_PIPELINE
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DEPLOY = "minicpmo_4_5.yaml"


def _resolve_stages(monkeypatch, platform: str, deploy_name: str = _DEPLOY):
    # Resolve platform overrides even when this test runs on a different host.
    monkeypatch.setattr(current_omni_platform, "device_name", platform)
    config = _apply_platform_overrides(load_deploy_config(get_deploy_config_path(deploy_name)), platform=platform)
    return config, merge_pipeline_deploy(MINICPMO_4_5_PIPELINE, config)


def test_duplex_mrv2_cuda_profile(monkeypatch) -> None:
    config, stages = _resolve_stages(monkeypatch, "cuda")
    assert config.session_mode == "duplex"
    assert [s.yaml_engine_args["use_v2_model_runner"] for s in stages] == [True, True, True]
    assert stages[1].yaml_engine_args["attention_backend"] == "FLASH_ATTN"
    assert [s.yaml_engine_args["async_chunk"] for s in stages] == [True, True, True]
    assert stages[0].yaml_engine_args["async_scheduling"] is True
    # K-step books its span with the synchronous runner contract.
    assert stages[1].yaml_engine_args["speculative_config"]["num_speculative_tokens"] == 15
    assert stages[1].yaml_engine_args["async_scheduling"] is False
    assert stages[2].yaml_engine_args.get("async_scheduling") is not False
    # Retain the base profile capacity and Talker KV budget.
    assert [s.yaml_engine_args["max_num_seqs"] for s in stages] == [16, 16, 8]
    assert stages[1].yaml_engine_args["kv_cache_memory_bytes"] == 4 * 1024**3


def test_duplex_mrv2_rocm_profile_preserves_shared_contract(monkeypatch) -> None:
    config, stages = _resolve_stages(monkeypatch, "rocm")
    assert config.model_runner == "v2"
    assert config.session_mode == "duplex"
    assert [s.yaml_engine_args["use_v2_model_runner"] for s in stages] == [True, True, True]
    assert [s.yaml_engine_args["async_chunk"] for s in stages] == [True, True, True]
    assert [s.yaml_runtime["devices"] for s in stages] == ["0", "0", "0"]
    assert stages[2].yaml_engine_args["enforce_eager"] is True
    assert stages[2].input_sources == [1]

    structured = VllmOmniConfig.from_pipeline_config(MINICPMO_4_5_PIPELINE, user_deploy_config=config)
    assert [s.model_config.use_v2_model_runner for s in structured.stage_configs] == [True, True, True]
    assert structured.stage_by_id(2).model_config.enforce_eager is True


@pytest.mark.parametrize("platform", ["xpu", "musa"])
def test_duplex_mrv2_profile_keeps_deferred_platforms_on_v1(monkeypatch, platform) -> None:
    monkeypatch.setattr(current_omni_platform, "device_name", platform)
    config = _apply_platform_overrides(load_deploy_config(get_deploy_config_path(_DEPLOY)), platform=platform)
    assert config.model_runner == "v1"
    with pytest.raises(ValueError, match="Model Runner V1 is not supported"):
        merge_pipeline_deploy(MINICPMO_4_5_PIPELINE, config)


def test_duplex_mrv2_profile_npu_full_mrv2(monkeypatch) -> None:
    # PR #10062 enables full-MRv2 three-stage serving on NPU.
    monkeypatch.setattr(current_omni_platform, "device_name", "npu")
    config = _apply_platform_overrides(load_deploy_config(get_deploy_config_path(_DEPLOY)), platform="npu")
    stages = merge_pipeline_deploy(MINICPMO_4_5_PIPELINE, config)
    assert [s.yaml_engine_args["use_v2_model_runner"] for s in stages] == [True, True, True]


def test_two_card_deploy_inherits_common_npu_settings(monkeypatch) -> None:
    config, stages = _resolve_stages(monkeypatch, "npu", "minicpmo_4_5_2gpu.yaml")
    thinker, talker, vocoder = stages
    assert thinker.yaml_engine_args["max_model_len"] == 40960
    assert thinker.yaml_runtime["env"]["MINICPMO_NPU_DSPARK_NATIVE_GATHER"] == "1"
    assert thinker.yaml_engine_args["interleave_mm_strings"] is True
    assert thinker.yaml_engine_args["media_io_kwargs"]["video"] == {"fps": 1, "num_frames": 128}
    assert talker.yaml_runtime["num_replicas"] == 2
    assert talker.yaml_runtime["env"]["MINICPMO_NPU_KSTEP_SKIP_TERMINAL_TAIL"] == "1"
    options = talker.yaml_engine_args["additional_config"]
    assert options["minicpmo_talker_fused_decode"] is True
    assert options["minicpmo_talker_compact_decode"] is True
    assert options["minicpmo_talker_hybrid_context"] is True
    assert options["minicpmo_talker_replica_route"] == "half"
    assert vocoder.yaml_runtime["num_replicas"] == 2
    assert vocoder.yaml_runtime["env"] == {
        "MINICPMO_CODE2WAV_SECONDARY_TIGHT_MEMORY": "1",
    }
    assert [stage.yaml_engine_args["max_num_seqs"] for stage in stages] == [16, 16, 32]
    extra = config.connectors["connector_of_shared_memory"]["extra"]
    assert extra["token2wav_n_timesteps"] == 5
    assert (extra["micro_batch_size"], extra["secondary_vocoder_micro_batch_size"]) == (8, 4)
    assert (extra["token2wav_setup_cache_size"], extra["token2wav_setup_host_cache_size"]) == (64, 512)
    assert extra["cfm_graph_capture_frames"] == [16, 32, 48, 64, 80]
    assert extra["token2wav_runtime_prompt_cache_size"] == 512


def test_two_card_prefill_startup_uses_private_markers(monkeypatch) -> None:
    keys = ("MINICPMO_NPU_THINKER_PREFILL_STARTUP", "MINICPMO_NPU_TALKER_PREFILL_STARTUP")
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    with npu_prefill_startup_environment(get_deploy_config_path(_DEPLOY)):
        assert all(key not in os.environ for key in keys)

    with npu_prefill_startup_environment(get_deploy_config_path("minicpmo_4_5_2gpu.yaml")):
        thinker, talker = (Path(os.environ[key]) for key in keys)
        assert thinker.parent == talker.parent and thinker != talker
        assert not thinker.exists() and not talker.exists()
        talker.touch()

    assert all(key not in os.environ for key in keys)
    assert not talker.parent.exists()


@pytest.mark.parametrize(("cards", "code2wav_device"), [(2, "1"), (3, "2")])
def test_multigpu_deploy_inherits_cuda_serving_defaults(monkeypatch, cards, code2wav_device) -> None:
    config, stages = _resolve_stages(monkeypatch, "cuda", f"minicpmo_4_5_{cards}gpu.yaml")
    thinker, talker, vocoder = stages
    assert config.duplex_session.max_sessions == 16
    assert [stage.yaml_engine_args["max_num_seqs"] for stage in stages] == [16, 16, 16]
    assert all(stage.yaml_engine_args["use_v2_model_runner"] for stage in stages)
    assert thinker.yaml_engine_args["limit_mm_per_prompt"] == {"image": 64, "audio": 64, "video": 1}
    assert thinker.yaml_engine_args["media_io_kwargs"]["video"] == {"fps": 1, "num_frames": 128}
    assert thinker.yaml_extras["default_sampling_params"]["repetition_penalty"] == 1.0
    assert thinker.yaml_engine_args["speculative_config"]["num_speculative_tokens"] == 7
    assert (vocoder.yaml_runtime["devices"], vocoder.yaml_runtime["num_replicas"]) == (code2wav_device, 1)
    assert talker.yaml_engine_args["attention_backend"] == "FLASH_ATTN"
    assert talker.yaml_engine_args["speculative_config"]["num_speculative_tokens"] == 15
    assert not vocoder.yaml_engine_args.get("additional_config")
    extra = config.connectors["connector_of_shared_memory"]["extra"]
    assert extra["token2wav_float16"] is True
    assert extra["token2wav_n_timesteps"] == 5
    assert "code2wav_encoder_padded_flow_history" not in extra


def test_two_card_deploy_accepts_single_vocoder_stage_override(monkeypatch) -> None:
    monkeypatch.setattr(current_omni_platform, "device_name", "npu")
    config = VllmOmniConfig.from_pipeline_config(
        MINICPMO_4_5_PIPELINE,
        user_deploy_config=load_deploy_config(get_deploy_config_path("minicpmo_4_5_2gpu.yaml")),
        cli_overrides={"stage_2_devices": "0", "stage_2_num_replicas": 1},
    )
    assert [(s.runtime_config.devices, s.runtime_config.num_replicas) for s in config.stage_configs] == [
        ("1", 1),
        ("1,0", 2),
        ("0", 1),
    ]


@pytest.mark.parametrize("deploy_name", [_DEPLOY, "minicpmo_4_5_turn.yaml"])
@pytest.mark.parametrize("platform", ["cuda", "npu", "rocm", "xpu", "musa", "cpu"])
def test_reference_optimizations_are_cuda_only(monkeypatch, deploy_name, platform):
    monkeypatch.setattr(current_omni_platform, "device_name", platform)
    path = get_deploy_config_path(deploy_name)
    deploy = _apply_platform_overrides(load_deploy_config(path), platform=platform)
    transfer = load_omni_transfer_config(config_dict=resolve_deploy_yaml(path))
    stage_extra = get_stage_connector_spec(transfer, stage_id=2, async_chunk=True)["extra"]
    resolved_extra = deploy.connectors["connector_of_shared_memory"]["extra"]
    for extra in (stage_extra, resolved_extra):
        for flag in (
            "token2wav_float16",
            "token2wav_spk_cuda_graph",
            "token2wav_s3_cuda_graph",
        ):
            assert extra.get(flag, False) is (platform == "cuda")
        assert extra.get("cfm_drop_upstream_att_buffer", False) is (platform in ("cuda", "npu"))
        if platform == "npu":
            assert (extra["token2wav_setup_cache_size"], extra["token2wav_setup_host_cache_size"]) == (4, 64)
            assert extra["cfm_graph_capture_frames"] == [64, 80]
        assert extra["cfm_npu_cross_reference_batching"] is True
        # The Ascend profile uses 32-frame output chunks.
        assert extra["codec_chunk_frames"] == (32 if platform == "npu" else 25)


@pytest.mark.parametrize("deploy_name", [_DEPLOY, "minicpmo_4_5_turn.yaml"])
@pytest.mark.parametrize("platform", ["cuda", "npu"])
@pytest.mark.parametrize("depth", [1, 2])
def test_platform_connector_overlay_preserves_inherited_options(tmp_path, monkeypatch, deploy_name, platform, depth):
    monkeypatch.setattr(current_omni_platform, "device_name", platform)
    path = get_deploy_config_path(deploy_name)
    expected = _apply_platform_overrides(load_deploy_config(path), platform=platform).connectors
    options = ("token2wav_float16", "token2wav_spk_cuda_graph")
    for level, option in enumerate(options[:depth]):
        overlay = tmp_path / f"overlay-{level}.yaml"
        overlay.write_text(
            f"base_config: {path}\n"
            "platforms:\n"
            "  cuda:\n"
            "    connectors:\n"
            "      connector_of_shared_memory:\n"
            "        extra:\n"
            f"          {option}: false\n"
        )
        path = overlay
        if platform == "cuda":
            expected["connector_of_shared_memory"]["extra"][option] = False

    deploy = _apply_platform_overrides(load_deploy_config(path), platform=platform)
    transfer = load_omni_transfer_config(config_dict=resolve_deploy_yaml(path))
    stage_extra = get_stage_connector_spec(transfer, stage_id=2, async_chunk=True)["extra"]
    assert deploy.connectors == expected
    # Worker resolution also adds transport-owned role/wakeup metadata.
    for key, value in expected["connector_of_shared_memory"]["extra"].items():
        assert stage_extra[key] == value
