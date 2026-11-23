# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pinned root configuration and routing for Realtime-Venus."""

import hashlib
import json

import pytest
import yaml

from vllm_omni.transformers_utils.configs.realtime_venus import RealtimeVenusOmniConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def checkpoint(tmp_path):
    model = tmp_path / "renamed-checkpoint"
    model.mkdir()
    full = {
        "model_type": "realtime_venus_omni",
        "architectures": ["RealtimeVenusOmni"],
        "version": "4.5",
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 2,
        "intermediate_size": 256,
        "vision_config": {"hidden_size": 64},
        "audio_config": {"d_model": 64},
        "tts_config": {"hidden_size": 32},
        "slice_config": {"max_slice_nums": 1},
    }
    raw = yaml.safe_dump(
        {
            "schema_version": 3,
            "name": "Realtime-Venus",
            "models": {"omni": {"path": "Realtime-Venus-Omni", "hf_config": full}},
        }
    ).encode()
    (tmp_path / "config.yaml").write_bytes(raw)
    child = {
        "model_type": "realtime_venus_omni",
        "architectures": ["RealtimeVenusOmni"],
        "venus_config_ref": {
            "repo_id": "inclusionAI/Realtime-Venus",
            "repo_type": "model",
            "filename": "config.yaml",
            "model": "omni",
            "revision": "a" * 40,
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
    }
    (model / "config.json").write_text(json.dumps(child))
    return model, child


def test_expands_verified_sibling_yaml(checkpoint):
    model, _ = checkpoint
    config = RealtimeVenusOmniConfig.from_pretrained(model, local_files_only=True)
    assert config.hidden_size == 128
    assert config.vision_config.hidden_size == 64
    assert config.audio_config.d_model == 64
    assert config.tts_config.hidden_size == 32
    assert config.slice_config.max_slice_nums == 1
    assert config.version == "4.5"


@pytest.mark.parametrize(
    "key,value",
    [
        ("revision", "main"),
        ("sha256", "invalid"),
        ("model", "audio"),
        ("filename", "../config.yaml"),
        ("repo_id", "other/model"),
    ],
)
def test_rejects_invalid_reference(checkpoint, key, value):
    model, child = checkpoint
    child["venus_config_ref"][key] = value
    (model / "config.json").write_text(json.dumps(child))
    with pytest.raises(ValueError):
        RealtimeVenusOmniConfig.from_pretrained(model, local_files_only=True)


def test_wrong_local_digest_is_rejected(checkpoint):
    model, _ = checkpoint
    (model.parent / "config.yaml").write_text("tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        RealtimeVenusOmniConfig.from_pretrained(model, local_files_only=True)


def test_missing_root_has_actionable_error(checkpoint):
    model, _ = checkpoint
    (model.parent / "config.yaml").unlink()
    with pytest.raises(OSError, match="Download the checkpoint subdirectory"):
        RealtimeVenusOmniConfig.from_pretrained(model, local_files_only=True)


def test_pipeline_routes_by_architecture(checkpoint, monkeypatch):
    from vllm_omni.config.config_factory import StageConfigFactory

    config = RealtimeVenusOmniConfig.from_pretrained(checkpoint[0], local_files_only=True)
    monkeypatch.setattr(StageConfigFactory, "get_hf_config", lambda **kw: config)
    monkeypatch.setattr(StageConfigFactory, "try_infer_model_type", lambda **kw: "unknown_model_type")
    pipeline = StageConfigFactory.get_pipeline_config("renamed", trust_remote_code=True)
    assert pipeline.model_type == "realtime_venus_omni"
    assert len(pipeline.stages) == 3
    assert pipeline.duplex_plugin == "vllm_omni.model_executor.models.realtime_venus.duplex.RealtimeVenusDuplexPlugin"
    config.version = "2.6"
    assert StageConfigFactory.get_pipeline_config("renamed", trust_remote_code=True) is None


def test_autoconfig_expands_reference(checkpoint):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(checkpoint[0], local_files_only=True, trust_remote_code=False)
    assert config.hidden_size == 128
    assert config.version == "4.5"


def test_legacy_processor_registration_restored_on_failure(monkeypatch):
    from transformers import AutoImageProcessor

    from vllm_omni.model_executor.models.minicpmo_4_5.processor import legacy_image_processor_registration

    registered = []

    def register(config_class, *args, **kwargs):
        registered.append(config_class)

    monkeypatch.setattr(AutoImageProcessor, "register", staticmethod(register))
    with pytest.raises(RuntimeError, match="load failed"):
        with legacy_image_processor_registration():
            AutoImageProcessor.register("RealtimeVenusOmniImageProcessor", object)
            AutoImageProcessor.register("MiniCPMVImageProcessor", object)
            AutoImageProcessor.register(RealtimeVenusOmniConfig, object)
            raise RuntimeError("load failed")
    assert AutoImageProcessor.register is register
    assert registered == [RealtimeVenusOmniConfig]


@pytest.mark.parametrize("mode,suffix", [("duplex", ""), ("turn", "_turn")])
@pytest.mark.parametrize("platform", ["cuda", "npu", "rocm", "xpu", "musa", "cpu"])
def test_venus_deploy_inherits_minicpm_settings(mode, suffix, platform):
    from vllm_omni.config.stage_config import _apply_platform_overrides, load_deploy_config

    config = _apply_platform_overrides(
        load_deploy_config(f"vllm_omni/deploy/realtime_venus_omni{suffix}.yaml"), platform=platform
    )
    base_suffix = "_turn" if mode == "turn" else ""
    expected = _apply_platform_overrides(
        load_deploy_config(f"vllm_omni/deploy/minicpmo_4_5{base_suffix}.yaml"), platform=platform
    )
    assert config.session_mode == mode
    expected.pipeline = "realtime_venus_omni"
    # Compare the applied settings; inheritance can add empty platform stage lists.
    config.platforms = expected.platforms = None
    assert config == expected
