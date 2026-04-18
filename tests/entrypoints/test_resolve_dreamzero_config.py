import os

import pytest

from vllm_omni.entrypoints.utils import load_stage_configs_from_model, resolve_model_config_path

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_dreamzero_vla_resolves_to_dreamzero_config(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.entrypoints.utils.get_config",
        lambda _model, trust_remote_code=True: type("Cfg", (), {"model_type": "vla"})(),
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.utils._looks_like_dreamzero",
        lambda _model: True,
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.utils.current_omni_platform.get_default_stage_config_path",
        lambda: "vllm_omni/model_executor/stage_configs",
    )

    original_exists = os.path.exists

    def mock_exists(path):
        if "dreamzero.yaml" in str(path):
            return True
        return original_exists(path)

    monkeypatch.setattr(os.path, "exists", mock_exists)

    result = resolve_model_config_path("GEAR-Dreams/DreamZero-DROID")

    assert result is not None
    assert "dreamzero.yaml" in result


def test_dreamzero_config_sets_model_class_and_policy_config(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.entrypoints.utils.resolve_model_config_path",
        lambda _model: "vllm_omni/model_executor/stage_configs/dreamzero.yaml",
    )

    stage_configs = load_stage_configs_from_model("GEAR-Dreams/DreamZero-DROID")
    engine_args = stage_configs[0].engine_args

    assert engine_args.model_class_name == "DreamZeroPipeline"
    assert engine_args.model_config.policy_server_config.action_space == "joint_position"
