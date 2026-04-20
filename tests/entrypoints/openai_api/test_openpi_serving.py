from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from vllm_omni.diffusion.models.dreamzero import transform as dreamzero_transform
from vllm_omni.entrypoints.openai.realtime.robot import openpi_serving

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

TEST_POLICY_SERVER_CONFIG = {
    "image_resolution": (180, 320),
    "n_external_cameras": 2,
    "needs_wrist_camera": True,
    "needs_stereo_camera": False,
    "needs_session_id": True,
    "action_space": "joint_position",
}


def _engine_with_policy_config(policy_config=None):
    od_config = SimpleNamespace(model_config={"policy_server_config": policy_config or TEST_POLICY_SERVER_CONFIG})
    return SimpleNamespace(get_diffusion_od_config=lambda: od_config)


def test_ensure_transforms_loaded_fails_fast_on_import_error(monkeypatch):
    def fail_import(module_name):
        raise ModuleNotFoundError(f"missing module: {module_name}")

    monkeypatch.setattr(dreamzero_transform.importlib, "import_module", fail_import)

    with pytest.raises(RuntimeError) as exc_info:
        dreamzero_transform.ensure_transforms_loaded()

    assert "Failed to import DreamZero transform module" in str(exc_info.value)


def test_ensure_transforms_loaded_fails_when_default_transform_missing(monkeypatch):
    monkeypatch.setattr(dreamzero_transform.importlib, "import_module", lambda _module_name: None)
    monkeypatch.setattr(dreamzero_transform, "TRANSFORMS", {})

    with pytest.raises(RuntimeError) as exc_info:
        dreamzero_transform.ensure_transforms_loaded()

    assert "roboarena" in str(exc_info.value)
    assert "not registered" in str(exc_info.value)


def test_policy_server_config_reads_diffusion_model_config():
    policy_config = {
        "image_resolution": [64, 64],
        "n_external_cameras": 1,
        "custom_model_key": {"nested": True},
    }
    od_config = SimpleNamespace(model_config={"policy_server_config": policy_config})
    engine_client = SimpleNamespace(get_diffusion_od_config=lambda: od_config)

    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=engine_client)

    assert serving.policy_server_config.to_dict() == policy_config


def test_policy_server_config_reads_stage_config_model_config():
    policy_config = {"custom_model_key": "from-stage-config"}
    engine_client = SimpleNamespace(
        get_diffusion_od_config=lambda: None,
        stage_configs=[
            SimpleNamespace(
                stage_type="diffusion",
                engine_args=SimpleNamespace(model_config={"policy_server_config": policy_config}),
            )
        ],
    )

    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=engine_client)

    assert serving.policy_server_config.to_dict() == policy_config


def test_policy_server_config_reads_omegaconf_stage_config():
    engine_client = SimpleNamespace(
        get_diffusion_od_config=lambda: None,
        stage_configs=[
            SimpleNamespace(
                stage_type="diffusion",
                engine_args=SimpleNamespace(
                    model_config=OmegaConf.create({"policy_server_config": {"custom_model_key": "from-omegaconf"}})
                ),
            )
        ],
    )

    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=engine_client)

    assert serving.policy_server_config.to_dict() == {"custom_model_key": "from-omegaconf"}


def test_policy_server_config_is_required():
    od_config = SimpleNamespace(model_config={})
    engine_client = SimpleNamespace(get_diffusion_od_config=lambda: od_config)

    with pytest.raises(ValueError) as exc_info:
        openpi_serving.ServingRealtimeRobotOpenPI(engine_client=engine_client)

    assert "policy_server_config" in str(exc_info.value)


def test_create_policy_server_returns_none_without_policy_config():
    od_config = SimpleNamespace(model_config={})
    engine_client = SimpleNamespace(get_diffusion_od_config=lambda: od_config)

    serving = openpi_serving.ServingRealtimeRobotOpenPI.create_policy_server(
        engine_client=engine_client,
        model_name="k2-fsa/OmniVoice",
    )

    assert serving is None


def test_policy_server_config_reads_engine_model_config():
    policy_config = {"custom_model_key": "custom-value"}
    engine_client = SimpleNamespace(model_config=SimpleNamespace(policy_server_config=policy_config))

    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=engine_client)

    assert serving.policy_server_config.to_dict() == policy_config


def test_reset_marks_next_request_for_engine_state_reset():
    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=_engine_with_policy_config())
    serving._call_count = 3

    serving.reset({})
    serving._call_count += 1
    request = serving._build_request({"prompt": "pick up the object"})

    assert request.sampling_params.extra_args["reset"] is True
    assert request.sampling_params.extra_args["robot_obs"]["prompt"] == "pick up the object"
