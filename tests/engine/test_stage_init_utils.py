from types import SimpleNamespace

from vllm_omni.engine.stage_init_utils import build_engine_args_dict


def test_build_engine_args_dict_preserves_stage_model_override():
    stage_cfg = SimpleNamespace(
        stage_id=1,
        stage_type="llm",
        engine_args=SimpleNamespace(model="stage-model", worker_type="ar"),
    )

    engine_args = build_engine_args_dict(stage_cfg, model="cli-model")

    assert engine_args["model"] == "stage-model"
    assert engine_args["stage_id"] == 1


def test_build_engine_args_dict_falls_back_to_cli_model():
    stage_cfg = SimpleNamespace(
        stage_id=0,
        stage_type="llm",
        engine_args=SimpleNamespace(worker_type="ar"),
    )

    engine_args = build_engine_args_dict(stage_cfg, model="cli-model")

    assert engine_args["model"] == "cli-model"
    assert engine_args["stage_id"] == 0
