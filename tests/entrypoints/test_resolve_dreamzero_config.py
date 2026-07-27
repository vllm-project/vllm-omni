import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.stage_diffusion_proc import StageDiffusionProc
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
    result = resolve_model_config_path("GEAR-Dreams/DreamZero-DROID")

    assert result is not None
    assert result.endswith("vllm_omni/deploy/dreamzero.yaml")


def test_dreamzero_config_sets_model_class_and_policy_config(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.config.config_factory.StageConfigFactory._try_infer_model_type",
        classmethod(lambda _cls, model, trust_remote_code=True: "vla"),
    )
    monkeypatch.setattr(
        "vllm_omni.config.config_factory.StageConfigFactory.get_hf_config",
        classmethod(lambda _cls, model, trust_remote_code=True: None),
    )
    monkeypatch.setattr(
        "vllm_omni.config.config_factory._looks_like_dreamzero",
        lambda _model: True,
    )

    stage_configs, _ = load_stage_configs_from_model(
        "GEAR-Dreams/DreamZero-DROID",
        trust_remote_code=False,
    )
    engine_args = stage_configs[0].engine_args

    assert engine_args.model_class_name == "DreamZeroPipeline"
    assert engine_args.model_config.policy_server_config.action_space == "joint_position"


def test_dreamzero_enrich_config_preserves_explicit_model_class_name(monkeypatch):
    monkeypatch.setattr(
        "vllm.transformers_utils.config.get_hf_file_to_dict",
        lambda path, _model: None if path == "model_index.json" else {"model_type": "vla", "architectures": ["VLA"]},
    )

    od_config = OmniDiffusionConfig(
        model="GEAR-Dreams/DreamZero-DROID",
        model_class_name="DreamZeroPipeline",
    )
    proc = StageDiffusionProc(od_config.model, od_config)

    proc._enrich_config()

    assert od_config.model_class_name == "DreamZeroPipeline"
