import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig, resolve_model_class_name
from vllm_omni.diffusion.stage_diffusion_proc import StageDiffusionProc

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("cfg", "expected"),
    [
        ({"model_type": "bagel", "architectures": []}, "BagelPipeline"),
        ({"model_type": "neo_chat", "architectures": []}, "SenseNovaU1Pipeline"),
        (
            {"model_type": "ming_flash_omni", "architectures": []},
            "MingImagePipeline",
        ),
        ({"model_type": "nextstep", "architectures": []}, "NextStep11Pipeline"),
        ({"model_type": "s2v", "architectures": []}, "WanS2VPipeline"),
    ],
)
def test_standard_model_families_resolve_model_class(monkeypatch, cfg, expected):
    monkeypatch.setattr(
        "vllm.transformers_utils.config.get_hf_file_to_dict",
        lambda path, _model: None if path == "model_index.json" else cfg,
    )

    assert resolve_model_class_name("fake-model") == expected


def test_resolve_model_class_name_falls_back_to_single_architecture(monkeypatch):
    monkeypatch.setattr(
        "vllm.transformers_utils.config.get_hf_file_to_dict",
        lambda path, _model: None
        if path == "model_index.json"
        else {"model_type": "unknown", "architectures": ["FooPipeline"]},
    )

    assert resolve_model_class_name("fake-model") == "FooPipeline"


def test_gr00t_not_resolved_by_resolve_model_class_name(monkeypatch):
    monkeypatch.setattr(
        "vllm.transformers_utils.config.get_hf_file_to_dict",
        lambda path, _model: None
        if path == "model_index.json"
        else {"model_type": "other", "architectures": ["Gr00tN1d7"]},
    )

    # Gr00tN1d7 remains enrich-only by design and should not be surfaced
    # through the read-only resolve_model_class_name() helper.
    assert resolve_model_class_name("nvidia/GR00T-N1.5") is None


def test_nextstep_enrich_config_preserves_explicit_model_class_name(monkeypatch):
    monkeypatch.setattr(
        "vllm.transformers_utils.config.get_hf_file_to_dict",
        lambda path, _model: None if path == "model_index.json" else {"model_type": "nextstep", "architectures": []},
    )

    od_config = OmniDiffusionConfig(
        model="bytedance-research/NextStep-1-Large",
        model_class_name="UserProvidedPipeline",
    )
    proc = StageDiffusionProc(od_config.model, od_config)

    proc._enrich_config()

    assert od_config.model_class_name == "UserProvidedPipeline"


def test_gr00t_enrich_config_sets_gr00t_pipeline(monkeypatch):
    monkeypatch.setattr(
        "vllm.transformers_utils.config.get_hf_file_to_dict",
        lambda path, _model: None
        if path == "model_index.json"
        else {"model_type": "other", "architectures": ["Gr00tN1d7"]},
    )

    od_config = OmniDiffusionConfig(model="nvidia/GR00T-N1.5")
    proc = StageDiffusionProc(od_config.model, od_config)

    proc._enrich_config()

    assert od_config.model_class_name == "Gr00tN1d7Pipeline"


def test_s2v_enrich_config_preserves_explicit_model_class_name(monkeypatch):
    monkeypatch.setattr(
        "vllm.transformers_utils.config.get_hf_file_to_dict",
        lambda path, _model: None if path == "model_index.json" else {"model_type": "s2v", "architectures": []},
    )

    od_config = OmniDiffusionConfig(
        model="Wan-AI/Wan2.2-S2V-14B",
        model_class_name="UserProvidedPipeline",
    )
    proc = StageDiffusionProc(od_config.model, od_config)

    proc._enrich_config()

    assert od_config.model_class_name == "UserProvidedPipeline"
