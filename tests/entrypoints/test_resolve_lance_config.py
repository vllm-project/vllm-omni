import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig, resolve_model_class_name
from vllm_omni.diffusion.stage_diffusion_proc import StageDiffusionProc

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_lance_subfolder_resolves_model_class(monkeypatch):
    monkeypatch.setattr(
        "vllm.transformers_utils.config.get_hf_file_to_dict",
        lambda _path, _model: None,
    )

    result = resolve_model_class_name("/fake/path/Lance_3B")

    assert result == "LancePipeline"


def test_lance_enrich_config_from_subfolder(monkeypatch):
    monkeypatch.setattr(
        "vllm.transformers_utils.config.get_hf_file_to_dict",
        lambda _path, _model: None,
    )

    od_config = OmniDiffusionConfig(model="/fake/path/Lance_3B")
    proc = StageDiffusionProc(od_config.model, od_config)

    proc._enrich_config()

    assert od_config.model_class_name == "LancePipeline"
