# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

import vllm_omni.diffusion.models.ernie_image.pipeline_ernie_image as pe_module
from vllm_omni.diffusion.models.ernie_image.pipeline_ernie_image import ErnieImagePipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    "model, expected",
    [
        ("baidu/ERNIE-Image-Turbo", True),
        ("baidu/ERNIE-Image", False),
        ("/data/models/ernie-image-turbo", True),
        ("/data/models/ernie-image-turbo/", True),
        ("/data/models/ernie-image-base", False),
    ],
)
def test_detect_distilled_by_name(model, expected, monkeypatch):
    monkeypatch.setattr(ErnieImagePipeline, "_read_model_index", staticmethod(lambda _model: {}))
    assert ErnieImagePipeline._detect_distilled(model) is expected


def test_detect_distilled_marker_takes_precedence_over_name():
    assert ErnieImagePipeline._detect_distilled("baidu/ERNIE-Image", {"is_distilled": True}) is True
    assert ErnieImagePipeline._detect_distilled("baidu/ERNIE-Image-Turbo", {"is_distilled": False}) is False


def test_detect_distilled_tolerates_unreadable_model_index(monkeypatch):
    def raise_offline(*_args, **_kwargs):
        raise OSError("offline")

    monkeypatch.setattr(pe_module, "get_hf_file_to_dict", raise_offline)
    assert ErnieImagePipeline._detect_distilled("baidu/ERNIE-Image-Turbo") is True
    assert ErnieImagePipeline._detect_distilled("baidu/ERNIE-Image") is False


def _make_pipe(is_distilled: bool, guidance_scale: float) -> ErnieImagePipeline:
    pipe = ErnieImagePipeline.__new__(ErnieImagePipeline)
    pipe.is_distilled = is_distilled
    pipe._guidance_scale = guidance_scale
    return pipe


def test_distilled_disables_cfg_at_default_guidance_scale():
    assert _make_pipe(True, 4.0).do_classifier_free_guidance is False


def test_base_keeps_cfg_at_default_guidance_scale():
    assert _make_pipe(False, 4.0).do_classifier_free_guidance is True


def test_base_cfg_off_below_threshold():
    assert _make_pipe(False, 1.0).do_classifier_free_guidance is False
