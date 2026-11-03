# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import vllm_omni.diffusion.torch_tweak as torch_tweak_module
from vllm_omni.diffusion.torch_tweak import serve_with_torch_tweak

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _Pipeline(nn.Module):
    _dit_modules = ("transformer",)

    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Linear(2, 1)
        self.text_encoder = nn.Linear(2, 1)


def _od_config(torch_tweak):
    return SimpleNamespace(model="org/model", additional_config={"torch_tweak": torch_tweak})


@pytest.fixture
def on_xpu(monkeypatch):
    monkeypatch.setattr(torch_tweak_module.current_omni_platform, "is_xpu", lambda: True)


def test_stage_without_torch_tweak_is_not_served():
    pipeline = _Pipeline()

    served = serve_with_torch_tweak(pipeline, _od_config({}), torch.device("cpu"))

    assert served is False


def test_torch_tweak_is_ignored_outside_xpu(monkeypatch, tmp_path):
    monkeypatch.setattr(torch_tweak_module.current_omni_platform, "is_xpu", lambda: False)

    served = serve_with_torch_tweak(_Pipeline(), _od_config({"workdir": tmp_path}), torch.device("cpu"))

    assert served is False


def test_dit_modules_are_recorded_by_default(on_xpu, tmp_path):
    pytest.importorskip("torch_tweak")
    pipeline = _Pipeline()
    serve_with_torch_tweak(pipeline, _od_config({"workdir": tmp_path, "backends": ["eager"]}), torch.device("cpu"))

    pipeline.transformer(torch.ones(1, 2))
    pipeline.text_encoder(torch.ones(1, 2))

    assert [path.name for path in tmp_path.iterdir()] == ["org_model.transformer.rank0"]


def test_modules_setting_replaces_the_dit_modules(on_xpu, tmp_path):
    pytest.importorskip("torch_tweak")
    pipeline = _Pipeline()
    config = _od_config({"workdir": tmp_path, "backends": ["eager"], "modules": ["text_encoder"]})
    serve_with_torch_tweak(pipeline, config, torch.device("cpu"))

    pipeline.transformer(torch.ones(1, 2))
    pipeline.text_encoder(torch.ones(1, 2))

    assert [path.name for path in tmp_path.iterdir()] == ["org_model.text_encoder.rank0"]
