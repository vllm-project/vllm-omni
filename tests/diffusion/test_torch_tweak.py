# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn as nn

import vllm_omni.diffusion.torch_tweak as torch_tweak_module
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.torch_tweak import serve_with_torch_tweak

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _Pipeline(nn.Module):
    _dit_modules = ("transformer",)

    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Linear(2, 1)
        self.text_encoder = nn.Linear(2, 1)


@pytest.fixture
def on_xpu(monkeypatch):
    monkeypatch.setattr(torch_tweak_module.current_omni_platform, "is_xpu", lambda: True)


def test_stage_without_torch_tweak_is_not_served():
    served = serve_with_torch_tweak(_Pipeline(), OmniDiffusionConfig(model="org/model"), torch.device("cpu"))

    assert served is False


def test_torch_tweak_outside_xpu_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(torch_tweak_module.current_omni_platform, "is_xpu", lambda: False)
    od_config = OmniDiffusionConfig(model="org/model", additional_config={"torch_tweak": {"workdir": tmp_path}})

    with pytest.raises(ValueError, match="requires an Intel XPU"):
        serve_with_torch_tweak(_Pipeline(), od_config, torch.device("cpu"))


def test_torch_tweak_with_several_gpus_is_rejected(on_xpu, tmp_path):
    od_config = OmniDiffusionConfig(
        model="org/model",
        parallel_config=DiffusionParallelConfig(tensor_parallel_size=2),
        additional_config={"torch_tweak": {"workdir": tmp_path}},
    )

    with pytest.raises(ValueError, match="single-GPU"):
        serve_with_torch_tweak(_Pipeline(), od_config, torch.device("cpu"))


def test_torch_tweak_with_cpu_offload_is_rejected(on_xpu, tmp_path):
    od_config = OmniDiffusionConfig(
        model="org/model", enable_cpu_offload=True, additional_config={"torch_tweak": {"workdir": tmp_path}}
    )

    with pytest.raises(ValueError, match="CPU offloading"):
        serve_with_torch_tweak(_Pipeline(), od_config, torch.device("cpu"))


def test_torch_tweak_with_cache_backend_is_rejected(on_xpu, tmp_path):
    od_config = OmniDiffusionConfig(
        model="org/model", cache_backend="cache_dit", additional_config={"torch_tweak": {"workdir": tmp_path}}
    )

    with pytest.raises(ValueError, match="cache_backend='none'"):
        serve_with_torch_tweak(_Pipeline(), od_config, torch.device("cpu"))


def test_dit_modules_are_recorded_by_default(on_xpu, tmp_path):
    pytest.importorskip("torch_tweak")
    pipeline = _Pipeline()
    torch_tweak = {"workdir": tmp_path, "backends": ["eager"]}
    od_config = OmniDiffusionConfig(model="org/model", additional_config={"torch_tweak": torch_tweak})
    serve_with_torch_tweak(pipeline, od_config, torch.device("cpu"))

    pipeline.transformer(torch.ones(1, 2))
    pipeline.text_encoder(torch.ones(1, 2))

    assert [path.name for path in tmp_path.iterdir()] == ["transformer"]


def test_modules_setting_replaces_the_dit_modules(on_xpu, tmp_path):
    pytest.importorskip("torch_tweak")
    pipeline = _Pipeline()
    torch_tweak = {"workdir": tmp_path, "backends": ["eager"], "modules": ["text_encoder"]}
    od_config = OmniDiffusionConfig(model="org/model", additional_config={"torch_tweak": torch_tweak})
    serve_with_torch_tweak(pipeline, od_config, torch.device("cpu"))

    pipeline.transformer(torch.ones(1, 2))
    pipeline.text_encoder(torch.ones(1, 2))

    assert [path.name for path in tmp_path.iterdir()] == ["text_encoder"]
