# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm_omni.model_executor.models.llama_omni2.llama_omni2 as wrapper_module
from vllm_omni.model_executor.models.llama_omni2.llama_omni2 import (
    Omni2Speech2SQwen2ForCausalLM,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeStage(nn.Module):
    def __init__(self, *, vllm_config, prefix):
        super().__init__()
        self.vllm_config = vllm_config
        self.prefix = prefix
        self.make_empty_intermediate_tensors = object()
        self.sampler = object()

    def forward(self, value):
        return ("forward", value)

    def compute_logits(self, value):
        return ("logits", value)

    def load_weights(self, weights):
        return {name for name, _ in weights}

    def embed_multimodal(self, **kwargs):
        return ("multimodal", kwargs)

    def get_language_model(self):
        return "language-model"


def _config(stage):
    return SimpleNamespace(
        model_config=SimpleNamespace(model_stage=stage),
    )


@pytest.mark.parametrize("stage", ("thinker", "talker", "code2wav"))
def test_wrapper_constructs_only_selected_stage(monkeypatch, stage):
    calls = []

    def factory(*, vllm_config, prefix):
        calls.append((vllm_config.model_config.model_stage, prefix))
        return _FakeStage(vllm_config=vllm_config, prefix=prefix)

    monkeypatch.setitem(wrapper_module._STAGE_FACTORIES, stage, factory)
    model = Omni2Speech2SQwen2ForCausalLM(
        vllm_config=_config(stage),
        prefix="root",
    )

    assert isinstance(model.model, _FakeStage)
    assert calls == [(stage, "root")]


def test_wrapper_rejects_unknown_stage():
    with pytest.raises(
        ValueError,
        match=r"Unsupported LLaMA-Omni 2 model stage 'other'.*thinker.*talker.*code2wav",
    ):
        Omni2Speech2SQwen2ForCausalLM(vllm_config=_config("other"))


def test_wrapper_delegates_runtime_methods(monkeypatch):
    monkeypatch.setitem(wrapper_module._STAGE_FACTORIES, "thinker", _FakeStage)
    model = Omni2Speech2SQwen2ForCausalLM(vllm_config=_config("thinker"))
    value = torch.tensor([1.0])

    assert model(value) == ("forward", value)
    assert model.compute_logits(value) == ("logits", value)
    assert model.load_weights([("weight", value)]) == {"model.weight"}
    assert model.embed_multimodal(audio=value) == (
        "multimodal",
        {"audio": value},
    )
    assert model.get_language_model() == "language-model"
    assert model.make_empty_intermediate_tensors is model.model.make_empty_intermediate_tensors
    assert model.sampler is model.model.sampler
