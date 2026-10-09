# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.chatterbox import chatterbox
from vllm_omni.model_executor.models.chatterbox.chatterbox import STAGES, ChatterboxForConditionalGeneration
from vllm_omni.model_executor.models.registry import _OMNI_MODELS

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class StubStage(nn.Module):
    """A stage with one parameter, two runner flags and two hooks."""

    has_preprocess = True
    omni_pooler_payload_include_hidden = False

    def __init__(self, *, vllm_config: SimpleNamespace, prefix: str = "") -> None:
        super().__init__()
        self.prefix = prefix
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **runner_kwargs):
        return runner_kwargs

    def preprocess_decode_batch(self, *, input_ids, req_infos):
        return input_ids, input_ids, req_infos

    def load_weights(self, weights):
        return {"weight"}


@pytest.fixture
def unified(monkeypatch: pytest.MonkeyPatch) -> ChatterboxForConditionalGeneration:
    monkeypatch.setitem(chatterbox.STAGES, "stub", StubStage)
    config = SimpleNamespace(model_config=SimpleNamespace(model_stage="stub"))
    return ChatterboxForConditionalGeneration(vllm_config=config)


def test_stage_attributes_resolve_on_the_stage_001(unified: ChatterboxForConditionalGeneration) -> None:
    """The runner probes the registered class for flags and hooks the stage owns."""
    assert unified.has_preprocess is True
    assert unified.model.prefix == "model"
    # The runner reads both of these on the registered class, and without
    # them falls back to shipping hidden states and to one embedding call per row.
    assert getattr(unified, "omni_pooler_payload_include_hidden", True) is False
    assert getattr(unified, "preprocess_decode_batch", None) == unified.model.preprocess_decode_batch
    # What a stage does not define stays undefined, so runner defaults apply.
    assert not hasattr(unified, "on_requests_finished")


def test_loaded_names_carry_the_wrapper_prefix_001(unified: ChatterboxForConditionalGeneration) -> None:
    """vLLM fails the load on any parameter the returned set does not name."""
    assert unified.load_weights(iter(())) == {name for name, _ in unified.named_parameters()} == {"model.weight"}


def test_forward_passes_runner_keywords_through_001(unified: ChatterboxForConditionalGeneration) -> None:
    assert unified(torch.zeros(1), torch.zeros(1), seq_token_counts=[1]) == {"seq_token_counts": [1]}


def test_unknown_stage_is_refused_001() -> None:
    config = SimpleNamespace(model_config=SimpleNamespace(model_stage="chatterbox_vocoder"))
    with pytest.raises(ValueError, match="chatterbox_vocoder"):
        ChatterboxForConditionalGeneration(vllm_config=config)


def test_all_three_classes_are_registered_and_importable_001() -> None:
    assert set(STAGES) == {"chatterbox_t3", "chatterbox_s3gen"}
    for name in ("ChatterboxForConditionalGeneration", "ChatterboxT3ForConditionalGeneration", "ChatterboxS3Gen"):
        package, module, class_name = _OMNI_MODELS[name]
        loaded = importlib.import_module(f"vllm_omni.model_executor.models.{package}.{module}")
        assert getattr(loaded, class_name).__name__ == name
