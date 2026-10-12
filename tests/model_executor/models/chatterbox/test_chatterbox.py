# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.chatterbox import chatterbox
from vllm_omni.model_executor.models.chatterbox.chatterbox import STAGES, ChatterboxForConditionalGeneration
from vllm_omni.model_executor.models.chatterbox.chatterbox_s3gen import ChatterboxS3Gen
from vllm_omni.model_executor.models.chatterbox.chatterbox_t3 import ChatterboxT3ForConditionalGeneration
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


@pytest.mark.parametrize(
    ("stage", "defined", "undefined"),
    [
        (
            ChatterboxT3ForConditionalGeneration,
            [
                "has_preprocess",
                "have_multimodal_outputs",
                "omni_pooler_payload_include_hidden",
                "preprocess",
                "preprocess_decode_batch",
                "preprocess_decode_batch_mrv2",
                "make_omni_output",
            ],
            # Either would change which of the hooks above Model Runner V2 calls.
            ["mrv2_decode_preprocess_is_identity", "make_omni_output_mrv2", "on_requests_finished"],
        ),
        (
            ChatterboxS3Gen,
            ["have_multimodal_outputs", "requires_request_ids", "on_requests_finished", "decode_step"],
            ["has_preprocess", "make_omni_output"],
        ),
    ],
    ids=["chatterbox_t3", "chatterbox_s3gen"],
)
def test_the_runners_find_each_stages_hooks_on_the_registered_class_001(
    stage: type[nn.Module], defined: list[str], undefined: list[str]
) -> None:
    """Both runners look hooks up on the class the registry names, with ``getattr`` and ``hasattr``."""
    unified = object.__new__(ChatterboxForConditionalGeneration)
    nn.Module.__init__(unified)
    unified.model = object.__new__(stage)
    nn.Module.__init__(unified.model)

    for name in defined:
        assert getattr(unified, name) == getattr(unified.model, name), name
    for name in undefined:
        assert not hasattr(unified, name), name


def test_loaded_names_carry_the_wrapper_prefix_001(unified: ChatterboxForConditionalGeneration) -> None:
    """vLLM fails the load on any parameter the returned set does not name."""
    assert unified.load_weights(iter(())) == {name for name, _ in unified.named_parameters()} == {"model.weight"}


def test_forward_passes_runner_keywords_through_001(unified: ChatterboxForConditionalGeneration) -> None:
    assert unified(torch.zeros(1), torch.zeros(1), seq_token_counts=[1]) == {"seq_token_counts": [1]}


def test_unknown_stage_is_refused_by_name_001() -> None:
    config = SimpleNamespace(model_config=SimpleNamespace(model_stage="chatterbox_vocoder"))
    with pytest.raises(KeyError, match="chatterbox_vocoder"):
        ChatterboxForConditionalGeneration(vllm_config=config)


def test_all_three_classes_are_registered_and_importable_001() -> None:
    assert set(STAGES) == {"chatterbox_t3", "chatterbox_original_t3", "chatterbox_s3gen"}
    for name in ("ChatterboxForConditionalGeneration", "ChatterboxT3ForConditionalGeneration", "ChatterboxS3Gen"):
        package, module, class_name = _OMNI_MODELS[name]
        loaded = importlib.import_module(f"vllm_omni.model_executor.models.{package}.{module}")
        assert getattr(loaded, class_name).__name__ == name
