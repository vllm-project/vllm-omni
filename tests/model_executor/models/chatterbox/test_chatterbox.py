# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.model_executor.models.interfaces import supports_pp
from vllm.model_executor.models.interfaces_base import is_text_generation_model

from vllm_omni.model_executor.models.chatterbox import chatterbox
from vllm_omni.model_executor.models.chatterbox.chatterbox import STAGES, ChatterboxForConditionalGeneration
from vllm_omni.model_executor.models.chatterbox.chatterbox_original_t3 import ChatterboxOriginalT3
from vllm_omni.model_executor.models.chatterbox.chatterbox_s3gen import ChatterboxS3Gen
from vllm_omni.model_executor.models.chatterbox.chatterbox_t3 import ChatterboxT3ForConditionalGeneration
from vllm_omni.model_executor.models.chatterbox.pipeline import CHATTERBOX_ORIGINAL_PIPELINE, CHATTERBOX_TURBO_PIPELINE
from vllm_omni.model_executor.models.registry import _OMNI_MODELS, OmniModelRegistry

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class StubStage(nn.Module):
    """A stage with one parameter, a forward and a loader."""

    def __init__(self, *, vllm_config: SimpleNamespace, prefix: str = "") -> None:
        super().__init__()
        self.prefix = prefix
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **runner_kwargs):
        return runner_kwargs

    def load_weights(self, weights):
        return {"weight"}


@pytest.fixture
def unified(monkeypatch: pytest.MonkeyPatch) -> ChatterboxForConditionalGeneration:
    monkeypatch.setitem(chatterbox.STAGES, "stub", StubStage)
    config = SimpleNamespace(model_config=SimpleNamespace(model_stage="stub"))
    return ChatterboxForConditionalGeneration(vllm_config=config)


@pytest.mark.parametrize(
    ("stage", "flags", "hooks", "undefined"),
    [
        (
            ChatterboxT3ForConditionalGeneration,
            # Without the first the runner never calls preprocess; without the
            # second it drops make_omni_output's payload; with the third true it
            # ships every hidden state to stage 1, which reads none.
            {"has_preprocess": True, "have_multimodal_outputs": True, "omni_pooler_payload_include_hidden": False},
            ["preprocess", "preprocess_decode_batch", "preprocess_decode_batch_mrv2", "make_omni_output"],
            # Either would change which of the hooks above Model Runner V2 calls.
            ["mrv2_decode_preprocess_is_identity", "make_omni_output_mrv2", "on_requests_finished"],
        ),
        (
            ChatterboxOriginalT3,
            # The last two have the runner sample through the stage's guidance
            # sampler and hand it each request's guidance pair.
            {
                "has_preprocess": True,
                "have_multimodal_outputs": True,
                "omni_pooler_payload_include_hidden": False,
                "prefer_model_sampler": True,
                "model_sampler_wants_extra_args": True,
            },
            ["preprocess", "preprocess_decode_batch", "preprocess_decode_batch_mrv2", "sample", "on_requests_finished"],
            ["mrv2_decode_preprocess_is_identity", "make_omni_output_mrv2"],
        ),
        (
            ChatterboxS3Gen,
            # Without requires_request_ids the runner passes no ids, and
            # decode_step returns empty audio for every request with no error.
            {
                "have_multimodal_outputs": True,
                "requires_request_ids": True,
                "enable_update_additional_information": True,
            },
            ["on_requests_finished"],
            ["has_preprocess", "make_omni_output"],
        ),
    ],
    ids=["chatterbox_t3", "chatterbox_original_t3", "chatterbox_s3gen"],
)
def test_the_runners_find_each_stages_hooks_on_the_registered_class(
    stage: type[nn.Module], flags: dict[str, bool], hooks: list[str], undefined: list[str]
) -> None:
    """Both runners look hooks up on the class the registry names, with ``getattr`` and ``hasattr``."""
    unified = object.__new__(ChatterboxForConditionalGeneration)
    nn.Module.__init__(unified)
    unified.model = object.__new__(stage)
    nn.Module.__init__(unified.model)

    assert {name: getattr(unified, name) for name in flags} == flags
    for name in hooks:
        assert getattr(unified, name) == getattr(unified.model, name), name
    for name in undefined:
        assert not hasattr(unified, name), name


def test_loaded_names_carry_the_wrapper_prefix(unified: ChatterboxForConditionalGeneration) -> None:
    """vLLM fails the load on any parameter the returned set does not name."""
    assert unified.model.prefix == "model"
    assert unified.load_weights(iter(())) == {name for name, _ in unified.named_parameters()} == {"model.weight"}


def test_forward_passes_runner_keywords_through(unified: ChatterboxForConditionalGeneration) -> None:
    assert unified(torch.zeros(1), torch.zeros(1), seq_token_counts=[1]) == {"seq_token_counts": [1]}


def test_unknown_stage_is_refused_by_name() -> None:
    config = SimpleNamespace(model_config=SimpleNamespace(model_stage="chatterbox_vocoder"))
    with pytest.raises(KeyError, match="chatterbox_vocoder"):
        ChatterboxForConditionalGeneration(vllm_config=config)


def test_one_registered_architecture_wraps_every_stage() -> None:
    """The pipelines name the wrapper for every stage; the stage classes are not architectures."""
    pipelines = (CHATTERBOX_TURBO_PIPELINE, CHATTERBOX_ORIGINAL_PIPELINE)
    assert {stage.model_stage for pipeline in pipelines for stage in pipeline.stages} == set(STAGES)
    assert {pipeline.model_arch for pipeline in pipelines} == {"ChatterboxForConditionalGeneration"}
    assert [name for name in _OMNI_MODELS if name.startswith("Chatterbox")] == ["ChatterboxForConditionalGeneration"]
    model_cls = OmniModelRegistry._try_load_model_cls("ChatterboxForConditionalGeneration")
    assert model_cls is ChatterboxForConditionalGeneration


def test_vllm_gives_the_registered_class_the_generate_runner() -> None:
    """vLLM inspects the class, not an instance, so ``__getattr__`` cannot answer for it.

    ``embed_input_ids``, ``forward`` and ``compute_logits`` on the class make
    it a text-generation model; ``make_empty_intermediate_tensors`` makes it
    pipeline-parallel.
    """
    assert is_text_generation_model(ChatterboxForConditionalGeneration)
    assert supports_pp(ChatterboxForConditionalGeneration)
