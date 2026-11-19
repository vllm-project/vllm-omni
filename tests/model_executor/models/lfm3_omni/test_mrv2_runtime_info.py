# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass

import pytest
import torch
from torch import nn
from vllm import SamplingParams

from vllm_omni.model_executor.models.lfm3_omni.backbone import RequestState
from vllm_omni.model_executor.models.lfm3_omni.model import Lfm3OmniForConditionalGeneration
from vllm_omni.model_executor.models.lfm3_omni.pipeline import LFM3_OMNI_PIPELINE, LFM3_OMNI_TEXT_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class TokenConfig:
    num_codebooks: int = 12
    audio_start_token_id: int = 5
    mixed_start_token_id: int = 6
    audio_out_token_id: int = 7
    text_end_token_id: int = 8
    interleaved_n_text: int = 2
    interleaved_n_audio: int = 3


class RecordingBackbone(nn.Module):
    def forward(self, x, positions, requests):
        self.requests = requests
        return x


def make_model():
    model = Lfm3OmniForConditionalGeneration.__new__(Lfm3OmniForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = TokenConfig()
    model.model = nn.Module()
    model.model.language_model = RecordingBackbone()
    model.model.embed_tokens = nn.Embedding(16, 4)
    model._states = {}
    model._speech_defaults = {}
    model._supports_mm_inputs = False
    return model


def embed(model, input_ids):
    return model.model.embed_tokens(input_ids)


def test_profile_placeholder_rows_run_like_an_empty_batch():
    model = make_model()
    hidden = model(
        torch.tensor([1, 2]),
        torch.arange(2),
        inputs_embeds=torch.zeros(2, 4),
        runtime_additional_information=[{}, {}],
        seq_token_counts=[1, 1],
    )
    assert isinstance(hidden, torch.Tensor)
    assert model.model.language_model.requests == []


def test_placeholder_rows_are_skipped_in_place():
    model = make_model()
    model.lm_head = nn.Linear(4, 16, bias=False)
    model._placeholder_token_ids = []
    state = RequestState()
    model._states = {"a": state}
    live = {"req_id": "a", "lfm3_span": 2, "sampling_params": SamplingParams(extra_args={"generation_mode": "text"})}
    output = model(
        torch.arange(6),
        torch.arange(6),
        inputs_embeds=torch.zeros(6, 4),
        runtime_additional_information=[{}, live, {}],
        seq_token_counts=[3, 2, 1],
    )
    assert model.model.language_model.requests == [(3, 5, state)]
    assert [frame.shape for frame in output.multimodal_outputs["codes"]["audio"]] == [(0, 12)] * 3
    logits = model.compute_logits(torch.zeros(3, 4))
    masked = [model.config.audio_start_token_id, model.config.mixed_start_token_id]
    assert torch.isinf(logits[1, masked]).all()
    assert not torch.isinf(logits[[0, 2]]).any()


def test_placeholder_rows_need_seq_token_counts():
    model = make_model()
    with pytest.raises(ValueError, match="seq_token_counts"):
        model(torch.tensor([1]), torch.arange(1), inputs_embeds=torch.zeros(1, 4), runtime_additional_information=[{}])


def test_rows_without_runner_extra_args_use_their_sampling_params():
    model = make_model()
    model._states = {"a": RequestState(), "b": RequestState()}
    infos = [
        {
            "req_id": "a",
            "lfm3_span": 1,
            "sampling_params": SamplingParams(extra_args={"generation_mode": "text", "audio_temperature": 0.0}),
        },
        {"req_id": "b", "lfm3_span": 1, "sampling_params": SamplingParams()},
    ]
    model(torch.tensor([1, 2]), torch.arange(2), inputs_embeds=torch.zeros(2, 4), runtime_additional_information=infos)
    assert model._states["a"].text_only
    assert model._states["a"].audio_temperature == 0.0
    assert not model._states["b"].text_only
    assert model._states["b"].audio_temperature == 0.4


def test_runner_extra_args_take_precedence_over_sampling_params():
    model = make_model()
    model._states = {"a": RequestState()}
    infos = [{"req_id": "a", "lfm3_span": 1, "sampling_params": SamplingParams()}]
    model(
        torch.tensor([1]),
        torch.arange(1),
        inputs_embeds=torch.zeros(1, 4),
        runtime_additional_information=infos,
        sampling_extra_args=[{"generation_mode": "text"}],
    )
    assert model._states["a"].text_only


@pytest.mark.parametrize("key", ["request_id", "req_id"])
def test_preprocess_accepts_either_request_id_key(key, monkeypatch):
    model = make_model()
    monkeypatch.setattr(model, "embed_input_ids", lambda ids: embed(model, ids))
    _, _, updates = model.preprocess(torch.tensor([3, 4]), None, **{key: "a"}, _omni_is_prefill=True, _omni_seed=7)
    assert updates == {"lfm3_span": 2, "request_id": "a"}
    assert set(model._states) == {"a"}


def test_preprocess_rejects_a_row_without_request_id(monkeypatch):
    model = make_model()
    monkeypatch.setattr(model, "embed_input_ids", lambda ids: embed(model, ids))
    with pytest.raises(ValueError, match="request id"):
        model.preprocess(torch.tensor([3]), None, _omni_is_prefill=True)


def test_model_runner_v2_contract_declarations():
    # Without these, MRv2 drops media inputs and never sends the full codec payload.
    assert Lfm3OmniForConditionalGeneration.preprocess_keeps_mm_inputs
    for pipeline in (LFM3_OMNI_PIPELINE, LFM3_OMNI_TEXT_PIPELINE):
        assert all(stage.supports_native_mrv2_data_plane for stage in pipeline.stages)
