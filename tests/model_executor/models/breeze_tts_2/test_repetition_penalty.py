# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from typing import Any

import pytest
import torch
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata

from tests.model_executor.models.breeze_tts_2.test_talker_state_machine import _depth_codes, _talker

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _model(first_score=11.0, second_score=10.5):
    model = _talker()
    model._async_chunk = True
    logits = torch.full((1, 9), -100.0)
    logits[0, 2:4] = torch.tensor([first_score, second_score])
    model._main_head_logits = lambda hidden: logits.expand(hidden.shape[0], -1).clone()
    model._generate_depth_codes = lambda hidden, code0: _depth_codes(model, hidden, code0)
    return model


def _step(model, infos, penalties, *, no_penalties=False):
    return model.make_omni_output(
        torch.zeros(len(infos), 16),
        model_intermediate_buffer=infos,
        request_token_spans=[(i, i + 1) for i in range(len(infos))],
        sampling_metadata=SamplingMetadata(
            temperature=None,
            all_greedy=True,
            all_random=False,
            top_p=None,
            top_k=None,
            generators={},
            max_num_logprobs=None,
            no_penalties=no_penalties,
            frequency_penalties=torch.zeros(len(infos)),
            presence_penalties=torch.zeros(len(infos)),
            repetition_penalties=torch.tensor(penalties),
            # Text/reference conditioning must never be counted as generated
            # codec history, even if the scheduler ids overlap the codec ids.
            prompt_token_ids=torch.tensor([[2, 2]] * len(infos)),
            output_token_ids=[list(info.get("breeze_generated_code0_ids", [])) for info in infos],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
        ),
    )


@pytest.mark.parametrize("scores", [(11.0, 10.5), (-1.0, -1.05)])
def test_penalty_applies_once_per_generated_token_and_respects_logit_sign(scores):
    model = _model(*scores)
    info = {"prompt_ids": torch.tensor([2]), "input_values": torch.tensor([[2, 3, 4, 5]])}
    chosen = []

    for _ in range(4):
        output = _step(model, [info], [1.1])
        chosen.append(output.multimodal_outputs["codes"]["audio"][0][0, 0].item())

    assert chosen == [2, 3, 2, 2]


def test_request_history_and_penalty_are_independent():
    model = _model()
    first: dict[str, Any] = {}
    second: dict[str, Any] = {}
    _step(model, [first], [1.1])
    output = _step(model, [first, second], [1.1, 1.1])
    assert [codes[0, 0].item() for codes in output.multimodal_outputs["codes"]["audio"]] == [3, 2]

    # Reverse the batch order; a request's history follows its state, while
    # each penalty follows the current sampling-metadata row.
    output = _step(model, [second, first], [1.1, 1.0])
    assert [codes[0, 0].item() for codes in output.multimodal_outputs["codes"]["audio"]] == [3, 2]


def test_disabled_penalties_do_not_read_uninitialized_sampling_buffer():
    model = _model()
    info: dict[str, Any] = {}
    for _ in range(2):
        output = _step(model, [info], [float("nan")], no_penalties=True)
        assert output.multimodal_outputs["codes"]["audio"][0][0, 0].item() == 2


def test_scheduler_logits_preserve_selected_frame_under_further_sampling_transforms():
    model = _model()
    info: dict[str, Any] = {}
    _step(model, [info], [1.1])
    output = _step(model, [info], [1.1])
    selected = output.multimodal_outputs["codes"]["audio"][0][0, 0].item()
    logits = model.compute_logits(torch.zeros(1, 16))

    assert selected == 3
    assert logits.argmax(dim=-1).item() == selected
    assert torch.isfinite(logits).sum().item() == 1
    assert logits[0, selected].item() == 0.0
    # The scheduler may apply repetition again and rescale temperature. A
    # finite zero for the selected token preserves a valid point mass.
    for temperature in (0.1, 0.9, 2.0):
        probabilities = torch.softmax(logits / 1.1 / temperature, dim=-1)
        assert torch.isfinite(probabilities).all()
        assert probabilities[0, selected].item() == 1.0


def test_repetition_penalty_can_select_eos_without_emitting_a_frame():
    model = _model()
    info: dict[str, Any] = {}
    _step(model, [info], [1.1])
    logits = torch.full((1, 9), -100.0)
    logits[0, 2] = 11.0
    logits[0, 8] = 10.5
    model._main_head_logits = lambda hidden: logits.expand(hidden.shape[0], -1).clone()

    output = _step(model, [info], [1.1])

    assert output.multimodal_outputs["codes"]["audio"][0].numel() == 0
    assert info["breeze_generated_frames"] == 1
    assert model.compute_logits(torch.zeros(1, 16)).argmax(dim=-1).item() == 8
