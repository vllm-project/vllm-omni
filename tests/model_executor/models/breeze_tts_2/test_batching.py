# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from typing import Any

import pytest
import torch

from vllm_omni.model_executor.models.breeze_tts_2.modeling_breeze_tts_2_talker import (
    BreezeTTS2TalkerForGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _RequestHead:
    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        # Hidden row 0 selects codec id 2; row 1 selects codec id 3.
        code0 = 2 + int(hidden.reshape(-1)[0].item())
        logits = torch.zeros((hidden.shape[0], 9), dtype=hidden.dtype)
        logits[:, code0] = 1.0
        return logits


def _model() -> BreezeTTS2TalkerForGeneration:
    model = object.__new__(BreezeTTS2TalkerForGeneration)
    model.num_codebooks = 4
    model.codebook_vocab_size = 8
    model.codebook_size = 8
    model.codebook_eos_token_id = 0
    model.audio_token_id = 100
    model.audio_eos_token_id = 101
    model.lm_head = _RequestHead()
    model.logits_processor = lambda head, hidden: head(hidden)
    model._codec_disallowed_mask = None
    model._batch_state = None
    model._batch_state_spans = None
    model._async_chunk = True
    model._golden_dump_dir = None
    model._generate_depth_codes = lambda _hidden, code0: _frame(code0).unsqueeze(0)
    return model


def _frame(code0: int) -> torch.Tensor:
    return torch.tensor([code0, code0 + 1, code0 + 2, code0 + 3], dtype=torch.long)


def _step(model: BreezeTTS2TalkerForGeneration, infos: list[dict[str, Any]], hidden: torch.Tensor):
    return model.make_omni_output(
        hidden,
        model_intermediate_buffer=infos,
        request_token_spans=[(0, 1), (1, 2)],
    )


def test_two_requests_accumulate_and_emit_independent_frames():
    model = _model()
    infos = [
        {
            "breeze_generated_frames": 1,
            "breeze_max_new_frames": 2,
            "breeze_audio_codes": _frame(2).reshape(1, 4),
        },
        {"breeze_generated_frames": 0, "breeze_max_new_frames": 3},
    ]
    hidden = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

    first = _step(model, infos, hidden)
    assert [tuple(item.shape) for item in first.multimodal_outputs["codes"]["audio"]] == [(1, 4), (1, 4)]
    assert torch.equal(first.multimodal_outputs["codes"]["audio"][0][0], _frame(2))
    assert torch.equal(first.multimodal_outputs["codes"]["audio"][1][0], _frame(3))

    second = _step(model, infos, hidden)
    # Request 0 reaches its two-frame budget and emits only a finish marker;
    # request 1 still has budget and emits its next tail frame.
    assert second.multimodal_outputs["codes"]["audio"][0].numel() == 0
    assert torch.equal(second.multimodal_outputs["codes"]["audio"][1][0], _frame(3))
    # Async streaming only maintains the per-frame tail; the cumulative
    # buffer is deliberately not grown (request 0 keeps its pre-seeded
    # entry untouched, request 1 never gains one).
    assert infos[0]["breeze_audio_codes"].shape == (1, 4)
    assert "breeze_audio_codes" not in infos[1]
    assert infos[0]["breeze_force_eos"] is True
    assert infos[1]["breeze_force_eos"] is False


def test_finished_requests_only_clear_model_owned_batch_mapping():
    model = _model()
    model._batch_state = [{"id": "a"}, {"id": "b"}]
    model._batch_state_spans = [(0, 1), (1, 2)]

    model.on_requests_finished(["a", "b"])

    assert model._batch_state is None
    assert model._batch_state_spans is None
