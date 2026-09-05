# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from typing import Any

import pytest
import torch

from vllm_omni.model_executor.models.breeze_tts_2.modeling_breeze_tts_2_talker import (
    BreezeTTS2TalkerForGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Head:
    def __init__(self) -> None:
        self.code0 = 2

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((hidden.shape[0], 9), dtype=hidden.dtype)
        logits[:, self.code0] = 1.0
        return logits


class _Embedding:
    def __init__(self) -> None:
        self.weight = torch.randn(4 * 8, 16)

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        return self.weight[ids.reshape(-1)].reshape(*ids.shape, -1)


def _talker() -> BreezeTTS2TalkerForGeneration:
    model = object.__new__(BreezeTTS2TalkerForGeneration)
    model.num_codebooks = 4
    model.codebook_vocab_size = 8
    model.codebook_size = 8
    model.hidden_size = 16
    model.codebook_eos_token_id = 0
    model.audio_token_id = 100
    model.audio_eos_token_id = 101
    # Bypass nn.Module registration; this fake is created with object.__new__.
    model.__dict__["embed_audio_tokens"] = _Embedding()
    model.audio_token_offsets = torch.arange(4) * 8
    model.lm_head = _Head()
    model.logits_processor = lambda head, hidden: head(hidden)
    model._codec_disallowed_mask = None
    model._batch_state = None
    model._batch_state_spans = None
    model._async_chunk = False
    model._golden_dump_dir = None
    return model


def _depth_codes(_model: Any, _hidden: torch.Tensor, code0: int) -> torch.Tensor:
    if code0 >= 8:
        return torch.empty(0, dtype=torch.long)
    return torch.tensor([code0, code0 + 1, code0 + 2, code0 + 3], dtype=torch.long).unsqueeze(0)


def test_prefill_hidden_generates_and_stores_first_complete_frame():
    model = _talker()
    model._generate_depth_codes = lambda hidden, code0: _depth_codes(model, hidden, code0)
    info = {"_omni_is_prefill": True, "breeze_generated_frames": 0, "breeze_max_new_frames": 3}
    hidden = torch.arange(16, dtype=torch.float32).reshape(1, 16)

    output = model.make_omni_output(
        hidden,
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    )

    codes = output.multimodal_outputs["codes"]["audio"][0]
    assert torch.equal(codes, torch.tensor([[2, 3, 4, 5]]))
    assert torch.equal(info["breeze_current_frame"], codes[0])
    assert info["breeze_generated_frames"] == 1
    assert info["breeze_force_eos"] is False


def test_first_decode_consumes_full_frame_instead_of_scalar_eos_fallback():
    model = _talker()
    model._generate_depth_codes = lambda hidden, code0: _depth_codes(model, hidden, code0)
    info = {"_omni_is_prefill": True, "breeze_generated_frames": 0, "breeze_max_new_frames": 3}
    hidden = torch.arange(16, dtype=torch.float32).reshape(1, 16)
    output = model.make_omni_output(
        hidden,
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    )
    frame = output.multimodal_outputs["codes"]["audio"][0][0]
    decode_info = dict(info)
    decode_info.pop("_omni_is_prefill", None)

    safe_id, decode_embeds, update = model.preprocess(
        torch.tensor([2], dtype=torch.long),
        None,
        **decode_info,
    )

    assert safe_id.item() == 2
    assert update == {}
    expected = model.embed_input_ids(frame.reshape(1, 1, -1)).reshape(1, -1)
    assert torch.equal(decode_embeds, expected)


def test_real_decode_rejects_missing_current_frame():
    model = _talker()
    info = {
        "breeze_generated_frames": 1,
        "breeze_audio_codes": torch.tensor([[2, 3, 4, 5]]),
    }

    try:
        model.preprocess(torch.tensor([2], dtype=torch.long), None, **info)
    except RuntimeError as exc:
        assert "breeze_current_frame" in str(exc)
        assert "codebooks 1..15" in str(exc)
    else:
        raise AssertionError("missing breeze_current_frame must fail explicitly")


def test_embed_input_ids_rejects_scalar_codec_id():
    model = _talker()

    try:
        model.embed_input_ids(torch.tensor([[2]]))
    except ValueError as exc:
        assert "complete codec frames" in str(exc)
    else:
        raise AssertionError("scalar codec ids must not be embedded")


def test_decode_accumulates_frames_then_natural_eos_stops_without_fake_frame():
    model = _talker()
    model._generate_depth_codes = lambda hidden, code0: _depth_codes(model, hidden, code0)
    info = {"breeze_generated_frames": 1, "breeze_max_new_frames": 3}
    info["breeze_audio_codes"] = torch.tensor([[2, 3, 4, 5]])
    model.lm_head.code0 = 3
    hidden = torch.arange(16, dtype=torch.float32).reshape(1, 16)

    output = model.make_omni_output(
        hidden,
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    )
    assert output.multimodal_outputs["codes"]["audio"][0].shape == (2, 4)

    # Main-head class 8 is the Breeze backbone EOS (one past vocab_size=8).
    model.lm_head.code0 = 8
    output = model.make_omni_output(
        hidden,
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    )
    assert output.multimodal_outputs["codes"]["audio"][0].shape == (2, 4)
    assert info["breeze_force_eos"] is True
    assert info["breeze_force_eos"] is True

    logits = model.compute_logits(hidden)
    assert torch.equal(logits[0, :8], torch.full((8,), float("-inf")))
    assert logits[0, 8].item() == 0.0


def test_frame_budget_forces_eos_on_followup_step():
    model = _talker()
    model._generate_depth_codes = lambda hidden, code0: _depth_codes(model, hidden, code0)
    info = {"breeze_generated_frames": 2, "breeze_max_new_frames": 2}
    info["breeze_audio_codes"] = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]])
    hidden = torch.arange(16, dtype=torch.float32).reshape(1, 16)

    output = model.make_omni_output(
        hidden,
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    )

    assert output.multimodal_outputs["codes"]["audio"][0].shape == (2, 4)
    assert info["breeze_generated_frames"] == 2
    assert info["breeze_force_eos"] is True


def test_opt_in_golden_dump_writes_terminal_frames(tmp_path):
    model = _talker()
    model._golden_dump_dir = str(tmp_path)
    info = {
        "template": "tts_plain",
        "prompt_ids": torch.tensor([1, 2, 3]),
        "breeze_audio_codes": torch.tensor([[1, 2, 3, 4]]),
        "breeze_generated_frames": 1,
        "breeze_force_eos": True,
    }

    model._dump_golden_frames(info)

    dump_path = next(tmp_path.glob("breeze_*.pt"))
    payload = torch.load(dump_path, weights_only=False)
    assert payload["template"] == "tts_plain"
    assert torch.equal(payload["prompt_ids"], torch.tensor([1, 2, 3]))
    assert torch.equal(payload["codes"], torch.tensor([[1, 2, 3, 4]]))
    assert payload["finished"] is True
