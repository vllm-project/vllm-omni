# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for cross-request ``codes.ref`` leakage (#4370).

``make_omni_output`` used to collapse every request's reference codec
frames into a single last-writer-wins slot and emit it as a length-1
list. ``to_payload_element`` indexes list payloads with
``element[idx] if idx < len(element) else element[0]``, so that length-1
list was broadcast to every request in the batch: each utterance's first
vocoder chunks then decoded with another request's reference voice as
context (audible onset timbre deformation at any concurrency > 1).
"""

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
    Qwen3TTSTalkerForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_NUM_CODE_GROUPS = 16


def _make_talker() -> Qwen3TTSTalkerForConditionalGeneration:
    # make_omni_output reads no instance state, so skip __init__ (no real
    # config / checkpoint needed), same as the preprocess tests.
    return Qwen3TTSTalkerForConditionalGeneration.__new__(Qwen3TTSTalkerForConditionalGeneration)


def _info(span_frames: int, ref_code: torch.Tensor | None) -> dict:
    codes: dict = {"audio": torch.zeros((span_frames, _NUM_CODE_GROUPS), dtype=torch.long)}
    meta: dict = {}
    if ref_code is not None:
        codes["ref"] = ref_code
        meta["ref_code_len"] = int(ref_code.shape[0])
    return {"codes": codes, "meta": meta}


def test_make_omni_output_keeps_ref_codes_per_request() -> None:
    ref_a = torch.ones((3, _NUM_CODE_GROUPS), dtype=torch.long)
    ref_b = torch.full((5, _NUM_CODE_GROUPS), 2, dtype=torch.long)

    out = _make_talker().make_omni_output(
        torch.zeros((4, 8)),
        model_intermediate_buffer=[_info(2, ref_a), _info(2, ref_b)],
    )

    ref_list = out.multimodal_outputs["codes"]["ref"]
    assert len(ref_list) == 2, "codes.ref must stay batch-aligned (one entry per request)"
    assert torch.equal(ref_list[0], ref_a)
    assert torch.equal(ref_list[1], ref_b)


def test_make_omni_output_pads_requests_without_ref_code() -> None:
    ref_b = torch.full((5, _NUM_CODE_GROUPS), 2, dtype=torch.long)

    out = _make_talker().make_omni_output(
        torch.zeros((4, 8)),
        model_intermediate_buffer=[_info(2, None), _info(2, ref_b)],
    )

    ref_list = out.multimodal_outputs["codes"]["ref"]
    assert len(ref_list) == 2
    assert ref_list[0].numel() == 0, "no-ref request must get an empty placeholder, not a neighbor's ref"
    assert torch.equal(ref_list[1], ref_b)


def test_make_omni_output_omits_ref_key_when_no_request_has_one() -> None:
    out = _make_talker().make_omni_output(
        torch.zeros((4, 8)),
        model_intermediate_buffer=[_info(2, None), _info(2, None)],
    )

    assert "ref" not in out.multimodal_outputs["codes"]


def test_make_omni_output_keeps_hidden_states_for_replay_spans_without_audio_codes() -> None:
    hidden = torch.arange(6 * 8, dtype=torch.float32).reshape(6, 8)

    out = _make_talker().make_omni_output(
        hidden,
        model_intermediate_buffer=[
            {"meta": {"codec_streaming": True}},
            _info(1, None),
        ],
    )

    assert torch.equal(out.text_hidden_states, hidden)
    assert out.multimodal_outputs["codes"]["audio"].shape == (1, _NUM_CODE_GROUPS)
