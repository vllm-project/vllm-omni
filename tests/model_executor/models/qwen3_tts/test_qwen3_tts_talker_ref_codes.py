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

from types import SimpleNamespace

import pytest
import torch

import vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker as qwen3_tts_talker
from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
    Qwen3TTSTalkerForConditionalGeneration,
    _qwen3_tts_gpu_resident_buffer_keys,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_NUM_CODE_GROUPS = 16


def test_large_tts_prefill_artifacts_stay_gpu_resident_only_on_mrv2() -> None:
    v1_keys = _qwen3_tts_gpu_resident_buffer_keys(False)
    v2_keys = _qwen3_tts_gpu_resident_buffer_keys(True)

    assert ("embed", "prefill") not in v1_keys
    assert ("codes", "ref") not in v1_keys
    assert ("meta", "codec_frame_valid") not in v1_keys
    assert ("embed", "prefill") in v2_keys
    assert ("codes", "ref") in v2_keys
    assert ("meta", "codec_frame_valid") in v2_keys


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


def test_make_omni_output_materializes_uniform_metadata_once_per_batch(monkeypatch) -> None:
    ref_code = torch.ones((5, _NUM_CODE_GROUPS), dtype=torch.long)
    infos = [_info(2, ref_code), _info(3, ref_code), _info(1, ref_code)]
    for info in infos:
        info["meta"]["codec_streaming"] = True

    original_full = torch.full
    calls: list[tuple[tuple[int, ...], torch.dtype]] = []

    def counted_full(size, fill_value, **kwargs):
        calls.append((tuple(size), kwargs.get("dtype")))
        return original_full(size, fill_value, **kwargs)

    monkeypatch.setattr(qwen3_tts_talker.torch, "full", counted_full)

    out = _make_talker().make_omni_output(
        torch.zeros((6, 8)),
        model_intermediate_buffer=infos,
    )

    assert calls == [((6,), torch.int32), ((6,), torch.int8)]
    assert out.multimodal_outputs["meta"]["ref_code_len"].tolist() == [5] * 6
    assert out.multimodal_outputs["meta"]["codec_streaming"].tolist() == [1] * 6


def test_make_omni_output_avoids_repeat_interleave_for_variable_ref_lengths(monkeypatch) -> None:
    ref_a = torch.ones((3, _NUM_CODE_GROUPS), dtype=torch.long)
    ref_b = torch.ones((5, _NUM_CODE_GROUPS), dtype=torch.long)
    infos = [_info(2, ref_a), _info(3, ref_b)]

    def fail_repeat_interleave(*_args, **_kwargs):
        raise AssertionError("variable ref lengths must not use CUDA repeat_interleave")

    monkeypatch.setattr(qwen3_tts_talker.torch, "repeat_interleave", fail_repeat_interleave)

    out = _make_talker().make_omni_output(
        torch.zeros((5, 8)),
        model_intermediate_buffer=infos,
    )

    assert out.multimodal_outputs["meta"]["ref_code_len"].tolist() == [3, 3, 5, 5, 5]


def test_make_omni_output_omits_redundant_metadata_for_async_chunk() -> None:
    talker = _make_talker()
    talker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(async_chunk=True),
    )
    ref_code = torch.ones((5, _NUM_CODE_GROUPS), dtype=torch.long)
    info = _info(2, ref_code)
    info["meta"]["codec_streaming"] = True
    info["meta"]["codec_frame_valid"] = True

    out = talker.make_omni_output(
        torch.zeros((2, 8)),
        model_intermediate_buffer=[info],
    )

    assert out.multimodal_outputs["codes"]["ref"][0] is ref_code
    assert out.multimodal_outputs["meta"]["codec_frame_valid"].tolist() == [1, 1]


def test_make_omni_output_preserves_per_request_codec_frame_validity() -> None:
    talker = _make_talker()
    talker.vllm_config = SimpleNamespace(model_config=SimpleNamespace(async_chunk=True))
    valid = _info(1, None)
    invalid = _info(1, None)
    valid["meta"]["codec_frame_valid"] = True
    invalid["meta"]["codec_frame_valid"] = False

    out = talker.make_omni_output(
        torch.zeros((2, 8)),
        model_intermediate_buffer=[valid, invalid],
    )

    assert out.multimodal_outputs["meta"]["codec_frame_valid"].tolist() == [1, 0]


def test_make_omni_output_publishes_ref_once_for_kv_resumed_first_decode() -> None:
    talker = _make_talker()
    talker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(async_chunk=True),
    )
    ref_code = torch.ones((5, _NUM_CODE_GROUPS), dtype=torch.long)
    info = _info(1, ref_code)
    info["_omni_num_computed_tokens"] = 9
    info["_omni_prompt_len"] = 5

    first = talker.make_omni_output(
        torch.zeros((1, 8)),
        model_intermediate_buffer=[info],
    )
    second = talker.make_omni_output(
        torch.zeros((1, 8)),
        model_intermediate_buffer=[info],
    )

    assert first.multimodal_outputs["codes"]["ref"][0] is ref_code
    assert "ref" not in second.multimodal_outputs["codes"]
