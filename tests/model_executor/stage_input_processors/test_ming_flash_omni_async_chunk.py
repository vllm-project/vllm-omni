# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.ming_flash_omni import (
    thinker2talker_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Sentences long enough (>12 / >8 tokens) to cross the min-token gate.
S1 = "The quick brown fox jumps over the lazy dog today."
S2 = "Now the second sentence here is also quite long indeed."


def _make_transfer_manager() -> SimpleNamespace:
    # Ming passes a self-contained text sentence per chunk
    # Bare manager: the producer attaches ming_text_buf / ming_flushed_segments.
    return SimpleNamespace()


def _make_request(req_id="r0", *, finished=False, additional_information=None) -> SimpleNamespace:
    return SimpleNamespace(
        external_req_id=req_id,
        additional_information=additional_information or {},
        is_finished=lambda: finished,
    )


def _step(tm, req, text, is_finished=False):
    """Drive one producer step with cumulative text"""
    return thinker2talker_async_chunk(
        transfer_manager=tm,
        pooling_output={"text": text},
        request=req,
        is_finished=is_finished,
    )


def _text(payload):
    return payload.kv_metadata["text"]


def _info(payload):
    return payload.kv_metadata["additional_information"]


def test_no_flush_mid_sentence():
    transfer_manager = _make_transfer_manager()
    req = _make_request()
    # No terminator yet -> back-pressure (None).
    assert _step(transfer_manager, req, "The quick brown fox jumps over") is None
    assert _step(transfer_manager, req, "The quick brown fox jumps over the lazy") is None


def test_min_token_gate_holds_short_sentence():
    transfer_manager = _make_transfer_manager()
    req = _make_request()
    # Terminator present but < 12 tokens -> gate holds, no flush.
    assert _step(transfer_manager, req, "Hello world. How are you") is None


def test_one_emit_per_completed_sentence():
    transfer_manager = _make_transfer_manager()
    req = _make_request()

    # First completed sentence flushes; the growing tail is held back.
    payload = _step(transfer_manager, req, S1 + " Now the second")
    assert payload is not None
    assert _text(payload) == S1
    assert _info(payload)["text"] == S1
    assert bool(payload.meta.finished.item()) is False

    # Same completed sentence, tail still growing -> nothing new.
    assert _step(transfer_manager, req, S1 + " " + S2[:20]) is None

    # Second sentence completes -> only the new sentence is emitted.
    out2 = _step(transfer_manager, req, S1 + " " + S2 + " Tail")
    assert out2 is not None
    assert _text(out2) == S2


def test_final_flush_of_trailing_partial_on_finish():
    transfer_manager = _make_transfer_manager()
    req = _make_request()

    payload = _step(transfer_manager, req, S1 + " Now the second")
    assert payload is not None and _text(payload) == S1

    # Finish: the held trailing fragment is flushed even without a terminator,
    # and the chunk is marked finished.
    out_final = _step(transfer_manager, req, S1 + " Now the second", is_finished=True)
    assert out_final is not None
    assert _text(out_final) == "Now the second"
    assert bool(out_final.meta.finished.item()) is True


def test_finish_with_no_output_emits_terminal_marker():
    transfer_manager = _make_transfer_manager()
    req = _make_request()
    payload = _step(transfer_manager, req, "", is_finished=True)
    assert payload is not None
    assert payload.kv_metadata is None
    assert bool(payload.meta.finished.item()) is True


def test_voice_metadata_preserved_on_every_chunk():
    transfer_manager = _make_transfer_manager()
    req = _make_request(
        additional_information={
            "voice_name": "Custom",
            "prompt_text": "ref",
            "max_text_length": 40,
        }
    )
    payload = _step(transfer_manager, req, S1 + " more text following along here")
    assert payload is not None
    info = _info(payload)
    assert info["ming_task"] == "omni"
    assert info["voice_name"] == "Custom"
    assert info["prompt_text"] == "ref"
    assert info["max_text_length"] == 40


def test_spk_emb_list_is_coerced_to_tensor():
    transfer_manager = _make_transfer_manager()
    req = _make_request(additional_information={"spk_emb": [0.1, 0.2, 0.3]})
    payload = _step(transfer_manager, req, S1 + " more text following along here")
    assert payload is not None
    spk = _info(payload)["spk_emb"]
    assert isinstance(spk, torch.Tensor)
    assert spk.shape == (1, 3)


def test_multiple_completed_sentences_flush_together():
    transfer_manager = _make_transfer_manager()
    req = _make_request()
    # Two sentences complete between steps -> both emitted in one chunk.
    payload = _step(transfer_manager, req, S1 + " " + S2 + " Tail")
    assert payload is not None
    assert _text(payload) == S1 + " " + S2


def test_per_request_isolation():
    transfer_manager = _make_transfer_manager()
    a = _make_request("a")
    b = _make_request("b")
    # a flushes S1, holds the trailing fragment.
    assert _step(transfer_manager, a, S1 + " trailing fragment a") is not None
    # b is independent: its first completed sentence still flushes.
    out_b = _step(transfer_manager, b, S2 + " trailing fragment b")
    assert out_b is not None and _text(out_b) == S2
    # a does not re-flush S1.
    assert _step(transfer_manager, a, S1 + " trailing fragment a") is None


def test_state_reclaimed_on_terminal_flush():
    # Per-request producer state must be reclaimed on finish; the connector's
    # cleanup_sender does not know about ming_text_buf / ming_flushed_segments.
    transfer_manager = _make_transfer_manager()
    req = _make_request("leak0")
    payload = _step(transfer_manager, req, S1 + " Now the second")
    assert payload is not None and _text(payload) == S1
    assert "leak0" in transfer_manager.ming_text_buf
    assert "leak0" in transfer_manager.ming_flushed_segments
    # Finish flushes the held trailing fragment AND drops the state.
    out_final = _step(transfer_manager, req, S1 + " Now the second", is_finished=True)
    assert out_final is not None and _text(out_final) == "Now the second"
    assert "leak0" not in transfer_manager.ming_text_buf
    assert "leak0" not in transfer_manager.ming_flushed_segments


def test_state_reclaimed_on_empty_finish():
    # Request that finishes before any sentence crosses the gate: still emit a
    # terminal marker and still reclaim per-request state.
    transfer_manager = _make_transfer_manager()
    req = _make_request("leak1")
    payload = _step(transfer_manager, req, "", is_finished=True)
    assert payload is not None
    assert bool(payload.meta.finished.item()) is True
    assert "leak1" not in transfer_manager.ming_text_buf
    assert "leak1" not in transfer_manager.ming_flushed_segments
