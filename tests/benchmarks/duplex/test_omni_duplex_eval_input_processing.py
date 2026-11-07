# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""A delayed listen from earlier input must not complete a committed clip."""

import pytest

from vllm_omni.benchmarks.duplex.omni_duplex_eval_runner import _residual_unit_decided

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]


def commit():
    return {
        "type": "input_audio_buffer.committed",
        "item_id": "clip-end",
        "event": {"epoch": 0, "processing_required": True},
    }


def listen():
    return {"type": "response.listen", "response": {"metadata": {"model_listen": True}}}


def processed(*, item_id="clip-end", epoch=0, model_listen=True):
    return {
        "type": "duplex.input.processed",
        "event": {"item_id": item_id, "epoch": epoch, "model_listen": model_listen},
    }


@pytest.mark.parametrize("timeout_fallback", [False, True])
def test_old_listen_cannot_close_clip_with_unconsumed_input(timeout_fallback):
    # Reproduced cold C8: commit arrives at 20.6 s, but the first input unit's
    # listen only arrives at 23.6 s. Event arrival order does not identify the unit.
    events = [commit(), listen()]
    assert not _residual_unit_decided(events, 0, settled_silence_is_terminal=timeout_fallback)
    events.append(processed())
    assert _residual_unit_decided(events, 0, settled_silence_is_terminal=timeout_fallback)


@pytest.mark.parametrize("wrong", [processed(item_id="previous-clip"), processed(epoch=1)])
def test_consumption_ack_must_match_committed_item_and_epoch(wrong):
    assert not _residual_unit_decided([commit(), wrong, listen()], 0)


def test_old_response_done_does_not_satisfy_input_barrier():
    events = [
        commit(),
        {"type": "response.created", "response_id": "old"},
        {"type": "response.done", "response": {"id": "old", "status": "completed"}},
    ]
    assert not _residual_unit_decided(events, 0)
    events.append(processed(model_listen=False))
    assert not _residual_unit_decided(events, 0)
    events.extend(
        [
            {"type": "response.created", "response_id": "last"},
            {"type": "response.done", "response": {"id": "last", "status": "completed"}},
        ]
    )
    assert _residual_unit_decided(events, 0)


def test_processing_ack_does_not_finish_open_response():
    events = [{"type": "response.created", "response_id": "speaking"}, commit(), processed()]
    assert not _residual_unit_decided(events, 1)
    events.append({"type": "response.done", "response": {"id": "speaking", "status": "completed"}})
    assert _residual_unit_decided(events, 1)


def test_legacy_server_without_processing_contract_keeps_existing_behavior():
    assert _residual_unit_decided([{"type": "input_audio_buffer.committed"}, listen()], 0)


def test_failed_response_is_terminal_even_without_processing_ack():
    events = [commit(), {"type": "response.done", "response": {"id": "failed", "status": "failed"}}]
    assert _residual_unit_decided(events, 0)
