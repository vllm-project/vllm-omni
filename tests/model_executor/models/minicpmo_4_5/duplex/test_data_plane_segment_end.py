# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Deduplicate output boundaries independently of input submission timing."""

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.engine.duplex.plugin import DuplexDataPlaneContext
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.data_plane import (
    MiniCPMO45DataPlaneSession,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _boundary(seq: int, *, samples: int = 0, cache_epoch: int = 0):
    return SimpleNamespace(
        request_id="request",
        finished=False,
        outputs=[SimpleNamespace(text="hello" if samples else "", token_ids=[], multimodal_output={})],
        multimodal_output={
            "audio": np.zeros(samples, dtype=np.float32),
            "sr": 24000,
            "meta.duplex_turn_id": 0,
            "meta.duplex_epoch": 0,
            "meta.tts_is_last_chunk": True,
            "meta.duplex_audio_chunk_seq": seq,
            "meta.duplex_audio_cache_epoch": cache_epoch,
        },
    )


def _project(plane, output):
    context = DuplexDataPlaneContext(auto_responds=True, active_response_id="response", active_response_turn_id=0)
    return list(plane.project_output(output, context=context))


@pytest.mark.parametrize("samples", [0, 24000])
def test_queued_inputs_do_not_suppress_later_segment_boundaries(samples):
    plane = MiniCPMO45DataPlaneSession(
        lambda audio, *args: "audio" if audio is not None and np.asarray(audio).size else ""
    )
    # Two inputs can already be in flight before the first TTS segment ends.
    plane.begin_request("request")
    plane.begin_request("request")
    first = _project(plane, _boundary(0, samples=samples))
    second = _project(plane, _boundary(1, samples=2 * samples))
    assert sum(bool(item.get("abort_data_plane_request")) for item in first) == 1
    assert sum(bool(item.get("abort_data_plane_request")) for item in second) == 1
    assert all(not item["end_of_turn"] for item in first + second)


def test_new_input_does_not_rearm_an_already_delivered_boundary():
    plane = MiniCPMO45DataPlaneSession(
        lambda audio, *args: "audio" if audio is not None and np.asarray(audio).size else ""
    )
    plane.begin_request("request")
    assert _project(plane, _boundary(3))
    plane.begin_request("request")
    assert _project(plane, _boundary(3)) == []
    assert _project(plane, _boundary(2)) == []
    assert _project(plane, _boundary(4))


@pytest.mark.parametrize("turn_end", [False, True])
@pytest.mark.parametrize("text_without_audio", [None, False, True])
def test_zero_frame_text_delivery_is_opt_in_and_deduplicated(turn_end, text_without_audio):
    plane = MiniCPMO45DataPlaneSession(lambda *args: "")
    output = _boundary(0)
    output.outputs[0].text = "<delegate>look up the weather</delegate>Done."
    if text_without_audio is not None:
        output.multimodal_output["meta.text_without_audio"] = text_without_audio
    output.multimodal_output["meta.turn_end"] = turn_end
    events = _project(plane, output)
    assert len(events) == 1
    assert events[0]["text"] == (output.outputs[0].text if text_without_audio else "")
    assert events[0]["audio_data"] == ""
    assert events[0]["end_of_turn"] is turn_end
    assert events[0]["abort_data_plane_request"] is True
    assert _project(plane, output) == []


def test_chunk_sequence_can_restart_in_a_new_audio_cache_epoch():
    plane = MiniCPMO45DataPlaneSession(
        lambda audio, *args: "audio" if audio is not None and np.asarray(audio).size else ""
    )
    plane.begin_request("request")
    assert _project(plane, _boundary(8))
    assert _project(plane, _boundary(0, cache_epoch=1))
    assert _project(plane, _boundary(9, cache_epoch=0)) == []


@pytest.mark.parametrize("texts", [("1", "1"), ("1", "11"), ("绿色", "绿色的")])
def test_distinct_chunk_text_is_not_deduplicated_by_content(texts):
    plane = MiniCPMO45DataPlaneSession(
        lambda audio, *args: "audio" if audio is not None and np.asarray(audio).size else ""
    )
    emitted = []
    for seq, text in enumerate(texts):
        output = _boundary(seq, samples=(seq + 1) * 24000)
        output.outputs[0].text = text
        emitted.extend(_project(plane, output))
        assert _project(plane, output) == []  # Replaying the same chunk is still deduplicated.
    assert "".join(str(item.get("text", "")) for item in emitted) == "".join(texts)


def test_text_chunk_identity_rejects_replays_and_is_scoped_to_turn():
    plane = MiniCPMO45DataPlaneSession(lambda *args: "audio")
    delta = plane.segment_text_delta
    assert delta("request", "1", turn_id=0, chunk_key=(0, 4)) == "1"
    assert delta("request", "11", turn_id=0, chunk_key=(0, 4)) == ""
    assert delta("request", "stale", turn_id=0, chunk_key=(0, 3)) == ""
    assert delta("request", "1", turn_id=0, chunk_key=(1, 0)) == "1"
    assert delta("request", "stale", turn_id=0, chunk_key=(0, 5)) == ""
    assert delta("request", "1", turn_id=1, chunk_key=(0, 0)) == "1"


def test_text_without_chunk_identity_retains_snapshot_semantics():
    plane = MiniCPMO45DataPlaneSession(lambda *args: "audio")
    assert plane.segment_text_delta("request", "1") == "1"
    assert plane.segment_text_delta("request", "1") == ""
    assert plane.segment_text_delta("request", "11") == "1"


def test_later_three_stage_turn_does_not_end_on_a_previous_turns_eos():
    plane = MiniCPMO45DataPlaneSession(lambda *args: "")
    first = _boundary(0)
    first.multimodal_output["meta.tts_is_last_chunk"] = False
    first.outputs[0].token_ids = [151645]
    first.outputs[0].cumulative_token_ids = [151645]
    assert [item["abort_data_plane_request"] for item in _project(plane, first)] == [True]

    later = _boundary(0)
    later.multimodal_output.update({"meta.tts_is_last_chunk": False, "meta.duplex_turn_id": 1})
    later.outputs[0].token_ids = [7]
    later.outputs[0].cumulative_token_ids = [151645, 7]
    assert _project(plane, later) == []

    # A fresh EOS still ends this turn's segment, exactly once.
    later.outputs[0].token_ids = [151645]
    later.outputs[0].cumulative_token_ids.append(151645)
    assert [item["abort_data_plane_request"] for item in _project(plane, later)] == [True]
    assert _project(plane, later) == []


def test_three_stage_empty_step_retains_cumulative_eos_fallback():
    plane = MiniCPMO45DataPlaneSession(lambda *args: "")
    output = _boundary(0)
    output.multimodal_output["meta.tts_is_last_chunk"] = False
    output.outputs[0].cumulative_token_ids = [151645]
    assert [item["abort_data_plane_request"] for item in _project(plane, output)] == [True]


@pytest.mark.parametrize("empty_step_ids", [[], None])
def test_later_three_stage_turn_with_empty_step_ignores_previous_turns_cumulative_eos(empty_step_ids):
    plane = MiniCPMO45DataPlaneSession(lambda *args: "")
    first = _boundary(0)
    first.multimodal_output["meta.tts_is_last_chunk"] = False
    first.outputs[0].token_ids = [151645]
    first.outputs[0].cumulative_token_ids = [151645]
    assert [item["abort_data_plane_request"] for item in _project(plane, first)] == [True]

    later = _boundary(0)
    later.multimodal_output.update({"meta.tts_is_last_chunk": False, "meta.duplex_turn_id": 1})
    later.outputs[0].token_ids = empty_step_ids
    later.outputs[0].cumulative_token_ids = [151645, 7]
    assert _project(plane, later) == []

    # A fresh EOS arriving only in the cumulative list still ends this turn's segment.
    later.outputs[0].cumulative_token_ids = [151645, 7, 151645]
    assert [item["abort_data_plane_request"] for item in _project(plane, later)] == [True]


@pytest.mark.parametrize("cumulative_ids", [None, [], [151645]], ids=["missing", "empty", "short"])
def test_incomplete_cumulative_snapshot_does_not_replay_a_previous_turns_eos(cumulative_ids):
    plane = MiniCPMO45DataPlaneSession(lambda *args: "")
    first = _boundary(0)
    first.multimodal_output["meta.tts_is_last_chunk"] = False
    first.outputs[0].cumulative_token_ids = [151645]
    assert [item["abort_data_plane_request"] for item in _project(plane, first)] == [True]

    later = _boundary(0)
    later.multimodal_output.update({"meta.tts_is_last_chunk": False, "meta.duplex_turn_id": 1})
    later.outputs[0].cumulative_token_ids = [151645, 7]
    assert _project(plane, later) == []

    # A metadata-only or truncated snapshot must not rewind the EOS cursor.
    intervening = _boundary(0)
    intervening.multimodal_output.update({"meta.tts_is_last_chunk": False, "meta.duplex_turn_id": 1})
    if cumulative_ids is not None:
        intervening.outputs[0].cumulative_token_ids = cumulative_ids
    assert _project(plane, intervening) == []

    later.outputs[0].cumulative_token_ids = [151645, 7, 8]
    assert _project(plane, later) == []

    # A fresh EOS arriving only in the restored cumulative list still works.
    later.outputs[0].cumulative_token_ids.append(151645)
    assert [item["abort_data_plane_request"] for item in _project(plane, later)] == [True]
    assert _project(plane, later) == []


@pytest.mark.parametrize("cumulative_ids", [None, [], [7]], ids=["missing", "empty", "short"])
def test_incomplete_cumulative_snapshot_still_scans_current_step_eos(cumulative_ids):
    plane = MiniCPMO45DataPlaneSession(lambda *args: "")
    output = _boundary(0)
    output.multimodal_output["meta.tts_is_last_chunk"] = False
    output.outputs[0].cumulative_token_ids = [7, 8]
    assert _project(plane, output) == []

    output.outputs[0].token_ids = [151645]
    if cumulative_ids is None:
        del output.outputs[0].cumulative_token_ids
    else:
        output.outputs[0].cumulative_token_ids = cumulative_ids
    assert [item["abort_data_plane_request"] for item in _project(plane, output)] == [True]


@pytest.mark.parametrize("replay", [False, True], ids=["once", "replayed"])
@pytest.mark.parametrize("cumulative_ids", [None, [], [7]], ids=["missing", "empty", "short"])
def test_restored_cumulative_does_not_replay_an_eos_already_seen_in_step_ids(cumulative_ids, replay):
    plane = MiniCPMO45DataPlaneSession(lambda *args: "")
    first = _boundary(0)
    first.multimodal_output["meta.tts_is_last_chunk"] = False
    first.outputs[0].cumulative_token_ids = [7, 8]
    assert _project(plane, first) == []

    gap = _boundary(0)
    gap.multimodal_output["meta.tts_is_last_chunk"] = False
    gap.outputs[0].token_ids = [151645]
    if cumulative_ids is not None:
        gap.outputs[0].cumulative_token_ids = cumulative_ids
    assert _aborts(_project(plane, gap)) == [True]
    if replay:
        # Replays record the gap step twice, and a same-length snapshot restores nothing.
        assert _project(plane, gap) == []
        assert _project(plane, first) == []

    later = _boundary(0)
    later.multimodal_output.update({"meta.tts_is_last_chunk": False, "meta.duplex_turn_id": 1})
    later.outputs[0].cumulative_token_ids = [7, 8, 151645]
    context = DuplexDataPlaneContext(auto_responds=True, active_response_id="response", active_response_turn_id=1)
    assert list(plane.project_output(later, context=context)) == []

    # A new EOS right after the restored history still ends the later turn.
    later.outputs[0].cumulative_token_ids.append(151645)
    assert _aborts(plane.project_output(later, context=context)) == [True]


def test_closing_request_resets_its_cumulative_eos_cursor():
    plane = MiniCPMO45DataPlaneSession(lambda *args: "")
    output = _boundary(0)
    output.multimodal_output["meta.tts_is_last_chunk"] = False
    output.outputs[0].cumulative_token_ids = [7, 151645]
    assert [item["abort_data_plane_request"] for item in _project(plane, output)] == [True]

    plane.close_request("request")
    plane.begin_request("request")
    output.outputs[0].cumulative_token_ids = [151645]
    assert [item["abort_data_plane_request"] for item in _project(plane, output)] == [True]


def test_skipped_stale_output_does_not_leave_its_eos_for_a_later_empty_step():
    plane = MiniCPMO45DataPlaneSession(lambda *args: "")
    context = DuplexDataPlaneContext(auto_responds=True, active_response_id="response", active_response_turn_id=1)
    stale = _boundary(0)
    stale.multimodal_output["meta.tts_is_last_chunk"] = False
    stale.outputs[0].token_ids = [151645]
    stale.outputs[0].cumulative_token_ids = [151645]
    assert list(plane.project_output(stale, context=context)) == []

    later = _boundary(0)
    later.multimodal_output.update({"meta.tts_is_last_chunk": False, "meta.duplex_turn_id": 1})
    later.outputs[0].cumulative_token_ids = [151645, 7]
    assert list(plane.project_output(later, context=context)) == []


def test_stage0_cursor_does_not_hide_a_final_stage_eos_on_the_same_request():
    # Downstream stages reuse the Stage0 request id, so the EOS cursor is per stage.
    plane = MiniCPMO45DataPlaneSession(lambda *args: "")
    stage0 = _boundary(0)
    stage0.stage_id = 0
    stage0.multimodal_output["meta.tts_is_last_chunk"] = False
    stage0.outputs[0].cumulative_token_ids = list(range(1, 12))
    assert _project(plane, stage0) == []

    final = _boundary(0)
    final.stage_id = 2
    final.multimodal_output["meta.tts_is_last_chunk"] = False
    final.outputs[0].cumulative_token_ids = [7, 151645]
    assert [item["abort_data_plane_request"] for item in _project(plane, final)] == [True]


def _audio_plane():
    return MiniCPMO45DataPlaneSession(
        lambda audio, *args: "audio" if audio is not None and np.asarray(audio).size else ""
    )


def _audio_step_with_eos(turn_id):
    output = _boundary(0, samples=24000)
    output.multimodal_output.update({"meta.tts_is_last_chunk": False, "meta.duplex_turn_id": turn_id})
    output.outputs[0].cumulative_token_ids = [7, 151645]
    return output


def _aborts(results):
    return [item["abort_data_plane_request"] for item in results if item.get("abort_data_plane_request")]


def test_eos_arriving_with_audio_ends_the_segment_on_the_next_empty_step():
    plane = _audio_plane()
    assert _aborts(_project(plane, _audio_step_with_eos(0))) == []
    # Same samples again: no new audio, so the pending EOS ends the segment once.
    assert _aborts(_project(plane, _audio_step_with_eos(0))) == [True]
    assert _aborts(_project(plane, _audio_step_with_eos(0))) == []


def test_pending_eos_does_not_end_a_later_turn():
    plane = _audio_plane()
    assert _aborts(_project(plane, _audio_step_with_eos(0))) == []
    later = _audio_step_with_eos(1)
    later.outputs[0].cumulative_token_ids.append(8)
    context = DuplexDataPlaneContext(auto_responds=True, active_response_id="response", active_response_turn_id=1)
    assert list(plane.project_output(later, context=context)) == []


def test_pending_eos_without_turn_metadata_is_not_kept_across_steps():
    plane = _audio_plane()
    assert _aborts(_project(plane, _audio_step_with_eos(None))) == []
    assert _aborts(_project(plane, _audio_step_with_eos(None))) == []


def test_closing_the_stream_drops_a_pending_eos():
    plane = _audio_plane()
    assert _aborts(_project(plane, _audio_step_with_eos(0))) == []
    plane.close_stream("request")
    drained = _boundary(0)
    drained.multimodal_output["meta.tts_is_last_chunk"] = False
    drained.outputs[0].cumulative_token_ids = [7, 151645]
    assert _project(plane, drained) == []
