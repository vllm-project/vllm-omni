# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Accepted force-listen inputs remain canonical for later full KV rebuilds."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from tests.model_executor.models.lychee_fd.test_host_audio_delta import _bridge, _commit, _plan, _record_through
from vllm_omni.engine.duplex.contracts import DuplexFence, duplex_resource_request_id
from vllm_omni.model_executor.models.lychee_fd.duplex.plugin import LycheeDuplexPlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _record(history, tick, *, epoch=0, text=101, speech=151694, control=158357):
    history.record_outputs(
        {
            "lychee_tick": torch.tensor([tick]),
            "lychee_text_token_ids": torch.tensor([text]),
            "lychee_speech_token_ids": torch.tensor([speech]),
            "lychee_control_token_ids": torch.tensor([control]),
            "lychee_execution_epoch": torch.tensor([epoch]),
        }
    )


def _forced_plan():
    plugin = LycheeDuplexPlugin(lambda *args: None)
    _commit(plugin, _plan(plugin))
    history = plugin.histories["session"]
    _record_through(history, 7)
    _record(history, 8)
    plugin.data_plane.close_stream("owner")
    return plugin, history, _plan(plugin, owner="replacement", epoch=1)


def _channels(history):
    return deepcopy((history.text, history.speech, history.control))


def _raw(history):
    return deepcopy((history.raw_text, history.raw_speech, history.raw_control))


def test_unaccepted_retry_preserves_original_frontier_until_accepted_commit():
    plugin, history, first = _forced_plan()
    original, raw = _channels(history), _raw(history)
    snapshot = deepcopy(_bridge(first)["lychee_history"])
    retry = _plan(plugin, owner="replacement", epoch=1)
    assert _channels(history) == original and _raw(history) == raw
    assert history.force_listen_at_frontier is True
    assert history.pending_eos_rebuild_tick == 8
    assert _bridge(retry)["lychee_history"] == snapshot
    assert len(history.audio_windows) == 2
    _commit(plugin, retry, owner="replacement", epoch=1)
    position = len(snapshot["logical_ticks"]) - 1
    assert tuple(channel[position] for channel in _channels(history)) == (
        history.text_pad,
        history.speech_pad,
        history.sleep,
    )
    assert _raw(history) == raw
    assert _bridge(first)["lychee_history"] == snapshot
    assert history.force_listen_at_frontier is False
    committed = _channels(history)
    _commit(plugin, first, owner="replacement", epoch=1)
    assert _channels(history) == committed and _raw(history) == raw


def test_later_outputs_do_not_move_the_accepted_forced_frontier_coordinate():
    plugin, history, plan = _forced_plan()
    position = len(_bridge(plan)["lychee_history"]["logical_ticks"]) - 1
    _record_through(history, 12, epoch=1)
    _record(history, 13, epoch=1, text=202, speech=151702, control=158350)
    later = tuple(deepcopy(channel[position + 1 :]) for channel in _channels(history))
    raw, ticks, audio = _raw(history), list(history.ticks), deepcopy(history.audio_windows)
    _commit(plugin, plan, owner="replacement", epoch=1)
    assert history.frontier_tick == 13
    assert tuple(channel[position] for channel in _channels(history)) == (
        history.text_pad,
        history.speech_pad,
        history.sleep,
    )
    assert tuple(channel[position + 1 :] for channel in _channels(history)) == later
    assert _raw(history) == raw and history.ticks == ticks and history.audio_windows == audio


def test_later_natural_eos_rebuild_uses_the_consumed_frontier_but_preserves_raw_merge_sample():
    plugin, history, plan = _forced_plan()
    position = len(_bridge(plan)["lychee_history"]["logical_ticks"]) - 1
    original_raw = tuple(channel[position] for channel in _raw(history))
    _commit(plugin, plan, owner="replacement", epoch=1)
    _record_through(history, 17, epoch=1)
    _record(history, 18, epoch=1)
    later = _plan(plugin, owner="replacement", epoch=1, seq=1)
    bridge = _bridge(later)
    assert bridge["lychee_kv_rebuild"]["eos_tick"] == 18
    full = bridge["lychee_history"]
    assert full["force_listen_at_frontier"] is False
    assert tuple(full[field][position] for field in ("text_input_ids", "speech_input_ids", "control_input_ids")) == (
        history.text_pad,
        history.speech_pad,
        history.sleep,
    )
    assert (
        tuple(
            full[field][position]
            for field in ("raw_text_output_ids", "raw_speech_output_ids", "raw_control_output_ids")
        )
        == original_raw
    )


@pytest.mark.parametrize("invalid", ["owner", "fence", "session", "epoch", "snapshot_epoch", "coordinate", "canonical"])
def test_mismatched_snapshot_fails_before_canonical_binding_or_marker_changes(invalid):
    plugin, history, plan = _forced_plan()
    info = plan.prompt["model_intermediate_buffer"]
    bridge = info["duplex"]
    snapshot = bridge["lychee_history"]
    if invalid == "owner":
        info["request_id"] = "foreign"
    elif invalid == "fence":
        bridge["fence"] = DuplexFence("session", epoch=2)
    elif invalid == "session":
        bridge["session_id"] = "foreign"
    elif invalid == "epoch":
        bridge["epoch"] = 2
    elif invalid == "snapshot_epoch":
        snapshot["execution_epoch"] = 2
    elif invalid == "coordinate":
        snapshot["logical_ticks"][-1] = 7
    else:
        history.text[-1] = 999
    before = (
        _channels(history),
        _raw(history),
        history.bound_request_id,
        history.bound_execution_epoch,
        set(history.request_ids),
        history.pending_eos_rebuild_tick,
    )
    with pytest.raises(ValueError, match="forced-listen snapshot"):
        _commit(plugin, plan, owner="replacement", epoch=1)
    assert (
        _channels(history),
        _raw(history),
        history.bound_request_id,
        history.bound_execution_epoch,
        history.request_ids,
        history.pending_eos_rebuild_tick,
    ) == before
    assert history.force_listen_at_frontier is True


def test_forced_commit_does_not_modify_another_session_history():
    plugin, history, plan = _forced_plan()
    other_plugin = LycheeDuplexPlugin(lambda *args: None)
    _commit(other_plugin, _plan(other_plugin))
    other = other_plugin.histories["session"]
    plugin.histories["other"] = other
    before = deepcopy(other)
    _commit(plugin, plan, owner="replacement", epoch=1)
    assert other == before
    assert history.bound_request_id == "replacement"


def test_pre_registered_fresh_owner_outputs_preserve_later_eos_and_forced_snapshot_row():
    plugin, history, _ = _forced_plan()
    owner = duplex_resource_request_id(DuplexFence("session", epoch=1), "stage0")
    plan = _plan(plugin, owner=owner, epoch=1)
    position = len(_bridge(plan)["lychee_history"]["logical_ticks"]) - 1
    ticks = torch.arange(9, 19)
    payload = {
        "lychee_tick": ticks,
        "lychee_text_token_ids": torch.full_like(ticks, 202),
        "lychee_speech_token_ids": torch.full_like(ticks, 151702),
        "lychee_control_token_ids": torch.full_like(ticks, 158357),
        "lychee_execution_epoch": torch.ones_like(ticks),
    }
    payload["lychee_speech_token_ids"][-1] = history.speech_eos
    assert owner not in history.request_ids
    output = SimpleNamespace(request_id=owner, outputs=[SimpleNamespace(multimodal_output=payload)])
    list(plugin.data_plane.project({"data_plane_outputs": [output]}))
    assert history.frontier_tick == 18 and history.pending_eos_rebuild_tick == 18
    raw = _raw(history)
    later = tuple(deepcopy(channel[position + 1 :]) for channel in _channels(history))
    _commit(plugin, plan, owner=owner, epoch=1)
    assert tuple(channel[position] for channel in _channels(history)) == (
        history.text_pad,
        history.speech_pad,
        history.sleep,
    )
    assert tuple(channel[position + 1 :] for channel in _channels(history)) == later
    assert _raw(history) == raw
    assert history.pending_eos_rebuild_tick == 18
    following = _plan(plugin, owner=owner, epoch=1, seq=1)
    assert _bridge(following)["lychee_kv_rebuild"]["eos_tick"] == 18


def test_accepted_input_fact_is_independent_of_binding_and_current_epoch():
    plugin, history, plan = _forced_plan()
    position = len(_bridge(plan)["lychee_history"]["logical_ticks"]) - 1
    original_raw = _raw(history)
    _record(history, 9, epoch=1, text=202, speech=151702)
    # Cancellation may advance host execution before the submit ACK returns.
    history.execution_epoch = 2
    before = (
        history.bound_request_id,
        history.bound_execution_epoch,
        set(history.request_ids),
        history.pending_eos_rebuild_tick,
        history.force_listen_at_frontier,
        history.execution_epoch,
    )
    later = tuple(channel[position + 1 :] for channel in _channels(history))
    for _ in range(2):
        plugin.record_accepted_append_plan(request_id="replacement", fence=DuplexFence("session", epoch=1), plan=plan)
    assert tuple(channel[position] for channel in _channels(history)) == (
        history.text_pad,
        history.speech_pad,
        history.sleep,
    )
    assert tuple(channel[position + 1 :] for channel in _channels(history)) == later
    assert tuple(channel[: position + 1] for channel in _raw(history)) == tuple(
        channel[: position + 1] for channel in original_raw
    )
    assert (
        history.bound_request_id,
        history.bound_execution_epoch,
        history.request_ids,
        history.pending_eos_rebuild_tick,
        history.force_listen_at_frontier,
        history.execution_epoch,
    ) == before
    rebuilt = _plan(plugin, owner="second-replacement", epoch=2)
    snapshot = _bridge(rebuilt)["lychee_history"]
    assert tuple(
        snapshot[field][position] for field in ("text_input_ids", "speech_input_ids", "control_input_ids")
    ) == (history.text_pad, history.speech_pad, history.sleep)
    assert tuple(
        snapshot[field][position]
        for field in ("raw_text_output_ids", "raw_speech_output_ids", "raw_control_output_ids")
    ) == tuple(channel[position] for channel in original_raw)


def test_accepted_input_fact_for_deleted_session_is_noop():
    plugin, _, plan = _forced_plan()
    del plugin.histories["session"]
    plugin.record_accepted_append_plan(request_id="replacement", fence=DuplexFence("session", epoch=1), plan=plan)
    assert plugin.histories == {}


def test_resident_delta_accepted_input_fact_does_not_force_current_frontier():
    plugin = LycheeDuplexPlugin(lambda *args: None)
    _commit(plugin, _plan(plugin))
    history = plugin.histories["session"]
    _record_through(history, 8)
    plan = _plan(plugin, seq=1)
    before = deepcopy(history)
    assert "lychee_history" not in _bridge(plan)
    plugin.record_accepted_append_plan(request_id="owner", fence=DuplexFence("session", epoch=0), plan=plan)
    assert history == before
