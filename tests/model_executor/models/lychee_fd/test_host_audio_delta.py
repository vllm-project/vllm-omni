# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Host bootstrap and resident PCM wire contracts for native Lychee KV."""

from __future__ import annotations

import base64
import struct

import msgspec
import pytest
import torch
from vllm.sampling_params import SamplingParams

from tools.lychee_session_lifecycle_probe import SYSTEM_PREFIX
from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.model_executor.models.lychee_fd.duplex.plugin import LycheeDuplexPlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _payload(value=0.05):
    return {
        "audio": base64.b64encode(struct.pack("<6400f", *([value] * 6400))).decode("ascii"),
        "format": "pcm_f32le",
        "sample_rate_hz": 16000,
        "lychee_audio_ledger": {
            "consumable_tick_start": 0,
            "consumable_tick_end": 10,
            "window_sample_end": 6400,
        },
    }


def _plan(plugin, *, owner="owner", epoch=0, seq=0, payload=None):
    return plugin.plan_append(
        request_id=owner,
        fence=DuplexFence("session", epoch=epoch),
        session_config={},
        runtime_config={"lychee_system_token_ids": list(SYSTEM_PREFIX), "duplex_scheduler_token_id": 158358},
        seq=seq,
        turn_seq=seq,
        payload=_payload() if payload is None else payload,
        final=False,
        sampling_params=SamplingParams(max_tokens=10, min_tokens=10, ignore_eos=True),
    )


def _bridge(plan):
    return plan.prompt["model_intermediate_buffer"]["duplex"]


def _commit(plugin, plan, *, owner="owner", epoch=0):
    plugin.commit_append_plan(request_id=owner, fence=DuplexFence("session", epoch=epoch), plan=plan)


def _record_through(history, last_tick, *, epoch=0):
    ticks = torch.arange(history.frontier_tick + 1, last_tick + 1)
    history.record_outputs(
        {
            "lychee_tick": ticks,
            "lychee_text_token_ids": torch.full_like(ticks, 101),
            "lychee_speech_token_ids": torch.full_like(ticks, 158359),
            "lychee_control_token_ids": torch.full_like(ticks, 158357),
            "lychee_execution_epoch": torch.full_like(ticks, epoch),
            "lychee_next_text_token_ids": torch.full_like(ticks, 158358),
        }
    )


def test_initial_bootstrap_and_resident_delta_use_exact_owner_coordinates():
    plugin = LycheeDuplexPlugin(lambda *args: None)
    first = _plan(plugin)
    history = plugin.histories["session"]
    bridge = _bridge(first)
    assert "lychee_audio_delta" not in bridge
    assert bridge["lychee_history"]["version"] == 1
    assert first.prompt["prompt_token_ids"] == SYSTEM_PREFIX + [158358]
    assert first.sampling_params.max_tokens == 9
    _commit(plugin, first)
    _record_through(history, 9)

    second = _plan(plugin, seq=1, payload=_payload(0.2))
    bridge = _bridge(second)
    assert "lychee_history" not in bridge
    assert bridge["lychee_audio_delta"] == {
        "version": 1,
        "kind": "resident_append",
        "request_id": "owner",
        "session_epoch": 0,
        "execution_epoch": 0,
        "op_seq": 1,
        "audio_window_seq": 2,
        "previous_audio_window_seq": 1,
        "start_tick": 10,
        "window_ticks": 10,
    }
    assert second.prompt["prompt_token_ids"] == [158358]
    assert second.sampling_params.max_tokens == second.sampling_params.min_tokens == 10
    assert bridge["payload"]["lychee_audio_ledger"]["consumable_tick_start"] == 10
    assert bridge["payload"]["lychee_audio_ledger"]["consumable_tick_end"] == 20
    assert struct.unpack_from("<f", base64.b64decode(bridge["payload"]["audio"]))[0] == pytest.approx(0.2)
    assert len(history.audio_windows) == 2
    assert history.raw_text[-1] == 101 and history.text[-1] == 158358


@pytest.mark.parametrize("reason", ["new_owner", "new_epoch", "retired_owner"])
def test_changed_binding_always_bootstraps_full_canonical_history(reason):
    plugin = LycheeDuplexPlugin(lambda *args: None)
    first = _plan(plugin)
    _commit(plugin, first)
    history = plugin.histories["session"]
    _record_through(history, 5)
    if reason == "retired_owner":
        history.request_ids.discard("owner")
    owner = "new-owner" if reason == "new_owner" else "owner"
    epoch = 1 if reason == "new_epoch" else 0
    seq = 0 if epoch else 1
    rebuilt = _plan(plugin, owner=owner, epoch=epoch, seq=seq)
    bridge = _bridge(rebuilt)
    assert "lychee_audio_delta" not in bridge
    full = bridge["lychee_history"]
    assert full["execution_epoch"] == epoch
    assert full["logical_ticks"][-1] == 5
    assert full["raw_text_output_ids"][-1] == 101
    assert full["text_input_ids"][-1] == 158358
    assert [window["start_tick"] for window in full["audio_windows"]] == [0, 10]
    assert rebuilt.prompt["prompt_token_ids"] == history.text
    assert rebuilt.sampling_params.max_tokens == rebuilt.sampling_params.min_tokens == 14
    _commit(plugin, rebuilt, owner=owner, epoch=epoch)
    resident = _plan(plugin, owner=owner, epoch=epoch, seq=seq + 1)
    assert "lychee_history" not in _bridge(resident)
    assert _bridge(resident)["lychee_audio_delta"]["audio_window_seq"] == 3


@pytest.mark.parametrize("resident", [False, True])
def test_unaccepted_plan_retry_is_idempotent_and_never_establishes_binding(resident):
    plugin = LycheeDuplexPlugin(lambda *args: None)
    first = _plan(plugin)
    if resident:
        _commit(plugin, first)
    seq = int(resident)
    planned = _plan(plugin, seq=seq)
    retry = _plan(plugin, seq=seq)
    history = plugin.histories["session"]
    assert len(history.audio_windows) == (2 if resident else 1)
    assert history.last_audio_operation == (0, seq)
    assert bool("lychee_audio_delta" in _bridge(planned)) is resident
    assert bool("lychee_audio_delta" in _bridge(retry)) is resident
    assert (history.bound_request_id is not None) is resident
    _commit(plugin, retry)
    next_plan = _plan(plugin, seq=seq + 1)
    assert _bridge(next_plan)["lychee_audio_delta"]["audio_window_seq"] == (3 if resident else 2)


def test_resident_wire_stays_bounded_while_full_recovery_evidence_grows():
    plugin = LycheeDuplexPlugin(lambda *args: None)
    first = _plan(plugin)
    _commit(plugin, first)
    history = plugin.histories["session"]
    _record_through(history, 9)
    wire_sizes = []
    for seq in range(1, 301):
        plan = _plan(plugin, seq=seq)
        bridge = _bridge(plan)
        assert "lychee_history" not in bridge
        delta = bridge["lychee_audio_delta"]
        assert "payload" not in delta and "audio" not in delta
        assert delta["audio_window_seq"] == seq + 1
        assert delta["previous_audio_window_seq"] == seq
        assert delta["start_tick"] == seq * 10
        wire_sizes.append(len(msgspec.msgpack.encode(plan.prompt)))
        _commit(plugin, plan)
        _record_through(history, (seq + 1) * 10 - 1)
    assert min(wire_sizes) > 34136
    assert max(wire_sizes) < 36000
    assert max(wire_sizes) - min(wire_sizes) < 128
    assert len(history.audio_windows) == 301
    assert history.frontier_tick == 3009
    recovered = _plan(plugin, owner="recovered", epoch=1)
    full = _bridge(recovered)["lychee_history"]
    assert len(full["audio_windows"]) == 302
    assert len(full["logical_ticks"]) == len(SYSTEM_PREFIX) + 3010
    assert len(msgspec.msgpack.encode(recovered.prompt)) > 10_000_000


def test_delta_coordinate_config_is_server_owned():
    plugin = LycheeDuplexPlugin(lambda *args: None)
    with pytest.raises(ValueError, match="server-owned: lychee_audio_delta"):
        plugin.validate_client_extra_body({"lychee_audio_delta": {"op_seq": 4}})
