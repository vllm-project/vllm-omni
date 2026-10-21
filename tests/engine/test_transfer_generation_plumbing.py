# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Generation-token plumbing for connector transfer keys (issue 24 of #7636).

Covers the two engine-side seams of the ``transfer_gen`` token:

- orchestrator stamping (``_stamp_transfer_gen`` / ``OrchestratorRequestState``)
  — the token must ride ``payload_sender_info`` and default to ``"0"``;
- the stage-engine release RPC handler — it must accept ``(id, gen)`` pairs,
  tolerate bare-id entries from older senders, and survive adapter failures.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from vllm_omni.engine.orchestrator import (
    OrchestratorRequestState,
    _stamp_transfer_gen,
)
from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestStampTransferGen:
    def test_merges_into_existing_dict(self):
        req = SimpleNamespace(payload_sender_info={"rank": 1})
        out = _stamp_transfer_gen(req, "abcd1234")
        assert out is req
        assert req.payload_sender_info == {"rank": 1, "transfer_gen": "abcd1234"}

    def test_sets_attribute_when_missing(self):
        req = SimpleNamespace()
        out = _stamp_transfer_gen(req, "abcd1234")
        assert out is req
        assert req.payload_sender_info == {"transfer_gen": "abcd1234"}

    def test_replaces_non_dict_info(self):
        req = SimpleNamespace(payload_sender_info=None)
        _stamp_transfer_gen(req, "abcd1234")
        assert req.payload_sender_info == {"transfer_gen": "abcd1234"}

    def test_zero_gen_is_short_circuit(self):
        req = SimpleNamespace(payload_sender_info={})
        out = _stamp_transfer_gen(req, "0")
        assert out is req
        assert req.payload_sender_info == {}

    def test_state_defaults_to_zero_gen(self):
        state = OrchestratorRequestState(request_id="r1")
        assert state.transfer_generation == "0"


class _RecordingAdapter:
    def __init__(self):
        self.calls: list[tuple[str, str | None]] = []

    def release_shm_resources(self, request_id, transfer_gen=None):
        self.calls.append((request_id, transfer_gen))


class TestReleaseRpcHandler:
    def _self_with(self, adapter) -> Any:
        return SimpleNamespace(scheduler=SimpleNamespace(chunk_transfer_adapter=adapter))

    def test_pairs_carry_generation(self):
        adapter = _RecordingAdapter()
        StageEngineCoreProc.omni_release_request_resources(self._self_with(adapter), [("r1", "aa11"), ("r2", None)])
        assert adapter.calls == [("r1", "aa11"), ("r2", None)]

    def test_bare_ids_fall_back_to_none(self):
        adapter = _RecordingAdapter()
        StageEngineCoreProc.omni_release_request_resources(self._self_with(adapter), ["r1", "r2"])
        assert adapter.calls == [("r1", None), ("r2", None)]

    def test_mixed_entries(self):
        adapter = _RecordingAdapter()
        entries: list[Any] = [("r1", "aa11"), "r2", ["r3", "bb22"]]
        StageEngineCoreProc.omni_release_request_resources(self._self_with(adapter), entries)
        assert adapter.calls == [("r1", "aa11"), ("r2", None), ("r3", "bb22")]

    def test_missing_adapter_is_noop(self):
        fake_self: Any = SimpleNamespace(scheduler=SimpleNamespace())
        entries: list[tuple[str, str | None]] = [("r1", "aa11")]
        StageEngineCoreProc.omni_release_request_resources(fake_self, entries)  # must not raise

    def test_adapter_failure_does_not_break_remaining_entries(self):
        seen: list[str] = []

        class _Flaky:
            def release_shm_resources(self, request_id, transfer_gen=None):
                seen.append(request_id)
                if request_id == "r1":
                    raise RuntimeError("boom")

        entries: list[tuple[str, str | None]] = [("r1", "aa11"), ("r2", "bb22")]
        StageEngineCoreProc.omni_release_request_resources(self._self_with(_Flaky()), entries)
        assert seen == ["r1", "r2"]  # r2 released despite r1 raising

    def test_empty_release_list(self):
        adapter = _RecordingAdapter()
        StageEngineCoreProc.omni_release_request_resources(self._self_with(adapter), [])
        assert adapter.calls == []
