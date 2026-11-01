# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ephemeral Stage0 bind uses r.stage0-turn{N}, not resumable r.stage0."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.engine.duplex.test_session_runner import close_harness, open_harness
from vllm_omni.engine.duplex.contracts import (
    DuplexFence,
    duplex_ephemeral_stage_request_id,
    duplex_resource_request_id,
)
from vllm_omni.engine.duplex.session.manager import DuplexSessionManager

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "resumable,draining,same_stream,cleared",
    [(False, False, True, True), (True, False, True, False), (False, True, True, False), (False, False, False, False)],
)
async def test_ephemeral_cleanup_clears_only_its_finished_stream(resumable, draining, same_stream, cleared):
    h = await open_harness()
    try:
        h.session.capabilities = replace(h.session.capabilities, supports_core_resumable_request=resumable)
        request_id = "finished-request"
        stream_id = request_id if same_stream else "newer-request"
        h.runner.run.stream_request_id = stream_id
        if draining:
            h.session.bind_draining_request(request_id, "older-response")
        await h.runner.model._release_ephemeral_request(request_id, whole_turn=True)
        assert h.runner.run.stream_request_id == (None if cleared else stream_id)
    finally:
        await close_harness(h)


def test_stage0_request_id_ephemeral_when_not_resumable() -> None:
    fence = DuplexFence("duplex-test", epoch=0, turn_id=3)
    rid = DuplexSessionManager.stage_request_id(fence, stage_id=0, resumable=False)
    expected = duplex_ephemeral_stage_request_id(fence, stage_id=0)
    assert rid == expected
    assert rid.endswith(".r.stage0-turn3")
    assert not rid.endswith(".r.stage0")


def test_stage0_request_id_resumable_keeps_stable_role() -> None:
    fence = DuplexFence("duplex-test", epoch=1, turn_id=3)
    rid = DuplexSessionManager.stage_request_id(fence, stage_id=0, resumable=True)
    expected = duplex_resource_request_id(fence, "stage0")
    assert rid == expected
    assert rid.endswith(".r.stage0")
