# SPDX-License-Identifier: Apache-2.0
"""Ephemeral Stage0 bind uses r.stage0-turn{N}, not resumable r.stage0."""

from __future__ import annotations

from vllm_omni.engine.duplex.contracts import (
    DuplexFence,
    duplex_ephemeral_stage_request_id,
    duplex_resource_request_id,
)
from vllm_omni.engine.duplex.session.manager import DuplexSessionManager


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
