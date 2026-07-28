# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest

from vllm_omni.experimental.fullduplex.engine.duplex_control_client import (
    DuplexControlClient,
    DuplexControlRequestError,
)
from vllm_omni.experimental.fullduplex.engine.lease import DuplexLeaseActivity
from vllm_omni.experimental.fullduplex.engine.messages import (
    DuplexControlError,
    DuplexControlResultMessage,
    DuplexFence,
    ResumeDuplexSessionMessage,
    TouchDuplexSessionMessage,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Transport:
    def __init__(self) -> None:
        self.calls = []

    def execute(self, key, message, **kwargs):
        self.calls.append((key, message, kwargs))
        return DuplexControlResultMessage(
            control_id=message.control_id,
            fence=message.fence,
            operation="touch" if isinstance(message, TouchDuplexSessionMessage) else "resume",
            session_id=message.session_id,
            ok=True,
            stage_results=[],
        )


class _UnsupportedTransport:
    def execute(self, key, message, **kwargs):
        del key, kwargs
        return DuplexControlResultMessage(
            control_id=message.control_id,
            fence=message.fence,
            operation="touch",
            session_id=message.session_id,
            ok=False,
            stage_results=[{"result": {"supported": False}}],
            unsupported_count=1,
            error=DuplexControlError(code="invalid_capability", message="unsupported", retryable=False),
            accepted_fence=DuplexFence(message.session_id, epoch=2),
            lease_generation=7,
        )


def test_control_client_routes_touch_and_resume_by_control_id() -> None:
    transport = _Transport()
    control_ids = iter(("touch-id", "resume-id"))
    client = DuplexControlClient(transport, control_id_factory=lambda: next(control_ids))
    fence = DuplexFence("sid-client")

    assert (
        client.touch(
            fence.session_id,
            fence=fence,
            activity=DuplexLeaseActivity.PLAYBACK_ACK,
            timeout=2.0,
        )["ok"]
        is True
    )
    assert (
        client.resume(
            fence.session_id,
            fence=fence,
            expected_lease_generation=7,
            timeout=3.0,
        )["ok"]
        is True
    )

    touch_key, touch_message, touch_kwargs = transport.calls[0]
    assert touch_key == ("duplex", "touch-id")
    assert isinstance(touch_message, TouchDuplexSessionMessage)
    assert touch_message.activity == DuplexLeaseActivity.PLAYBACK_ACK.value
    assert touch_kwargs["timeout"] == 2.0

    resume_key, resume_message, resume_kwargs = transport.calls[1]
    assert resume_key == ("duplex", "resume-id")
    assert isinstance(resume_message, ResumeDuplexSessionMessage)
    assert resume_message.expected_lease_generation == 7
    assert resume_kwargs["timeout"] == 3.0


def test_control_client_treats_ok_false_as_authoritative_failure() -> None:
    client = DuplexControlClient(_UnsupportedTransport(), control_id_factory=lambda: "unsupported")
    fence = DuplexFence("sid-unsupported")

    with pytest.raises(DuplexControlRequestError, match="duplex touch failed") as exc_info:
        client.touch(
            fence.session_id,
            fence=fence,
            activity=DuplexLeaseActivity.HEARTBEAT,
            timeout=1.0,
        )

    assert exc_info.value.code == "invalid_capability"
    assert exc_info.value.retryable is False
    assert exc_info.value.accepted_fence == DuplexFence(fence.session_id, epoch=2)
    assert exc_info.value.lease_generation == 7
