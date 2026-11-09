# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Auto-response commits must consume the same object they retain."""

from __future__ import annotations

import base64
import struct
from types import MethodType, SimpleNamespace

import pytest

from vllm_omni.engine.duplex.config import DuplexSessionState
from vllm_omni.engine.duplex.session.append_task import AppendAttempt
from vllm_omni.engine.duplex.session.runner import DuplexSessionRunner
from vllm_omni.model_executor.models.aura_omni.duplex.session import AuraServingSessionState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _pcm(samples: list[float]) -> str:
    raw = b"".join(struct.pack("<f", sample) for sample in samples)
    return base64.b64encode(raw).decode("ascii")


def _samples(audio: str) -> list[float]:
    raw = base64.b64decode(audio)
    return [struct.unpack_from("<f", raw, offset)[0] for offset in range(0, len(raw), 4)]


class _Reservation:
    def __init__(self, payload: dict[str, object], operation_id: str) -> None:
        self.payload = payload
        self.operation_id = operation_id
        self.byte_count = 4

    def commit(self) -> None:
        return None


class _Buffer:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self._payloads = list(payloads)
        self._index = 0

    def prepare_commit(self, *, operation_id: str, chunk_period_ms: int) -> _Reservation:
        del chunk_period_ms
        payload = self._payloads[self._index]
        self._index += 1
        return _Reservation(payload, operation_id)

    def prepare_commit_prefix(self, *, operation_id: str, chunk_period_ms: int) -> None:
        del operation_id, chunk_period_ms
        return None


class _Model:
    def __init__(self) -> None:
        self.submitted: list[str] = []

    async def append_runtime_input(self, payload: dict[str, object], **kwargs: object) -> tuple[bool, bool]:
        del kwargs
        audio = payload["audio"]
        assert isinstance(audio, str)
        self.submitted.append(audio)
        return True, True


@pytest.mark.asyncio
@pytest.mark.parametrize("item_ids", [("item-a", "item-b"), (None, None)])
async def test_auto_response_commit_consumes_retained_pcm(
    item_ids: tuple[str | None, str | None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Copying the payload after retain skips consume, so the next commit merges it.

    Successive commits of [1.0] then [2.0] must submit those payloads, with or
    without a realtime item id. Stamping the item id after retain used to submit
    [1.0] followed by [1.0, 2.0].
    """
    import vllm_omni.engine.duplex.session.runner as runner_mod

    monkeypatch.setattr(runner_mod.helpers, "commit_audio_input", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner_mod.helpers,
        "audio_committed_payload",
        lambda *args, **kwargs: {"type": "input.committed"},
    )
    state = AuraServingSessionState()
    state.audio_buffer = _Buffer(
        [
            {"format": "pcm_f32le", "sample_rate_hz": 16000, "audio": _pcm([1.0]), "is_speech": True},
            {"format": "pcm_f32le", "sample_rate_hz": 16000, "audio": _pcm([2.0]), "is_speech": True},
        ]
    )
    session = SimpleNamespace(
        turn_id=0,
        epoch=0,
        state=DuplexSessionState.OPEN,
        capabilities=SimpleNamespace(chunk_period_ms=1000),
        release_input_bytes=lambda count: None,
    )
    model = _Model()

    async def start_append(payload: dict[str, object], **kwargs: object) -> bool:
        retained = kwargs["retained_committed_payload"]
        assert retained is state.committed_audio_payload
        item_id = retained.get("realtime_item_id") if isinstance(retained, dict) else None
        expected = item_ids[len(model.submitted)]
        assert item_id == expected
        attempt = AppendAttempt(
            ctx=SimpleNamespace(session=session, model_state=state, run=SimpleNamespace()),
            out=SimpleNamespace(),
            model=model,
            fail_session=lambda reason: None,
            payload=payload,
            epoch=0,
            request_id="req",
            final=True,
            pcm_reservation=None,
            operation_id=kwargs.get("operation_id") if isinstance(kwargs.get("operation_id"), str) else None,
            retained_committed_payload=retained if isinstance(retained, dict) else None,
            precreated_response_id=None,
        )
        return await attempt._submit()

    runner = SimpleNamespace(
        session=session,
        model_state=state,
        emit=lambda payload: None,
        _start_append=start_append,
    )
    for name in (
        "_reserve_commit_prefixes",
        "_rollback_commit_reservations",
        "_hold_commit_prefixes",
        "_submit_held_commit_prefixes",
    ):
        setattr(runner, name, MethodType(getattr(DuplexSessionRunner, name), runner))
    for item_id in item_ids:
        await DuplexSessionRunner._commit_and_start_auto_response(runner, {}, realtime_item_id=item_id)

    assert [_samples(audio) for audio in model.submitted] == [[1.0], [2.0]]
    assert state.committed_audio_payload is None
    assert state.committed_audio_consume_started is False
