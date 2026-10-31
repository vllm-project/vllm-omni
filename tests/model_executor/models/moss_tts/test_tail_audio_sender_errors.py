# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tail sender initialization and delivery failures belong to the owner."""

from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts.local_model_state import MossLocalModelState
from vllm_omni.model_executor.models.moss_tts.tail_audio_sender import TailAudioSender

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("failure", ["device", "stream"])
def test_startup_failure_is_reported_to_engine(monkeypatch, failure):
    def fail(*args, **kwargs):
        raise RuntimeError("initialization unavailable")

    monkeypatch.setattr(torch.accelerator, "set_device_index", fail if failure == "device" else lambda *args: None)
    monkeypatch.setattr(torch.cuda, "Stream", fail)
    with pytest.raises(RuntimeError, match="initialization failed") as exc:
        TailAudioSender(SimpleNamespace(), torch.device("cuda", 0))
    assert "initialization unavailable" in str(exc.value.__cause__)


@pytest.mark.parametrize("failure", ["wait", "allocation", "copy", "synchronize"])
def test_copy_failure_is_queued_as_terminal_error(monkeypatch, failure):
    def fail(*args, **kwargs):
        raise RuntimeError(f"injected {failure} failure")

    def noop(*args, **kwargs) -> None:
        pass

    stream = SimpleNamespace(
        wait_event=fail if failure == "wait" else noop,
        synchronize=fail if failure == "synchronize" else noop,
    )
    monkeypatch.setattr(torch.accelerator, "set_device_index", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda *args: nullcontext())
    host = SimpleNamespace(copy_=fail if failure == "copy" else lambda *args, **kwargs: None)
    monkeypatch.setattr(torch, "empty", fail if failure == "allocation" else lambda *args, **kwargs: host)
    scheduler = SimpleNamespace(omni_tail_outputs=deque())
    sender = TailAudioSender(scheduler, torch.device("cuda", 0))
    sender.submit("r", [(torch.ones(2, 4), object())], torch.tensor(48000), False, after_first=True)
    sender.close()
    ((request, payload, final, error),) = scheduler.omni_tail_outputs
    assert request == "r" and payload is None and final
    assert f"injected {failure} failure" in error


def test_model_closes_tail_sender_even_when_first_sender_close_fails(mocker):
    first, tail = mocker.Mock(), mocker.Mock()
    first.close.side_effect = RuntimeError("first close failed")
    state = MossLocalModelState.__new__(MossLocalModelState)
    state._early_first_audio = object()
    state._first_audio_sender = first
    state._mtp_state = SimpleNamespace(first_audio_sender=None)
    state._fused_audio = SimpleNamespace(tail_sender=tail)
    with pytest.raises(RuntimeError, match="first close failed"):
        state.close_first_audio_sender()
    tail.close.assert_called_once_with()
