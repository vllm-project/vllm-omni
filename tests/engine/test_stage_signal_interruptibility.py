# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The owned shutdown handlers must interrupt the next native idle wait."""

import signal

import pytest
from vllm.v1.engine.core import EngineCoreProc

from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def owned_engine(mocker):
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine._omni_shutdown_signal_handler = mocker.Mock()
    mocker.patch("signal.getsignal", return_value=engine._omni_shutdown_signal_handler)
    return engine


def test_restores_interruptibility_before_waiting(owned_engine, mocker):
    calls: list[tuple[int, bool] | str] = []
    mocker.patch("signal.siginterrupt", side_effect=lambda signum, flag: calls.append((signum, flag)), create=True)
    mocker.patch.object(EngineCoreProc, "_process_input_queue", side_effect=lambda: calls.append("wait"))

    owned_engine._process_input_queue()

    assert calls == [(signal.SIGTERM, True), (signal.SIGINT, True), "wait"]


@pytest.mark.parametrize("new_handler", [signal.SIG_DFL, signal.SIG_IGN, lambda signum, frame: None])
def test_leaves_replaced_handler_policy_alone(owned_engine, new_handler, mocker):
    mocker.patch("signal.getsignal", side_effect=[new_handler, owned_engine._omni_shutdown_signal_handler])
    interrupt = mocker.patch("signal.siginterrupt", create=True)
    wait = mocker.patch.object(EngineCoreProc, "_process_input_queue")

    owned_engine._process_input_queue()

    interrupt.assert_called_once_with(signal.SIGINT, True)
    wait.assert_called_once_with()


def test_unowned_engine_keeps_inherited_wait(mocker):
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    interrupt = mocker.patch("signal.siginterrupt", create=True)
    get_handler = mocker.patch("signal.getsignal")
    wait = mocker.patch.object(EngineCoreProc, "_process_input_queue")

    engine._process_input_queue()

    interrupt.assert_not_called()
    get_handler.assert_not_called()
    wait.assert_called_once_with()


def test_platform_without_siginterrupt_keeps_inherited_wait(owned_engine, monkeypatch, mocker):
    monkeypatch.delattr(signal, "siginterrupt", raising=False)
    wait = mocker.patch.object(EngineCoreProc, "_process_input_queue")

    owned_engine._process_input_queue()

    wait.assert_called_once_with()


def test_restores_again_after_each_native_step(owned_engine, mocker):
    interrupt = mocker.patch("signal.siginterrupt", create=True)
    wait = mocker.patch.object(EngineCoreProc, "_process_input_queue")

    owned_engine._process_input_queue()
    owned_engine._process_input_queue()

    assert interrupt.call_args_list == [
        mocker.call(signal.SIGTERM, True),
        mocker.call(signal.SIGINT, True),
        mocker.call(signal.SIGTERM, True),
        mocker.call(signal.SIGINT, True),
    ]
    assert wait.call_count == 2


def test_inherited_wait_exception_is_not_swallowed(owned_engine, mocker):
    mocker.patch("signal.siginterrupt", create=True)
    mocker.patch.object(EngineCoreProc, "_process_input_queue", side_effect=RuntimeError("native wait failed"))

    with pytest.raises(RuntimeError, match="native wait failed"):
        owned_engine._process_input_queue()
