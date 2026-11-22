# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.helpers.fixtures.cpu_threads import _single_threaded_cpu

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _ThreadController:
    def __init__(self, num_threads: int, *, fail_when_bounding: bool = False):
        self.num_threads = num_threads
        self.fail_when_bounding = fail_when_bounding
        self.transitions: list[int] = []

    def get_num_threads(self) -> int:
        return self.num_threads

    def set_num_threads(self, num_threads: int) -> None:
        self.num_threads = num_threads
        self.transitions.append(num_threads)
        if self.fail_when_bounding and num_threads == 1:
            raise RuntimeError("could not set thread count")


def _request(*, cpu_marked: bool):
    marker = object() if cpu_marked else None
    node = SimpleNamespace(
        nodeid="tests/helpers/tests/test_cpu_threads.py::probe",
        get_closest_marker=lambda name: marker if name == "cpu" else None,
    )
    return SimpleNamespace(node=node)


def _finish_fixture(fixture) -> None:
    with pytest.raises(StopIteration):
        next(fixture)


def test_registered_fixture_caps_cpu_marked_test(single_threaded_cpu):
    import torch

    assert torch.get_num_threads() == 1


def test_single_threaded_cpu_caps_and_restores(capsys):
    torch = _ThreadController(8)
    fixture = _single_threaded_cpu(_request(cpu_marked=True), lambda: torch)

    next(fixture)
    assert torch.num_threads == 1
    _finish_fixture(fixture)

    assert torch.num_threads == 8
    assert torch.transitions == [1, 8]
    output = capsys.readouterr().out
    assert "CPU reference threads 8 -> 1" in output
    assert "restored threads 8" in output


def test_single_threaded_cpu_is_noop_without_cpu_marker():
    loaded = False

    def load_torch():
        nonlocal loaded
        loaded = True
        return _ThreadController(8)

    fixture = _single_threaded_cpu(_request(cpu_marked=False), load_torch)
    next(fixture)
    _finish_fixture(fixture)

    assert not loaded


def test_single_threaded_cpu_restores_if_bounding_raises():
    torch = _ThreadController(8, fail_when_bounding=True)
    fixture = _single_threaded_cpu(_request(cpu_marked=True), lambda: torch)

    with pytest.raises(RuntimeError, match="could not set thread count"):
        next(fixture)

    assert torch.num_threads == 8
    assert torch.transitions == [1, 8]
