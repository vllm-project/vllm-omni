# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import os
import time
from collections.abc import Callable, Generator
from typing import Protocol

import pytest


class _TorchThreadAPI(Protocol):
    def get_num_threads(self) -> int: ...

    def set_num_threads(self, num_threads: int) -> None: ...


def _single_threaded_cpu(
    request: pytest.FixtureRequest,
    load_torch: Callable[[], _TorchThreadAPI],
) -> Generator[None, None, None]:
    """Implement the marker guard separately so its state handling is testable."""
    if request.node.get_closest_marker("cpu") is None:
        yield
        return

    torch = load_torch()
    previous_threads = torch.get_num_threads()
    thread_env = ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
    previous_env = {name: os.environ.get(name) for name in thread_env}
    started = time.perf_counter()
    try:
        for name in thread_env:
            os.environ[name] = "1"
        torch.set_num_threads(1)
        print(f"{request.node.nodeid}: CPU reference threads {previous_threads} -> {torch.get_num_threads()}")
        yield
    finally:
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        torch.set_num_threads(previous_threads)
        print(
            f"{request.node.nodeid}: CPU reference elapsed {time.perf_counter() - started:.3f}s; "
            f"restored threads {torch.get_num_threads()}"
        )


@pytest.fixture
def single_threaded_cpu(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """Bound opted-in CPU tests while leaving GPU-marked cases unchanged."""

    def load_torch() -> _TorchThreadAPI:
        import torch

        return torch

    yield from _single_threaded_cpu(request, load_torch)
