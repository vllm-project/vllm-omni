# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

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
    started = time.perf_counter()
    try:
        torch.set_num_threads(1)
        print(f"{request.node.nodeid}: CPU reference threads {previous_threads} -> {torch.get_num_threads()}")
        yield
    finally:
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
