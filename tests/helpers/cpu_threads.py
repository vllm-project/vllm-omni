# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import time

import pytest
import torch


@pytest.fixture
def single_threaded_cpu(request):
    """Bound small CPU reference models without changing GPU test settings."""
    if request.node.get_closest_marker("cpu") is None:
        yield
        return
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    print(f"{request.node.nodeid}: CPU reference threads {previous_threads} -> {torch.get_num_threads()}")
    started = time.perf_counter()
    try:
        yield
    finally:
        torch.set_num_threads(previous_threads)
        print(
            f"{request.node.nodeid}: CPU reference elapsed {time.perf_counter() - started:.3f}s; "
            f"restored threads {torch.get_num_threads()}"
        )
