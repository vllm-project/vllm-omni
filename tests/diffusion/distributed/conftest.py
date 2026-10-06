# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Keep CPU collective workers independent of the host's GPU count."""

import pytest


@pytest.fixture(autouse=True)
def cpu_worker_platform(request, monkeypatch):
    # Spawn imports vLLM afresh. CPU/gloo workers must select its CPU platform,
    # otherwise its group coordinator assigns a CUDA device to each CPU rank.
    # The parent retains its already-selected platform for other model tests.
    if request.node.get_closest_marker("cpu") is not None:
        monkeypatch.setenv("VLLM_TARGET_DEVICE", "cpu")
