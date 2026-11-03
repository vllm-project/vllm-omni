# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib

import pytest

from tests.distributed.weight_sharing.helpers import load_weight_sharing


@pytest.fixture(scope="session")
def weight_sharing():
    return load_weight_sharing()


@pytest.fixture
def contracts(weight_sharing):
    return importlib.import_module(f"{weight_sharing.__name__}.contracts")


@pytest.fixture(params=[False, True], ids=["native", "expandable"])
def ipc_allocator(request, monkeypatch):
    # Set before spawning: each CUDA worker initializes its own allocator.
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", f"backend:native,expandable_segments:{request.param}")
    monkeypatch.setenv("TORCH_CUDA_EXPANDABLE_SEGMENTS_IPC", "1")
    return request.param
