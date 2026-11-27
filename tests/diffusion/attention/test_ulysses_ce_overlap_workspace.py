# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Workspace sizing and teardown of the copy-engine overlap runners, without GPUs."""

import sys

import pytest
import torch

from vllm_omni.diffusion.attention.parallel import ulysses_ce_overlap
from vllm_omni.diffusion.distributed import a2a_permute

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Group:
    group_name = "ulysses"


class _Handle:
    def __init__(self, alloc):
        self.buffer_ptrs = [alloc.data_ptr(), alloc.data_ptr()]


@pytest.fixture
def fake_runtime(monkeypatch):
    allocations, events = [], []

    def empty_strided_p2p(size, stride, dtype, device):
        alloc = torch.empty(size, dtype=dtype)
        allocations.append(alloc)
        return alloc

    monkeypatch.setattr(ulysses_ce_overlap.symm_mem._SymmetricMemory, "empty_strided_p2p", empty_strided_p2p)
    monkeypatch.setattr(ulysses_ce_overlap.symm_mem, "rendezvous", lambda alloc, name: _Handle(alloc))
    monkeypatch.setattr(ulysses_ce_overlap.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(ulysses_ce_overlap.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(ulysses_ce_overlap.dist, "barrier", lambda group: events.append("barrier"))
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda device=None: events.append("sync"))
    monkeypatch.setattr(torch.cuda, "Stream", lambda device=None: object())
    monkeypatch.setattr(torch.cuda, "Event", lambda: object())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(ulysses_ce_overlap, "_driver", lambda: None)
    monkeypatch.setattr(ulysses_ce_overlap, "_ce_runtime", lambda: None)
    monkeypatch.setattr(ulysses_ce_overlap, "_RUNNERS", {})
    return allocations, events


def _configure(shape):
    seq, heads, chunks = shape
    key = ("ulysses", torch.device("cpu"))
    runner = ulysses_ce_overlap._RUNNERS.get(key)
    if runner is None:
        runner = ulysses_ce_overlap._RUNNERS[key] = ulysses_ce_overlap.CEOverlapUlysses(
            _Group(), device=torch.device("cpu")
        )
    runner._configure(seq, heads, 128, chunks, torch.bfloat16)
    return runner


def test_mixed_shapes_share_one_grow_only_workspace(fake_runtime):
    allocations, events = fake_runtime
    runner = _configure((384, 56, 7))
    assert len(allocations) == 1
    first = runner.alloc
    # Smaller sequences, fewer chunks and fewer heads reuse the same allocation.
    for shape in ((256, 56, 7), (384, 56, 2), (128, 16, 4), (384, 56, 7)):
        assert _configure(shape) is runner
        assert runner.alloc is first
    assert len(allocations) == 1
    # A larger sequence grows it once, after synchronizing so nothing still reads the old allocation.
    events.clear()
    _configure((768, 56, 7))
    assert len(allocations) == 2 and runner.alloc is allocations[-1]
    assert events[0] == "sync"
    # More chunks than the flag region holds also grows it.
    _configure((768, 56, 14))
    assert len(allocations) == 3
    assert len(ulysses_ce_overlap._RUNNERS) == 1


def test_views_fit_in_the_workspace_after_the_flags(fake_runtime):
    runner = _configure((384, 56, 7))
    _configure((128, 16, 4))
    base = runner.alloc.data_ptr()
    end = base + runner.capacity
    flags_end = base + 4 * runner.n * (runner.max_chunks + 2)
    views = [*(t for chunk in runner.inputs for t in chunk), runner.output, *runner.self_inputs]
    for view in views:
        assert view.data_ptr() >= flags_end
        span = sum((size - 1) * stride for size, stride in zip(view.shape, view.stride()))
        assert view.data_ptr() + span * view.element_size() < end
    # Flag slots stay at fixed addresses below the data, whatever the chunk count.
    assert runner._slot("in", 1, runner.k - 1) < runner.data_off
    assert runner._slot("out", 1) < runner.data_off


def test_a2a_permute_teardown_releases_the_runners(fake_runtime, monkeypatch):
    allocations, events = fake_runtime
    _configure((384, 56, 7))
    assert ulysses_ce_overlap._RUNNERS
    monkeypatch.setitem(sys.modules, ulysses_ce_overlap.__name__, ulysses_ce_overlap)
    events.clear()
    a2a_permute.clear_a2a_permute_workspaces()
    assert not ulysses_ce_overlap._RUNNERS
    assert events == ["sync"]
