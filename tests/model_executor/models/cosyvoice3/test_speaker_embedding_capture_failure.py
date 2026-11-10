# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A failed first replay must leave no graph bucket behind (review r7).

``CampplusTorch.__call__`` publishes ``_graphs[frames]`` only after the
post-capture ``graph.replay()`` + ``clone()`` succeed. This test fakes
``torch.npu`` on CPU with a graph whose ``replay()`` raises on the first
call and asserts the three consequences that keep later requests safe:
the bucket is absent, the frame count is recorded in ``_graph_failed``,
and the next same-length call returns the eager result.
"""

import contextlib

import pytest
import torch

from vllm_omni.model_executor.models.cosyvoice3.speaker_embedding_torch import (
    CampplusTorch,
)
from vllm_omni.model_executor.models.indextts2.utils.campplus.dtdnn import CAMPPlus

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeNPUDevice:
    """Stands in for ``torch.device("npu")`` on a CPU-only box.

    ``type`` reads ``"npu"`` so the capture branch is entered, while
    equality matches cpu tensors (and the fake itself, which the test
    plugs into the ``freqs_cis`` device check) so nothing is moved to a
    device that does not exist.
    """

    type = "npu"

    def __eq__(self, other):
        if other is self:
            return True
        return isinstance(other, torch.device) and other.type == "cpu"

    def __hash__(self):
        return hash(("fake-npu", id(self)))


class _FakeStream:
    def wait_event(self, event):
        pass

    def record_event(self):
        return object()


class _BrokenGraph:
    """NPUGraph whose first ``replay()`` raises: the exact failure the
    registration-order fix must survive without publishing a bucket."""

    instances = []

    def __init__(self):
        self.replays = 0
        _BrokenGraph.instances.append(self)

    def replay(self):
        self.replays += 1
        raise RuntimeError("simulated first-replay failure")


class _FakeNPUModule:
    """Minimal ``torch.npu`` stand-in: real kernels run eagerly on CPU,
    only the graph bookkeeping is faked."""

    class config:
        """Mimic ``torch.npu.config`` (the prepare step flips this)."""

        allow_internal_format = True

    @staticmethod
    def set_compile_mode(jit_compile=False):
        pass

    graph = staticmethod(contextlib.nullcontext)

    @staticmethod
    def current_stream(device):
        return _FakeStream()

    @staticmethod
    def synchronize():
        pass

    NPUGraph = _BrokenGraph


@pytest.fixture()
def fake_npu(monkeypatch):
    monkeypatch.setattr(torch, "npu", _FakeNPUModule, raising=False)
    _BrokenGraph.instances.clear()
    return _FakeNPUModule


def _wrapper() -> CampplusTorch:
    # Random weights: the failure path is a control-flow check, not a
    # golden-output comparison, so no pretrained asset is needed.
    wrapper = CampplusTorch(CAMPPlus(feat_dim=80, embedding_size=192).eval(), "cpu")
    wrapper.device = _FakeNPUDevice()
    return wrapper


def test_first_replay_failure_leaves_no_bucket(fake_npu) -> None:
    wrapper = _wrapper()
    frames = 97
    feat = torch.randn(frames, 80)

    with torch.inference_mode():
        out = wrapper(feat)  # capture branch: replay raises -> eager fallback

    assert frames not in wrapper._graphs
    assert frames in wrapper._graph_failed
    assert _BrokenGraph.instances[0].replays == 1

    with torch.inference_mode():
        again = wrapper(feat)  # next same-length call must run eager
        ref = wrapper.model(feat.unsqueeze(0).float())

    assert torch.equal(again, ref)
    assert torch.equal(out, ref)
    assert wrapper._eager_calls == 1  # only the retry; the fallback does not count
    assert wrapper._replay_calls == 0
