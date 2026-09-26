"""Regression tests for latent accumulation with terminal snapshots.

Latent emissions are per-step chunks, except that a stop-token finish
additionally delivers the full cumulative snapshot. The accumulator replaces
the accumulated chunks only when the incoming payload provably contains them
(at least as many rows, prefix bitwise-equal); anything else — single-row
steps, equal-sized prefill slices, growing chunks — concatenates as before.
"""

import pytest
import torch

from vllm_omni.outputs.mm_outputs import MultimodalPayload
from vllm_omni.outputs.output_modality import OutputModality
from vllm_omni.outputs.output_processor import OmniRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeState:
    """Carries just the attributes add_multimodal_tensor needs."""

    def __init__(self):
        self.mm_type = None
        self.mm_accumulated = MultimodalPayload()

    def add(self, tensor):
        OmniRequestState.add_multimodal_tensor(self, {"latent": tensor}, "latent")

    def consolidated(self):
        self.mm_accumulated.consolidate_tensors(OutputModality.LATENT)
        return self.mm_accumulated.tensors["latent"]


def test_per_step_chunks_concatenate():
    state = _FakeState()
    prefill = torch.randn(40, 8)
    chunks = [torch.randn(1, 8) for _ in range(5)]
    state.add(prefill)
    for c in chunks:
        state.add(c)
    out = state.consolidated()
    assert out.shape == (45, 8)
    assert torch.equal(out, torch.cat([prefill, *chunks], dim=0))


def test_one_row_then_one_row_concatenates():
    # A one-token prompt: one prefill row, then a one-row decode step. The
    # decode step has equal length but a different prefix, so it must append.
    state = _FakeState()
    prefill = torch.randn(1, 8)
    step = torch.randn(1, 8)
    state.add(prefill.clone())
    state.add(step.clone())
    out = state.consolidated()
    assert out.shape == (2, 8)
    assert torch.equal(out, torch.cat([prefill, step], dim=0))


def test_equal_sized_prefill_slices_concatenate():
    # Two 32-row chunked-prefill slices with distinct contents.
    state = _FakeState()
    s1, s2 = torch.randn(32, 8), torch.randn(32, 8)
    state.add(s1.clone())
    state.add(s2.clone())
    out = state.consolidated()
    assert out.shape == (64, 8)
    assert torch.equal(out, torch.cat([s1, s2], dim=0))


def test_growing_chunks_concatenate():
    # A longer chunk that does not duplicate the accumulated prefix appends.
    state = _FakeState()
    s1, s2 = torch.randn(4, 8), torch.randn(8, 8)
    state.add(s1.clone())
    state.add(s2.clone())
    out = state.consolidated()
    assert out.shape == (12, 8)
    assert torch.equal(out, torch.cat([s1, s2], dim=0))


def test_terminal_snapshot_supersedes_chunks():
    # Stop-token finish: the final flush delivers the full cumulative
    # snapshot (prefix bitwise-equal to the accumulated chunks, plus the
    # trailing steps). It must replace the chunks, not be appended.
    state = _FakeState()
    prefill = torch.randn(40, 8)
    steps = [torch.randn(1, 8) for _ in range(10)]
    state.add(prefill.clone())
    for c in steps:
        state.add(c.clone())
    snapshot = torch.cat([prefill, *steps, torch.randn(2, 8)], dim=0)  # + 2 trailing rows
    state.add(snapshot.clone())
    out = state.consolidated()
    assert out.shape == (52, 8)
    assert torch.equal(out, snapshot)


def test_single_snapshot_passthrough():
    state = _FakeState()
    snapshot = torch.randn(4199, 8)
    state.add(snapshot.clone())
    assert torch.equal(state.consolidated(), snapshot)
