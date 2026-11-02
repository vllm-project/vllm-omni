"""Regression tests for latent accumulation with terminal snapshots.

Latent emissions are per-step chunks, except that a stop-token finish
additionally delivers the full cumulative snapshot. The producer declares the
snapshot explicitly (``meta.full_payload_snapshot``, set where the full
payload is materialized); the accumulator never infers emission semantics
from tensor values, because values cannot distinguish a snapshot from a
legitimate identical next chunk.
"""

import pytest
import torch

from vllm_omni.outputs.mm_outputs import MultimodalPayload
from vllm_omni.outputs.output_modality import OutputModality
from vllm_omni.outputs.output_processor import OmniRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_SNAPSHOT_FLAG = {"meta.full_payload_snapshot": torch.ones((), dtype=torch.bool)}


class _FakeState:
    """Carries just the attributes add_multimodal_tensor needs."""

    def __init__(self):
        self.mm_type = None
        self.mm_accumulated = MultimodalPayload()

    def add(self, tensor, *, snapshot=False):
        payload = {"latent": tensor, **(_SNAPSHOT_FLAG if snapshot else {})}
        OmniRequestState.add_multimodal_tensor(self, payload, "latent")

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


def test_identical_consecutive_chunks_concatenate():
    # Two legitimate chunks with bitwise-identical values must both land:
    # no value heuristic may mistake the second for a cumulative snapshot.
    state = _FakeState()
    row = torch.randn(1, 8)
    state.add(row.clone())
    state.add(row.clone())
    out = state.consolidated()
    assert out.shape == (2, 8)
    assert torch.equal(out, torch.cat([row, row], dim=0))


def test_identical_equal_sized_prefill_slices_concatenate():
    state = _FakeState()
    s = torch.randn(32, 8)
    state.add(s.clone())
    state.add(s.clone())
    out = state.consolidated()
    assert out.shape == (64, 8)
    assert torch.equal(out, torch.cat([s, s], dim=0))


def test_unflagged_superset_still_concatenates():
    # Even a payload that bitwise-contains the accumulated prefix appends
    # unless the producer flagged it: emission semantics come from the
    # producer, not from tensor values.
    state = _FakeState()
    s1 = torch.randn(4, 8)
    tail = torch.randn(2, 8)
    state.add(s1.clone())
    state.add(torch.cat([s1, tail], dim=0))
    out = state.consolidated()
    assert out.shape == (10, 8)


def test_flagged_terminal_snapshot_supersedes_chunks():
    # Stop-token finish: the materialized full payload carries the explicit
    # snapshot flag and replaces the accumulated chunks.
    state = _FakeState()
    prefill = torch.randn(40, 8)
    steps = [torch.randn(1, 8) for _ in range(10)]
    state.add(prefill.clone())
    for c in steps:
        state.add(c.clone())
    snapshot = torch.cat([prefill, *steps, torch.randn(2, 8)], dim=0)
    state.add(snapshot.clone(), snapshot=True)
    out = state.consolidated()
    assert out.shape == (52, 8)
    assert torch.equal(out, snapshot)


def test_single_snapshot_passthrough():
    state = _FakeState()
    snapshot = torch.randn(4199, 8)
    state.add(snapshot.clone(), snapshot=True)
    assert torch.equal(state.consolidated(), snapshot)


def test_materialize_full_payload_entry_sets_flag():
    from vllm_omni.distributed.omni_connectors.model_runner.omni_connector_payload_transport import (
        _OmniConnectorPayloadTransportMixin,
    )

    chunks = {"hidden": [torch.randn(3, 8), torch.randn(1, 8)]}
    output, request = _OmniConnectorPayloadTransportMixin._materialize_full_payload_entry((chunks, {}, 4, "req"))
    assert output["hidden"].shape == (4, 8)
    flag = output["meta.full_payload_snapshot"]
    assert isinstance(flag, torch.Tensor) and bool(flag)
    assert request == "req"
