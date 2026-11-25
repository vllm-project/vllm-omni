# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Bounded FlashInfer Ulysses output registrations across request shapes (no GPU or FlashInfer needed)."""

import pytest
import torch

from vllm_omni.diffusion.distributed import flashinfer_ulysses

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _Communicator:
    """Mimics the direct route: each registration allocates ``max_bytes`` and hands back a view at its base."""

    def __init__(self, max_bytes, world_size=2, registered=True):
        self.max_bytes = max_bytes
        self.world_size = world_size
        self.registered = registered
        self.registrations = []

    def allocate_output(self, x, op):
        shape = flashinfer_ulysses._output_shape(x, op, self.world_size)
        if not self.registered:
            out = torch.empty(shape, dtype=x.dtype)
        else:
            flat = torch.empty(self.max_bytes // x.element_size(), dtype=x.dtype)
            out = flat.narrow(0, 0, torch.Size(shape).numel()).view(shape)
        self.registrations.append(out.data_ptr())
        return out


@pytest.fixture(autouse=True)
def _clean_outputs(monkeypatch):
    monkeypatch.setattr(flashinfer_ulysses, "_OUTPUTS", {})


def _shapes():
    # A large request first, then several distinct smaller geometries, then the large one again.
    return [(1, 64, 16, 8), (1, 32, 16, 8), (1, 48, 8, 8), (2, 8, 4, 8), (1, 16, 16, 8), (1, 64, 16, 8)]


def test_registered_outputs_stay_one_per_slot_across_shapes():
    large = torch.Size(_shapes()[0]).numel() * 2
    comm = _Communicator(max_bytes=large)
    for shape in _shapes():
        x = torch.empty(shape, dtype=torch.bfloat16)
        for slot in range(3):
            out = flashinfer_ulysses._output(comm, "sp", x, "scatter_heads", slot)
            assert tuple(out.shape) == flashinfer_ulysses._output_shape(x, "scatter_heads", 2)
            assert out.is_contiguous()
        y = torch.empty(shape, dtype=torch.bfloat16)
        out = flashinfer_ulysses._output(comm, "sp", y, "gather_heads", 0)
        assert tuple(out.shape) == flashinfer_ulysses._output_shape(y, "gather_heads", 2)
    # One registration per (op, slot); later shapes reuse the same base pointer, which keeps the direct route.
    assert len(comm.registrations) == 4
    assert len(flashinfer_ulysses._OUTPUTS) == 4
    for (_, op, slot), out in flashinfer_ulysses._OUTPUTS.items():
        x = torch.empty(_shapes()[3], dtype=torch.bfloat16)
        assert flashinfer_ulysses._output(comm, "sp", x, op, slot).data_ptr() == out.data_ptr()


def test_staged_outputs_are_replaced_not_accumulated():
    # Small, then large, then small again: the larger shape replaces the staged output once, and the last
    # small request reuses that second output as a view instead of allocating a third time.
    comm = _Communicator(max_bytes=1 << 20, registered=False)
    small, large = (1, 16, 16, 8), (1, 64, 16, 8)
    outs = [
        flashinfer_ulysses._output(comm, "sp", torch.empty(shape, dtype=torch.float16), "gather_heads", 0)
        for shape in (small, large, small)
    ]
    assert len(comm.registrations) == 2
    assert outs[1].data_ptr() != outs[0].data_ptr()
    assert outs[2].data_ptr() == outs[1].data_ptr()
    assert tuple(outs[2].shape) == flashinfer_ulysses._output_shape(torch.empty(small), "gather_heads", 2)
    assert outs[2].is_contiguous()
    assert len(flashinfer_ulysses._OUTPUTS) == 1


def test_dtype_change_takes_a_new_output():
    comm = _Communicator(max_bytes=1 << 20)
    first = flashinfer_ulysses._output(comm, "sp", torch.empty(1, 8, 4, 8, dtype=torch.float16), "scatter_heads", 0)
    second = flashinfer_ulysses._output(comm, "sp", torch.empty(1, 8, 4, 8, dtype=torch.bfloat16), "scatter_heads", 0)
    assert second.dtype == torch.bfloat16 and first.dtype == torch.float16
    assert len(flashinfer_ulysses._OUTPUTS) == 1


def test_growing_the_communicator_synchronizes_before_closing_the_old_one(monkeypatch):
    events = []

    class _Closable:
        def __init__(self, group, max_bytes, dtype, backend, device):
            self.max_bytes = max_bytes

        def close(self):
            events.append("close")

    flashinfer_comm = pytest.importorskip("flashinfer.comm", exc_type=ImportError)
    monkeypatch.setattr(flashinfer_comm, "UlyssesCommunicator", _Closable, raising=False)
    monkeypatch.setattr(flashinfer_ulysses, "_COMMUNICATORS", {})
    monkeypatch.setattr(flashinfer_ulysses, "_resolve_process_group", lambda name: name)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda device=None: events.append("sync"))
    flashinfer_ulysses._OUTPUTS[("sp", "scatter_heads", 0)] = torch.empty(4)
    flashinfer_ulysses._communicator("sp", 1024, torch.bfloat16)
    assert events == []  # first creation releases nothing
    flashinfer_ulysses._communicator("sp", 512, torch.bfloat16)
    assert events == []  # fits: reused
    flashinfer_ulysses._communicator("sp", 4096, torch.bfloat16)
    # The previous layer may still read the old registered outputs: synchronize, then drop and close them.
    assert events == ["sync", "close"]
    assert ("sp", "scatter_heads", 0) not in flashinfer_ulysses._OUTPUTS
