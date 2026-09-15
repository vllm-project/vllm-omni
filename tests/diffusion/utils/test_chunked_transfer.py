# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ownership and backpressure contract for chunk transfers to the host.

The ring hands a chunk one reusable host slot for its whole trip. Getting the
release point wrong is silent: a slot reused while its copy is still landing,
or while an encoder still holds a view of it, corrupts frames rather than
raising. These cover the release point, the backpressure it creates, and the
escape path out of a wait.
"""

from __future__ import annotations

import threading

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.utils.chunked_video import (
    ChunkedVideoMP4Session,
    ChunkLease,
    PinnedChunkRing,
    quantize_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _frames(value: int = 7, *, batch: int = 1) -> torch.Tensor:
    """One quantized ``BTHWC`` chunk, the shape the ring transfers."""
    return torch.full((batch, 2, 4, 6, 3), value, dtype=torch.uint8)


def _cpu_ring(depth: int) -> PinnedChunkRing:
    return PinnedChunkRing(depth=depth, device=torch.device("cpu"))


def test_slot_returns_only_after_its_last_reader():
    """Each batch entry reads the same buffer, so one release must not free it."""
    ring = _cpu_ring(2)
    lease = ring.transfer(_frames(batch=3), readers=3)

    assert ring.slots_in_use == 1
    view = lease.wait()
    assert view.shape == (3, 2, 4, 6, 3)
    # Waiting is not consuming: an encoder still holds a view of this buffer.
    assert ring.slots_in_use == 1

    lease.release()
    lease.release()
    assert ring.slots_in_use == 1

    lease.release()
    assert ring.slots_in_use == 0
    # A late extra release must not free the slot a second time.
    lease.release()
    assert ring.slots_in_use == 0


def test_transfer_waits_for_a_slot_instead_of_growing_the_pool():
    """Depth is the bound on in-flight host buffers, so a full ring blocks."""
    ring = _cpu_ring(1)
    held = ring.transfer(_frames(1), readers=1)
    second: list[object] = []

    def take_a_slot() -> None:
        second.append(ring.transfer(_frames(2), readers=1))

    waiter = threading.Thread(target=take_a_slot)
    waiter.start()
    waiter.join(timeout=0.3)
    assert waiter.is_alive(), "a full ring must not hand out a second slot"

    held.wait()
    held.release()
    waiter.join(timeout=5)
    assert not waiter.is_alive()
    assert ring.slots_in_use == 1
    assert ring.stats.slot_wait_seconds > 0


def test_abort_wakes_a_transfer_waiting_for_a_slot():
    """A dead consumer must surface as an error, not as a stuck producer."""
    ring = _cpu_ring(1)
    ring.transfer(_frames(), readers=1)
    failures: list[BaseException] = []

    def take_a_slot() -> None:
        try:
            ring.transfer(_frames(), readers=1)
        except BaseException as exc:  # noqa: BLE001
            failures.append(exc)

    waiter = threading.Thread(target=take_a_slot)
    waiter.start()
    waiter.join(timeout=0.3)
    assert waiter.is_alive()

    ring.abort(RuntimeError("encoder died"))
    waiter.join(timeout=5)
    assert [str(exc) for exc in failures] == ["encoder died"]
    # Later transfers stay failed rather than silently resuming.
    with pytest.raises(RuntimeError, match="encoder died"):
        ring.transfer(_frames(), readers=1)


class _RecordingEncoder:
    """Stands in for one batch entry's encoder, honouring the release contract."""

    instances: list[_RecordingEncoder] = []

    def __init__(self, **kwargs):
        self.pushes: list[int] = []
        self.released = 0
        self.fail_on_push = False
        self.__class__.instances.append(self)

    def push(self, frames, *, on_consumed=None):
        if self.fail_on_push:
            if on_consumed is not None:
                on_consumed()
                self.released += 1
            raise RuntimeError("encoder rejected the chunk")
        self.pushes.append(int(frames[0, 0, 0, 0]))
        if on_consumed is not None:
            on_consumed()
            self.released += 1

    def finish(self) -> bytes:
        return b"mp4"

    def abort(self) -> None:
        return None


@pytest.fixture
def recording_encoders(monkeypatch):
    _RecordingEncoder.instances = []
    monkeypatch.setattr(
        "vllm_omni.diffusion.utils.chunked_video.ChunkedMP4Encoder",
        _RecordingEncoder,
    )
    return _RecordingEncoder.instances


def _push_ramp(session: ChunkedVideoMP4Session, values: list[int]) -> None:
    for value in values:
        session.push(torch.full((1, 3, 1, 4, 6), value / 255.0))


def test_session_feeds_chunks_in_producer_order(recording_encoders):
    """Copies complete in issue order, and the encoder must see that order."""
    session = ChunkedVideoMP4Session(value_range=(0.0, 1.0), fps=24)
    _push_ramp(session, [10, 20, 30, 40])

    assert session.finish() == [b"mp4"]
    assert recording_encoders[0].pushes == [10, 20, 30, 40]


def test_session_returns_the_slot_when_an_encoder_rejects_a_chunk(recording_encoders):
    """A rejected chunk must not strand the buffer it was lent."""
    session = ChunkedVideoMP4Session(value_range=(0.0, 1.0), fps=24, transfer_slots=1)
    session.push(torch.zeros(1, 3, 1, 4, 6))
    recording_encoders[0].fail_on_push = True

    with pytest.raises(RuntimeError, match="rejected"):
        session.push(torch.zeros(1, 3, 1, 4, 6))

    assert session.ring is not None
    assert session.ring.slots_in_use == 0


def test_session_abort_releases_every_slot_still_in_flight(recording_encoders):
    """Teardown returns the pool even when nothing consumed the chunks."""
    session = ChunkedVideoMP4Session(value_range=(0.0, 1.0), fps=24, transfer_slots=2)
    session.push(torch.zeros(1, 3, 1, 4, 6))

    session.abort()

    assert session.ring is not None
    assert session.ring.slots_in_use == 0


@hardware_test(res={"cuda": ["H100", "B200"]})
def test_device_chunks_land_intact_through_the_pinned_ring():
    """On an accelerator the copy is deferred, so its event must order the read."""
    if not torch.accelerator.is_available():
        pytest.skip("no accelerator available")

    device = torch.device(torch.accelerator.current_accelerator().type, 0)
    ring = PinnedChunkRing(depth=2, device=device)
    pending: list[tuple[int, ChunkLease]] = []
    for step in range(6):
        value = 10 * step + 1
        # Quantize on the compute stream immediately before handing the chunk
        # over: without the source-ready event the copy would race this work.
        chunk = torch.full((1, 3, 2, 4, 6), value / 255.0, device=device)
        pending.append((value, ring.transfer(quantize_chunk(chunk, (0.0, 1.0)), readers=1)))
        if len(pending) == 2:
            for expected, lease in pending:
                assert int(lease.wait()[0, 0, 0, 0, 0]) == expected
                lease.release()
            pending.clear()

    assert ring.stats.transfers == 6
    assert ring.stats.peak_slots_in_use == 2
    assert ring.stats.peak_pending_bytes == 2 * 2 * 4 * 6 * 3
