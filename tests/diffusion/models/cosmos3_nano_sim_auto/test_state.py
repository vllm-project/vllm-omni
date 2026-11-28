# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ring slot assignment and the dense ring oracle."""

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3_nano_sim_auto.layout import TokenMetadata
from vllm_omni.diffusion.models.cosmos3_nano_sim_auto.state_cosmos3_nano_sim_auto import (
    Cosmos3NanoSimAutoSessionState,
    DenseReplayRing,
    ring_slot_for_step,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_reference_slot_assignment_pins_sink_and_rotates_the_rest() -> None:
    assert [ring_slot_for_step(step, cache_chunks=2, sink_chunks=1) for step in range(6)] == [0, 1, 1, 1, 1, 1]
    assert [ring_slot_for_step(step, cache_chunks=8, sink_chunks=1) for step in range(10)] == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        1,
        2,
    ]
    with pytest.raises(ValueError):
        ring_slot_for_step(0, cache_chunks=1, sink_chunks=1)


def test_dense_ring_slots_and_history_metadata() -> None:
    ring = DenseReplayRing(
        num_layers=2, cache_chunks=2, frame_unit=8, num_kv_heads=1, head_dim=4, device="cpu", dtype=torch.float32
    )
    slots = ring.write_slots(1, 3, torch.device("cpu"))
    assert slots.tolist() == [8, 9, 10]
    ring.keys[0][slots] = 1.0
    ring.clear_slot(1)
    assert float(ring.keys[0].abs().sum()) == 0.0
    with pytest.raises(ValueError, match="frame unit"):
        ring.write_slots(0, 9, torch.device("cpu"))

    state = Cosmos3NanoSimAutoSessionState(session_id="s", frame_unit=8)
    metadata = TokenMetadata(
        view_id=torch.zeros(3, dtype=torch.long),
        frame_id=torch.arange(3),
        timestamp=torch.zeros(3),
        role=torch.full((3,), 4),
        step=torch.zeros(3, dtype=torch.long),
        caption_scope=torch.ones(3, dtype=torch.long),
    )
    state.record_commit(slot=1, step=3, metadata=metadata)
    state.record_commit(slot=0, step=0, metadata=metadata)
    history = state.history_slots(frame_unit=8)
    assert [(slot.slot, slot.base) for slot in history] == [(0, 0), (1, 8)]
    state.reset()
    assert state.history_slots(frame_unit=8) == [] and state.next_chunk == 0 and state.fingerprint is None
