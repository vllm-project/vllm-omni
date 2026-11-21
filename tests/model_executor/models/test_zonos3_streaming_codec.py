# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GPU handoff: delayed RVQ coverage, absorbing flush and real SWITCH replay."""

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.zonos3.model_state import Zonos3RVQBuffer
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model]


@pytest.mark.cpu
@pytest.mark.parametrize("first", [1, 10])
@torch.inference_mode()
def test_delayed_rows_ramp_final_and_slot_reuse(first):
    state = Zonos3RVQBuffer(2, 8, torch.device("cpu"))
    state.reset(0, torch.empty(0, 10, dtype=torch.long), 8, (1, 2, 4, 8), first)
    slots = torch.tensor([0, 1])
    emitted: list[torch.Tensor] = []
    start = 0
    # Request 1 never becomes ready and must always use its own scratch slot.
    for step in range(100):
        columns = torch.arange(10)
        codes = (step - columns).clamp_min(0)[:, None].expand(10, 1).T + columns
        final = step == 99
        packed, destinations, lengths, flush, _, _ = state.prepare(
            torch.cat((codes, codes)), slots, torch.tensor([True, False]), torch.tensor([final, False]), 16
        )
        n = int(lengths[0])
        assert destinations[1] == 3 and lengths[1] == 0 and not flush[1]
        if n:
            expected = torch.arange(start, start + n)[:, None] + columns
            # The partial first packet intentionally omits not-yet-arrived books.
            if first == 1 and not emitted:
                expected = torch.where(torch.arange(start, start + n)[:, None] + columns <= step, expected, -1)
            torch.testing.assert_close(packed[:n], expected)
            emitted.append(packed[:n].clone())
            start += n
    assert start == 91 and state.flushed[0]
    before = state.consumed_frames.clone()
    _, destination, lengths, flush, _, order = state.prepare(
        torch.zeros(2, 10, dtype=torch.long), slots, torch.ones(2, dtype=torch.bool), torch.tensor([True, False]), 16
    )
    row = int((order == 0).nonzero()[0])
    assert destination[row] >= 2 and lengths[row] == 0 and not flush[row]
    assert state.consumed_frames[0] == before[0]
    state.reset(0, torch.empty(0, 10, dtype=torch.long), 8, (1, 2), first)
    assert not state.flushed[0] and state.received[0] == state.consumed_frames[0] == 0


@pytest.mark.parametrize("batch", [pytest.param(n, marks=hardware_marks(res={"cuda": "L4"})) for n in (2, 32)])
@pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="Requires CUDA conditional graph nodes")
@torch.inference_mode()
def test_switch_graph_matches_eager_and_preserves_owned_output(batch):
    from tests.model_executor.models.test_zonos3_codec import graph_network
    from vllm_omni.model_executor.models.zonos3.cuda_graph_wrapper import ZACCodecGraphWrapper
    from vllm_omni.model_executor.models.zonos3.model_state import Zonos3AudioSnapshot, _AudioLease

    network = graph_network("cuda")
    assemblies = [Zonos3RVQBuffer(batch, 4, torch.device("cuda")) for _ in range(2)]
    eager, graph = [
        ZACCodecGraphWrapper(network, state, batch, 4, mode) for state, mode in zip(assemblies, (True, False))
    ]
    graph.capture()
    for state in assemblies:
        for slot in range(batch):
            state.reset(slot, torch.empty(0, 10, dtype=torch.long, device="cuda"), 4, (1, 2, 4), 10)
    slots = torch.arange(batch - 1, -1, -1, device="cuda")
    even = torch.arange(batch, device="cuda").remainder(2) == 0
    retained: list[torch.Tensor] = []
    pending: list[tuple[Zonos3AudioSnapshot, torch.Tensor, list[int], list[int]]] = []
    leases = [_AudioLease() for _ in range(batch)]
    copy_stream = torch.cuda.Stream()
    metadata_stream = torch.cuda.Stream()

    def materialize(item):
        snapshot, reference, boundaries, order = item
        with torch.cuda.stream(metadata_stream):
            cpu = snapshot.to_cpu(metadata_stream, lambda value: value.cpu())
        payloads = cpu.get_output()
        for row, original in enumerate(order):
            start, end = boundaries[row : row + 2]
            if end > start:
                torch.testing.assert_close(
                    payloads[original]["model_outputs"], reference[start:end], atol=2e-5, rtol=2e-4
                )
                retained.append(payloads[original]["model_outputs"])
            else:
                assert payloads[original] is None

    for step in range(30):
        scheduled = 1 if step % 3 == 0 else batch
        codes = torch.full((scheduled, 10), step, dtype=torch.long, device="cuda")
        active = ~even[:scheduled] | (step < 21)
        final = (even[:scheduled] & (step >= 20)) | (step == 29)
        consumed = assemblies[0].consumed_frames[:batch].clone()
        flushed = assemblies[0].flushed[:batch].clone()
        expected = eager(codes, slots[:scheduled], active, final)
        actual = graph(codes, slots[:scheduled], active, final)
        delta = assemblies[0].consumed_frames[:batch] - consumed
        ready = int(((delta > 0) | (assemblies[0].flushed[:batch] & ~flushed)).sum())
        tokens = int(delta.sum())
        selected = next(
            (i for i, (b, t) in enumerate(graph.capture_keys) if ready > 0 and ready <= b and tokens <= t),
            len(graph.capture_keys),
        )
        assert int(graph.case_index) == selected
        expected_boundaries, boundaries = expected[1].tolist(), actual[1].tolist()
        assert expected_boundaries == boundaries
        torch.testing.assert_close(expected[2], actual[2])
        for i in range(scheduled):
            start, end = boundaries[i : i + 2]
            torch.testing.assert_close(actual[0][start:end], expected[0][start:end], atol=2e-5, rtol=2e-4)
        event = torch.cuda.Event()
        snapshot = Zonos3AudioSnapshot(
            actual[0].clone(),
            actual[1].clone(),
            actual[2].clone(),
            actual[3].clone(),
            leases[:scheduled],
            actual[4].clone(),
            event,
            copy_stream,
        )
        event.record()
        pending.append((snapshot, expected[0][: boundaries[-1]].cpu(), boundaries, actual[4].tolist()))
        # Keep GPU snapshots alive across subsequent replays before D2H. A
        # borrowed graph buffer or metadata view would be overwritten here.
        if len(pending) >= 3:
            materialize(pending.pop(0))
    for item in pending:
        materialize(item)
    saved = [audio.clone() for audio in retained]
    # Finalized real slots must never re-enter synthesis, even with final=True.
    before = graph.codec.synthesis.emitted.clone()
    result = graph(codes, slots, torch.ones_like(active), torch.ones_like(final))
    assert result[2].eq(0).all()
    assert result[1].eq(0).all() and int(graph.case_index) == len(graph.capture_keys)
    torch.testing.assert_close(graph.codec.synthesis.emitted[:batch], before[:batch], atol=0, rtol=0)
    for audio, snapshot in zip(retained, saved):
        torch.testing.assert_close(audio, snapshot, atol=0, rtol=0)

    # Reuse a real slot after cancellation, with old output still in flight.
    # A one-frame partial packet followed by inactive final needs a zero-frame
    # flush: ready_count must include flush, not just packed token coverage.
    slot = torch.zeros(1, dtype=torch.long, device="cuda")
    one = torch.ones(1, dtype=torch.bool, device="cuda")
    code = torch.full((1, 10), 7, dtype=torch.long, device="cuda")

    def reset_slot():
        for wrapper, state in zip((eager, graph), assemblies):
            wrapper.codec.reset(slot)
            state.reset(0, torch.empty(0, 10, dtype=torch.long, device="cuda"), 4, (1, 2, 4), 1)

    reset_slot()
    expected = eager(code, slot, one, ~one)
    actual = graph(code, slot, one, ~one)
    old_lease = _AudioLease()
    cancelled = Zonos3AudioSnapshot(
        actual[0].clone(),
        actual[1].clone(),
        actual[2].clone(),
        actual[3].clone(),
        [old_lease],
        actual[4].clone(),
        pcm_copy_stream=copy_stream,
    )
    expected = eager(code, slot, ~one, one)
    actual = graph(code, slot, ~one, one)
    assert int(graph.case_index) == 0 and actual[2].sum() > 0
    assert assemblies[1].consumed_frames[0] == 1
    torch.testing.assert_close(actual[1], expected[1])
    torch.testing.assert_close(actual[0][: actual[1][-1]], expected[0][: expected[1][-1]], atol=2e-5, rtol=2e-4)
    with old_lease.lock:
        old_lease.active = False
    reset_slot()
    expected = eager(code, slot, one, ~one)
    actual = graph(code, slot, one, ~one)
    torch.testing.assert_close(actual[1], expected[1])
    torch.testing.assert_close(actual[0][: actual[1][-1]], expected[0][: expected[1][-1]], atol=2e-5, rtol=2e-4)
    # Materialization after both final and reuse cannot resurrect the old lease.
    cpu = cancelled.to_cpu(metadata_stream, lambda value: value.cpu())
    assert cpu.get_output() == [None]
