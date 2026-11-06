# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""World retrieval ranking, attention, byte bounds, and transactional CPU tests."""

import math

import pytest
import torch
import torch.nn.functional as F

from tests.diffusion.ar_diffusion.test_worldkv import absolute_keys, raw, worldkv

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def pose(x=0.0, yaw=0.0):
    result = torch.eye(4, dtype=torch.float64)
    c, s = math.cos(yaw), math.sin(yaw)
    result[:3, :3] = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float64)
    result[0, 3] = x
    return result


def state(**kwargs):
    return worldkv.WorldKV(
        worldkv.WorldKVConfig(
            window_frames=8,
            sink_frames=2,
            retrieval_frames=2,
            bank_bytes=kwargs.pop("bank_bytes", 1 << 20),
            **kwargs,
        ),
        num_layers=2,
        tokens_per_frame=2,
    )


def append(memory, start, *, camera=None):
    memory.set_pose(pose(start) if camera is None else camera)
    with memory.forward(start, 2, commit=True):
        for layer in range(2):
            memory.assemble(
                layer,
                raw(start),
                absolute_keys(raw(start) + layer, torch.arange(start, start + 2).repeat_interleave(2)),
                raw(start) * 2,
            )


def test_pose_scores_against_analytic_translation_and_rotation():
    candidates = torch.stack((pose(0, 0), pose(1, math.pi / 2), pose(2, math.pi)))
    torch.testing.assert_close(
        worldkv.camera_distances(pose(), candidates), torch.tensor([0, 0.75, 2.0], dtype=torch.float64)
    )
    assert torch.equal(worldkv.camera_distances(pose(), torch.stack([pose()] * 3)), torch.zeros(3, dtype=torch.float64))


def test_revisit_attention_matches_manual_chunks_without_duplicates():
    memory = state()
    for start in range(0, 8, 2):
        append(memory, start)
    original = memory.bank[2].layers[0].key.clone()
    for _ in range(3):
        memory.set_pose(pose(2))
        with memory.forward(8, 2, commit=False):
            assert [chunk.start for chunk in memory._retrieved] == [2]
            for layer in range(2):
                positions = torch.arange(8, 10).repeat_interleave(2)
                current = absolute_keys(raw(8) + layer, positions)
                k, v = memory.assemble(layer, raw(8), current, raw(8) * 2)
                key_parts, value_parts = [], []
                for chunk in (0, 2, 6, 8):
                    target = chunk
                    key_parts.append(
                        absolute_keys(raw(chunk) + layer, torch.arange(target, target + 2).repeat_interleave(2))
                    )
                    value_parts.append(raw(chunk) * 2)
                expected_k, expected_v = torch.cat(key_parts, 1), torch.cat(value_parts, 1)
                torch.testing.assert_close(k, expected_k)
                torch.testing.assert_close(v, expected_v)
                expected = F.scaled_dot_product_attention(
                    raw(8).transpose(1, 2), expected_k.transpose(1, 2), expected_v.transpose(1, 2)
                )
                actual = F.scaled_dot_product_attention(raw(8).transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
                torch.testing.assert_close(actual, expected)
                assert k.shape[1] == memory.config.window_frames * 2
    assert torch.equal(memory.bank[2].layers[0].key, original)


def test_bank_byte_budget_eviction_and_noisy_read_only():
    sample = state()
    append(sample, 0)
    append(sample, 2)
    entry_bytes = sample.bank_nbytes
    memory = state(bank_bytes=entry_bytes * 2)
    for start in range(0, 30, 2):
        append(memory, start)
        assert memory.bank_nbytes <= entry_bytes * 2
        assert len(memory.bank) <= 2
        for entry in memory.bank.values():
            assert all(rows.key.device.type == "cpu" for rows in entry.layers)
    assert list(memory.bank) == [26, 28]
    before = memory.bank
    memory.set_pose(pose(26))
    with memory.forward(30, 2, commit=False):
        for layer in range(2):
            memory.assemble(layer, raw(30), raw(30), raw(30))
    assert memory.bank is before and memory.next_frame == 30
    memory.close()
    assert memory.bank_nbytes == 0


def test_first_eviction_warns_once_and_counts_evicted_chunks(monkeypatch):
    sample = state()
    append(sample, 0)
    append(sample, 2)
    memory = state(bank_bytes=sample.bank_nbytes * 2)
    warnings = []
    monkeypatch.setattr(worldkv.logger, "warning", lambda *args: warnings.append(args))
    for start in range(0, 12, 2):
        append(memory, start)
    # Chunks 2..10 were banked and only the newest two remain.
    assert list(memory.bank) == [8, 10]
    assert memory.evicted_chunks == 3
    assert len(warnings) == 1


def test_host_bank_reservations_admit_and_release_memories(monkeypatch):
    ledger = worldkv._HostBankLedger()
    monkeypatch.setattr(worldkv, "_HOST_BANKS", ledger)
    config = worldkv.WorldKVConfig(
        window_frames=8, sink_frames=2, retrieval_frames=2, bank_bytes=1 << 20, host_bank_limit_bytes=2 << 20
    )

    def create():
        return worldkv.WorldKV(config, num_layers=2, tokens_per_frame=2)

    first, second = create(), create()
    with pytest.raises(RuntimeError, match="host limit"):
        create()
    assert ledger.reserved == 2 << 20
    first.close()
    first.close()
    assert ledger.reserved == 1 << 20
    third = create()
    # Request-local caches are dropped without close().
    del second, third
    assert ledger.reserved == 0
    # Memories without retrieval reserve nothing.
    worldkv.WorldKV(worldkv.WorldKVConfig(window_frames=8, sink_frames=2), num_layers=2, tokens_per_frame=2)
    assert ledger.reserved == 0


def test_oversized_entry_fails_before_allocating_or_publishing(monkeypatch):
    memory = state(bank_bytes=1)

    def unexpected(*args, **kwargs):
        pytest.fail("compression must not run for an undersized bank")

    monkeypatch.setattr(worldkv, "world_compress", unexpected)
    with pytest.raises(ValueError, match="cannot fit one compressed chunk"):
        append(memory, 0)
    assert memory.bank_nbytes == 0 and memory.next_frame == 0
    assert memory.history == [None, None] and not memory._pending


def test_bank_is_transactional_and_copies_camera(monkeypatch):
    memory = state()
    camera = pose(2)
    append(memory, 0)
    append(memory, 2, camera=camera)
    camera[0, 3] = 99
    assert memory.bank[2].pose[0, 3] == 2
    old_bank, old_history = memory.bank, memory.history

    def fail(replacement):
        raise RuntimeError("CPU bank copy failed")

    monkeypatch.setattr(memory, "_publish_bank", fail)
    with pytest.raises(RuntimeError, match="copy failed"):
        append(memory, 4)
    assert memory.bank is old_bank and memory.history is old_history and memory.next_frame == 4


@pytest.mark.parametrize("bad", [torch.eye(3), torch.full((4, 4), float("nan")), torch.ones(4, 4), -torch.eye(4)])
def test_invalid_camera(bad):
    with pytest.raises(ValueError):
        state().set_pose(bad)


def test_missing_pose_rejected_and_pose_required_again_after_commit():
    memory = state()
    with pytest.raises(ValueError, match="absolute camera pose"):
        with memory.forward(0, 2, commit=False):
            pass
    append(memory, 0)
    with pytest.raises(ValueError, match="absolute camera pose"):
        with memory.forward(2, 2, commit=False):
            pass


def test_ties_are_deterministic_and_sink_recent_never_retrieved():
    memory = state()
    for start in range(0, 12, 2):
        append(memory, start, camera=pose())
    memory.set_pose(pose())
    assert [chunk.start for chunk in memory._retrieve(2)] == [2]
    assert 0 not in memory.bank


@pytest.mark.parametrize(
    "config",
    [
        {"retrieval_frames": -1},
        {"retrieval_frames": 3},
        {"sink_frames": 9, "retrieval_frames": 9, "bank_bytes": 100},
        {"bank_bytes": True},
        {"host_bank_limit_bytes": 1 << 20},
        {"retrieval_frames": 2, "bank_bytes": 10, "host_bank_limit_bytes": 5},
    ],
)
def test_invalid_retrieval_config(config):
    with pytest.raises(ValueError):
        worldkv.WorldKVConfig(**config)


def test_worldkv_small_fifo_bank_retrieves_over_long_rollout():
    config = worldkv.WorldKVConfig(
        window_frames=18,
        sink_frames=3,
        retrieval_frames=3,
        bank_bytes=4 * (2 * 24 * (2 * 8 * 4 + 16) + 128),
        world_keep_ratio=0.25,
    )
    memory = worldkv.WorldKV(config, num_layers=2, tokens_per_frame=16)
    generator = torch.Generator().manual_seed(7)
    hits = 0
    for start in range(0, 120, 3):
        data = [torch.randn(1, 48, 1, 8, generator=generator) for _ in range(2)]
        memory.set_pose(pose())
        for commit in (False, True):
            with memory.forward(start, 3, commit=commit):
                hits += len(memory._retrieved) if not commit else 0
                for layer, x in enumerate(data):
                    memory.assemble(layer, x, x, x)
                    previous = memory.history[layer]
                    if previous is not None:
                        tail_start = max(3, start + 3 - 18 + 3 + 3)
                        visible_ids = previous.token_ids[(previous.positions < 3) | (previous.positions >= tail_start)]
                        for chunk in memory._retrieved:
                            assert not torch.isin(chunk.layers[layer].token_ids, visible_ids).any()
        assert len(memory.bank) <= 4
    assert hits > 0


@pytest.mark.parametrize("ratio", [0.0, 0.25, 1.0])
def test_bank_capacity_exact_boundary_includes_ids_and_pose(ratio):
    sample = state(world_keep_ratio=ratio)
    append(sample, 0)
    append(sample, 2)
    exact = sample.bank_nbytes
    config = state(world_keep_ratio=ratio, bank_bytes=exact)
    config.config.validate_bank_capacity(2, 2, 2, 8, 8, 2)
    append(config, 0)
    append(config, 2)
    assert config.bank_nbytes == exact
    too_small = state(world_keep_ratio=ratio, bank_bytes=exact - 1)
    with pytest.raises(ValueError, match="cannot fit"):
        too_small.config.reservation_bytes(2, 2, 2, 8, 8, chunk_frames=2)


def test_real_geometry_rejects_one_gib_bank_and_reserves_layer_local_scratch():
    config = worldkv.WorldKVConfig(
        window_frames=18, sink_frames=3, retrieval_frames=3, bank_bytes=1 << 30, world_keep_ratio=0.25
    )
    with pytest.raises(ValueError, match="cannot fit"):
        config.reservation_bytes(40, 1560, 40, 128, 2, chunk_frames=3)
    policy = worldkv.WorldKVConfig(window_frames=18, sink_frames=3)
    reserve = policy.reservation_bytes(40, 1560, 40, 128, 2, chunk_frames=3)
    old = 4 * 40 * 18 * 1560 * (2 * 40 * 128 * 2 + 8)
    assert reserve < old
    # Publication retains old and replacement canonical windows.
    assert reserve >= 40 * 1560 * 36 * (2 * 40 * 128 * 2)


def test_retrieved_rows_transfer_once_per_chunk_and_clear_on_commit(monkeypatch):
    memory = state()
    for start in range(0, 8, 2):
        append(memory, start)
    transfers = []
    original = worldkv.MemoryRows.to_kv_device

    def record(rows, device):
        transfers.append(rows.token_ids.tolist())
        return original(rows, device)

    monkeypatch.setattr(worldkv.MemoryRows, "to_kv_device", record)
    outputs = []
    for commit in (False, False, False, True):
        memory.set_pose(pose(2))
        with memory.forward(8, 2, commit=commit):
            for layer in range(2):
                k, v = memory.assemble(layer, raw(8), raw(8), raw(8))
                outputs.append((k.clone(), v.clone()))
    assert len(transfers) == 2  # one chunk, two layers, four forwards
    for i in range(2, len(outputs)):
        torch.testing.assert_close(outputs[i], outputs[i % 2])
    assert not memory._device_retrieved and not memory._retrieved
    assert memory._retrieval_frames == 0


@pytest.mark.parametrize("end", ["failure", "pose", "close"])
def test_retrieved_device_cache_invalidated_on_lifecycle_changes(end):
    memory = state()
    for start in range(0, 8, 2):
        append(memory, start)
    memory.set_pose(pose(2))
    with memory.forward(8, 2, commit=False):
        for layer in range(2):
            memory.assemble(layer, raw(8), raw(8), raw(8))
    assert memory._device_retrieved and memory._retrieved
    if end == "failure":
        with pytest.raises(RuntimeError, match="injected"):
            with memory.forward(8, 2, commit=True):
                memory.assemble(0, raw(8), raw(8), raw(8))
                raise RuntimeError("injected")
        assert memory.next_frame == 8
    elif end == "pose":
        memory.set_pose(pose(4))
    else:
        memory.close()
    assert not memory._device_retrieved and not memory._retrieved
    assert memory._retrieval_frames == 0


def test_clean_publication_releases_device_views_and_consumes_pending_transactionally(monkeypatch):
    memory = state()
    for start in range(0, 8, 2):
        append(memory, start)
    old_history, old_bank = memory.history, memory.bank
    publish, retain = memory._publish_bank, memory._retained
    calls = []

    def checked_publish(replacement):
        assert not memory._device_retrieved and not memory._pending
        assert memory.history is old_history and memory.bank is old_bank
        calls.append("bank")
        return publish(replacement)

    def checked_retain(layer, current):
        assert "bank" not in calls
        assert layer not in memory._pending
        assert len(memory._pending) == 1 - layer
        assert memory.history is old_history and memory.bank is old_bank
        calls.append(layer)
        if layer == 1:
            raise RuntimeError("late retention failed")
        return retain(layer, current)

    monkeypatch.setattr(memory, "_publish_bank", checked_publish)
    monkeypatch.setattr(memory, "_retained", checked_retain)
    with pytest.raises(RuntimeError, match="late retention"):
        append(memory, 8)
    assert memory.history is old_history and memory.bank is old_bank and memory.next_frame == 8
    assert not memory._pending and not memory._device_retrieved
    assert calls == [0, 1]
    monkeypatch.setattr(memory, "_retained", retain)
    append(memory, 8)
    assert memory.next_frame == 10 and calls[-1] == "bank"


def test_recent_retrieval_reuses_history_without_host_roundtrip(monkeypatch):
    memory = state()
    for start in range(0, 8, 2):
        append(memory, start)
    memory.set_pose(pose(4))

    def forbidden(*args, **kwargs):
        pytest.fail("recent retrieved KV is already resident in history")

    monkeypatch.setattr(worldkv.MemoryRows, "to_kv_device", forbidden)
    with memory.forward(8, 2, commit=False):
        assert [chunk.start for chunk in memory._retrieved] == [4]
        for layer in range(2):
            retrieved = memory._retrieved_rows(layer, torch.device("cpu"))[0]
            expected = memory.bank[4].layers[layer]
            for name in ("key", "value", "positions", "token_ids"):
                actual = getattr(retrieved, name)
                assert torch.equal(actual, getattr(expected, name))
                assert (
                    actual.untyped_storage().data_ptr()
                    == getattr(memory.history[layer], name).untyped_storage().data_ptr()
                )
            memory.assemble(layer, raw(8), raw(8), raw(8))


def test_host_reader_waits_for_publication_before_access(monkeypatch):
    from dataclasses import replace
    from unittest.mock import Mock

    memory = state()
    for start in range(0, 8, 2):
        append(memory, start)
    completed = []
    ready = Mock(spec=torch.Event)
    ready.synchronize.side_effect = lambda: completed.append(True)
    memory.bank[2] = replace(memory.bank[2], ready=ready)
    original = worldkv.MemoryRows.to_kv_device

    def checked_copy(rows, device):
        assert completed, "host rows cannot be read while D2H is still pending"
        return original(rows, device)

    monkeypatch.setattr(worldkv.MemoryRows, "to_kv_device", checked_copy)
    memory.set_pose(pose(2))
    with memory.forward(8, 2, commit=False):
        for layer in range(2):
            memory.assemble(layer, raw(8), raw(8), raw(8))


@pytest.mark.parametrize("ratio", [0.0, 0.5, 1.0])
def test_resident_shortcut_preserves_long_rollout_and_variable_chunk_sizes(monkeypatch, ratio):
    fast, reference = state(world_keep_ratio=ratio), state(world_keep_ratio=ratio)
    monkeypatch.setattr(reference, "_resident_rows", lambda *args: None)
    start = 0
    for frames in [2, 1, 2, 1] * 5:
        for memory in (fast, reference):
            memory.set_pose(pose(start - 3 if start % 2 else 2))
        for commit in (False, True):
            outputs = []
            for memory in (fast, reference):
                with memory.forward(start, frames, commit=commit):
                    for layer in range(2):
                        x = raw(start, frames=frames) + layer
                        outputs.append(memory.assemble(layer, x, x, x * 2))
            for lhs, rhs in zip(outputs[:2], outputs[2:]):
                assert all(torch.equal(a, b) for a, b in zip(lhs, rhs))
        start += frames


@pytest.mark.parametrize("ratio", [0.0, 0.5, 1.0])
def test_reservation_accounts_for_compressed_async_sources_only(ratio):
    config = worldkv.WorldKVConfig(
        window_frames=18, sink_frames=9, retrieval_frames=3, bank_bytes=48 << 30, world_keep_ratio=ratio
    )
    layers, tokens, heads, dim, itemsize, chunk = 40, 960, 40, 128, 2, 3
    reserve = config.reservation_bytes(layers, tokens, heads, dim, itemsize, chunk_frames=chunk)
    row = 2 * heads * dim * itemsize + 16
    scratch = 4 * config.window_frames * tokens * (2 * heads * dim * max(4, itemsize) + 16)
    extra = 0 if ratio == 1 else tokens + math.ceil((chunk - 1) * tokens * ratio)
    assert reserve - scratch == layers * (2 * (config.window_frames - config.retrieval_frames) * tokens + extra) * row


@pytest.mark.parametrize(
    "sink_frames,retrieval_frames,ratio",
    [(0, 0, 1.0), (1, 0, 1.0), (3, 0, 1.0)] + [(sink, 3, ratio) for sink in (0, 6) for ratio in (0.0, 0.5, 1.0)],
)
def test_assembly_matches_position_mask_across_sink_and_chunk_boundaries(sink_frames, retrieval_frames, ratio):
    memory = worldkv.WorldKV(
        worldkv.WorldKVConfig(
            window_frames=12,
            sink_frames=sink_frames,
            retrieval_frames=retrieval_frames,
            bank_bytes=1 << 20 if retrieval_frames else 0,
            world_keep_ratio=ratio,
        ),
        num_layers=2,
        tokens_per_frame=2,
    )
    start = 0
    for frames in [1, 3, 2, 1] * 5:
        memory.set_pose(pose(start // 3))
        for commit in (False, True):
            with memory.forward(start, frames, commit=commit):
                for layer in range(2):
                    current = raw(start, frames=frames) + layer
                    previous = memory.history[layer]
                    expected_k, expected_v = current, current * 2
                    if previous is not None:
                        recent_start = max(sink_frames, start + frames - 12 + sink_frames + retrieval_frames)
                        mask = (previous.positions < sink_frames) | (previous.positions >= recent_start)
                        visible = previous.select(torch.where(mask)[0])
                        sink_count = int((visible.positions < sink_frames).sum())
                        retrieved = [chunk.layers[layer] for chunk in memory._retrieved]
                        expected_k = torch.cat(
                            [
                                visible.key[:, :sink_count],
                                *(rows.key for rows in retrieved),
                                visible.key[:, sink_count:],
                                current,
                            ],
                            1,
                        )
                        expected_v = torch.cat(
                            [
                                visible.value[:, :sink_count],
                                *(rows.value for rows in retrieved),
                                visible.value[:, sink_count:],
                                current * 2,
                            ],
                            1,
                        )
                    actual_k, actual_v = memory.assemble(layer, current, current, current * 2)
                    assert torch.equal(actual_k, expected_k)
                    assert torch.equal(actual_v, expected_v)
        start += frames
    memory.close()


@pytest.mark.parametrize("ratio", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("mixed_dtype", [False, True])
def test_bank_storage_is_packed_without_aliasing_across_layers_or_chunks(ratio, mixed_dtype):
    memory = state(world_keep_ratio=ratio)
    expected = {}
    for start in range(0, 12, 2):
        memory.set_pose(pose(start))
        with memory.forward(start, 2, commit=True):
            layers = []
            for layer in range(2):
                dtype = torch.float32 if mixed_dtype and layer else torch.float64
                # Different geometry must not imply equal layer sizes.
                key = raw(start, dtype=dtype)[:, :, : layer + 1]
                value = key * 2
                positions = torch.arange(start, start + 2).repeat_interleave(2)
                rows = worldkv.MemoryRows(key, value, positions, torch.arange(start * 2, (start + 2) * 2))
                layers.append(worldkv.world_compress(rows, 2, ratio, "pairwise_mean"))
                memory.assemble(layer, key, key, value)
            expected[start] = layers
        live_storage = set()
        for chunk in memory.bank.values():
            by_dtype: dict[torch.dtype, list[torch.Tensor]] = {}
            for actual, reference in zip(chunk.layers, expected[chunk.start]):
                for name in ("key", "value", "positions", "token_ids"):
                    assert torch.equal(getattr(actual, name), getattr(reference, name))
                for tensor in (actual.key, actual.value):
                    by_dtype.setdefault(tensor.dtype, []).append(tensor)
            for tensors in by_dtype.values():
                storage = tensors[0].untyped_storage()
                assert storage.data_ptr() not in live_storage
                live_storage.add(storage.data_ptr())
                offset = 0
                for tensor in tensors:
                    assert tensor.untyped_storage().data_ptr() == storage.data_ptr()
                    assert tensor.storage_offset() == offset
                    assert tensor.is_contiguous()
                    offset += tensor.numel()
                assert storage.nbytes() == offset * tensors[0].element_size()
    memory.close()


@pytest.mark.parametrize("fail_allocation", [1, 2])
def test_packed_bank_allocation_failure_keeps_published_state(monkeypatch, fail_allocation):
    memory = state()

    def commit(start):
        memory.set_pose(pose(start))
        with memory.forward(start, 2, commit=True):
            for layer, dtype in enumerate((torch.float32, torch.float64)):
                key = raw(start, dtype=dtype)
                memory.assemble(layer, key, key, key * 2)

    commit(0)
    commit(2)
    history, bank = memory.history, memory.bank
    saved = [(rows.key.clone(), rows.value.clone()) for chunk in bank.values() for rows in chunk.layers]
    empty = torch.empty
    allocations = 0

    def fail(*args, **kwargs):
        nonlocal allocations
        if "pin_memory" in kwargs:
            allocations += 1
            if allocations == fail_allocation:
                raise RuntimeError("bank allocation failed")
        return empty(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", fail)
        with pytest.raises(RuntimeError, match="bank allocation failed"):
            commit(4)
    assert memory.history is history and memory.bank is bank
    assert memory.next_frame == 4 and not memory._active and memory._publication is None
    for rows, (key, value) in zip((rows for chunk in bank.values() for rows in chunk.layers), saved):
        assert torch.equal(rows.key, key) and torch.equal(rows.value, value)
    commit(4)
    memory.close()
