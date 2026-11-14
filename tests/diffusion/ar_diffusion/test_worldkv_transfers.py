# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA bank transfer ordering, retention rollback and CPU numerical oracle."""

from dataclasses import replace

import pytest
import torch

from tests.diffusion.ar_diffusion.test_worldkv import raw, worldkv
from tests.diffusion.ar_diffusion.test_worldkv_retrieval import pose
from tests.helpers.mark import hardware_marks

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    *hardware_marks(res={"cuda": ["H100", "B200"]}, num_cards=1),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for pinned async transfers"),
]


@pytest.mark.parametrize("layerwise", [False, True])
@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("reuse,prefetch", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("ratio", [0.0, 0.5, 1.0])
def test_async_bank_matches_cpu_through_retrieval_eviction_and_rollback(
    monkeypatch, ratio, reuse, prefetch, background, layerwise
):
    config = worldkv.WorldKVConfig(
        window_frames=8, sink_frames=2, retrieval_frames=2, bank_bytes=9000, world_keep_ratio=ratio
    )
    cpu = worldkv.WorldKV(config, num_layers=2, tokens_per_frame=2)
    gpu = worldkv.WorldKV(
        replace(
            config,
            reuse_attention_kv=reuse,
            prefetch_kv=prefetch,
            prefetch_bank_allocation=background,
            layerwise_bank_copy=layerwise,
        ),
        num_layers=2,
        tokens_per_frame=2,
    )

    def run(memory, start, commit, device, step=0):
        outputs = []
        # Alternate nearest-history hits and older host-bank hits.
        memory.set_pose(pose(start - 4 if start % 4 else 2))
        with memory.forward(start, 2, commit=commit):
            for layer in range(2):
                x = (raw(start) + layer + step / 8).to(device)
                outputs.append(memory.assemble(layer, x, x, x * 2))
        return outputs

    try:
        # Exercise producer/consumer ordering on a non-default compute stream.
        with torch.Stream(device="cuda"):
            for start in range(0, 24, 2):
                for step, commit in enumerate((False, False, False, True)):
                    expected = run(cpu, start, commit, "cpu", step)
                    actual = run(gpu, start, commit, "cuda", step)
                    for lhs, rhs in zip(actual, expected):
                        for tensor, reference in zip(lhs, rhs):
                            assert torch.equal(tensor.cpu(), reference)
                assert list(cpu.bank) == list(gpu.bank)
                assert gpu.bank_nbytes <= config.bank_bytes
            assert gpu.evicted_chunks > 0
            for chunk in gpu.bank.values():
                assert chunk.ready is not None
                chunk.ready.synchronize()
                backing = chunk.layers[0].key.untyped_storage().data_ptr()
                for actual, expected in zip(chunk.layers, cpu.bank[chunk.start].layers):
                    assert actual.key.untyped_storage().data_ptr() == backing
                    assert actual.value.untyped_storage().data_ptr() == backing
                    assert actual.key.is_pinned() and actual.value.is_pinned()
                    for name in ("key", "value", "positions", "token_ids"):
                        assert torch.equal(getattr(actual, name), getattr(expected, name))

            old_bank, old_history = gpu.bank, gpu.history
            old_keys = {start: chunk.layers[0].key.clone() for start, chunk in old_bank.items()}
            retain = gpu._retained

            def fail(layer, current):
                if layer == 1:
                    raise RuntimeError("late retention failure")
                return retain(layer, current)

            monkeypatch.setattr(gpu, "_retained", fail)
            with pytest.raises(RuntimeError, match="late retention failure"):
                run(gpu, 24, True, "cuda")
            assert gpu.bank is old_bank and gpu.history is old_history
            assert gpu.next_frame == 24 and gpu._publication is None
            for start, chunk in gpu.bank.items():
                assert torch.equal(chunk.layers[0].key, old_keys[start])
            monkeypatch.setattr(gpu, "_retained", retain)
            run(gpu, 24, True, "cuda")

            old_bank, old_history = gpu.bank, gpu.history
            nbytes = worldkv.BankChunk.nbytes.fget

            def fail_after_copy(chunk):
                if chunk.start == 26:
                    raise RuntimeError("publication failed after copy enqueue")
                return nbytes(chunk)

            with monkeypatch.context() as patch:
                patch.setattr(worldkv.BankChunk, "nbytes", property(fail_after_copy))
                with pytest.raises(RuntimeError, match="after copy enqueue"):
                    run(gpu, 26, True, "cuda")
            assert gpu.bank is old_bank and gpu.history is old_history
            assert gpu.next_frame == 26 and gpu._publication is None
            gpu.close()
            assert gpu._publication is None and not gpu.bank
    finally:
        gpu.close()
        cpu.close()


def test_retrieval_moves_only_kv_and_keeps_metadata_on_cpu():
    key = raw(2).pin_memory()
    positions = torch.arange(2, 4).repeat_interleave(2)
    token_ids = torch.arange(4, 8)
    source = worldkv.MemoryRows(key, key * 2, positions, token_ids)
    actual = source.to_kv_device(torch.device("cuda"))
    assert actual.key.is_cuda and actual.value.is_cuda
    assert actual.positions is positions and actual.token_ids is token_ids
    assert torch.equal(actual.key.cpu(), source.key)
    assert torch.equal(actual.value.cpu(), source.value)


def test_uncompressed_publication_does_not_read_back_gpu_metadata(monkeypatch):
    memory = worldkv.WorldKV(
        worldkv.WorldKVConfig(window_frames=8, sink_frames=2, retrieval_frames=2, bank_bytes=9000),
        num_layers=2,
        tokens_per_frame=2,
    )
    original = torch.Tensor.to

    def checked_to(tensor, *args, **kwargs):
        if tensor.is_cuda and tensor.dtype == torch.int64:
            raise AssertionError("publication must not read back GPU metadata")
        return original(tensor, *args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "to", checked_to)
            for start in (0, 2):
                memory.set_pose(pose(start))
                with memory.forward(start, 2, commit=True):
                    for layer in range(2):
                        key = (raw(start) + layer).to("cuda")
                        memory.assemble(layer, key, key, key * 2)
        rows = memory.bank[2].layers
        assert rows[0].positions is rows[1].positions
        assert rows[0].token_ids is rows[1].token_ids
        assert rows[0].positions.device.type == "cpu"
        assert torch.equal(rows[0].positions, torch.tensor([2, 2, 3, 3]))
        assert torch.equal(rows[0].token_ids, torch.arange(4, 8))
    finally:
        memory.close()


def test_prefetch_failure_preserves_history_and_allows_retry(monkeypatch):
    config = worldkv.WorldKVConfig(window_frames=8, sink_frames=2, retrieval_frames=2, bank_bytes=9000)
    cpu = worldkv.WorldKV(config, num_layers=2, tokens_per_frame=2)
    gpu = worldkv.WorldKV(replace(config, prefetch_kv=True, reuse_attention_kv=True), num_layers=2, tokens_per_frame=2)

    def run(memory, start, commit, device):
        memory.set_pose(pose(2))
        outputs = []
        with memory.forward(start, 2, commit=commit):
            for layer in range(2):
                key = (raw(start) + layer).to(device)
                outputs.append(memory.assemble(layer, key, key, key * 2))
        return outputs

    original = worldkv.MemoryRows.to_kv_device
    calls = 0

    def fail_second(rows, device):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("prefetch interrupted after first layer")
        return original(rows, device)

    try:
        with torch.Stream(device="cuda"):
            for start in range(0, 8, 2):
                run(cpu, start, True, "cpu")
                run(gpu, start, True, "cuda")
            history, bank = gpu.history, gpu.bank
            with monkeypatch.context() as patch:
                patch.setattr(worldkv.MemoryRows, "to_kv_device", fail_second)
                with pytest.raises(RuntimeError, match="prefetch interrupted"):
                    run(gpu, 8, False, "cuda")
            assert calls == 2
            assert gpu.history is history and gpu.bank is bank and gpu.next_frame == 8
            assert not gpu._active and not gpu._device_retrieved and not gpu._prefetch_ready and not gpu._assembly
            actual, expected = run(gpu, 8, False, "cuda"), run(cpu, 8, False, "cpu")
            for lhs, rhs in zip(actual, expected):
                for tensor, reference in zip(lhs, rhs):
                    assert torch.equal(tensor.cpu(), reference)
    finally:
        gpu.close()
        cpu.close()


def test_bank_allocation_overlaps_denoise_and_is_consumed_at_commit(monkeypatch):
    import threading

    started, release = threading.Event(), threading.Event()
    allocate = worldkv._allocate_pinned_bank

    def delayed(sizes, device):
        started.set()
        assert release.wait(10), "allocation worker was not released"
        return allocate(sizes, device)

    monkeypatch.setattr(worldkv, "_allocate_pinned_bank", delayed)
    memory = worldkv.WorldKV(
        worldkv.WorldKVConfig(
            window_frames=8, sink_frames=2, retrieval_frames=2, bank_bytes=9000, prefetch_bank_allocation=True
        ),
        num_layers=2,
        tokens_per_frame=2,
    )

    def run(start, commit):
        memory.set_pose(pose(start))
        with memory.forward(start, 2, commit=commit):
            for layer in range(2):
                key = (raw(start) + layer).to("cuda")
                memory.assemble(layer, key, key, key * 2)

    try:
        run(0, True)
        run(2, False)
        assert started.wait(5)
        allocation = memory._bank_allocation
        assert allocation is not None and not allocation.done()
        assert not memory.bank  # Denoising does not publish the prefetched storage.
        release.set()
        run(2, True)
        assert memory._bank_allocation is None
        assert memory.bank[2].layers[0].key.is_pinned()
        memory.bank[2].ready.synchronize()
        assert torch.equal(memory.bank[2].layers[0].key, raw(2))
    finally:
        release.set()
        memory.close()
    assert memory._bank_allocator is None


def test_background_allocation_failure_rolls_back_and_retry_succeeds(monkeypatch):
    memory = worldkv.WorldKV(
        worldkv.WorldKVConfig(
            window_frames=8, sink_frames=2, retrieval_frames=2, bank_bytes=9000, prefetch_bank_allocation=True
        ),
        num_layers=2,
        tokens_per_frame=2,
    )

    def run(start):
        memory.set_pose(pose(start))
        with memory.forward(start, 2, commit=True):
            for layer in range(2):
                key = (raw(start) + layer).to("cuda")
                memory.assemble(layer, key, key, key * 2)

    def fail(*args):
        raise RuntimeError("background allocation failed")

    try:
        run(0)
        history, bank = memory.history, memory.bank
        with monkeypatch.context() as patch:
            patch.setattr(worldkv, "_allocate_pinned_bank", fail)
            with pytest.raises(RuntimeError, match="background allocation failed"):
                run(2)
        assert memory.history is history and memory.bank is bank
        assert memory.next_frame == 2 and memory._bank_allocation is None
        run(2)
        assert memory.next_frame == 4
    finally:
        memory.close()


def test_layerwise_copy_starts_before_last_layer_and_rolls_back():
    memory = worldkv.WorldKV(
        worldkv.WorldKVConfig(
            window_frames=8,
            sink_frames=2,
            retrieval_frames=2,
            bank_bytes=9000,
            prefetch_bank_allocation=True,
            layerwise_bank_copy=True,
        ),
        num_layers=2,
        tokens_per_frame=2,
    )

    def layer(index, start):
        key = (raw(start) + index).to("cuda")
        memory.assemble(index, key, key, key * 2)

    try:
        with torch.Stream(device="cuda"):
            memory.set_pose(pose(0))
            with memory.forward(0, 2, commit=True):
                layer(0, 0)
                layer(1, 0)
            previous = memory.history
            memory.set_pose(pose(2))
            with pytest.raises(RuntimeError, match="later layer failed"):
                with memory.forward(2, 2, commit=True):
                    layer(0, 2)
                    assert memory._publication is not None
                    memory._publication.synchronize()
                    assert torch.equal(memory._staged_bank[0].key, raw(2))
                    assert not memory.bank and memory.history is previous
                    raise RuntimeError("later layer failed")
            assert memory._staged_bank is None
            assert memory._publication is None and memory.history is previous
            assert memory.next_frame == 2 and not memory.bank
            memory.set_pose(pose(2))
            with memory.forward(2, 2, commit=True):
                layer(0, 2)
                layer(1, 2)
                assert memory._publication is not None
                assert not memory.bank
            memory.bank[2].ready.synchronize()
            for index, rows in enumerate(memory.bank[2].layers):
                assert torch.equal(rows.key, raw(2) + index)
                assert torch.equal(rows.value, (raw(2) + index) * 2)
    finally:
        memory.close()


@pytest.mark.parametrize("layerwise", [False, True])
def test_prefetched_bank_handles_short_chunks_and_changed_denoise_geometry(layerwise):
    config = worldkv.WorldKVConfig(window_frames=8, sink_frames=2, retrieval_frames=2, bank_bytes=9000)
    cpu = worldkv.WorldKV(config, num_layers=2, tokens_per_frame=2)
    gpu = worldkv.WorldKV(
        replace(config, prefetch_bank_allocation=True, layerwise_bank_copy=layerwise),
        num_layers=2,
        tokens_per_frame=2,
    )

    def run(memory, start, frames, commit, device):
        memory.set_pose(pose(2))
        outputs = []
        with memory.forward(start, frames, commit=commit):
            for layer in range(2):
                key = (raw(start)[:, : frames * 2] + layer).to(device)
                outputs.append(memory.assemble(layer, key, key, key * 2))
        return outputs

    try:
        with torch.Stream(device="cuda"):
            start = 0
            for frames in (2, 1, 1, 2, 1, 1):
                # The short commit must replace a larger allocation prefetched
                # by the first noisy forward, without publishing either early.
                for count, commit in ((2, False), (frames, False), (frames, True)):
                    expected = run(cpu, start, count, commit, "cpu")
                    actual = run(gpu, start, count, commit, "cuda")
                    for lhs, rhs in zip(actual, expected):
                        for tensor, reference in zip(lhs, rhs):
                            assert torch.equal(tensor.cpu(), reference)
                assert list(gpu.bank) == list(cpu.bank)
                if start in gpu.bank:
                    chunk = gpu.bank[start]
                    chunk.ready.synchronize()
                    assert chunk.frames == frames
                    for rows, reference in zip(chunk.layers, cpu.bank[start].layers):
                        for name in ("key", "value", "positions", "token_ids"):
                            assert torch.equal(getattr(rows, name), getattr(reference, name))
                start += frames
    finally:
        gpu.close()
        cpu.close()
