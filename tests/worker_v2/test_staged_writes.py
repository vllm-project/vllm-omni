# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""NumPy/UVA staged writes land exactly where vLLM's list-based writes do."""

import numpy as np
import pytest
import torch

from tests.helpers.mark import hardware_test

pytestmark = pytest.mark.core_model


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
@pytest.mark.parametrize("uva", [False, True])
def test_staged_writes_match_a_host_model(dtype, uva):
    from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor

    from vllm_omni.worker_v2 import staged_writes

    staged_writes.install()
    device = torch.device("cuda", 0)
    rows, cols = 8, 2048
    t = StagedWriteTensor((rows, cols), dtype=dtype, device=device, uva_instead_of_gpu=uva)
    expected = np.zeros((rows, cols), dtype=torch.empty((), dtype=dtype).numpy().dtype)
    rng = np.random.default_rng(0)
    for step in range(6):
        # Writes applied together must not overlap (the kernel applies them in parallel).
        for row in rng.permutation(rows)[:5].tolist():
            start = int(rng.integers(1, 64))
            n = int(rng.integers(1, 1500))
            values = rng.integers(0, 1000, n)
            kind = step % 4
            if kind == 0:
                src = values.tolist()
            elif kind == 1:
                src = values
            elif kind == 2:
                src = torch.from_numpy(values)
            else:
                src = (int(v) for v in values)
            t.stage_write(row, start, src)
            expected[row, start : start + n] = values
            if kind == 1:
                values[:] = -1  # the staged chunk is a private copy
        elem_row = int(rng.integers(rows))
        t.stage_write_elem(elem_row, 7 + step)
        expected[elem_row, 0] = 7 + step
        t.stage_write(0, 0, [])
        t.apply_write()
        torch.accelerator.synchronize()
        np.testing.assert_array_equal(t.gpu.cpu().numpy(), expected)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_fused_staged_writes_match_a_host_model():
    from vllm.v1.worker.gpu.buffer_utils import FusedStagedWriter, StagedWriteTensor

    from vllm_omni.worker_v2 import staged_writes

    staged_writes.install()
    device = torch.device("cuda", 0)
    tables = [StagedWriteTensor((4, 64), dtype=torch.int32, device=device) for _ in range(3)]
    ptrs = torch.tensor([t.gpu.data_ptr() for t in tables], dtype=torch.uint64, device=device)
    strides = torch.tensor([t.gpu.stride(0) for t in tables], dtype=torch.int64, device=device)
    writer = FusedStagedWriter(device, 12)
    expected = [np.zeros((4, 64), dtype=np.int32) for _ in tables]
    rng = np.random.default_rng(1)
    for _ in range(4):
        for g in (0, 2):  # group 1 has nothing staged
            for row in rng.permutation(4)[:2].tolist():
                start, n = int(rng.integers(8)), int(rng.integers(1, 40))
                values = rng.integers(0, 100, n).tolist()
                tables[g].stage_write(row, start, values)
                expected[g][row, start : start + n] = values
        writer.apply(tables, ptrs, strides)
        torch.accelerator.synchronize()
        for t, e in zip(tables, expected):
            np.testing.assert_array_equal(t.gpu.cpu().numpy(), e)
            assert not t._staged_write_indices and t._staged_len == 0


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_deferred_writes_land_together_like_immediate_writes():
    from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor

    from vllm_omni.worker_v2 import staged_writes

    staged_writes.install()
    device = torch.device("cuda", 0)
    specs = [
        ((6, 300), torch.int64, False),
        ((6, 40), torch.int32, True),
        (16, torch.float32, False),
        (16, torch.int32, True),
    ]
    tensors = [
        StagedWriteTensor(shape, dtype=dtype, device=device, uva_instead_of_gpu=uva) for shape, dtype, uva in specs
    ]
    expected = [np.zeros(shape, dtype=torch.empty((), dtype=dtype).numpy().dtype) for shape, dtype, _ in specs]
    rng = np.random.default_rng(2)

    def stage(i: int) -> None:
        t, e = tensors[i], expected[i]
        if e.ndim == 2:
            for row in rng.permutation(e.shape[0])[:3].tolist():
                start, n = int(rng.integers(4)), int(rng.integers(1, e.shape[1] - 4))
                values = rng.integers(0, 1 << 20, n)
                t.stage_write(row, start, values if i % 2 else values.tolist())
                e[row, start : start + n] = values
        else:
            for idx in rng.permutation(e.shape[0])[:5].tolist():
                value = int(rng.integers(1 << 20)) if e.dtype == np.int32 else float(rng.random())
                t.stage_write_elem(idx, value)
                e[idx] = value

    for _ in range(3):
        with staged_writes.deferred_writes():
            for i in range(len(tensors)):
                stage(i)
                tensors[i].apply_write()
            assert len(staged_writes._Deferred.pending) == len(tensors), "applied on exit, not per tensor"
            # A second batch for the same tensor stays ordered after the first.
            stage(0)
            tensors[0].apply_write()
        torch.accelerator.synchronize()
        for t, e in zip(tensors, expected):
            np.testing.assert_array_equal(t.gpu.cpu().numpy(), e)
            assert not t._staged_write_indices and t._staged_len == 0
    # Outside the window writes apply immediately.
    stage(2)
    tensors[2].apply_write()
    torch.accelerator.synchronize()
    np.testing.assert_array_equal(tensors[2].gpu.cpu().numpy(), expected[2])


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_repeated_flushes_survive_a_slow_stream():
    from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor

    from vllm_omni.worker_v2 import staged_writes

    staged_writes.install()
    device = torch.device("cuda", 0)
    t = StagedWriteTensor((64, 32), dtype=torch.int64, device=device)
    expected = np.zeros((64, 32), dtype=np.int64)
    rng = np.random.default_rng(3)
    # Compile the flush kernel first, so the timed window below only queues launches.
    with staged_writes.deferred_writes():
        t.stage_write(0, 0, np.zeros(32, dtype=np.int64))
        t.apply_write()
    torch.accelerator.synchronize()
    with staged_writes.deferred_writes():
        torch.cuda._sleep(1_000_000_000)  # ~0.5 s: the flushes queue behind it while the host keeps going
        # Each re-apply of the same tensor flushes the window: many flushes in one window.
        for row in range(64):
            values = rng.integers(0, 1 << 30, 32)
            t.stage_write(row, 0, values)
            t.apply_write()
            expected[row] = values
    torch.accelerator.synchronize()
    np.testing.assert_array_equal(t.gpu.cpu().numpy(), expected)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_staged_h2d_matches_the_pinned_copy_and_survives_slow_readers():
    from vllm.utils.torch_utils import async_tensor_h2d as original
    from vllm.v1.worker.gpu import model_runner

    from vllm_omni.worker_v2 import staged_writes

    staged_writes.install()
    staged = model_runner.async_tensor_h2d
    assert staged is not original
    device = torch.device("cuda", 0)
    rng = np.random.default_rng(2)
    results, expected = [], []
    for step in range(100):  # more calls than ring slots
        n = int(rng.integers(1, 300))
        values = rng.integers(0, 1 << 20, n)
        torch.cuda._sleep(200_000)  # the copies queue behind slow work
        if step % 3 == 0:
            results.append(staged(values.astype(np.int32), device=device, dtype=torch.int32))
        elif step % 3 == 1:
            results.append(staged(values.tolist(), device=device, dtype=torch.int64))
        else:
            out = torch.empty(n, dtype=torch.int32, device=device)
            results.append(staged(values, out=out))
        expected.append(original(values, device=device, dtype=results[-1].dtype))
    floats = rng.random(17).astype(np.float32)
    results.append(staged(floats, device=device))
    expected.append(original(floats, device=device))
    torch.accelerator.synchronize()
    for got, want in zip(results, expected):
        assert got.dtype == want.dtype and got.device == want.device
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    # Tensors and untyped lists keep the original path.
    host = torch.arange(5)
    torch.testing.assert_close(staged(host, device=device).cpu(), host)
    torch.testing.assert_close(staged([1, 2, 3], device=device).cpu(), torch.tensor([1, 2, 3]))


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_fused_bincount_matches_a_host_count():
    from vllm.v1.worker.gpu.sample import penalties

    from vllm_omni.worker_v2 import staged_writes

    staged_writes.install()
    device = torch.device("cuda", 0)
    vocab, rows, max_len = 3072, 8, 1500
    rng = np.random.default_rng(3)
    token_ids = torch.from_numpy(rng.integers(0, vocab, (rows, max_len)).astype(np.int32)).to(device)
    prompt_len = torch.from_numpy(rng.integers(1, 1200, rows).astype(np.int32)).to(device)
    # Resumed requests carry output tokens in their prefill.
    prefill_len = prompt_len + torch.from_numpy(rng.integers(0, 300, rows).astype(np.int32)).to(device)
    mask = torch.full((rows, (vocab + 31) // 32), -1, dtype=torch.int32, device=device)
    counts = torch.full((rows, vocab), 7, dtype=torch.int32, device=device)
    idx = torch.tensor([5, 1, 6], dtype=torch.int32, device=device)
    penalties.bincount(idx, token_ids, prompt_len, prefill_len, mask, counts, int(prefill_len.max()))
    torch.accelerator.synchronize()
    tok, pl, fl = token_ids.cpu().numpy(), prompt_len.cpu().numpy(), prefill_len.cpu().numpy()
    got_mask, got_counts = mask.cpu().numpy().view(np.uint32), counts.cpu().numpy()
    for r in range(rows):
        if r not in (5, 1, 6):
            assert (got_mask[r] == np.uint32(0xFFFFFFFF)).all() and (got_counts[r] == 7).all()
            continue
        bits = np.zeros(vocab, dtype=bool)
        bits[tok[r, : pl[r]]] = True
        want_mask = np.packbits(bits.reshape(-1, 32)[:, ::-1], axis=1, bitorder="big").view(">u4").reshape(-1)
        np.testing.assert_array_equal(got_mask[r], want_mask.astype(np.uint32))
        np.testing.assert_array_equal(got_counts[r], np.bincount(tok[r, pl[r] : fl[r]], minlength=vocab))
