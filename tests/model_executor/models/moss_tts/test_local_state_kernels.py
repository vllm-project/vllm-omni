# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""UVA-indexed fused MOSS Local slot-state copies equal the eager ops they replace."""

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]

H, NVQ, CAP, PAD = 2560, 12, 128, 1024


def _uva(array: np.ndarray) -> torch.Tensor:
    from vllm.v1.worker.gpu.buffer_utils import UvaBufferPool

    pool = UvaBufferPool(array.shape if array.ndim > 1 else array.shape[0], torch.int64, 2)
    return pool.copy_to_uva(array)


@pytest.mark.parametrize("bsz", [1, 7, 64, 128])
def test_gather_and_scatter_mtp_match_eager(bsz):
    from vllm_omni.model_executor.models.moss_tts.local_state_kernels import gather_mtp, scatter_mtp

    gen = torch.Generator(device="cuda").manual_seed(bsz)
    tokens = 3 * bsz + 5
    input_ids = torch.randint(0, 150000, (tokens,), device="cuda", dtype=torch.int32, generator=gen)
    embeds = torch.randn((tokens, H), device="cuda", generator=gen).to(torch.bfloat16)
    hidden_pool = torch.randn((CAP, H), device="cuda", generator=gen).to(torch.bfloat16)
    active_pool = torch.rand(CAP, device="cuda", generator=gen) > 0.5
    codes_pool = torch.randint(0, PAD + 1, (CAP, NVQ), device="cuda", generator=gen)
    slots = torch.randperm(CAP, device="cuda", generator=gen)[:bsz]
    offsets = torch.randperm(tokens, device="cuda", generator=gen)[:bsz]
    index = _uva(np.stack([slots.cpu().numpy(), offsets.cpu().numpy()], axis=1).astype(np.int64))

    ids = torch.zeros(bsz, dtype=torch.long, device="cuda")
    emb = torch.zeros((bsz, H), dtype=torch.bfloat16, device="cuda")
    hidden = torch.zeros_like(emb)
    step = torch.zeros((bsz, H), dtype=torch.bfloat16, device="cuda")
    gather_mtp(index, input_ids, embeds, hidden_pool, active_pool, ids, emb, hidden, step)

    ref_step = torch.zeros_like(step)
    ref_step[:, 0].copy_(active_pool.index_select(0, slots))
    assert torch.equal(ids, input_ids.index_select(0, offsets).long())
    assert torch.equal(emb, embeds.index_select(0, offsets))
    assert torch.equal(hidden, hidden_pool.index_select(0, slots))
    assert torch.equal(step, ref_step)

    new_emb = torch.randn((bsz, H), device="cuda", generator=gen).to(torch.bfloat16)
    codes = torch.randint(0, PAD, (bsz, NVQ), device="cuda", generator=gen)
    ref_embeds, ref_codes = embeds.clone(), codes_pool.clone()
    ref_embeds.index_copy_(0, offsets, new_emb)
    ref_codes.index_copy_(0, slots, codes)
    scatter_mtp(index, new_emb, codes, embeds, codes_pool)
    assert torch.equal(embeds, ref_embeds)
    assert torch.equal(codes_pool, ref_codes)


def test_reset_slots_matches_eager_fills():
    from vllm_omni.model_executor.models.moss_tts.local_state_kernels import reset_slots

    gen = torch.Generator(device="cuda").manual_seed(0)
    hidden = torch.randn((CAP, H), device="cuda", generator=gen).to(torch.bfloat16)
    eager = torch.randn_like(hidden)
    codes = torch.randint(0, PAD, (CAP, NVQ), device="cuda", generator=gen)
    active = torch.zeros(CAP, dtype=torch.bool, device="cuda")
    keep = torch.ones(CAP, dtype=torch.bool, device="cuda")
    slots = [5, 0, 127, 64]
    ref = [t.clone() for t in (hidden, eager, codes, active, keep)]
    for slot in slots:
        ref[0][slot].zero_()
        ref[1][slot].zero_()
        ref[2][slot].fill_(PAD)
        ref[3][slot].fill_(True)
        ref[4][slot].fill_(False)
    reset_slots(_uva(np.asarray(slots, dtype=np.int64)), hidden, codes, active, keep, eager, PAD)
    for actual, expected in zip((hidden, eager, codes, active, keep), ref):
        assert torch.equal(actual, expected)


def test_uva_index_uploads_survive_steps_in_flight():
    """A step uploads row lists through per-site pools; later steps must not overwrite them.

    The GPU is held back, so every read below happens after the CPU issued
    three whole steps (two in flight plus the next), as with async scheduling.
    """
    from vllm_omni.model_executor.models.moss_tts.local_model_state import MossLocalModelState, _UvaIndexPools

    state = MossLocalModelState.__new__(MossLocalModelState)
    depth = 4  # vLLM's default of two concurrent batches, plus two
    state._uva = uva = _UvaIndexPools(16, depth)
    table = torch.arange(1000, device="cuda", dtype=torch.long)
    sites = [uva.eager_slots, uva.eager_offsets]
    # Load the kernels first: a lazy module load would wait for the sleep.
    table.index_select(0, table[:2]).long()
    torch.accelerator.synchronize()
    torch.cuda._sleep(200_000_000)
    reads = []
    for step in range(depth - 1):
        for site, pool in enumerate(sites):
            rows = [100 * step + 10 * site + 1, 100 * step + 10 * site + 3]
            reads.append((rows, state._select_rows(table, rows, pool)))
    torch.accelerator.synchronize()
    for rows, selected in reads:
        assert selected.tolist() == rows


def _rows(gen, n, limit):
    return torch.randperm(limit, device="cuda", generator=gen)[:n]


@pytest.mark.parametrize("bsz", [1, 7, 128])
def test_add_rows_matches_select_add_copy(bsz):
    from vllm_omni.model_executor.models.moss_tts.local_state_kernels import add_rows

    gen = torch.Generator(device="cuda").manual_seed(bsz)
    embeds = torch.randn((3 * bsz + 5, H), device="cuda", generator=gen).to(torch.bfloat16)
    pool = torch.randn((CAP, H), device="cuda", generator=gen).to(torch.bfloat16)
    slots, offsets = _rows(gen, bsz, CAP), _rows(gen, bsz, embeds.shape[0])
    expected = embeds.clone()
    expected.index_copy_(0, offsets, expected.index_select(0, offsets) + pool.index_select(0, slots))
    index = _uva(np.stack([slots.cpu().numpy(), offsets.cpu().numpy()], axis=1).astype(np.int64))
    add_rows(index[:, 1], pool, embeds, src_index=index[:, 0])
    assert torch.equal(embeds, expected)

    # Implicit source rows and a strided device index (reference conditioning).
    audio = torch.randn((bsz, H), device="cuda", generator=gen).to(torch.bfloat16)
    packed = torch.zeros((bsz, NVQ + 1), dtype=torch.long, device="cuda")
    packed[:, 0] = offsets
    expected.index_copy_(0, offsets, expected.index_select(0, offsets) + audio)
    add_rows(packed[:, 0], audio, embeds)
    assert torch.equal(embeds, expected)


def test_snapshot_hidden_matches_last_token_index_copy():
    from vllm_omni.model_executor.models.moss_tts.local_state_kernels import snapshot_hidden

    gen = torch.Generator(device="cuda").manual_seed(0)
    counts = [5, 1, 1, 3, 1]
    starts = torch.tensor(np.cumsum([0, *counts]), dtype=torch.int32, device="cuda")
    slots = _rows(gen, len(counts), CAP).to(torch.int32)
    hidden = torch.randn((sum(counts) + 7, H), device="cuda", generator=gen).to(torch.bfloat16)
    pool = torch.randn((CAP, H), device="cuda", generator=gen).to(torch.bfloat16)
    expected = pool.clone()
    expected.index_copy_(0, slots.long(), hidden.index_select(0, starts[1:] - 1))
    snapshot_hidden(starts, slots, hidden, pool, len(counts))
    assert torch.equal(pool, expected)


@pytest.mark.parametrize("bsz", [1, 7, 128])
def test_eager_mtp_gather_scatter_match_eager_ops(bsz):
    from vllm_omni.model_executor.models.moss_tts.local_state_kernels import gather_eager_mtp, scatter_eager_mtp

    gen = torch.Generator(device="cuda").manual_seed(bsz)
    hidden_pool = torch.randn((CAP, H), device="cuda", generator=gen).to(torch.bfloat16)
    keep, active = (torch.rand(CAP, device="cuda", generator=gen) > 0.5 for _ in range(2))
    slots = _rows(gen, bsz, CAP)
    first = torch.rand(bsz, device="cuda", generator=gen) > 0.5
    ids = torch.full((bsz,), 9, dtype=torch.long, device="cuda")
    emb, hidden, step = (torch.randn((bsz, H), device="cuda", generator=gen).to(torch.bfloat16) for _ in range(3))
    expected_step = step.clone()
    expected_step[:, 0].copy_(active.index_select(0, slots) & (keep.index_select(0, slots) | first))
    index = _uva(np.stack([slots.cpu().numpy(), first.cpu().numpy()], axis=1).astype(np.int64))
    gather_eager_mtp(index, hidden_pool, keep, active, ids, emb, hidden, step)
    assert not ids.any() and not emb.any()
    assert torch.equal(hidden, hidden_pool.index_select(0, slots))
    assert torch.equal(step, expected_step)

    # Graph outputs are padded past bsz; codes may be a strided view.
    contribution = torch.randn((bsz + 3, H), device="cuda", generator=gen).to(torch.bfloat16)
    codes = torch.randint(0, PAD + 1, (bsz + 3, 2 * NVQ), device="cuda", generator=gen)[:, :NVQ]
    codes[: bsz // 2] = PAD
    eager_emb = torch.randn((CAP, H), device="cuda", generator=gen).to(torch.bfloat16)
    codes_pool = torch.randint(0, PAD + 1, (CAP, NVQ), device="cuda", generator=gen)
    expected = [eager_emb.clone(), codes_pool.clone(), keep.clone()]
    expected[0].index_copy_(0, slots, contribution[:bsz])
    expected[1].index_copy_(0, slots, codes[:bsz])
    expected[2].index_copy_(0, slots, codes[:bsz].ne(PAD).any(dim=-1))
    scatter_eager_mtp(index, contribution[:bsz], codes[:bsz], eager_emb, codes_pool, keep, PAD)
    for actual, reference in zip((eager_emb, codes_pool, keep), expected):
        assert torch.equal(actual, reference)


@pytest.mark.parametrize("all_rows", [False, True])
def test_gather_codes_matches_index_select_and_keep(all_rows):
    from vllm_omni.model_executor.models.moss_tts.local_state_kernels import gather_codes

    gen = torch.Generator(device="cuda").manual_seed(int(all_rows))
    codes_pool = torch.randint(0, PAD + 1, (CAP, NVQ), device="cuda", generator=gen)
    codes_pool[::3] = PAD
    num_reqs = 40
    rows = list(range(num_reqs)) if all_rows else [0, 3, 4, 17, 39]
    slots = _rows(gen, len(rows), CAP)
    expected_codes = codes_pool.index_select(0, slots)
    expected_keep = torch.ones(num_reqs, dtype=torch.bool, device="cuda")
    expected_keep[rows] = expected_codes.ne(PAD).any(dim=-1)
    out = torch.empty_like(expected_codes)
    keep = torch.ones(num_reqs, dtype=torch.bool, device="cuda")
    gather_codes(_uva(np.stack([slots.cpu().numpy(), rows], axis=1).astype(np.int64)), codes_pool, out, keep, PAD)
    assert torch.equal(out, expected_codes)
    assert torch.equal(keep, expected_keep)
