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
    """A mixed eager step uploads four row lists; later steps must not overwrite them.

    The GPU is held back, so every read below happens after the CPU issued
    three whole steps (two in flight plus the next), as with async scheduling.
    """
    from vllm_omni.model_executor.models.moss_tts.local_model_state import MossLocalModelState, _UvaIndexPools

    state = MossLocalModelState.__new__(MossLocalModelState)
    depth = 4  # vLLM's default of two concurrent batches, plus two
    state._uva = uva = _UvaIndexPools(16, depth)
    table = torch.arange(1000, device="cuda", dtype=torch.long)
    sites = [uva.eager_slots, uva.eager_offsets, uva.eager_mtp_slots, uva.rows]
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
