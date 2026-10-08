# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused Higgs sampling kernels must be bit-identical to the eager code they replace."""

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_test
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="requires CUDA")]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("numel", [8 * 1026, 1000, 300000])
def test_batched_generator_noise_matches_per_request_exponential(numel):
    from vllm_omni.utils.seeded_exponential import fill_exponential_rows

    device = torch.device("cuda")
    seeds = [42, 42, None, 2**63 + 5]

    def generators():
        gens = [None if s is None else torch.Generator(device=device).manual_seed(s) for s in seeds]
        resumed = gens[1]
        assert resumed is not None
        resumed.set_offset(resumed.get_offset() + 4)  # a request that already sampled
        return gens

    default = torch.cuda.default_generators[device.index or 0]
    start = default.get_offset()
    expected_gens = generators()
    expected = torch.empty(len(seeds), numel, device=device)
    for row, generator in enumerate(expected_gens):
        expected[row].exponential_(generator=generator)
    expected_default = default.get_offset()

    default.set_offset(start)
    actual_gens = generators()
    actual = fill_exponential_rows(torch.empty_like(expected), actual_gens)
    assert torch.equal(actual, expected)
    assert default.get_offset() == expected_default
    for got, want in zip(actual_gens, expected_gens):
        if want is not None:
            assert got.get_offset() == want.get_offset()


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_fused_decode_embeddings_match_feedback_and_zero_padding():
    from vllm_omni.model_executor.models.higgs_audio_v3.step_kernels import decode_embeddings

    torch.manual_seed(0)
    vocab, hidden, books, audio_vocab = 1000, 96, 8, 17
    text = torch.randn(vocab, hidden, device="cuda", dtype=torch.bfloat16)
    audio = torch.randn(books * audio_vocab, hidden, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(0, vocab, (5,), device="cuda")
    ids[0] = -1
    codes = torch.randint(0, audio_vocab, (8, books), device="cuda")
    has = torch.tensor([True, False, True, False, True, True, False, False], device="cuda")
    offsets = torch.arange(books, device="cuda") * audio_vocab
    expected = torch.where(
        has[:5, None], F.embedding(codes[:5] + offsets, audio).sum(-2), F.embedding(ids.clamp_min(0), text)
    )
    out = torch.full((8, hidden), 3.0, device="cuda", dtype=torch.bfloat16)
    decode_embeddings(out, ids, text, audio, codes, has, 7, audio_vocab)
    assert torch.equal(out[:5], expected)
    assert torch.count_nonzero(out[5:7]) == 0
    assert torch.all(out[7] == 3)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("prompt_mode", [False, True])
def test_step_tail_guard_matches_eager_restore_reset_and_validation(prompt_mode):
    from vllm_omni.model_executor.models.higgs_audio_v3.step_kernels import step_tail_guard

    audio, eos = 7, 9
    # Rows: decode audio, decode EOS, 3-token prefill ending in audio, decode audio.
    ids = torch.tensor([audio, eos, 1, 2, audio, audio], device="cuda")
    qsl = torch.tensor([0, 1, 2, 5, 6], device="cuda", dtype=torch.int32)
    has = torch.tensor([True, True, True, False], device="cuda")
    done = torch.tensor([False, False, True, False], device="cuda")
    delay = torch.tensor([3, 4, 5, 6], device="cuda")
    eoc = torch.tensor([1, 2, 3, 4], device="cuda")
    error = torch.zeros(1, device="cuda", dtype=torch.int32)
    step_tail_guard(ids, qsl, has, done, delay, eoc, error, audio_id=audio, eos_id=eos, prompt_mode=prompt_mode)
    prefill = torch.tensor([False, False, True, False], device="cuda")
    tail = ids[(qsl[1:] - 1).long()]
    want_done = torch.tensor([False, False, True, False], device="cuda")
    if prompt_mode:
        want_done |= tail == eos
    want_done &= ~prefill
    assert torch.equal(done, want_done)
    assert torch.equal(has, torch.tensor([True, True, False, False], device="cuda"))
    assert delay.tolist() == [3, 4, 0, 6] and eoc.tolist() == [1, 2, -1, 4]
    # The EOS row is valid only as a terminal prompt-mode row.
    assert int(error.item()) == (0 if prompt_mode else 1)
