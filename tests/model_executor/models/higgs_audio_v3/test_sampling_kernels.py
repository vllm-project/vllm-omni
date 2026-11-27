# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused Higgs sampling kernels must be bit-identical to the eager code they replace."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.device import DEVICE_MODULE, DEVICE_TYPE, NVIDIA_CUDA_OR_XPU
from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model, pytest.mark.skipif(not NVIDIA_CUDA_OR_XPU, reason="requires NVIDIA CUDA or XPU")]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(DEVICE_TYPE != "cuda", reason="requires CUDA Graph replay")
def test_bonus_graph_keeps_sampler_workspace_when_batch_grows(monkeypatch):
    import weakref

    from vllm.v1.sample.ops import topk_topp_triton

    from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_talker import (
        HiggsAudioSamplingParams,
        HiggsAudioV3TalkerForConditionalGeneration,
    )
    from vllm_omni.model_executor.models.higgs_audio_v3.spec_decode import HiggsAudioSpeculator

    cache: dict[tuple[torch.device, torch.dtype, int], torch.Tensor] = {}
    monkeypatch.setattr(topk_topp_triton, "_TRITON_BUFFER_CACHE", cache)
    device = torch.device("cuda", torch.accelerator.current_device_index())
    books, vocab, max_reqs = 8, 1026, 4
    model = HiggsAudioV3TalkerForConditionalGeneration.__new__(HiggsAudioV3TalkerForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.num_codebooks = books
    model.multimodal_embedding = SimpleNamespace(vocab_size=vocab)
    model.modality_head = SimpleNamespace(generate=lambda hidden: hidden.clone())
    spec = object.__new__(HiggsAudioSpeculator)
    spec.vllm_config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=max_reqs))
    spec._bonus_graphs = {}
    generator = torch.Generator(device=device).manual_seed(42)
    codebooks = torch.arange(books, device=device)[None]

    def inputs(rows):
        metadata = HiggsAudioSamplingParams(
            temperature=torch.ones(rows, device=device),
            top_k=torch.full((rows,), 50, device=device),
            top_p=torch.full((rows,), 0.95, device=device),
            generators={},
            all_greedy=False,
        )
        return (
            torch.randn((rows, books, vocab), device=device, generator=generator),
            torch.full((rows,), books - 1, device=device),
            torch.full((rows,), -1, device=device),
            metadata,
            torch.ones((rows, books, vocab), device=device),
            codebooks,
        )

    small = inputs(1)
    expected = tuple(out.clone() for out in spec._replay_bonus(model, *small))
    key = device, torch.float32, vocab
    captured = weakref.ref(cache[key])
    logits, _, _, metadata, noise, _ = inputs(max_reqs)
    model._sample_audio_codes(logits.flatten(0, 1), metadata, books, noise=noise.flatten(0, 1))
    # Do not replay against a freed workspace if this regression returns.
    assert captured() is not None and captured() is cache[key]
    for actual, reference in zip(spec._replay_bonus(model, *small), expected, strict=True):
        assert torch.equal(actual, reference)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("numel", [8 * 1026, 1000, 300000])
def test_batched_generator_noise_matches_per_request_exponential(numel):
    from vllm_omni.utils.seeded_exponential import fill_exponential_rows

    device = torch.device(DEVICE_TYPE)
    seeds = [42, 42, None, 2**63 + 5]

    def generators():
        gens = [None if s is None else torch.Generator(device=device).manual_seed(s) for s in seeds]
        resumed = gens[1]
        assert resumed is not None
        resumed.set_offset(resumed.get_offset() + 4)  # a request that already sampled
        return gens

    default = DEVICE_MODULE.default_generators[device.index or 0]
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


@hardware_test(res={"cuda": "L4", "xpu": "B60"}, num_cards=1)
def test_fused_decode_embeddings_match_feedback_and_zero_padding():
    from vllm_omni.model_executor.models.higgs_audio_v3.step_kernels import decode_embeddings

    torch.manual_seed(0)
    vocab, hidden, books, audio_vocab = 1000, 96, 8, 17
    text = torch.randn(vocab, hidden, device=DEVICE_TYPE, dtype=torch.bfloat16)
    audio = torch.randn(books * audio_vocab, hidden, device=DEVICE_TYPE, dtype=torch.bfloat16)
    ids = torch.randint(0, vocab, (5,), device=DEVICE_TYPE)
    ids[0] = -1
    codes = torch.randint(0, audio_vocab, (8, books), device=DEVICE_TYPE)
    has = torch.tensor([True, False, True, False, True, True, False, False], device=DEVICE_TYPE)
    offsets = torch.arange(books, device=DEVICE_TYPE) * audio_vocab
    expected = torch.where(
        has[:5, None], F.embedding(codes[:5] + offsets, audio).sum(-2), F.embedding(ids.clamp_min(0), text)
    )
    out = torch.full((8, hidden), 3.0, device=DEVICE_TYPE, dtype=torch.bfloat16)
    decode_embeddings(out, ids, text, audio, codes, has, 7, audio_vocab)
    assert torch.equal(out[:5], expected)
    assert torch.count_nonzero(out[5:7]) == 0
    assert torch.all(out[7] == 3)


@hardware_test(res={"cuda": "L4", "xpu": "B60"}, num_cards=1)
@pytest.mark.parametrize("prompt_mode", [False, True])
def test_step_tail_guard_matches_eager_restore_reset_and_validation(prompt_mode):
    from vllm_omni.model_executor.models.higgs_audio_v3.step_kernels import step_tail_guard

    audio, eos = 7, 9
    # Rows: decode audio, decode EOS, 3-token prefill ending in audio, decode audio.
    ids = torch.tensor([audio, eos, 1, 2, audio, audio], device=DEVICE_TYPE)
    qsl = torch.tensor([0, 1, 2, 5, 6], device=DEVICE_TYPE, dtype=torch.int32)
    has = torch.tensor([True, True, True, False], device=DEVICE_TYPE)
    done = torch.tensor([False, False, True, False], device=DEVICE_TYPE)
    delay = torch.tensor([3, 4, 5, 6], device=DEVICE_TYPE)
    eoc = torch.tensor([1, 2, 3, 4], device=DEVICE_TYPE)
    error = torch.zeros(1, device=DEVICE_TYPE, dtype=torch.int32)
    step_tail_guard(ids, qsl, has, done, delay, eoc, error, audio_id=audio, eos_id=eos, prompt_mode=prompt_mode)
    prefill = torch.tensor([False, False, True, False], device=DEVICE_TYPE)
    tail = ids[(qsl[1:] - 1).long()]
    want_done = torch.tensor([False, False, True, False], device=DEVICE_TYPE)
    if prompt_mode:
        want_done |= tail == eos
    want_done &= ~prefill
    assert torch.equal(done, want_done)
    assert torch.equal(has, torch.tensor([True, True, False, False], device=DEVICE_TYPE))
    assert delay.tolist() == [3, 4, 0, 6] and eoc.tolist() == [1, 2, -1, 4]
    # The EOS row is valid only as a terminal prompt-mode row.
    assert int(error.item()) == (0 if prompt_mode else 1)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(DEVICE_TYPE != "cuda", reason="requires CUDA")
def test_audio_history_keeps_only_committed_rows_in_request_slots():
    from vllm_omni.model_executor.models.higgs_audio_v3.step_kernels import record_audio_history

    device = torch.device("cuda")
    history = torch.full((4, 5, 8), -1, device=device, dtype=torch.int16)
    lengths = torch.tensor([0, 1, 0, 0], device=device, dtype=torch.int32)
    first = torch.arange(32, device=device, dtype=torch.int32).view(4, 8)
    snapshots = torch.cat((first, torch.tensor([[1, 0], [1, 0], [1, 0], [0, 1]], device=device)), dim=1).int()
    drafts = torch.stack((first, first + 100), dim=1)
    counts = torch.tensor([2, 1, 0, 1], device=device)
    slots = torch.tensor([1, 3, 0, 2], device=device)
    total = torch.tensor([5, 7, 4, 7], device=device)
    prompt = torch.tensor([5, 6, 4, 7], device=device)
    record_audio_history(history, lengths, snapshots, drafts, counts, slots, total, prompt)
    assert lengths.tolist() == [0, 3, 0, 1]
    torch.testing.assert_close(history[1, 1:3], drafts[0].to(torch.int16))
    torch.testing.assert_close(history[3, 0], first[1].to(torch.int16))
    assert torch.all(history[3, 1:] == -1)
    assert torch.all(history[[0, 2]] == -1)
