# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The single-kernel Talker sampler draws exactly the upstream Gumbel sampler's tokens."""

import types

import numpy as np
import pytest
import torch

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]

VOCAB, MAX_REQS, EOS = 3072, 16, 2150


def _setup(params, prompt_len=4, use_fp64_gumbel=False):
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.states import RequestState

    device = torch.device("cuda", 0)
    req_states = RequestState(MAX_REQS, 256, 256, 0, VOCAB, device)
    base = Sampler(
        vllm_config=types.SimpleNamespace(reasoning_config=None),
        max_num_reqs=MAX_REQS,
        vocab_size=VOCAB,
        device=device,
        req_states=req_states,
        use_fp64_gumbel=use_fp64_gumbel,
    )
    # The fused kernel uses the upstream seeded Gumbel draw; compare against that path.
    base.use_flashinfer = False
    gen = torch.Generator().manual_seed(0)
    slots = []
    for i, sp in enumerate(params):
        prompt = torch.randint(0, VOCAB, (prompt_len,), generator=gen).tolist()
        req_states.add_request(f"r{i}", prompt_len, prompt, prompt_len, 64)
        slot = req_states.req_id_to_index[f"r{i}"]
        slots.append(slot)
    talker = types.SimpleNamespace(
        _codec_disallowed_mask=(torch.arange(VOCAB, device=device) >= 2048)
        & (torch.arange(VOCAB, device=device) != EOS)
    )
    from vllm_omni.model_executor.models.qwen3_tts.talker_sampler import Qwen3TTSTalkerSampler

    fused = Qwen3TTSTalkerSampler(base, talker)
    for slot, sp in zip(slots, params):
        fused.add_request(slot, sp)
    req_states.apply_staged_writes()
    fused.apply_staged_writes()
    # Earlier outputs for the penalties.
    base.penalties_state.output_bin_counts[slots] = torch.randint(
        0, 2, (len(slots), VOCAB), generator=gen, dtype=torch.int32
    ).to(device)
    return base, fused, talker, req_states, np.array(slots, dtype=np.int32), device


def _batch(slots, positions, seq_lens, device):
    n = len(slots)
    idx = torch.tensor(slots, dtype=torch.int32, device=device)
    return types.SimpleNamespace(
        num_reqs=n,
        idx_mapping=idx,
        idx_mapping_np=slots,
        expanded_idx_mapping=idx,
        expanded_local_pos=torch.zeros(n, dtype=torch.int32, device=device),
        positions=torch.tensor(positions, dtype=torch.int64, device=device),
        logits_indices=torch.arange(n, dtype=torch.int64, device=device),
        input_ids=torch.zeros(n, dtype=torch.int32, device=device),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        seq_lens_cpu_upper_bound=torch.tensor(seq_lens, dtype=torch.int32),
        cu_num_logits=torch.arange(n + 1, dtype=torch.int32, device=device),
        cu_num_logits_np=np.arange(n + 1, dtype=np.int32),
    )


def _talker_params(**overrides):
    from vllm.sampling_params import SamplingParams

    kw = dict(temperature=0.9, top_k=50, repetition_penalty=1.05, min_tokens=2, stop_token_ids=[EOS], seed=None)
    kw.update(overrides)
    return SamplingParams(**kw)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fused_talker_sampler_matches_upstream_tokens(dtype):
    params = [
        _talker_params(),
        _talker_params(seed=7),
        _talker_params(temperature=0.0),
        _talker_params(temperature=1.0, top_k=-1),
        _talker_params(top_k=1),
        _talker_params(repetition_penalty=1.0, frequency_penalty=0.2, presence_penalty=0.3),
        _talker_params(top_k=64, min_tokens=0),
    ]
    base, fused, talker, _, slots, device = _setup(params)
    assert fused._fast[slots].all()
    gen = torch.Generator(device=device).manual_seed(1)
    # Prompt length 4 and min_tokens 2: position 4 still bans EOS, position 5 allows it.
    for step, pos in enumerate((4, 5, 9, 40)):
        logits = (torch.randn(len(slots), VOCAB, device=device, generator=gen) * 3).to(dtype)
        logits[:, EOS] = 20.0  # EOS would win wherever it is allowed
        batch = _batch(slots, [pos] * len(slots), [pos + 1] * len(slots), device)
        out = fused(logits.clone(), batch)
        ref = base(logits.masked_fill(talker._codec_disallowed_mask, float("-inf")), batch)
        torch.testing.assert_close(out.sampled_token_ids, ref.sampled_token_ids, rtol=0, atol=0)
        torch.testing.assert_close(out.num_sampled, ref.num_sampled.to(out.num_sampled.dtype))
        assert out.num_rejected.tolist() == [0] * len(slots)
        tokens = out.sampled_token_ids.view(-1).tolist()
        assert all(t < 2048 or t == EOS for t in tokens)
        if pos == 4:
            # min_tokens rows must not stop yet; the row without min_tokens does.
            assert all(t != EOS for t in tokens[:-1]) and tokens[-1] == EOS, (step, tokens)


def test_fused_talker_sampler_counts_chunked_prefill_rows_as_unsampled():
    params = [_talker_params(), _talker_params()]
    base, fused, talker, req_states, slots, device = _setup(params)
    logits = torch.randn(2, VOCAB, device=device)
    # The first row is still inside its prefill (seq_len < prefill_len 4).
    batch = _batch(slots, [2, 4], [3, 5], device)
    out = fused(logits, batch)
    ref = base(logits.masked_fill(talker._codec_disallowed_mask, float("-inf")), batch)
    assert out.num_sampled.tolist() == [0, 1] == ref.num_sampled.tolist()


def test_unsupported_requests_take_the_upstream_sampler():
    params = [_talker_params(), _talker_params(top_p=0.8), _talker_params(top_k=100), _talker_params(logprobs=2)]
    base, fused, talker, _, slots, device = _setup(params)
    assert fused._fast[slots].tolist() == [True, False, False, False]
    logits = torch.randn(len(slots), VOCAB, device=device)
    batch = _batch(slots, [9] * len(slots), [10] * len(slots), device)
    out = fused(logits.clone(), batch)
    ref = base(logits.masked_fill(talker._codec_disallowed_mask, float("-inf")), batch)
    torch.testing.assert_close(out.sampled_token_ids, ref.sampled_token_ids, rtol=0, atol=0)
    assert out.logprobs_tensors is not None


def test_fp64_gumbel_sampler_stays_upstream():
    params = [_talker_params(seed=7), _talker_params(seed=8)]
    base, fused, talker, _, slots, device = _setup(params, use_fp64_gumbel=True)
    # The Talker's compute_logits then applies the codec mask itself.
    assert not fused._enabled and not fused.fused_disallowed_mask
    logits = torch.randn(len(slots), VOCAB, device=device).masked_fill(talker._codec_disallowed_mask, float("-inf"))
    batch = _batch(slots, [9] * len(slots), [10] * len(slots), device)
    out = fused(logits.clone(), batch)
    ref = base(logits.clone(), batch)
    torch.testing.assert_close(out.sampled_token_ids, ref.sampled_token_ids, rtol=0, atol=0)
