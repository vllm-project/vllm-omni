# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-state RVQ sampling and determined logical tokens, without an adapter."""

import logging
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm import SamplingParams
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

from tests.model_executor.models.test_zonos3_model_state import make_owner, materialize_codes
from vllm_omni.core.sched.output import OmniNewRequestData
from vllm_omni.model_executor.models.zonos3.ar_config import ZONOS3_SAMPLING_KEY
from vllm_omni.model_executor.models.zonos3.model_state import Zonos3SamplingOptions
from vllm_omni.model_executor.models.zonos3.sampling_state import CONTINUE, EOS
from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def cpu_non_uva_allocator(monkeypatch):
    monkeypatch.setattr("vllm.v1.worker.gpu.buffer_utils.is_uva_available", lambda: False)


def make_state(capacity=4, device="cpu", dtype=torch.float32):
    state, _, _ = make_owner(capacity, device, dtype)
    torch.manual_seed(42)
    with torch.no_grad():
        state.model.multi_output.weight.normal_(std=0.2)
    return state


def add_sampling(state, slot, params):
    if f"request-{slot}" in state._request_slots:
        state.remove_request(slot)
    request = OmniNewRequestData(
        req_id=f"request-{slot}",
        prompt_token_ids=[0],
        prefill_token_ids=[0],
        mm_features=[],
        sampling_params=params,
        pooling_params=None,
        block_ids=(),
        num_computed_tokens=0,
        lora_request=None,
    )
    state.add_request(slot, request)
    state.sampling_state.has_audio_prefix[slot] = True


def make_batch(slots, seq_lens=None, prefill_len=1, device="cpu"):
    batch = InputBatch.make_dummy(len(slots), len(slots), InputBuffers(len(slots), len(slots), torch.device(device)))
    batch.idx_mapping = torch.tensor(slots, dtype=torch.int64, device=device)
    batch.idx_mapping_np = np.array(slots, dtype=np.intp)
    batch.expanded_idx_mapping = batch.idx_mapping
    if seq_lens is not None:
        batch.seq_lens[:] = torch.tensor(seq_lens, device=device)
    lengths = UvaBackedTensor(max(slots) + 1, dtype=torch.int32)
    lengths.np.fill(prefill_len)
    lengths.copy_to_uva()
    return batch, SimpleNamespace(prefill_len=lengths)


def greedy_params(**extra):
    return SamplingParams(temperature=0, stop_token_ids=[EOS], extra_args={ZONOS3_SAMPLING_KEY: extra})


def run(state, slots, hidden, seq_lens=None, prefill_len=1):
    batch, requests = make_batch(slots, seq_lens, prefill_len, hidden.device)
    _, snapshot = state.postprocess_model_output(hidden, batch, requests)
    # Exercise the existing runner shortcut. This sampler has no __call__, so
    # any accidental fallback to standard text sampling fails the test.
    runner = SimpleNamespace(model_state=state, sampler=object(), batch_sharder=None, pp_handler=None)
    output, sampled, rejected = OmniARModelRunner.sample(runner, hidden, batch, None)
    assert sampled is output.num_sampled and rejected is output.num_rejected
    return output, snapshot


def codes(snapshot):
    return torch.stack(snapshot["codes"]["audio"])[:, 0]


@torch.inference_mode()
def test_parallel_codebooks_one_scheduler_token_and_feedback_contract():
    state = make_state(2)
    for slot in range(2):
        add_sampling(state, slot, greedy_params())
    state.apply_staged_writes()
    output, snapshot = run(state, [1, 0], torch.randn(2, 16))
    assert output.sampled_token_ids.shape == (2, 1)
    assert output.num_sampled.tolist() == [1, 1]
    sampled = codes(snapshot)
    assert sampled.shape == (2, 10) and not (sampled == state.model.config.audio_pad_id).any()
    frame = torch.cat((sampled, torch.full((2, 1), state.model.config.text_vocab)), -1)
    torch.testing.assert_close(state.next_embeddings[[1, 0]], state.model.multi_embedder(frame))
    materialized = materialize_codes(snapshot)
    assert [p["codes"]["audio"].shape for p in materialized] == [(1, 10), (1, 10)]
    assert all(bool(p["meta"]["codec_frame_valid"].item()) for p in materialized)


def test_rvq_top_k_is_not_clamped_to_three_logical_tokens():
    state = make_state(1)
    add_sampling(state, 0, SamplingParams(top_k=80, seed=17))
    state.apply_staged_writes()
    assert state.settings.top_k.np[0] == state.model.config.audio_vocab == 10


@torch.inference_mode()
def test_chunked_prefill_does_not_advance_physical_state_or_rng():
    state, single = make_state(1), make_state(1)
    params = SamplingParams(temperature=0.7, top_k=5, top_p=0.8, seed=27)
    for instance in (state, single):
        add_sampling(instance, 0, params)
        instance.apply_staged_writes()
    hidden = torch.randn(1, 16)
    output, first = run(state, [0], hidden, [2], prefill_len=5)
    assert output.num_sampled.item() == 0 and state.sampling_state.position[0].item() == 0
    assert not materialize_codes(first)[0]["meta"]["codec_frame_valid"].any()
    _, actual = run(state, [0], hidden, [5], prefill_len=5)
    _, expected = run(single, [0], hidden)
    assert torch.equal(codes(actual), codes(expected))


@torch.inference_mode()
def test_tail_drain_logical_eos_is_independent_and_absorbs_async_extra_step():
    state = make_state(2)
    state.model.multi_output.weight.zero_()
    c = state.model.config
    for cb in range(10):
        state.model.multi_output.weight[cb * c.audio_vocab + c.eoa_id, 0] = 12
    for slot in range(2):
        add_sampling(state, slot, greedy_params())
    state.apply_staged_writes()
    for step in range(15):
        hidden = torch.zeros((2, 16))
        hidden[:, 0] = torch.tensor([1.0, -1.0 if step < 4 else 1.0])
        output, snapshot = run(state, [0, 1], hidden)
        assert output.sampled_token_ids[:, 0].tolist() == [
            EOS if step >= 10 else CONTINUE,
            EOS if step >= 14 else CONTINUE,
        ]
    assert state.sampling_state.position.tolist()[:2] == [11, 15]
    assert not materialize_codes(snapshot)[0]["meta"]["codec_frame_valid"].any()


@pytest.mark.parametrize("concurrency", [1, 4, 8, 16, 32])
@torch.inference_mode()
def test_seeded_rvq_sampling_is_invariant_to_batch_reordering(concurrency):
    batched = make_state(concurrency)
    singles = [make_state(1) for _ in range(concurrency)]
    for slot, single in enumerate(singles):
        params = SamplingParams(
            temperature=0.7 + 0.01 * slot,
            top_k=5,
            top_p=0.85,
            seed=100 + slot,
            repetition_penalty=1.4,
            extra_args={ZONOS3_SAMPLING_KEY: {"repetition_window": 4}},
        )
        add_sampling(single, 0, params)
        single.apply_staged_writes()
        add_sampling(batched, slot, params)
    batched.apply_staged_writes()
    generator = torch.Generator().manual_seed(19)
    for step in range(4):
        hidden = torch.randn(concurrency, 16, generator=generator)
        order = list(reversed(range(concurrency))) if step % 2 else list(range(concurrency))
        _, actual = run(batched, order, hidden[order])
        for row, slot in enumerate(order):
            _, expected = run(singles[slot], [0], hidden[slot : slot + 1])
            assert torch.equal(codes(actual)[row], codes(expected)[0])


@torch.inference_mode()
def test_slot_reuse_resets_history_and_output_snapshot_stays_owned():
    state = make_state(1)
    params = SamplingParams(seed=21)
    add_sampling(state, 0, params)
    state.apply_staged_writes()
    _, output = run(state, [0], torch.randn(1, 16))
    saved = codes(output).clone()
    add_sampling(state, 0, params)
    state.apply_staged_writes()
    assert state.sampling_state.position[0].item() == 0 and (state.history == -1).all()
    run(state, [0], torch.randn(1, 16))
    assert torch.equal(codes(output), saved)


def test_state_rejects_unsupported_logical_constraints_and_dirty_staging():
    state = make_state(1)
    with pytest.raises(ValueError, match="Unsupported logical-token"):
        add_sampling(state, 0, SamplingParams(logprobs=2))
    add_sampling(state, 0, greedy_params())
    with pytest.raises(RuntimeError, match="staged writes"):
        run(state, [0], torch.zeros(1, 16))
    with pytest.raises(ValueError, match="eoa_policy"):
        Zonos3SamplingOptions.from_params(greedy_params(eoa_policy="bad"), state.model.config, 128)


@torch.inference_mode()
def test_model_state_warmup_covers_buckets_and_leaves_seeded_requests_unchanged():
    state, reference = make_state(4), make_state(4)
    _, requests = make_batch([0, 1, 2, 3])
    calls = []
    hook = state.model.multi_output.register_forward_pre_hook(lambda _module, args: calls.append(len(args[0])))
    try:
        state.warmup_sampling(torch.randn(4, 16), requests)
    finally:
        hook.remove()
    assert calls == [1, 2, 4]
    assert state.options == [None] * 4 and not state._generators
    assert state._sample_output is None and not state._sampling_pending
    assert not state.sampling_state.position.any()
    assert not state.next_embeddings.any() and (state.history == -1).all()
    params = SamplingParams(temperature=0.7, top_k=5, top_p=0.8, seed=42)
    for instance in (state, reference):
        add_sampling(instance, 0, params)
        instance.apply_staged_writes()
    hidden = torch.randn(1, 16)
    _, actual = run(state, [0], hidden)
    _, expected = run(reference, [0], hidden)
    assert torch.equal(codes(actual), codes(expected))


@torch.inference_mode()
def test_official_rvq_filter_probabilities_with_repetition_and_delayed_min_frames(monkeypatch):
    root = os.environ.get("ZONOS3_REFERENCE_ROOT")
    if not root:
        pytest.skip("Set ZONOS3_REFERENCE_ROOT for official RVQ sampling parity")
    source = Path(root) / "model/src/zonos3/tts/sampler.py"
    import sys
    from types import ModuleType

    module = ModuleType("_zonos3_official_sampler")
    logging.getLogger(module.__name__).setLevel(logging.WARNING)
    sys.modules[module.__name__] = module
    try:
        exec(compile(source.read_text(), str(source), "exec"), module.__dict__)
        state = make_state(2)
        c = state.model.config
        params = SamplingParams(
            temperature=0.7,
            top_k=5,
            top_p=0.8,
            min_p=0.05,
            repetition_penalty=1.4,
            seed=23,
            extra_args={ZONOS3_SAMPLING_KEY: {"repetition_window": 5, "repetition_codebooks": 6, "min_frames": 2}},
        )
        for slot in range(2):
            add_sampling(state, slot, params)
        state.apply_staged_writes()
        positions = torch.tensor([4, 6])
        state.sampling_state.position[:2] = positions
        ref_history = torch.full((2, 10, 5), -1, dtype=torch.int64)
        for slot, length in enumerate(positions.tolist()):
            history = torch.randint(0, c.audio_vocab, (10, length))
            state.history[slot, :, :length] = history
            width = min(length, 5)
            ref_history[slot, :6, -width:] = history[:6, -width:]
        ref_history[ref_history >= c.codebook_size] = -1
        logits = torch.randn(2, 10, c.audio_vocab)
        actual = state.probabilities(logits, torch.arange(2), 5)
        multinomial, captured = torch.multinomial, []

        def draw(probs, num_samples, **kwargs):
            captured.append(probs)
            return multinomial(probs, num_samples, **kwargs)

        monkeypatch.setattr(torch, "multinomial", draw)
        module.sample_tts(
            logits,
            torch.full((2,), 0.7),
            torch.full((2,), 5),
            torch.full((2,), 0.8),
            torch.full((2,), 0.05),
            codebook_size=c.codebook_size,
            audio_pad_id=c.audio_pad_id,
            stop_token_ids=torch.full((2,), c.eoa_id),
            repetition_token_ids=ref_history,
            repetition_penalties=torch.full((2,), 1.4),
            stop_blocked=positions[:, None] < 2 + torch.arange(10),
            generators=[torch.Generator().manual_seed(23), torch.Generator().manual_seed(24)],
        )
        torch.testing.assert_close(actual, torch.stack(captured), atol=1e-7, rtol=2e-5)
    finally:
        sys.modules.pop(module.__name__, None)
