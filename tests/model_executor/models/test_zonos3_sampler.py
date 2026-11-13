# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import torch
from vllm import SamplingParams
from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import SamplingStates
from vllm.v1.worker.gpu.states import RequestState

from tests.model_executor.models.test_zonos3_modeling import small_ar_config
from vllm_omni.model_executor.models.zonos3.modeling_zonos3_mini import MultiEmbedding, MultiOutputHead
from vllm_omni.model_executor.models.zonos3.sampler import (
    ZONOS3_SAMPLING_KEY,
    Zonos3Sampler,
    Zonos3SamplingOptions,
    finalize_zonos3_codes,
)
from vllm_omni.model_executor.models.zonos3.sampling_state import CONTINUE, EOS, Zonos3ARSamplingState
from vllm_omni.worker_v2.omni_sampler import sample_with_output

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def cpu_non_uva_allocator(monkeypatch):
    # vLLM's GPU buffer helper reports UVA available on CPU, but its pinned
    # allocation crashes in this macOS CPU build. Exercise the upstream
    # NonUvaBuffer path; sampling, settings and lifecycle logic remain real.
    monkeypatch.setattr("vllm.v1.worker.gpu.buffer_utils.is_uva_available", lambda: False)


def make_sampler(capacity=4):
    torch.manual_seed(42)
    config = small_ar_config()
    base = Mock(spec=Sampler)
    base.sampling_states = SamplingStates(capacity, 3)
    base.req_states = MagicMock(spec=RequestState)
    base.req_states.index_to_req_id = {}
    base.req_states.prefill_len = UvaBackedTensor(capacity, dtype=torch.int32)
    base.req_states.prefill_len.np.fill(1)
    base.req_states.prefill_len.copy_to_uva()
    base.add_request.side_effect = base.sampling_states.add_request
    base.apply_staged_writes.side_effect = base.sampling_states.apply_staged_writes
    state = Zonos3ARSamplingState(capacity, config.eoa_id, config.audio_pad_id, "cpu")
    state.has_audio_prefix.fill_(True)
    head, embed = MultiOutputHead(config), MultiEmbedding(config)
    with torch.no_grad():
        head.weight.normal_(std=0.2)
    feedback = torch.zeros((capacity, config.dim))
    return Zonos3Sampler(base, config, embed, head, state, feedback, 128)


def make_batch(slots, seq_lens=None, prefill_len=1):
    batch = InputBatch.make_dummy(len(slots), len(slots), InputBuffers(len(slots), len(slots), torch.device("cpu")))
    batch.idx_mapping = torch.tensor(slots, dtype=torch.int64)
    batch.idx_mapping_np = np.array(slots, dtype=np.intp)
    batch.expanded_idx_mapping = batch.idx_mapping
    if seq_lens is not None:
        batch.seq_lens[:] = torch.tensor(seq_lens)
    requests = MagicMock(spec=RequestState)
    requests.prefill_len = UvaBackedTensor(max(slots) + 1, dtype=torch.int32)
    requests.prefill_len.np.fill(prefill_len)
    requests.prefill_len.copy_to_uva()
    return batch, requests


def greedy_params(**extra):
    return SamplingParams(temperature=0, stop_token_ids=[EOS], extra_args={ZONOS3_SAMPLING_KEY: extra})


def run(sampler, slots, hidden, seq_lens=None, prefill_len=1):
    batch, requests = make_batch(slots, seq_lens, prefill_len)
    standard = Mock(side_effect=AssertionError("standard logical-token sampler must not resample RVQ output"))
    result = sample_with_output(sampler, standard, hidden, batch, requests, None)
    standard.assert_not_called()
    return result


@torch.inference_mode()
def test_parallel_codebooks_one_scheduler_token_and_gpu_feedback_contract():
    sampler = make_sampler(2)
    for slot in range(2):
        sampler.add_request(slot, greedy_params())
    sampler.apply_staged_writes()
    hidden = torch.randn(2, 16)
    result = run(sampler, [1, 0], hidden)
    assert result.sampler_output.sampled_token_ids.shape == (2, 1)
    assert result.num_sampled.tolist() == [1, 1]
    codes = result.multimodal_outputs["codes"]["audio"][:, 0]
    assert codes.shape == (2, 10) and not (codes == sampler.config.audio_pad_id).any()
    assert result.owns_multimodal_outputs and not result.include_hidden_states
    frame = torch.cat((codes, torch.full((2, 1), sampler.config.text_vocab)), -1)
    torch.testing.assert_close(sampler.next_embeddings[[1, 0]], sampler.embedder(frame))
    materialized = result.finalize_multimodal(result.multimodal_outputs, [1, 0])
    assert materialized.inter_stage[0]["codes"]["audio"].shape == (1, 10)
    assert materialized.inter_stage[1] is None


@torch.inference_mode()
def test_rvq_top_k_is_not_clamped_to_three_logical_tokens():
    sampler = make_sampler(1)
    sampler.add_request(0, SamplingParams(top_k=80, seed=17))
    sampler.apply_staged_writes()
    assert sampler.base_sampler.sampling_states.top_k.np[0] == 3
    assert sampler.settings.top_k.np[0] == sampler.config.audio_vocab == 10


@torch.inference_mode()
def test_chunked_prefill_does_not_advance_physical_state_or_rng():
    sampler, single = make_sampler(1), make_sampler(1)
    params = SamplingParams(temperature=0.7, top_k=5, top_p=0.8, seed=27)
    for instance in (sampler, single):
        instance.add_request(0, params)
        instance.apply_staged_writes()
    hidden = torch.randn(1, 16)
    first = run(sampler, [0], hidden, [2], prefill_len=5)
    assert first.num_sampled.item() == 0 and sampler.state.position.item() == 0
    assert first.finalize_multimodal(first.multimodal_outputs, [0]).inter_stage == [None]
    actual = run(sampler, [0], hidden, [5], prefill_len=5)
    expected = run(single, [0], hidden)
    assert torch.equal(actual.multimodal_outputs["codes"]["audio"], expected.multimodal_outputs["codes"]["audio"])


@torch.inference_mode()
def test_tail_drain_logical_eos_is_independent_and_absorbs_async_extra_step():
    sampler = make_sampler(2)
    sampler.head.weight.zero_()
    for cb in range(10):
        sampler.head.weight[cb * sampler.config.audio_vocab + sampler.config.eoa_id, 0] = 12
    for slot in range(2):
        sampler.add_request(slot, greedy_params())
    sampler.apply_staged_writes()
    for step in range(15):
        hidden = torch.zeros((2, 16))
        hidden[:, 0] = torch.tensor([1.0, -1.0 if step < 4 else 1.0])
        result = run(sampler, [0, 1], hidden)
        assert result.sampler_output.sampled_token_ids[:, 0].tolist() == [
            EOS if step >= 10 else CONTINUE,
            EOS if step >= 14 else CONTINUE,
        ]
    assert sampler.state.position.tolist() == [11, 15]
    assert result.finalize_multimodal(result.multimodal_outputs, [1, 1]).inter_stage[0] is None


@pytest.mark.parametrize("concurrency", [1, 4, 8, 16, 32])
@torch.inference_mode()
def test_seeded_rvq_sampling_is_invariant_to_batch_reordering(concurrency):
    batched = make_sampler(concurrency)
    singles = [make_sampler(1) for _ in range(concurrency)]
    for slot, single in enumerate(singles):
        params = SamplingParams(
            temperature=0.7 + 0.01 * slot,
            top_k=5,
            top_p=0.85,
            seed=100 + slot,
            repetition_penalty=1.4,
            extra_args={ZONOS3_SAMPLING_KEY: {"repetition_window": 4}},
        )
        single.add_request(0, params)
        single.apply_staged_writes()
        batched.add_request(slot, params)
    batched.apply_staged_writes()
    generator = torch.Generator().manual_seed(19)
    for step in range(4):
        hidden = torch.randn(concurrency, 16, generator=generator)
        order = list(reversed(range(concurrency))) if step % 2 else list(range(concurrency))
        actual = run(batched, order, hidden[order]).multimodal_outputs["codes"]["audio"]
        for row, slot in enumerate(order):
            expected = run(singles[slot], [0], hidden[slot : slot + 1]).multimodal_outputs["codes"]["audio"]
            assert torch.equal(actual[row], expected[0])


@torch.inference_mode()
def test_slot_reuse_resets_history_and_output_snapshot_stays_owned():
    sampler = make_sampler(1)
    params = SamplingParams(seed=21)
    sampler.add_request(0, params)
    sampler.apply_staged_writes()
    output = run(sampler, [0], torch.randn(1, 16)).multimodal_outputs
    snapshot = output["codes"]["audio"].clone()
    sampler.add_request(0, params)
    sampler.apply_staged_writes()
    assert sampler.state.position.item() == 0 and (sampler.history == -1).all()
    run(sampler, [0], torch.randn(1, 16))
    assert torch.equal(output["codes"]["audio"], snapshot)


def test_sampler_rejects_unsupported_logical_constraints_and_dirty_staging():
    sampler = make_sampler(1)
    with pytest.raises(ValueError, match="Unsupported logical-token"):
        sampler.add_request(0, SamplingParams(logprobs=2))
    sampler.add_request(0, greedy_params())
    with pytest.raises(RuntimeError, match="staged writes"):
        run(sampler, [0], torch.zeros(1, 16))
    sampler.apply_staged_writes()
    with pytest.raises(ValueError, match="eoa_policy"):
        Zonos3SamplingOptions.from_params(greedy_params(eoa_policy="bad"), sampler.config, 128)
    with pytest.raises(ValueError, match="at most one"):
        finalize_zonos3_codes(
            {
                "codes": {"audio": torch.zeros(1, 1, 10)},
                "meta": {"codec_frame_valid": torch.ones(1), "codec_eos_valid": torch.zeros(1, dtype=torch.bool)},
            },
            [2],
        )


@torch.inference_mode()
def test_native_startup_warmup_uses_sample_step_head_and_sanitizes_only_warmup_requests():
    sampler = make_sampler(2)
    calls = []
    handle = sampler.head.register_forward_pre_hook(lambda _, args: calls.append(args[0].shape))
    try:
        assert sampler.warmup(torch.randn(2, 16))
    finally:
        handle.remove()
    assert calls == [torch.Size([2, 16])]
    sampler.base_sampler.req_states.index_to_req_id[0] = "_warmup_request"
    sampler.add_request(0, SamplingParams.for_sampler_warmup())
    assert sampler.options[0].temperature == 0.9
    sampler.base_sampler.req_states.index_to_req_id[0] = "real-request"
    with pytest.raises(ValueError, match="Unsupported logical-token"):
        sampler.add_request(0, SamplingParams.for_sampler_warmup())


@torch.inference_mode()
def test_generic_warmup_grammar_is_ignored_only_for_native_warmup_ids():
    sampler = make_sampler(1)
    sampler.add_request(0, greedy_params())
    sampler.apply_staged_writes()
    data, requests = make_batch([0])
    data.req_ids = ["_warmup_0_"]
    grammar = GrammarOutput(
        structured_output_request_ids=data.req_ids, grammar_bitmask=np.full((1, 1), -1, dtype=np.int32)
    )
    output = sampler.sample_step(torch.randn(1, 16), data, requests, grammar, Mock())
    assert output.num_sampled.tolist() == [1]
    data.req_ids = ["real"]
    with pytest.raises(ValueError, match="grammar"):
        sampler.sample_step(torch.randn(1, 16), data, requests, grammar, Mock())


@torch.inference_mode()
def test_official_rvq_filter_probabilities_with_repetition_and_delayed_min_frames(monkeypatch):
    root = os.environ.get("ZONOS3_REFERENCE_ROOT")
    if not root:
        pytest.skip("Set ZONOS3_REFERENCE_ROOT for official RVQ sampling parity")
    source = Path(root) / "model/src/zonos3/tts/sampler.py"
    oracle = {"__name__": "_zonos3_official_sampler"}
    # The module has no server imports. Its DEBUG-only CUDA diagnostics are
    # disabled so the unchanged sampling body can run on the CPU oracle path.
    logging.getLogger(oracle["__name__"]).setLevel(logging.WARNING)
    import sys
    from types import ModuleType

    module = ModuleType(oracle["__name__"])
    sys.modules[module.__name__] = module
    try:
        exec(compile(source.read_text(), str(source), "exec"), module.__dict__)
        sampler = make_sampler(2)
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
            sampler.add_request(slot, params)
        sampler.apply_staged_writes()
        positions = torch.tensor([4, 6])
        sampler.state.position[:] = positions
        ref_history = torch.full((2, 10, 5), -1, dtype=torch.int64)
        for slot, length in enumerate(positions.tolist()):
            history = torch.randint(0, sampler.config.audio_vocab, (10, length))
            sampler.history[slot, :, :length] = history
            width = min(length, 5)
            ref_history[slot, :6, -width:] = history[:6, -width:]
        ref_history[ref_history >= sampler.config.codebook_size] = -1
        logits = torch.randn(2, 10, sampler.config.audio_vocab)
        actual = sampler.probabilities(logits, torch.arange(2), 5)
        multinomial = torch.multinomial
        captured = []

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
            codebook_size=sampler.config.codebook_size,
            audio_pad_id=sampler.config.audio_pad_id,
            stop_token_ids=torch.full((2,), sampler.config.eoa_id),
            repetition_token_ids=ref_history,
            repetition_penalties=torch.full((2,), 1.4),
            stop_blocked=positions[:, None] < 2 + torch.arange(10),
            generators=[torch.Generator().manual_seed(23), torch.Generator().manual_seed(24)],
        )
        torch.testing.assert_close(actual, torch.stack(captured), atol=1e-7, rtol=2e-5)
    finally:
        sys.modules.pop(module.__name__, None)
