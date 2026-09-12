# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Real vLLM sampler parity for mixed native-duplex and ordinary chat rows."""

from dataclasses import replace

import pytest
import torch
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.duplex_sampling import DuplexSamplingRow
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration


@pytest.mark.core_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_mixed_native_cuda_sampler_preserves_chat_logprobs_and_rng(monkeypatch):
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.__dict__["sampler"] = Sampler()
    tokens = {"unit_token_id": 1, "listen_token_id": 2, "chunk_eos_token_id": 3}
    model._minicpmo45_native_duplex_token_ids_cache = tokens
    monkeypatch.setattr(model, "_minicpmo45_native_forbidden_token_ids", lambda *a: [1, 3])
    monkeypatch.setattr(model, "_maybe_cut_minicpmo45_native_duplex_text_chunk", lambda sampled, *a: sampled)
    native_rng = torch.Generator(device="cuda").manual_seed(42)
    chat_rng = torch.Generator(device="cuda").manual_seed(43)
    before = native_rng.get_state().clone()
    metadata = SamplingMetadata(
        temperature=torch.tensor([0.0, 0.8], device="cuda"),
        all_greedy=False,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={0: native_rng, 1: chat_rng},
        max_num_logprobs=2,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(2, device="cuda"),
        presence_penalties=torch.zeros(2, device="cuda"),
        repetition_penalties=torch.ones(2, device="cuda"),
        output_token_ids=[[], []],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )
    logits = torch.zeros(2, 128, device="cuda")
    logits[0, 1], logits[0, 2] = 20, 10
    logits[1, 6] = 3
    reference_metadata = replace(
        metadata, generators={row: gen.clone_state() for row, gen in metadata.generators.items()}
    )
    expected = model.sampler(logits.clone(), reference_metadata)
    row = DuplexSamplingRow(0, "native", None, 0, 1, None, 20, temperature=0.0, top_k=-1, top_p=1.0)
    model.prepare_duplex_sampling(logits, metadata, (row,))
    actual = model.sample(logits, metadata)
    assert actual.sampled_token_ids[0].item() == 2  # forbidden unit 1 must not leak through the standard sampler
    torch.testing.assert_close(actual.sampled_token_ids[1], expected.sampled_token_ids[1], rtol=0, atol=0)
    torch.testing.assert_close(
        actual.logprobs_tensors.logprobs[1], expected.logprobs_tensors.logprobs[1], rtol=0, atol=0
    )
    torch.testing.assert_close(native_rng.get_state(), before, rtol=0, atol=0)
    torch.testing.assert_close(chat_rng.get_state(), reference_metadata.generators[1].get_state(), rtol=0, atol=0)
