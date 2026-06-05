# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn

from vllm_omni.model_executor.models.fish_speech.configuration_fish_speech import (
    FishSpeechFastARConfig,
    FishSpeechSlowARConfig,
)
from vllm_omni.model_executor.models.fish_speech.fish_speech_fast_ar import FishSpeechFastAR
from vllm_omni.model_executor.models.fish_speech.fish_speech_slow_ar import (
    FishSpeechSlowARForConditionalGeneration,
)


def test_fast_ar_reuses_dense_position_id_buffer(monkeypatch):
    fast_config = FishSpeechFastARConfig(
        vocab_size=16,
        num_codebooks=4,
        dim=8,
        n_head=2,
        n_local_heads=1,
        head_dim=4,
        n_layer=1,
        intermediate_size=16,
        max_seq_len=5,
    )
    slow_config = FishSpeechSlowARConfig(
        vocab_size=32,
        dim=8,
        n_head=2,
        n_local_heads=1,
        head_dim=4,
        n_layer=1,
        intermediate_size=16,
        codebook_size=16,
        num_codebooks=4,
        semantic_begin_id=4,
        semantic_end_id=19,
    )
    fast_ar = object.__new__(FishSpeechFastAR)
    nn.Module.__init__(fast_ar)
    fast_ar.config = fast_config
    fast_ar.slow_ar_config = slow_config
    fast_ar.fast_project_in = nn.Identity()
    fast_ar.fast_embeddings = nn.Embedding(fast_config.vocab_size, fast_config.hidden_size)
    fast_ar.fast_output = nn.Linear(fast_config.hidden_size, fast_config.vocab_size, bias=False)
    fast_ar.fast_norm = nn.Identity()
    fast_ar._num_codebooks = fast_config.num_codebooks
    fast_ar._fast_dim = fast_config.hidden_size
    fast_ar._embed_buf = None
    fast_ar._pos_ids = None
    fast_ar._k_cache = None
    fast_ar._v_cache = None
    fast_ar._compiled_model_fwd = None
    fast_ar._compile_attempted = True
    fast_ar._compile_failed = False
    fast_ar._disable_compile_for_graph = False

    seen_position_ids: list[torch.Tensor] = []

    def fake_run_model_one(step_input, step_pos_ids, cache_pos):
        seen_position_ids.append(step_pos_ids)
        return torch.zeros(
            step_input.shape[0],
            fast_config.hidden_size,
            dtype=step_input.dtype,
            device=step_input.device,
        )

    monkeypatch.setattr(fast_ar, "_run_model_one", fake_run_model_one)

    hidden = torch.zeros(3, slow_config.hidden_size, dtype=torch.float32)
    semantic = torch.full((3,), slow_config.semantic_begin_id, dtype=torch.long)
    fast_ar(hidden, semantic, do_sample=False)

    assert fast_ar._pos_ids is not None
    assert fast_ar._pos_ids.shape == (3, fast_config.num_codebooks + 1)
    assert len(seen_position_ids) == fast_config.num_codebooks
    for step, step_pos_ids in enumerate(seen_position_ids):
        assert step_pos_ids.shape == (3,)
        assert step_pos_ids.untyped_storage().data_ptr() == fast_ar._pos_ids.untyped_storage().data_ptr()
        assert step_pos_ids.stride() == (fast_ar._pos_ids.stride(0),)
        expected = torch.full((3,), step)
        assert torch.equal(step_pos_ids.cpu(), expected)


def test_talker_mtp_does_not_mutate_input():
    model = object.__new__(FishSpeechSlowARForConditionalGeneration)
    nn.Module.__init__(model)
    model._semantic_begin_id = 4
    model._semantic_end_id = 19
    model._codebook_size = 16
    model._num_codebooks = 4
    model.codebook_embeddings = nn.Embedding(model._num_codebooks * model._codebook_size, 8)
    model.fast_ar = lambda **_: torch.tensor(
        [
            [0, 1, 2, 3],
            [15, 14, 13, 12],
        ],
        dtype=torch.long,
    )
    input_ids = torch.tensor([4, 2], dtype=torch.long)
    input_embeds = torch.randn(2, 8)
    input_embeds_before = input_embeds.clone()
    last_hidden = torch.randn(2, 8, dtype=torch.bfloat16)
    text_step = torch.zeros(2, 8, dtype=torch.bfloat16)

    out1, _ = FishSpeechSlowARForConditionalGeneration.talker_mtp(
        model,
        input_ids,
        input_embeds,
        last_hidden,
        text_step,
    )

    assert torch.equal(input_embeds, input_embeds_before)
    assert out1.untyped_storage().data_ptr() != input_embeds.untyped_storage().data_ptr()


def test_talker_mtp_forwards_sampling_params():
    model = object.__new__(FishSpeechSlowARForConditionalGeneration)
    nn.Module.__init__(model)
    model._semantic_begin_id = 4
    model._semantic_end_id = 19
    model._codebook_size = 16
    model._num_codebooks = 4
    model.codebook_embeddings = nn.Embedding(model._num_codebooks * model._codebook_size, 8)
    seen = {}

    def fake_fast_ar(**kwargs):
        seen.update(kwargs)
        return torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

    model.fast_ar = fake_fast_ar
    input_ids = torch.tensor([4], dtype=torch.long)
    input_embeds = torch.randn(1, 8)
    last_hidden = torch.randn(1, 8, dtype=torch.bfloat16)
    text_step = torch.zeros(1, 8, dtype=torch.bfloat16)

    FishSpeechSlowARForConditionalGeneration.talker_mtp(
        model,
        input_ids,
        input_embeds,
        last_hidden,
        text_step,
        do_sample=False,
        temperature=0.1,
        top_k=5,
        top_p=0.7,
    )

    assert seen["do_sample"] is False
    assert seen["temperature"] == 0.1
    assert seen["top_k"] == 5
    assert seen["top_p"] == 0.7
