# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reference batching preserves coordinates and asynchronous source ownership."""

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from tests.model_executor.models.moss_tts.test_local_model_state import _batch, _state
from vllm_omni.model_executor.models.moss_tts.local_model_state import MossLocalModelState
from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import MossTTSLocalTalkerForGeneration

pytestmark = pytest.mark.core_model


@dataclass
class _PrefillPositions:
    prompt_len: np.ndarray
    num_computed_tokens: np.ndarray


@pytest.mark.cpu
@pytest.mark.parametrize("flat", [False, True])
def test_prefill_reuses_text_embeddings_and_noncontiguous_reference_offsets(mocker, flat):
    state = _state(MossLocalModelState, torch.device("cpu"))
    state._batch_prefill = True
    codes = torch.tensor([[1, 2], [3, 4], [5, 6], [2, 3]])
    for slot, offset in [(3, 1), (0, 2)]:
        state.intermediate_buffer.buffers[slot] = {
            "req_id": str(slot),
            "codes": {"ref": codes.flatten() if flat else codes},
            "ref_offset": offset,
        }
    batch = _batch(torch.device("cpu"), [3, 0], [2, 1])
    req = _PrefillPositions(prompt_len=np.full(5, 8), num_computed_tokens=np.zeros(5))
    req.num_computed_tokens[[3, 0]] = [1, 2]
    embeds = state._static_inputs_embeds[:3]
    expected = state.model.embed_input_ids(batch.input_ids) + state.model._audio_embed(codes[[1, 2, 2]])
    embed = mocker.spy(state.model.model.embed_tokens, "forward")
    with torch.inference_mode():
        state.run_preprocess(batch, {"input_ids": batch.input_ids, "inputs_embeds": embeds}, req)
    torch.testing.assert_close(embeds, expected, rtol=0, atol=0)
    assert embed.call_count == 1
    assert state.intermediate_buffer.buffers[3]["ref_offset"] == 3
    assert state.intermediate_buffer.buffers[0]["ref_offset"] == 3


@pytest.mark.cpu
@pytest.mark.parametrize("batch_prefill", [False, True])
@pytest.mark.parametrize("cached,count", [(2, 1), (1, 2)])
def test_cached_prefix_prefill_uses_absolute_reference_position(batch_prefill, cached, count):
    state = _state(MossLocalModelState, torch.device("cpu"))
    state._batch_prefill = batch_prefill
    codes = torch.tensor([[1, 2], [3, 4], [5, 6], [2, 3]])
    state.intermediate_buffer.buffers[0] = {"req_id": "prefix-hit", "codes": {"ref": codes}}
    batch = _batch(torch.device("cpu"), [0], [count])
    req = _PrefillPositions(prompt_len=np.full(5, 4), num_computed_tokens=np.full(5, cached))
    embeds = state._static_inputs_embeds[:count]
    expected = state.model.embed_input_ids(batch.input_ids) + state.model._audio_embed(codes[cached : cached + count])
    with torch.inference_mode():
        state.run_preprocess(batch, {"input_ids": batch.input_ids, "inputs_embeds": embeds}, req)
    torch.testing.assert_close(embeds, expected, rtol=0, atol=0)
    assert state.intermediate_buffer.buffers[0]["ref_offset"] == cached + count


@pytest.mark.cpu
def test_scalar_prefill_uses_cached_prompt_position():
    state = _state(MossLocalModelState, torch.device("cpu"))
    codes = torch.tensor([[1, 2], [3, 4], [5, 6], [2, 3]])
    ids = torch.tensor([2])
    expected = state.model.embed_input_ids(ids) + state.model._audio_embed(codes[2:3])
    with torch.inference_mode():
        _, actual, updates = state.model.preprocess(
            ids, None, codes={"ref": codes}, _omni_is_prefill=True, _omni_num_computed_tokens=2
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert updates["ref_offset"] == 3


@pytest.mark.cpu
@pytest.mark.parametrize("batch_prefill", [False, True])
@pytest.mark.parametrize("omit_trailing_pad", [False, True])
def test_prefill_chunk_crosses_reference_end(batch_prefill, omit_trailing_pad):
    state = _state(MossLocalModelState, torch.device("cpu"))
    state._batch_prefill = batch_prefill
    # codes.ref uses prompt coordinates, including PAD rows for text tokens.
    codes = torch.tensor([[8, 8], [1, 2], [3, 4], [5, 6], [8, 8], [8, 8]])
    if omit_trailing_pad:
        codes = codes[:4]
    state.model._audio_embed = lambda rows: rows[:, :1].masked_fill(rows[:, :1] == 8, 0).expand(-1, 4) / 8
    state.intermediate_buffer.buffers[0] = {"req_id": "reference-boundary", "codes": {"ref": codes}}
    req = _PrefillPositions(prompt_len=np.full(5, 6), num_computed_tokens=np.zeros(5))
    outputs = []
    with torch.inference_mode():
        for start, count in [(0, 2), (2, 4)]:
            req.num_computed_tokens[0] = start
            batch = _batch(torch.device("cpu"), [0], [count])
            embeds = state._static_inputs_embeds[:count]
            state.run_preprocess(batch, {"input_ids": batch.input_ids, "inputs_embeds": embeds}, req)
            outputs.append(embeds.clone())
    expected = state.model.embed_input_ids(torch.full((6,), 2))
    expected[1:4] += torch.tensor([1, 3, 5]).reshape(-1, 1) / 8
    torch.testing.assert_close(torch.cat(outputs), expected, rtol=0, atol=0)
    assert state.intermediate_buffer.buffers[0]["ref_offset"] == 6


@pytest.mark.cuda
def test_pending_reference_uploads_survive_staging_reuse_and_overflow():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    state = _state(MossLocalModelState, torch.device("cuda"))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    outputs = []
    with torch.cuda.stream(stream), torch.inference_mode():
        # Keep the first H2D pending while CPU prepares all subsequent batches.
        torch.cuda._sleep(100_000_000)
        for value in range(1, 8):
            embeds = torch.zeros(4, 4, device="cuda", dtype=state.dtype)
            codes = torch.full((2, 2), value)
            state._apply_reference_batch(embeds, [(1, codes)])
            outputs.append((embeds, value))
            codes.fill_(-1)
    stream.synchronize()
    assert len(state._prefill_staging) <= 3
    for output, value in outputs:
        expected = torch.zeros_like(output)
        expected[1:3] = value / 8
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.cuda
def test_batched_real_audio_embedding_matches_scalar_bfloat16_reduction():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    state = _state(MossLocalModelState, torch.device("cuda"))
    model = state.model
    model.n_vq = 12
    model.audio_vocab_size = model.audio_pad_token_id = 1024
    model._stacked_audio_emb_w = torch.randn(12, 1024, 3072, device="cuda", dtype=torch.bfloat16)
    model._audio_embed = MossTTSLocalTalkerForGeneration._audio_embed.__get__(model)
    state._static_inputs_embeds = torch.zeros(512, 3072, device="cuda", dtype=torch.bfloat16)
    references = [(start, torch.randint(0, 1025, (count, 12))) for start, count in [(0, 169), (201, 11), (220, 257)]]
    with torch.inference_mode():
        actual = torch.randn_like(state._static_inputs_embeds)
        expected = actual.clone()
        for start, codes in references:
            expected[start : start + len(codes)] += model._audio_embed(codes.to("cuda"))
        state._apply_reference_batch(actual, references)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
