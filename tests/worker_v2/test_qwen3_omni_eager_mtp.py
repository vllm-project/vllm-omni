# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU capability/orchestration tests; CUDA kernel parity requires GPU validation."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import Qwen3OmniMoeForConditionalGeneration
from vllm_omni.worker_v2.model_states.eager_mtp import EagerMTPState, FrameBatch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def make_state():
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_stage = "talker"
    model._codec_codebook_size = 8
    embed = nn.Embedding(12, 4)
    model.talker = SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(codec_embedding=embed)))
    owner = SimpleNamespace(
        model=model,
        vllm_config=SimpleNamespace(cache_config=SimpleNamespace(enable_prefix_caching=False)),
        _mtp_input_ids=torch.zeros(3, dtype=torch.long),
        _mtp_input_embeds=torch.zeros(3, 4),
        _mtp_hidden=torch.zeros(3, 4),
        _mtp_text_step=torch.zeros(3, 4),
    )
    state = EagerMTPState(owner)
    state.embeds = torch.zeros(4, 4)
    return state, embed


def test_qwen_capability_resolves_codec_embedding_not_text_embedding():
    state, embed = make_state()
    assert state._embedding_weight() is embed.weight
    assert state._codebook_size() == 8
    state.owner.model.model_stage = "thinker"
    assert state._embedding_weight() is None
    assert state._codebook_size() is None


@pytest.mark.parametrize("unsupported", ["tp", "quantized", "strided"])
def test_unsupported_embedding_falls_back(unsupported):
    state, embed = make_state()
    if unsupported == "tp":
        embed.tp_size = 2
    elif unsupported == "quantized":
        embed.quant_method = object()
    else:
        embed.weight = nn.Parameter(torch.zeros(4, 12).t())
    assert state._embedding_weight() is None


def test_fast_path_layout_and_cpu_fallback():
    state, _ = make_state()
    assert not state._fast_path_ok(torch.zeros(3, 2), torch.zeros(3, 4))
    # Exercise layout guards without allocating a CUDA tensor.
    state._fast_ok = True
    assert state._fast_path_ok(torch.zeros(3, 2), torch.zeros(3, 4))
    assert not state._fast_path_ok(torch.zeros(3, 2), torch.zeros(3, 5))
    assert not state._fast_path_ok(torch.zeros(2, 3).t(), torch.zeros(3, 4))
    state._fast_ok = None
    state.owner.vllm_config.cache_config.enable_prefix_caching = True
    assert not state._fast_path_ok(torch.zeros(3, 2), torch.zeros(3, 4))


@pytest.mark.parametrize("prefill", [False, True])
def test_fused_orchestration_preserves_slots_eos_and_first_audio(monkeypatch, prefill):
    from vllm_omni.worker_v2.model_states import eager_mtp_kernels as kernels

    state, embed = make_state()
    owner = state.owner
    state._stage_meta = lambda values, device: torch.as_tensor(values.copy())
    state._stage_scheduled = lambda values, device: torch.as_tensor(values.copy())
    state._first_audio_valid = torch.tensor([False, True, False, True])
    state.first_audio_requests = {"ended"}
    # Request order differs from token spans and storage slots. Sample EOS for
    # one row, and use EOS as the previous input for another asynchronous row.
    batch = FrameBatch([(2, 3, "ended", False), (0, 1, "fresh", prefill), (1, 0, "eos", False)])
    input_batch = SimpleNamespace(num_reqs=3, query_start_loc=torch.tensor([0, 2, 3, 4]))
    inputs = torch.tensor([0, 9, 2, 9])
    sampled = torch.tensor([[3], [8], [4]])
    hidden = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    codes_out = torch.full((4, 2), -1, dtype=torch.long)
    valid_out = torch.zeros(4, dtype=torch.int8)
    first = torch.zeros(4, dtype=torch.int8)
    outputs = {"meta": {"first_audio": first}}

    # CPU references stand in for the two existing CUDA kernels. This checks
    # the adapter's arguments/state flow, not Triton arithmetic or speed.
    def pre(meta, n, qsl, samples, hs, weight, ids, emb, mh, step):
        rows = meta[:n]
        last = qsl[rows + 1] - 1
        layer0 = samples[rows, 0]
        ids.copy_(layer0)
        emb.copy_(weight[layer0])
        mh.copy_(hs[last])
        step.zero_()
        return last, layer0

    def post(meta, n, last, layer0, frames, eager, codes, output, input_ids, validity, fa, fa_valid, vocab, **kwargs):
        slots = meta[n : 2 * n]
        is_prefill = meta[2 * n : 3 * n].bool()
        eager[slots] = frames
        output[last] = codes
        valid = (layer0 >= 0) & (layer0 < vocab) & (((input_ids[last] >= 0) & (input_ids[last] < vocab)) | is_prefill)
        validity[last] = valid.to(validity.dtype)
        if fa is not None:
            fa[last] = (meta[3 * n : 4 * n].bool() & fa_valid[slots]).to(fa.dtype)
        return valid

    monkeypatch.setattr(kernels, "eager_pre", pre)
    monkeypatch.setattr(kernels, "eager_post", post)

    def mtp(slots, ids, embs, hs, step, dispatch, *, req_ids):
        assert slots == [3, 1, 0] and req_ids == ["ended", "fresh", "eos"]
        assert torch.equal(hs, hidden[[3, 1, 2]])
        assert torch.equal(embs, embed.weight[[4, 3, 8]])
        assert not step.any()
        return embs + 10, torch.stack([ids, ids + 1], dim=1)

    owner._mtp_forward = mtp

    def publish(*args):
        state.first_audio_requests.add("fresh")

    state._publish_first_audio = Mock(side_effect=publish)
    state._run_eager_mtp_fused(input_batch, hidden, sampled, outputs, batch, inputs, codes_out, valid_out, None)
    assert valid_out.tolist() == [0, int(prefill), 0, 0]
    assert codes_out.tolist() == [[-1, -1], [3, 4], [8, 9], [4, 5]]
    assert torch.equal(state.embeds[[3, 1, 0]], embed.weight[[4, 3, 8]] + 10)
    assert state.ready == {3: "ended", 1: "fresh", 0: "eos"}
    assert state._publish_first_audio.call_count == int(prefill)
    assert first.tolist() == [0, int(prefill), 0, 1]
