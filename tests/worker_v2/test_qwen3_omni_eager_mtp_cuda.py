# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in CUDA parity for Qwen3-Omni's shared eager-MTP capability adapter."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import Qwen3OmniMoeForConditionalGeneration
from vllm_omni.worker_v2.model_states.eager_mtp import EagerMTPState

pytestmark = pytest.mark.core_model


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [1025, 2048])
@pytest.mark.parametrize("first_audio_valid", [False, True])
@torch.inference_mode()
def test_qwen_eager_kernels_match_native_indexing(dtype, hidden_size, first_audio_valid):
    from vllm_omni.worker_v2.model_states.eager_mtp_kernels import eager_post, eager_pre

    device = torch.device("cuda", 0)
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_stage = "talker"
    model._codec_codebook_size = 8
    embedding = nn.Embedding(12, hidden_size, device=device, dtype=dtype)
    model.talker = SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(codec_embedding=embedding)))
    state = EagerMTPState(
        SimpleNamespace(
            model=model,
            vllm_config=SimpleNamespace(cache_config=SimpleNamespace(enable_prefix_caching=False)),
        )
    )
    n, nq = 4, 16
    # Nonidentity batch-row and request-slot maps; mixed final prefill/decode.
    rows = torch.tensor([2, 0, 3, 1], device=device)
    slots = torch.tensor([5, 1, 4, 0], device=device)
    prefill = torch.tensor([0, 1, 0, 0], device=device)
    scheduled = torch.tensor([1, 1, 0, 1], device=device)
    meta = torch.stack((rows, slots, prefill, scheduled, torch.zeros_like(rows)))
    qsl = torch.tensor([0, 2, 3, 5, 6], device=device)
    # Nonunit sample stride also occurs when extra sampler columns exist.
    sampled = torch.tensor([[3, 99], [8, 99], [4, 99], [5, 99]], device=device)
    hidden = torch.randn(6, hidden_size, device=device, dtype=dtype)
    ids = torch.empty(n, device=device, dtype=torch.long)
    embs, mtp_hidden, step = [torch.empty(n, hidden_size, device=device, dtype=dtype) for _ in range(3)]
    codes_out = torch.full((6, nq), -17, device=device, dtype=torch.long)
    assert state._fast_path_ok(codes_out, hidden)
    last, layer0 = eager_pre(
        meta,
        n,
        qsl,
        sampled,
        hidden,
        state._embedding_weight(),
        ids,
        embs,
        mtp_hidden,
        step,
    )
    expected_last = qsl.index_select(0, rows + 1).long() - 1
    expected_ids = sampled[:, 0].index_select(0, rows)
    for got, expected in [
        (last, expected_last),
        (layer0, expected_ids),
        (ids, expected_ids),
        (embs, embedding(expected_ids)),
        (mtp_hidden, hidden.index_select(0, expected_last)),
        (step, torch.zeros_like(step)),
    ]:
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    # Use real-shaped synthetic predictor outputs: isolate staging arithmetic
    # from stochastic generation while checking all rows and untouched slots.
    frames = torch.randn(n, hidden_size, device=device, dtype=dtype)
    codes = torch.arange(n * nq, device=device).reshape(n, nq)
    eager = torch.full((7, hidden_size), -9, device=device, dtype=dtype)
    input_ids = torch.tensor([0, 9, 2, 1, 9, 6], device=device)
    validity = torch.full((6,), -1, device=device, dtype=torch.int8)
    fa = torch.full((6,), -1, device=device, dtype=torch.int8)
    fa_valid = (
        torch.tensor([True, False, False, False, True, True, False], device=device) if first_audio_valid else None
    )
    expected_eager, expected_codes, expected_validity, expected_fa = [
        t.clone() for t in (eager, codes_out, validity, fa)
    ]
    expected_eager.index_copy_(0, slots, frames)
    expected_codes.index_copy_(0, expected_last, codes)
    expected_valid = model.mtp_frame_valid(expected_ids) & (
        model.mtp_frame_valid(input_ids.index_select(0, expected_last)) | prefill.bool()
    )
    expected_validity.index_copy_(0, expected_last, expected_valid.to(validity.dtype))
    expected_scheduled = (
        scheduled * fa_valid.index_select(0, slots) if fa_valid is not None else torch.zeros_like(scheduled)
    )
    expected_fa.index_copy_(0, expected_last, expected_scheduled.to(fa.dtype))
    valid = eager_post(
        meta,
        n,
        last,
        layer0,
        frames,
        eager,
        codes,
        codes_out,
        input_ids,
        validity,
        fa,
        fa_valid,
        state._codebook_size(),
    )
    for got, expected in [
        (eager, expected_eager),
        (codes_out, expected_codes),
        (validity, expected_validity),
        (valid, expected_valid),
        (fa, expected_fa),
    ]:
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
