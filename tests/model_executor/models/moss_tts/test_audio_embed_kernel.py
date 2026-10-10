# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Preserve native embedding semantics and ownership across fused launches."""

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts.audio_embed_kernel import audio_embed

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows", [0, 1, 8, 127, 513])
@pytest.mark.parametrize("strided", [False, True])
def test_fused_embedding_matches_native_padding_clamping_and_reduction(dtype, rows, strided):
    torch.manual_seed(17)
    nq, vocab, hidden = 32, 128, 320
    weights = torch.randn(nq, vocab, hidden, device="cuda", dtype=dtype)
    codes = torch.randint(-2, vocab + 3, (rows, nq * (2 if strided else 1)), device="cuda")
    if strided:
        codes = codes[:, ::2]
    if rows:
        codes[0].fill_(vocab)
    valid = codes.ne(vocab)
    safe = codes.masked_fill(~valid, 0).clamp(0, vocab - 1)
    gathered = weights[torch.arange(nq, device="cuda")[:, None], safe.t()]
    expected = (gathered * valid.t().unsqueeze(-1)).sum(0)
    actual = audio_embed(codes, weights, vocab)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    saved = actual.clone()
    audio_embed(torch.zeros_like(codes), weights, vocab)
    torch.testing.assert_close(actual, saved, rtol=0, atol=0)


def test_compiled_embedding_has_static_shape_and_owned_outputs():
    weights = torch.randn(32, 128, 320, device="cuda", dtype=torch.bfloat16)
    codes = torch.ones(8, 32, device="cuda", dtype=torch.long)
    compiled = torch.compile(audio_embed, fullgraph=True, backend="aot_eager")
    expected = audio_embed(codes, weights, 128)
    actual = compiled(codes, weights, 128)
    codes.fill_(2)
    compiled(codes, weights, 128)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
