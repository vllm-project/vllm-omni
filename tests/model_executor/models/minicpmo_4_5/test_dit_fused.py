# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused Code2Wav DiT body matches the ragged reference."""

import pytest
import torch
import torch.nn as nn

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import BatchedToken2Wav
from vllm_omni.model_executor.models.minicpmo_4_5.dit_fused import blocks_forward_chunk_fused, supports_fused_body

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _upstream_dit(device: str) -> nn.Module:
    for name in ("cosyvoice2.flow.decoder_dit", "stepaudio2.cosyvoice2.flow.decoder_dit"):
        try:
            import importlib

            decoder_dit = importlib.import_module(name)
            break
        except ImportError:
            pass
    else:
        decoder_dit = pytest.importorskip("cosyvoice2.flow.decoder_dit")
    torch.manual_seed(0)
    estimator = decoder_dit.DiT(in_channels=320, out_channels=80, depth=2, num_heads=4, head_dim=16, hidden_size=64)
    with torch.no_grad():
        for parameter in estimator.parameters():
            parameter.normal_(0.0, 0.08)
    return estimator.eval().to(device)


def _run(body, estimator, *, rows: int, frames: int, cached: int, lengths):
    torch.manual_seed(1)
    depth = len(estimator.blocks)
    channels = estimator.blocks[0].conv.block[1].in_channels
    attn = estimator.blocks[0].attn
    device = next(estimator.parameters()).device
    estimator_input = torch.randn(rows, 320, frames, device=device)
    time_embedding = estimator.t_embedder(torch.rand((), device=device).expand(rows)).unsqueeze(1)
    mask = torch.rand(rows, frames, cached + frames, device=device) > 0.3
    mask[..., 0] = True
    cnn = [torch.randn(rows, 2 * channels, 2, device=device) for _ in range(depth)]
    att = [torch.randn(rows, attn.num_heads, cached, 2 * attn.head_dim, device=device) for _ in range(depth)]
    cnn_out = torch.zeros((depth, rows, 2 * channels, 2), device=device)
    att_out = torch.zeros((depth, rows, attn.num_heads, cached + frames, 2 * attn.head_dim), device=device)
    with torch.no_grad():
        out = body(estimator, estimator_input, time_embedding, mask, cnn, att, cnn_out, att_out, lengths)
    return out, cnn_out, att_out


@pytest.mark.parametrize("device", _DEVICES)
def test_fused_body_matches_ragged_body(device: str) -> None:
    estimator = _upstream_dit(device)
    assert supports_fused_body(estimator)
    expected = _run(
        BatchedToken2Wav._blocks_forward_chunk_ragged, estimator, rows=4, frames=6, cached=5, lengths=[6, 4]
    )
    actual = _run(blocks_forward_chunk_fused, estimator, rows=4, frames=6, cached=5, lengths=[6, 4])
    for want, got in zip(expected, actual, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-4, atol=2e-4)


def test_unsupported_estimator_is_refused() -> None:
    estimator = _upstream_dit("cpu")
    estimator.blocks[0].attn.q_norm = nn.Identity()
    assert not supports_fused_body(estimator)
