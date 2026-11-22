# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare real clean-refresh layer math and managed caches across window rolls."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.state_cosmos3_nano_sim_bimanual import (
    append_dense_kv_history,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.transformer_cosmos3_nano_sim_bimanual import (
    Cosmos3NanoSimBimanualGenDecoderLayer,
    Cosmos3NanoSimBimanualJointAttention,
    Cosmos3NanoSimBimanualTransformer,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]
BLOCK, WIDTH, HEAD, LAYERS = 16, 64, 64, 3


class DenseAttention(nn.Module):
    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), enable_gqa=True
        ).transpose(1, 2)


def layers(device, dtype):
    """Use production forward methods with small weights; no distributed init."""
    result = nn.ModuleList()
    for _ in range(LAYERS):
        attn = Cosmos3NanoSimBimanualJointAttention.__new__(Cosmos3NanoSimBimanualJointAttention)
        nn.Module.__init__(attn)
        attn.num_heads_local, attn.num_kv_heads_local, attn.head_dim = 4, 2, HEAD
        attn.qk_norm = True
        attn.to_q, attn.to_k, attn.to_v = [nn.Linear(WIDTH, n * HEAD, bias=False) for n in (4, 2, 2)]
        attn.to_out = nn.Linear(4 * HEAD, WIDTH, bias=False)
        attn.norm_q, attn.norm_k = nn.RMSNorm(HEAD), nn.RMSNorm(HEAD)
        attn.norm_q.variance_epsilon = attn.norm_k.variance_epsilon = 1e-6
        attn.attn = DenseAttention()
        layer = Cosmos3NanoSimBimanualGenDecoderLayer.__new__(Cosmos3NanoSimBimanualGenDecoderLayer)
        nn.Module.__init__(layer)
        layer.cross_attention = attn
        layer.input_layernorm = nn.RMSNorm(WIDTH, eps=1e-6)
        layer.post_attention_layernorm = nn.RMSNorm(WIDTH, eps=1e-6)
        layer.mlp = nn.Sequential(nn.Linear(WIDTH, 2 * WIDTH), nn.SiLU(), nn.Linear(2 * WIDTH, WIDTH))
        result.append(layer)
    return result.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def run_forward(
    net,
    h,
    text,
    start,
    *,
    dense=None,
    batched=False,
    window=5,
    sink=0,
    commit=True,
    dense_cache=None,
):
    frames = h.shape[1] // BLOCK
    nulls = tuple(i for i in range(frames) if (start + i) % 3 == 0)
    positions = torch.arange(start * BLOCK, (start + frames) * BLOCK, device=h.device).float()
    phases = positions.view(1, -1, 1, 1) * torch.arange(1, HEAD + 1, device=h.device).view(1, 1, 1, -1) * 0.001
    cos, sin = phases.cos().to(h.dtype), phases.sin().to(h.dtype)
    current = []
    for index, layer in enumerate(net):
        h, k, v = layer(
            h,
            text_k=text[index][0],
            text_v=text[index][1],
            real_text_kv_len=text[index][0].shape[1],
            freqs_cos=cos,
            freqs_sin=sin,
            dense_history=None if dense is None or dense_cache is not None else dense[index],
            dense_joint_kv=None if dense_cache is None else dense_cache.kv[index],
            dense_history_tokens=0 if dense_cache is None else dense_cache.history_length,
            num_frames=frames,
            tokens_per_frame=BLOCK,
            action_tokens_per_frame=2,
            null_action_frame_indexes=nulls,
            clean_history_window=(sink, window) if batched else None,
        )
        current.append((k, v))
    if commit and dense_cache is not None:
        dense = dense_cache.commit(frames * BLOCK)
    elif commit:
        for offset in range(frames):
            dense = append_dense_kv_history(
                dense,
                [
                    (k[:, offset * BLOCK : (offset + 1) * BLOCK], v[:, offset * BLOCK : (offset + 1) * BLOCK])
                    for k, v in current
                ],
                tokens_per_frame=BLOCK,
                sink_frames=sink,
                window_frames=window,
            )
    return h, dense


def compare(*, device, dtype, history, window, sink, frames):
    torch.manual_seed(123)
    net = layers(device, dtype)
    text = [
        (torch.randn(1, 5, 2, HEAD, device=device, dtype=dtype), torch.randn(1, 5, 2, HEAD, device=device, dtype=dtype))
        for _ in net
    ]
    inputs = torch.randn(1, (history + frames + 1) * BLOCK, WIDTH, device=device, dtype=dtype)
    da = db = None
    tolerance = 5e-2 if dtype == torch.bfloat16 else 2e-5
    for i in range(history):
        x = inputs[:, i * BLOCK : (i + 1) * BLOCK]
        _, da = run_forward(net, x, text, i, dense=da, window=window, sink=sink)
        _, db = run_forward(net, x, text, i, dense=db, window=window, sink=sink)
    sequential = []
    for i in range(history, history + frames):
        output, da = run_forward(
            net, inputs[:, i * BLOCK : (i + 1) * BLOCK], text, i, dense=da, window=window, sink=sink
        )
        sequential.append(output)
    actual, db = run_forward(
        net,
        inputs[:, history * BLOCK : (history + frames) * BLOCK],
        text,
        history,
        dense=db,
        batched=True,
        window=window,
        sink=sink,
    )
    torch.testing.assert_close(actual, torch.cat(sequential, dim=1), atol=tolerance, rtol=tolerance)
    assert da is not None and db is not None
    for pair_a, pair_b in zip(da, db, strict=True):
        for ka, kb in zip(pair_a, pair_b, strict=True):
            torch.testing.assert_close(ka, kb, atol=tolerance, rtol=tolerance)
    # The next denoising call consumes the newly committed history.
    x = inputs[:, -BLOCK:]
    oa, _ = run_forward(net, x, text, history + frames, dense=da, window=window, sink=sink, commit=False)
    ob, _ = run_forward(net, x, text, history + frames, dense=db, window=window, sink=sink, commit=False)
    torch.testing.assert_close(oa, ob, atol=tolerance, rtol=tolerance)


@pytest.mark.cpu
@pytest.mark.parametrize("frames", [1, 3, 4])
@pytest.mark.parametrize(
    "history,window,sink", [(0, 5, 0), (2, 5, 0), (4, 5, 0), (8, 5, 0), (8, 2, 0), (8, 2, 1), (0, 1, 2)]
)
def test_batched_clean_cache_matches_sequential_cpu(history, window, sink, frames):
    compare(
        device=torch.device("cpu"),
        dtype=torch.float32,
        history=history,
        window=window,
        sink=sink,
        frames=frames,
    )


@hardware_test(res={"cuda": "L4"})
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("history,window,sink,frames", [(0, 5, 0, 4), (4, 5, 0, 4), (8, 2, 0, 3), (8, 2, 1, 4)])
def test_batched_clean_cache_matches_sequential_cuda(history, window, sink, frames):
    compare(
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        history=history,
        window=window,
        sink=sink,
        frames=frames,
    )


@pytest.mark.cpu
@pytest.mark.parametrize("nulls", [(), (0,), (1, 3), (0, 1, 2, 3)])
@pytest.mark.parametrize("fps", [15.0, 29.97])
def test_clean_rope_matches_individual_frames(nulls, fps):
    # Return the position IDs themselves to inspect the real packing code.
    def rotary(hidden, *, position_ids):
        ids = position_ids[:, 0].T.unsqueeze(0)
        return ids, torch.zeros_like(ids)

    model = SimpleNamespace(
        manifest=SimpleNamespace(action_tokens_per_frame=4),
        temporal_modality_margin=2,
        base_fps=24.0,
        temporal_compression_factor=4,
        enable_fps_modulation=True,
        language_model=SimpleNamespace(rotary_emb=rotary),
    )
    common = dict(grid_h=2, grid_w=2, real_text_kv_len=5, fps=fps)
    hidden = torch.empty(1, 32, WIDTH)
    batched, _ = Cosmos3NanoSimBimanualTransformer._current_rope(
        model,
        hidden,
        frame_start=7,
        num_frames=4,
        null_action_frame_indexes=nulls,
        frame_causal=True,
        **common,
    )
    sequential = [
        Cosmos3NanoSimBimanualTransformer._current_rope(
            model,
            hidden[:, :8],
            frame_start=7 + i,
            num_frames=1,
            null_action_frame_indexes=(0,) if i in nulls else (),
            **common,
        )[0]
        for i in range(4)
    ]
    torch.testing.assert_close(batched, torch.cat(sequential, dim=1), rtol=0, atol=0)
