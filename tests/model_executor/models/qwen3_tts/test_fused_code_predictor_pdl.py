# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803 - kernel-style shape names
"""PDL chain of the fused residual-codebook predictor: chain GEMM epilogues and the whole predictor."""

import types

import pytest
import torch

from tests.helpers.mark import hardware_test

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.version.hip is not None or torch.cuda.get_device_capability()[0] < 9,
        reason="requires CUDA sm90+ (programmatic dependent launch)",
    ),
]

HID, HEADS, KV_HEADS, HEAD_DIM, INTER, VOCAB, GROUPS = 1024, 16, 8, 128, 3072, 2048, 16


def _launch(kind: str, a, w, out, ss_in, nt_in, ss_out, M, epi, a_rs=1, a_ro=0, eps=1e-6, rmax=None):
    from vllm.triton_utils import triton

    from vllm_omni.model_executor.models.qwen3_tts.fused_code_predictor import (
        _cp_gemm_kernel,
        _cp_gemm_stream_kernel,
    )

    N, K = w.shape
    rmax = rmax or a.shape[0]
    ws = torch.empty(8 * rmax * N, device=a.device, dtype=torch.float32)
    cnt = torch.zeros(4096, device=a.device, dtype=torch.int32)
    common = (a, a_rs, a_ro, w, ws, cnt, out, ss_in, ss_out, M, eps)
    if kind == "panel":
        # split-K 2 and two M tiles per CTA
        _cp_gemm_kernel[(N // 64, K // 512, 1)](
            *common, K=K, N=N, HID=HID, RMAX=rmax, BN=64, BK=256, NCH=2, BM=16, EPI=epi, NT_IN=nt_in, MSTAGES=1,
            num_warps=4, launch_pdl=True,
        )  # fmt: skip
    else:
        _cp_gemm_stream_kernel[(N // 64, 2, triton.cdiv(M, 32))](
            *common, K=K, N=N, HID=HID, RMAX=rmax, BN=64, BK=128, SK=2, BM=32, EPI=epi, NT_IN=nt_in, STAGES=2,
            NLP=triton.next_power_of_2(K // 2 // 64), num_warps=4, launch_pdl=True,
        )  # fmt: skip
    assert int(cnt.abs().sum()) == 0, "split-K arrival counters must return to zero"


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("kind", ["panel", "stream"])
@torch.inference_mode()
def test_chain_gemm_epilogues_match_reference(kind: str):
    torch.manual_seed(0)
    dev, bf = "cuda", torch.bfloat16
    M, eps = 37, 1e-6
    x = torch.randn(M, HID, device=dev, dtype=bf)
    # EPI 2: x += A @ W^T in place, plus per-tile row sums of squares of the new x
    att = torch.randn(M, 2048, device=dev, dtype=bf)
    w_o = torch.randn(HID, 2048, device=dev, dtype=bf) * 0.02
    x_new = x.clone()
    ss = torch.zeros(HID // 64, M, device=dev)
    _launch(kind, att, w_o, x_new, None, 0, ss, M, 2)
    ref = (x.float() + att.float() @ w_o.float().t()).to(bf)
    torch.testing.assert_close(x_new.float(), ref.float(), atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(ss.sum(0), x_new.float().square().sum(1), atol=1e-3, rtol=1e-4)

    # EPI 0: rows scaled by rsqrt(mean(x^2) + eps), RMS weight folded into W
    w_q = torch.randn(4096, HID, device=dev, dtype=bf) * 0.02
    q = torch.empty(M, 4096, device=dev, dtype=bf)
    _launch(kind, x_new, w_q, q, ss, HID // 64, None, M, 0, eps=eps)
    r = torch.rsqrt(x_new.float().square().mean(1, keepdim=True) + eps)
    torch.testing.assert_close(q.float(), (x_new.float() @ w_q.float().t()) * r, atol=2e-2, rtol=2e-2)

    # EPI 1: SiLU(gate) * up with gate/up rows interleaved per 64-row tile
    w_g = torch.randn(INTER, HID, device=dev, dtype=bf) * 0.02
    w_u = torch.randn(INTER, HID, device=dev, dtype=bf) * 0.02
    w_gu = torch.stack((w_g.view(-1, 32, HID), w_u.view(-1, 32, HID)), 1).reshape(-1, HID).contiguous()
    act = torch.empty(M, INTER, device=dev, dtype=bf)
    _launch(kind, x_new, w_gu, act, ss, HID // 64, None, M, 1, eps=eps)
    g = ((x_new.float() @ w_g.float().t()) * r).to(bf).float()
    u = ((x_new.float() @ w_u.float().t()) * r).to(bf).float()
    torch.testing.assert_close(act.float(), torch.nn.functional.silu(g).to(bf).float() * u, atol=2e-2, rtol=2e-2)

    # strided rows (the head reads the last of two rows per request)
    logits = torch.empty(M // 2, 4096, device=dev, dtype=bf)
    _launch(kind, x_new, w_q, logits, ss, HID // 64, None, M // 2, 0, a_rs=2, a_ro=1, eps=eps, rmax=M)
    torch.testing.assert_close(logits.float(), q[1::2][: M // 2].float(), atol=2e-2, rtol=2e-2)


def _ns(**kw):
    return types.SimpleNamespace(**kw)


def _predictor(dev: str):
    """Random-weight stand-in with the Qwen3-TTS 12Hz code predictor's shapes."""
    torch.manual_seed(0)
    bf = torch.bfloat16

    def lin(n, k, scale=0.03):
        return _ns(weight=torch.randn(n, k, device=dev, dtype=bf) * scale, bias=None)

    def norm(n):
        return _ns(weight=(1 + 0.1 * torch.randn(n, device=dev)).to(bf), variance_epsilon=1e-6)

    layers = []
    for _ in range(5):
        attn = _ns(
            qkv_proj=lin((HEADS + 2 * KV_HEADS) * HEAD_DIM, HID), q_norm=norm(HEAD_DIM), k_norm=norm(HEAD_DIM),
            o_proj=lin(HID, HEADS * HEAD_DIM), num_heads=HEADS, num_kv_heads=KV_HEADS, head_dim=HEAD_DIM,
            scaling=HEAD_DIM**-0.5, max_seq=GROUPS + 1,
        )  # fmt: skip
        layers.append(
            _ns(
                self_attn=attn,
                input_layernorm=norm(HID),
                post_attention_layernorm=norm(HID),
                mlp=_ns(gate_up_proj=lin(2 * INTER, HID), down_proj=lin(HID, INTER)),
            )  # fmt: skip
        )
    inv = 1.0 / (1e6 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    ang = torch.outer(torch.arange(GROUPS + 1, dtype=torch.float32), inv)
    ang = torch.cat((ang, ang), -1)
    model = _ns(
        layers=layers, norm=norm(HID), codec_embedding=[lin(VOCAB, 2048, 1.0) for _ in range(GROUPS - 1)],
        rotary_emb=_ns(cos_cached=ang.cos().to(dev), sin_cached=ang.sin().to(dev)),
    )  # fmt: skip
    model.parameters = lambda: iter([model.norm.weight])
    proj = torch.nn.Linear(2048, HID, device=dev, dtype=bf)
    return _ns(
        model=model, config=_ns(num_code_groups=GROUPS, vocab_size=VOCAB, hidden_size=HID),
        small_to_mtp_projection=proj, lm_head=[lin(VOCAB, HID, 0.1) for _ in range(GROUPS - 1)],
    )  # fmt: skip


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@torch.inference_mode()
def test_pdl_predictor_matches_cublas_path_and_replays_in_graphs(monkeypatch):
    from vllm_omni.model_executor.models.qwen3_tts import fused_code_predictor as fcp

    dev = "cuda"
    pred = _predictor(dev)
    pdl = fcp.FusedCodePredictor(pred, 64)
    assert pdl._pdl_configs, "sm90+ must take the PDL chain"
    # Every config bucket at its limit and just above the previous one, read before the configs are cleared.
    limits = [limit for limit, _ in fcp._PDL_CONFIGS]
    batches = sorted({1, 2, 5, *limits, *(limit + 1 for limit in limits[:-1])})
    monkeypatch.setattr(fcp, "_PDL_CONFIGS", ())
    ref = fcp.FusedCodePredictor(pred, 64)
    assert ref._pdl_config(1) is None

    for B in batches:
        assert pdl._pdl_config(B) is not None
        gen = torch.Generator(device=dev).manual_seed(B)
        code0 = torch.randint(0, VOCAB, (B, 1), device=dev, generator=gen)
        emb = torch.randn(B, 1, 2048, device=dev, generator=gen).to(torch.bfloat16)
        hid = torch.randn(B, 1, 2048, device=dev, generator=gen).to(torch.bfloat16)
        # greedy: constant Gumbel noise, no top-k
        half = torch.full((B, GROUPS - 1, VOCAB), 0.5, device=dev)
        got = pdl(code0, emb, hid, 1.0, 0, half)
        want = ref(code0, emb, hid, 1.0, 0, half)
        assert torch.equal(got[:, 0], code0[:, 0])
        # Rounding happens in different places (RMS weight folded into W), so near-ties may flip.
        assert (got[:, 1] == want[:, 1]).float().mean().item() >= 0.8
        assert torch.equal(pdl(code0, emb, hid, 1.0, 0, half), got), "the chain must be deterministic"

        u = torch.rand(B, GROUPS - 1, VOCAB, device=dev, generator=gen).clamp_(1e-6, 1 - 1e-6)
        eager = pdl(code0, emb, hid, 1 / 0.9, 50, u)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = pdl(code0, emb, hid, 1 / 0.9, 50, u)
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(captured, eager)
