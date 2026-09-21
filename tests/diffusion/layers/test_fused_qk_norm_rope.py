# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn.functional as F
from vllm.triton_utils import HAS_TRITON

pytestmark = [pytest.mark.core_model, pytest.mark.cuda, pytest.mark.diffusion]

_HEAD_DIM = 128
_ROTARY_DIM = 96
_EPS = 1e-5


def _reference(q, k, q_weight, k_weight, rope_table):
    q = F.rms_norm(q, (_HEAD_DIM,), q_weight, _EPS)
    k = F.rms_norm(k, (_HEAD_DIM,), k_weight, _EPS)
    half = _ROTARY_DIM // 2
    cos = rope_table[..., :half].unsqueeze(1)
    sin = rope_table[..., half:].unsqueeze(1)

    def apply(x):
        first = x[..., :half]
        second = x[..., half:_ROTARY_DIM]
        return torch.cat(
            (
                first * cos - second * sin,
                second * cos + first * sin,
                x[..., _ROTARY_DIM:],
            ),
            dim=-1,
        )

    return apply(q), apply(k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("seq_len", [1, 257, 1024])
def test_fused_qk_norm_rope_matches_bf16_reference(seq_len):
    from vllm_omni.diffusion.layers.fused_qk_norm_rope import (
        fused_qk_norm_rope,
    )

    torch.manual_seed(17)
    heads = 14
    qkv = torch.randn(
        seq_len,
        heads * _HEAD_DIM * 3,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q = qkv[:, : heads * _HEAD_DIM].view(seq_len, heads, _HEAD_DIM)
    k = qkv[:, heads * _HEAD_DIM : 2 * heads * _HEAD_DIM].view(
        seq_len,
        heads,
        _HEAD_DIM,
    )
    q_weight = torch.randn(_HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k_weight = torch.randn(_HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    freqs = torch.randn(seq_len, _ROTARY_DIM // 2, device="cuda")
    rope_table = torch.cat((torch.cos(freqs), torch.sin(freqs)), dim=-1).to(torch.bfloat16)

    expected_q, expected_k = _reference(
        q,
        k,
        q_weight,
        k_weight,
        rope_table,
    )
    actual_q, actual_k = fused_qk_norm_rope(
        q,
        k,
        q_weight,
        k_weight,
        rope_table,
        _EPS,
    )

    torch.testing.assert_close(actual_q, expected_q, atol=0.0625, rtol=0.02)
    torch.testing.assert_close(actual_k, expected_k, atol=0.0625, rtol=0.02)


# ---------------------------------------------------------------------------
# General geometry (any even head_dim, here Boogu-Image's 120) in both
# pairing modes, against the module's own eager reference at the same
# tolerance as the MiniMax-H3 test above.
# ---------------------------------------------------------------------------

_BOOGU_HEAD_DIM = 120


def _boogu_inputs(strided: bool):
    torch.manual_seed(11)
    if strided:
        # Slices of a wider head axis: the op must honour q/k strides
        # (merged-QKV projections hand the op such views).
        q = torch.randn(4139, 35, _BOOGU_HEAD_DIM, device="cuda", dtype=torch.bfloat16)[:, :28]
        k = torch.randn(4139, 35, _BOOGU_HEAD_DIM, device="cuda", dtype=torch.bfloat16)[:, :7]
    else:
        q = torch.randn(4139, 28, _BOOGU_HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(4139, 7, _BOOGU_HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    q_weight = torch.randn(_BOOGU_HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k_weight = torch.randn(_BOOGU_HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    freqs = torch.randn(4139, _BOOGU_HEAD_DIM // 2, device="cuda", dtype=torch.float32)
    rope_table = torch.cat((torch.cos(freqs), torch.sin(freqs)), dim=-1)
    return q, k, q_weight, k_weight, rope_table


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("strided", [False, True])
def test_fused_qk_norm_rope_interleaved(strided):
    from vllm_omni.diffusion.layers.fused_qk_norm_rope import (
        _eager_qk_norm_rope,
        _launch_fused_qk_norm_rope,
        fused_qk_norm_rope,
    )

    q, k, q_weight, k_weight, rope_table = _boogu_inputs(strided)
    expected = _eager_qk_norm_rope(q, k, q_weight, k_weight, rope_table, _EPS, _BOOGU_HEAD_DIM, _BOOGU_HEAD_DIM, True)
    # Check the public op AND the launcher directly: the latter cannot fall
    # back to eager, so a silent dispatch regression cannot go green here.
    for actual in (
        fused_qk_norm_rope(q, k, q_weight, k_weight, rope_table, _EPS, interleaved=True),
        _launch_fused_qk_norm_rope(q, k, q_weight, k_weight, rope_table, _EPS, interleaved=True),
    ):
        torch.testing.assert_close(actual[0], expected[0], atol=0.0625, rtol=0.02)
        torch.testing.assert_close(actual[1], expected[1], atol=0.0625, rtol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_fused_qk_norm_rope_half_split_general_dim():
    """head_dim=120 off the MiniMax-H3 128 pin, half-split pairing.

    Exercises the generalised combined kernel's half-split mode via the
    launcher. It is not routed in production (half-split traffic keeps the
    untouched pre-existing per-tensor kernel and its 128/96 contract, so the
    public op falls back to eager at this geometry), pending the
    maintainers' call.
    """
    from vllm_omni.diffusion.layers.fused_qk_norm_rope import (
        _eager_qk_norm_rope,
        _launch_fused_qk_norm_rope,
    )

    q, k, q_weight, k_weight, rope_table = _boogu_inputs(strided=False)
    expected = _eager_qk_norm_rope(q, k, q_weight, k_weight, rope_table, _EPS, _BOOGU_HEAD_DIM, _BOOGU_HEAD_DIM)
    actual = _launch_fused_qk_norm_rope(q, k, q_weight, k_weight, rope_table, _EPS, interleaved=False)
    torch.testing.assert_close(actual[0], expected[0], atol=0.0625, rtol=0.02)
    torch.testing.assert_close(actual[1], expected[1], atol=0.0625, rtol=0.02)


def test_fused_qk_norm_rope_min_tokens_resolution(monkeypatch):
    """The op-level token-gate override: unset or blank -> caller's default; a
    non-negative integer string overrides it (``0`` = always fuse); anything
    else is rejected naming the variable."""
    from vllm_omni.diffusion.layers.fused_qk_norm_rope import fused_qk_norm_rope_min_tokens

    env = "VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS"
    monkeypatch.delenv(env, raising=False)
    assert fused_qk_norm_rope_min_tokens(2048) == 2048
    for blank in ("", "  "):
        monkeypatch.setenv(env, blank)
        assert fused_qk_norm_rope_min_tokens(2048) == 2048
    monkeypatch.setenv(env, "0")
    assert fused_qk_norm_rope_min_tokens(2048) == 0
    monkeypatch.setenv(env, "4096")
    assert fused_qk_norm_rope_min_tokens(2048) == 4096
    for bad in ("-1", "abc"):
        monkeypatch.setenv(env, bad)
        with pytest.raises(ValueError, match=env):
            fused_qk_norm_rope_min_tokens(2048)


# ---- two-stream (joint) variant: Flux.2 double-block geometry ----

_FLUX2_HEAD_DIM = 128


def _flux2_inputs(batch: int, txt_len: int, img_len: int, heads: int = 48, strided: bool = True):
    """Text/image Q/K/V as chunked-QKV projection views plus a joint table."""
    torch.manual_seed(23)
    dim = heads * _FLUX2_HEAD_DIM

    def qkv(seq_len):
        proj = torch.randn(batch, seq_len, 3 * dim, device="cuda", dtype=torch.bfloat16)
        if strided:
            q, k, v = proj.chunk(3, dim=-1)
        else:
            q, k, v = (proj[..., i * dim : (i + 1) * dim].contiguous() for i in range(3))
        return tuple(t.unflatten(-1, (heads, -1)) for t in (q, k, v))

    q0, k0, v0 = qkv(txt_len)
    q1, k1, v1 = qkv(img_len)
    weights = [torch.rand(_FLUX2_HEAD_DIM, device="cuda", dtype=torch.bfloat16) + 0.5 for _ in range(4)]
    freqs = torch.randn(batch * (txt_len + img_len), _FLUX2_HEAD_DIM // 2, device="cuda", dtype=torch.float32)
    rope_table = torch.cat((torch.cos(freqs), torch.sin(freqs)), dim=-1)
    return (q0, k0, v0, q1, k1, v1), weights, rope_table


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("batch,txt_len,img_len", [(1, 512, 4096), (2, 77, 1030), (1, 0, 64), (3, 5, 0)])
@pytest.mark.parametrize("table_dtype", [torch.float32, torch.bfloat16])
def test_fused_joint_qkv_norm_rope_matches_single_stream_op(batch, txt_len, img_len, table_dtype):
    """Joint kernel == per-stream single kernel on the concatenated rows, bitwise.

    The joint op's Q/K arithmetic is the single-stream kernel's; the only new
    logic is the stream/row routing and the V gather, so equality must be
    exact.
    """
    from vllm_omni.diffusion.layers.fused_qk_norm_rope import (
        _launch_fused_joint_qkv_norm_rope,
        _launch_fused_qk_norm_rope,
        fused_joint_qkv_norm_rope,
    )

    (q0, k0, v0, q1, k1, v1), (q0_w, k0_w, q1_w, k1_w), rope_table = _flux2_inputs(batch, txt_len, img_len)
    rope_table = rope_table.to(table_dtype)
    seq_total = txt_len + img_len
    heads = q0.shape[2]

    def expected():
        # Per-stream single op rows, gathered into joint token order.
        table = rope_table.view(batch, seq_total, -1)
        parts_q, parts_k = [], []
        for b in range(batch):
            row_q, row_k = [], []
            for q, k, qw, kw, sl in (
                (q0, k0, q0_w, k0_w, slice(0, txt_len)),
                (q1, k1, q1_w, k1_w, slice(txt_len, None)),
            ):
                if q.shape[1] == 0:
                    continue
                oq, ok = _launch_fused_qk_norm_rope(
                    q[b], k[b], qw, kw, table[b, sl].contiguous(), _EPS, interleaved=True
                )
                row_q.append(oq)
                row_k.append(ok)
            parts_q.append(torch.cat(row_q, dim=0))
            parts_k.append(torch.cat(row_k, dim=0))
        return torch.stack(parts_q), torch.stack(parts_k), torch.cat((v0, v1), dim=1)

    exp_q, exp_k, exp_v = expected()
    for actual in (
        fused_joint_qkv_norm_rope(q0, k0, v0, q1, k1, v1, q0_w, k0_w, q1_w, k1_w, rope_table, _EPS),
        _launch_fused_joint_qkv_norm_rope(
            q0, k0, v0, q1, k1, v1, q0_w, k0_w, q1_w, k1_w, rope_table, _EPS, interleaved=True
        ),
    ):
        assert actual[0].shape == (batch, seq_total, heads, _FLUX2_HEAD_DIM)
        assert torch.equal(actual[0], exp_q)
        assert torch.equal(actual[1], exp_k)
        assert torch.equal(actual[2], exp_v)
        assert all(t.is_contiguous() for t in actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("interleaved", [True, False])
def test_fused_joint_qkv_norm_rope_matches_eager_reference(interleaved):
    from vllm_omni.diffusion.layers.fused_qk_norm_rope import (
        _eager_joint_qkv_norm_rope,
        _launch_fused_joint_qkv_norm_rope,
    )

    streams, weights, rope_table = _flux2_inputs(2, 64, 300, heads=6)
    expected = _eager_joint_qkv_norm_rope(
        *streams, *weights, rope_table, _EPS, _FLUX2_HEAD_DIM, _FLUX2_HEAD_DIM, interleaved
    )
    actual = _launch_fused_joint_qkv_norm_rope(*streams, *weights, rope_table, _EPS, interleaved=interleaved)
    torch.testing.assert_close(actual[0], expected[0], atol=0.0625, rtol=0.02)
    torch.testing.assert_close(actual[1], expected[1], atol=0.0625, rtol=0.02)
    assert torch.equal(actual[2], expected[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_fused_joint_qkv_norm_rope_matches_flux2_eager_chain():
    """Against the chain Flux.2's double block runs today: vLLM RMSNorm per
    stream, cat (text first), interleaved RoPE via RotaryEmbedding."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNorm

    from vllm_omni.diffusion.layers.fused_qk_norm_rope import fused_joint_qkv_norm_rope
    from vllm_omni.diffusion.layers.rope import RotaryEmbedding, apply_rope_to_qk

    batch, txt_len, img_len = 2, 128, 1024
    (q0, k0, v0, q1, k1, v1), weights, _ = _flux2_inputs(batch, txt_len, img_len, heads=8)
    norms = []
    with set_current_vllm_config(VllmConfig()):
        for w in weights:
            norm = RMSNorm(_FLUX2_HEAD_DIM, eps=1e-6).cuda().to(torch.bfloat16)
            norm.weight.data.copy_(w)
            norms.append(norm)
    norm_added_q, norm_added_k, norm_q, norm_k = norms
    # Flux2PosEmbed-style [S, D/2] theta-width cos/sin shared by the batch.
    freqs = torch.randn(txt_len + img_len, _FLUX2_HEAD_DIM // 2, device="cuda", dtype=torch.float32)
    cos, sin = torch.cos(freqs), torch.sin(freqs)
    rope = RotaryEmbedding(is_neox_style=False)

    exp_q = torch.cat([norm_added_q(q0), norm_q(q1)], dim=1)
    exp_k = torch.cat([norm_added_k(k0), norm_k(k1)], dim=1)
    exp_v = torch.cat([v0, v1], dim=1)
    exp_q, exp_k = apply_rope_to_qk(rope, exp_q, exp_k, (cos, sin))

    table = torch.cat((cos, sin), dim=-1).to(torch.bfloat16)
    table = table.unsqueeze(0).expand(batch, -1, -1).reshape(-1, _FLUX2_HEAD_DIM)
    act_q, act_k, act_v = fused_joint_qkv_norm_rope(
        q0,
        k0,
        v0,
        q1,
        k1,
        v1,
        norm_added_q.weight,
        norm_added_k.weight,
        norm_q.weight,
        norm_k.weight,
        table,
        1e-6,
    )
    torch.testing.assert_close(act_q, exp_q, atol=0.0625, rtol=0.02)
    torch.testing.assert_close(act_k, exp_k, atol=0.0625, rtol=0.02)
    assert torch.equal(act_v, exp_v)


def test_fused_joint_qkv_norm_rope_rejects_bad_shapes():
    from vllm_omni.diffusion.layers.fused_qk_norm_rope import fused_joint_qkv_norm_rope

    q0 = torch.randn(1, 4, 2, 8, dtype=torch.bfloat16)
    q1 = torch.randn(1, 6, 2, 8, dtype=torch.bfloat16)
    w = torch.ones(8, dtype=torch.bfloat16)
    table = torch.zeros(10, 8)
    with pytest.raises(ValueError, match="incompatible"):
        fused_joint_qkv_norm_rope(q0, q0, q0, q1, q1, q1[:, :5], w, w, w, w, table, 1e-6)
    with pytest.raises(ValueError, match="rope_table"):
        fused_joint_qkv_norm_rope(q0, q0, q0, q1, q1, q1, w, w, w, w, table[:9], 1e-6)
    with pytest.raises(ValueError, match="batch, seq, heads, head_dim"):
        fused_joint_qkv_norm_rope(q0[0], q0, q0, q1, q1, q1, w, w, w, w, table, 1e-6)


def test_pack_qk_norm_rope_table_skips_when_fused_path_unavailable(monkeypatch):
    """No table (and no allocation) on devices/dtypes the fused kernel cannot
    serve: CPU tensors, non-bf16 activations, unsupported geometry."""
    from vllm_omni.diffusion.layers.fused_qk_norm_rope import pack_qk_norm_rope_table

    monkeypatch.setenv("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", "0")
    cos, sin = torch.randn(16, 64), torch.randn(16, 64)  # CPU
    assert pack_qk_norm_rope_table(cos, sin, 1, dtype=torch.bfloat16, min_tokens=0) is None
    if torch.cuda.is_available() and HAS_TRITON:
        cos, sin = cos.cuda(), sin.cuda()
        assert pack_qk_norm_rope_table(cos, sin, 1, dtype=torch.float16, min_tokens=0) is None
        assert (
            pack_qk_norm_rope_table(cos, sin, 1, dtype=torch.float32, min_tokens=0, activation_dtype=torch.float16)
            is None
        )
        assert pack_qk_norm_rope_table(cos, sin, 1, dtype=torch.bfloat16, min_tokens=0, head_dim=512) is None  # > 256
        table = pack_qk_norm_rope_table(cos, sin, 2, dtype=torch.bfloat16, min_tokens=0)
        assert table is not None and table.shape == (32, 128) and table.dtype == torch.bfloat16
        # fp32 table for bf16 activations (Qwen-Image style) is allowed
        assert (
            pack_qk_norm_rope_table(cos, sin, 1, dtype=torch.float32, min_tokens=0, activation_dtype=torch.bfloat16)
            is not None
        )


def test_pack_qk_norm_rope_table_token_gate_boundary(monkeypatch):
    """``None`` strictly below the resolved token gate, a table at and above
    it; the env override wins over the consumer default. CPU-only: the
    device/dtype check is stubbed so only the gate decides."""
    from vllm_omni.diffusion.layers import fused_qk_norm_rope as mod

    monkeypatch.setattr(mod, "fused_qk_norm_rope_available", lambda *a, **k: True)
    env = "VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS"
    monkeypatch.delenv(env, raising=False)
    cos, sin = torch.randn(8, 4), torch.randn(8, 4)  # S=8, rotary_dim=8
    batch = 2  # 16 tokens

    assert mod.pack_qk_norm_rope_table(cos, sin, batch, dtype=torch.bfloat16, min_tokens=17) is None
    table = mod.pack_qk_norm_rope_table(cos, sin, batch, dtype=torch.bfloat16, min_tokens=16)
    assert table is not None and table.shape == (16, 8) and table.dtype == torch.bfloat16
    assert torch.equal(table.view(batch, 8, 8)[1], torch.cat((cos, sin), dim=-1).to(torch.bfloat16))
    assert mod.pack_qk_norm_rope_table(cos, sin, batch, dtype=torch.bfloat16, min_tokens=0) is not None

    monkeypatch.setenv(env, "17")  # env override: consumer default 0 no longer fuses
    assert mod.pack_qk_norm_rope_table(cos, sin, batch, dtype=torch.bfloat16, min_tokens=0) is None
    monkeypatch.setenv(env, "0")  # env override: consumer default 1000 fuses
    assert mod.pack_qk_norm_rope_table(cos, sin, batch, dtype=torch.bfloat16, min_tokens=1000) is not None


def test_pack_qk_norm_rope_table_skips_under_sequence_parallel(monkeypatch):
    """Under SP (``sequence_parallel_size > 1``) no table is packed: RoPE is
    applied per stream/shard and the consumer keeps its eager chain."""
    from vllm_omni.diffusion.layers import fused_qk_norm_rope as mod

    monkeypatch.setattr(mod, "fused_qk_norm_rope_available", lambda *a, **k: True)
    monkeypatch.setenv("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", "0")
    cos, sin = torch.randn(8, 4), torch.randn(8, 4)

    for sp_size in (2, 8):
        assert (
            mod.pack_qk_norm_rope_table(cos, sin, 1, dtype=torch.bfloat16, min_tokens=0, sequence_parallel_size=sp_size)
            is None
        )
    for inactive_sp_size in (None, 0, 1):
        assert (
            mod.pack_qk_norm_rope_table(
                cos, sin, 1, dtype=torch.bfloat16, min_tokens=0, sequence_parallel_size=inactive_sp_size
            )
            is not None
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_fused_ops_bitwise_under_torch_compile_and_cuda_graph():
    """Both custom ops under ``torch.compile(fullgraph=True)`` and under CUDA
    graph capture/replay produce exactly the eager op's outputs."""
    from vllm_omni.diffusion.layers.fused_qk_norm_rope import fused_joint_qkv_norm_rope, fused_qk_norm_rope

    torch.manual_seed(3)
    batch, txt_len, img_len = 2, 512, 4096
    (q0, k0, v0, q1, k1, v1), (q0_w, k0_w, q1_w, k1_w), rope_table = _flux2_inputs(batch, txt_len, img_len)
    seq_total = txt_len + img_len
    heads = q0.shape[2]

    def joint(q0, k0, v0, q1, k1, v1, table):
        return fused_joint_qkv_norm_rope(q0, k0, v0, q1, k1, v1, q0_w, k0_w, q1_w, k1_w, table, _EPS)

    # Single-stream op over the joint sequence (the single blocks' call).
    qkv = torch.randn(batch * seq_total, heads * _FLUX2_HEAD_DIM * 3, device="cuda", dtype=torch.bfloat16)
    sq, sk, _ = (t.unflatten(-1, (heads, -1)) for t in qkv.chunk(3, dim=-1))

    def single(q, k, table):
        return fused_qk_norm_rope(q, k, q0_w, k0_w, table, _EPS, interleaved=True)

    joint_inputs = (q0, k0, v0, q1, k1, v1, rope_table)
    single_inputs = (sq, sk, rope_table)
    eager_joint = joint(*joint_inputs)
    eager_single = single(*single_inputs)

    def check(actual, expected):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert torch.equal(a, e)

    compiled_joint = torch.compile(joint, fullgraph=True)
    compiled_single = torch.compile(single, fullgraph=True)
    check(compiled_joint(*joint_inputs), eager_joint)
    check(compiled_single(*single_inputs), eager_single)

    # CUDA graph: capture on a side stream after warmup, replay, compare.
    for fn, inputs, expected in ((joint, joint_inputs, eager_joint), (single, single_inputs, eager_single)):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                fn(*inputs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = fn(*inputs)
        graph.replay()
        torch.accelerator.synchronize()
        check(static_out, expected)
        graph.replay()
        torch.accelerator.synchronize()
        check(static_out, expected)


def test_pack_qk_norm_rope_table_identity_rows(monkeypatch):
    """``identity_rows`` appends ``cos = 1, sin = 0`` rows after the rotated
    rows (per batch element) and counts toward the token gate."""
    from vllm_omni.diffusion.layers import fused_qk_norm_rope as mod

    monkeypatch.setattr(mod, "fused_qk_norm_rope_available", lambda *a, **k: True)
    monkeypatch.delenv("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", raising=False)
    cos, sin = torch.randn(5, 4), torch.randn(5, 4)
    table = mod.pack_qk_norm_rope_table(cos, sin, 2, dtype=torch.bfloat16, min_tokens=0, identity_rows=3)
    assert table is not None and table.shape == (2 * 8, 8) and table.dtype == torch.bfloat16
    per_batch = table.view(2, 8, 8)
    assert torch.equal(per_batch[0], per_batch[1])
    assert torch.equal(per_batch[0, :5], torch.cat((cos, sin), dim=-1).to(torch.bfloat16))
    assert torch.equal(per_batch[0, 5:, :4], torch.ones(3, 4, dtype=torch.bfloat16))
    assert torch.equal(per_batch[0, 5:, 4:], torch.zeros(3, 4, dtype=torch.bfloat16))
    # Identity rows count toward the gate: 2 * (5 + 3) = 16 tokens.
    assert mod.pack_qk_norm_rope_table(cos, sin, 2, dtype=torch.bfloat16, min_tokens=17, identity_rows=3) is None
    assert mod.pack_qk_norm_rope_table(cos, sin, 2, dtype=torch.bfloat16, min_tokens=16, identity_rows=3) is not None


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TRITON, reason="CUDA and Triton required")
@pytest.mark.parametrize("interleaved", [False, True])
def test_fused_qk_norm_rope_large_storage_offsets(interleaved):
    """Only three tokens, but the last row lies beyond signed 32-bit indexing."""
    from vllm_omni.diffusion.layers.fused_qk_norm_rope import fused_qk_norm_rope

    # 4 GiB of backing storage; initialize only the three small visible rows.
    q = torch.empty_strided((3, 1, 128), (2**30, 128, 1), device="cuda", dtype=torch.bfloat16)
    q.fill_(1)
    weight = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    table = torch.zeros(3, 96, device="cuda", dtype=torch.bfloat16)
    table[:, :48] = 1
    actual = fused_qk_norm_rope(q, q, weight, weight, table, _EPS, interleaved=interleaved)
    for value in actual:
        torch.testing.assert_close(value, torch.ones_like(value))


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TRITON, reason="CUDA and Triton required")
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("large_stream", [0, 1])
def test_fused_joint_qkv_norm_rope_large_storage_offsets(interleaved, large_stream):
    """Both input streams must retain 64-bit addressing for Q, K, and V."""
    from vllm_omni.diffusion.layers.fused_qk_norm_rope import _launch_fused_joint_qkv_norm_rope

    # As in the single-stream regression, touch only three rows of a 4 GiB view.
    large = torch.empty_strided((1, 3, 1, 128), (3 * 2**30, 2**30, 128, 1), device="cuda", dtype=torch.bfloat16)
    large.fill_(1)
    small = torch.ones((1, 2, 1, 128), device="cuda", dtype=torch.bfloat16)
    q0, q1 = (large, small) if large_stream == 0 else (small, large)
    weight = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    table = torch.zeros(5, 96, device="cuda", dtype=torch.bfloat16)
    table[:, :48] = 1
    actual = _launch_fused_joint_qkv_norm_rope(
        q0, q0, q0, q1, q1, q1, weight, weight, weight, weight, table, _EPS, interleaved=interleaved
    )
    for value in actual:
        assert value.shape == (1, 5, 1, 128)
        torch.testing.assert_close(value, torch.ones_like(value), atol=0, rtol=0)
