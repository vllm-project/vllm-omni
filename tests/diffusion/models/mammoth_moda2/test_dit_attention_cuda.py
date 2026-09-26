# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real-shape checks for MammothModa2's fused QK norm/RoPE and shared CUDA
attention backend against the pre-change bf16 arithmetic."""

import pytest
import torch
from diffusers.models.attention_processor import Attention
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

import vllm_omni.diffusion.attention.backends.sdpa as sdpa_backend
import vllm_omni.diffusion.models.mammoth_moda2.mammothmoda2_dit_model as mammoth_dit
from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.models.mammoth_moda2.mammothmoda2_dit_model import TransformerBlock

from .test_dit_attention import _reference_attention

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.advanced_model,
    pytest.mark.cuda,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
]

# MammothModa2-Preview gen_dit_config: hidden 2520, 21 heads, 7 KV heads, head_dim 120.
DIM, HEADS, KV_HEADS = 2520, 21, 7


def _norm_attention(head_dim, dtype):
    attn = Attention(query_dim=head_dim * HEADS, heads=HEADS, kv_heads=KV_HEADS, dim_head=head_dim)
    attn.norm_q = Qwen2RMSNorm(head_dim)
    attn.norm_k = Qwen2RMSNorm(head_dim)
    return attn.to(device="cuda", dtype=dtype)


def test_unpaired_rope_preserves_native_arithmetic(monkeypatch):
    """A generic block accepts independently valued even and odd RoPE lanes."""
    torch.manual_seed(42)
    q = torch.randn(2, 17, HEADS, 120, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 17, KV_HEADS, 120, device="cuda", dtype=torch.bfloat16)
    attn = _norm_attention(120, torch.bfloat16)
    angles = torch.rand(2, 17, 120, device="cuda") * 6.283
    rotary = angles.cos().to(q.dtype), angles.sin().to(q.dtype)
    monkeypatch.setenv("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", "0")
    with torch.no_grad():
        got = mammoth_dit._apply_qk_norm_rope(attn, q, k, rotary)
        expected = (
            mammoth_dit.apply_real_rotary_emb(attn.norm_q(q), *rotary),
            mammoth_dit.apply_real_rotary_emb(attn.norm_k(k), *rotary),
        )
    for actual, want in zip(got, expected):
        torch.testing.assert_close(actual, want, atol=0, rtol=0)


def test_qk_norm_rope_benchmark_measures_inference(monkeypatch):
    from benchmarks.diffusion import benchmark_mammoth_moda2_qk_norm_rope as benchmark

    calls = []

    def measure(fn, warmup, iters):
        outputs = fn()
        assert not torch.is_grad_enabled()
        assert all(not output.requires_grad for output in outputs)
        calls.append(fn.__name__)
        return {"median_ms": 1.0}

    monkeypatch.setattr(benchmark, "_measure", measure)
    benchmark._run_shape(17, warmup=0, iters=1)
    assert calls == ["native", "fused"]


def _production_rotary(batch, seq, head_dim, dtype):
    """Mammoth's real RoPE repeats one angle across each adjacent pair."""
    angles = torch.rand(batch, seq, head_dim // 2, device="cuda") * 6.283
    return (
        angles.cos().repeat_interleave(2, dim=-1).to(dtype),
        angles.sin().repeat_interleave(2, dim=-1).to(dtype),
    )


@pytest.mark.parametrize("force_native", [False, True])
def test_production_rope_uses_fused_qk_norm_rope(monkeypatch, force_native):
    torch.manual_seed(0)
    seq = 512
    block = (
        TransformerBlock(
            DIM,
            HEADS,
            KV_HEADS,
            multiple_of=256,
            ffn_dim_multiplier=1.0,
            norm_eps=1e-5,
            rope_repeats_pairs=True,
        )
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )
    hidden = torch.randn(2, seq, DIM, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(2, seq, dtype=torch.bool, device="cuda")
    mask[0, -37:] = False
    rotary = _production_rotary(1, seq, block.head_dim, torch.bfloat16)
    calls = []
    fused_qk_norm_rope = mammoth_dit.fused_qk_norm_rope

    def record_fused(q, k, q_weight, k_weight, rope_table, eps, **kwargs):
        calls.append((q.shape, k.shape, rope_table.shape, kwargs))
        return fused_qk_norm_rope(q, k, q_weight, k_weight, rope_table, eps, **kwargs)

    monkeypatch.setenv("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", "0")
    if force_native:
        monkeypatch.setattr(mammoth_dit, "_fused_cuda_supported", lambda *args, **kwargs: False)
    monkeypatch.setattr(mammoth_dit, "fused_qk_norm_rope", record_fused)
    with torch.no_grad():
        got = block.attn(hidden, hidden, attention_mask=mask, image_rotary_emb=rotary)
        want = _reference_attention(block.attn, hidden, mask, rotary)

    supported = mammoth_dit._fused_cuda_supported(hidden, hidden, block.head_dim, block.head_dim, interleaved=True)
    expected_calls = (
        [
            (
                torch.Size((2 * seq, HEADS, block.head_dim)),
                torch.Size((2 * seq, KV_HEADS, block.head_dim)),
                torch.Size((2 * seq, block.head_dim)),
                {"head_dim": block.head_dim, "rotary_dim": block.head_dim, "interleaved": True},
            )
        ]
        if supported
        else []
    )
    assert calls == expected_calls
    diff = (got.float() - want.float()).abs()[mask]
    assert diff.max().item() < 2e-2, diff.max().item()
    assert diff.mean().item() < 1e-3, diff.mean().item()


@pytest.mark.parametrize("seq", [77 + 4096, 512], ids=["t2i_1024", "short"])
def test_real_shape_matches_previous_arithmetic_bf16(seq):
    torch.manual_seed(0)
    block = (
        TransformerBlock(
            DIM,
            HEADS,
            KV_HEADS,
            multiple_of=256,
            ffn_dim_multiplier=1.0,
            norm_eps=1e-5,
            rope_repeats_pairs=True,
        )
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )
    hidden = torch.randn(2, seq, DIM, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(2, seq, dtype=torch.bool, device="cuda")
    mask[0, seq - 300 :] = False
    mask[1, seq - 17 :] = False
    rotary = _production_rotary(1, seq, block.head_dim, torch.bfloat16)
    with torch.no_grad():
        got = block.attn(
            hidden_states=hidden, encoder_hidden_states=hidden, attention_mask=mask, image_rotary_emb=rotary
        )
        want = _reference_attention(block.attn, hidden, mask, rotary)
    assert torch.isfinite(got).all()
    assert torch.count_nonzero(got[~mask]) == 0
    diff = (got.float() - want.float()).abs()[mask]
    # bf16 kernels differ in accumulation order; measured on A800: max 1.0e-3, mean 5e-5.
    assert diff.max().item() < 2e-2, diff.max().item()
    assert diff.mean().item() < 1e-3, diff.mean().item()


def test_empty_text_stream_on_the_default_backend():
    """The recipe's text-to-image request with text_guidance_scale > 1 runs the
    context refiner on a zero-token unconditional prompt. Before the guard this
    raised ``RuntimeError: step must be nonzero`` from the FA varlen fallback."""
    torch.manual_seed(0)
    block = (
        TransformerBlock(DIM, HEADS, KV_HEADS, multiple_of=256, ffn_dim_multiplier=1.0, norm_eps=1e-5, modulation=False)
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )
    hidden = torch.randn(1, 0, DIM, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(1, 0, dtype=torch.bool, device="cuda")
    angles = torch.rand(1, 0, block.head_dim, device="cuda")
    with torch.no_grad():
        out = block.attn(
            hidden_states=hidden,
            encoder_hidden_states=hidden,
            attention_mask=mask,
            image_rotary_emb=(angles.cos().to(torch.bfloat16), angles.sin().to(torch.bfloat16)),
        )
    assert out.shape == (1, 0, DIM)


@pytest.mark.parametrize(
    ("seq", "force_gqa_fallback"),
    [(512, False), (77 + 4096, False), (32, True)],
    ids=["short", "t2i_1024", "unsupported_native_gqa"],
)
def test_fp32_falls_back_to_sdpa_and_matches_reference(monkeypatch, seq, force_gqa_fallback):
    """FP32 keeps its dtype and reaches shared SDPA's GQA compatibility check.

    Cover the long masked shape that otherwise falls back to quadratic math
    attention, and force the unsupported-GQA branch independently of hardware.
    """
    torch.manual_seed(0)
    dt = torch.float32
    block = (
        TransformerBlock(
            DIM,
            HEADS,
            KV_HEADS,
            multiple_of=256,
            ffn_dim_multiplier=1.0,
            norm_eps=1e-5,
            rope_repeats_pairs=True,
        )
        .cuda()
        .to(dt)
        .eval()
    )
    hidden = torch.randn(1, seq, DIM, device="cuda", dtype=dt)
    mask = torch.ones(1, seq, dtype=torch.bool, device="cuda")
    mask[:, -13:] = False
    angles = torch.rand(1, seq, block.head_dim, device="cuda")
    rotary = (angles.cos().to(dt), angles.sin().to(dt))
    capabilities, calls, phases = [], [], []
    can_use_fused_gqa = sdpa_backend.can_sdpa_use_fused_gqa
    sdpa = torch.nn.functional.scaled_dot_product_attention
    strategy = block.attn.omni_attn._get_active_parallel_strategy()
    pre_attention, post_attention = strategy.pre_attention, strategy.post_attention

    def prepare(*args):
        phases.append("pre")
        return pre_attention(*args)

    def restore(*args):
        phases.append("post")
        return post_attention(*args)

    def check_gqa(*args):
        supported = False if force_gqa_fallback else can_use_fused_gqa(*args)
        capabilities.append(supported)
        return supported

    def record_sdpa(query, key, value, **kwargs):
        phases.append("sdpa")
        calls.append((query.shape, key.shape, value.shape, query.dtype, kwargs))
        return sdpa(query, key, value, **kwargs)

    def reject_fused(*args, **kwargs):
        raise AssertionError("FP32 must keep Mammoth's native QK norm/RoPE path")

    with torch.no_grad(), monkeypatch.context() as patch:
        patch.setattr(mammoth_dit, "fused_qk_norm_rope", reject_fused)
        patch.setattr(sdpa_backend, "can_sdpa_use_fused_gqa", check_gqa)
        patch.setattr(torch.nn.functional, "scaled_dot_product_attention", record_sdpa)
        patch.setattr(strategy, "pre_attention", prepare)
        patch.setattr(strategy, "post_attention", restore)
        got = block.attn(hidden, hidden, attention_mask=mask, image_rotary_emb=rotary)

    assert phases == ["pre", "sdpa", "post"], "FP32 must preserve the shared parallel dispatch"
    assert len(capabilities) == len(calls) == 1, "FP32 must use shared SDPA's runtime GQA check"
    q_shape, k_shape, v_shape, dtype, kwargs = calls[0]
    expected_kv_heads = KV_HEADS if capabilities[0] else HEADS
    assert q_shape == (1, HEADS, seq, block.head_dim)
    assert k_shape == v_shape == (1, expected_kv_heads, seq, block.head_dim)
    assert dtype == got.dtype == torch.float32
    assert kwargs["enable_gqa"] == capabilities[0]
    assert kwargs["scale"] == block.attn.scale
    assert kwargs["is_causal"] is False
    assert torch.equal(kwargs["attn_mask"], mask[:, None, None, :])

    with torch.no_grad():
        want = _reference_attention(block.attn, hidden, mask, rotary)
    assert torch.isfinite(got).all()
    assert torch.count_nonzero(got[~mask]) == 0
    diff = (got.float() - want.float()).abs()[mask]
    assert diff.max().item() < 1e-3, diff.max().item()


@pytest.mark.parametrize("layout", ["unbatched", "broadcast", "per_example", "strided"])
@pytest.mark.parametrize("table_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("force_native", [False, True])
def test_paired_rope_with_learned_norm_weights(monkeypatch, layout, table_dtype, force_native):
    torch.manual_seed(42)
    batch, seq, head_dim = 2, 19, 120
    attn = _norm_attention(head_dim, torch.bfloat16)
    with torch.no_grad():
        attn.norm_q.weight.normal_(1.0, 0.2)
        attn.norm_k.weight.normal_(1.0, 0.2)
    # Mimic views into packed QKV projections, including their nontrivial stride.
    packed = torch.randn(batch, seq, HEADS + 2 * KV_HEADS, head_dim, device="cuda", dtype=torch.bfloat16)
    q, k, _ = packed.split((HEADS, KV_HEADS, KV_HEADS), dim=2)
    rotary = _production_rotary(batch if layout in ("per_example", "strided") else 1, seq, head_dim, table_dtype)
    if layout == "unbatched":
        rotary = tuple(t.squeeze(0) for t in rotary)
    elif layout == "strided":
        rotary = tuple(t.transpose(0, 1).contiguous().transpose(0, 1) for t in rotary)
        assert not rotary[0].is_contiguous()
    calls = []
    fused = mammoth_dit.fused_qk_norm_rope

    def record(*args, **kwargs):
        calls.append(args[4].shape)
        return fused(*args, **kwargs)

    monkeypatch.setattr(mammoth_dit, "fused_qk_norm_rope", record)
    if force_native:
        monkeypatch.setattr(mammoth_dit, "_fused_cuda_supported", lambda *args, **kwargs: False)
    monkeypatch.setenv("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", "0")
    with torch.no_grad():
        got = mammoth_dit._apply_qk_norm_rope(attn, q, k, rotary, rope_repeats_pairs=True)
        expected = (
            mammoth_dit.apply_real_rotary_emb(attn.norm_q(q), *rotary),
            mammoth_dit.apply_real_rotary_emb(attn.norm_k(k), *rotary),
        )
    supported = mammoth_dit._fused_cuda_supported(q, k, head_dim, head_dim, interleaved=True)
    assert calls == ([torch.Size((batch * seq, head_dim))] if supported else [])
    for actual, want in zip(got, expected):
        torch.testing.assert_close(actual, want.to(actual.dtype), atol=0.0625, rtol=0.02)


@pytest.mark.parametrize(
    "reason", ["support_query", "token_gate", "no_q_norm", "no_k_norm", "epsilon", "fp16", "head_dim", "no_rope"]
)
def test_qk_norm_rope_unsupported_inputs_keep_native_chain(monkeypatch, reason):
    torch.manual_seed(42)
    dtype = torch.float16 if reason == "fp16" else torch.bfloat16
    dim = 260 if reason == "head_dim" else 120
    attn = _norm_attention(dim, dtype)
    q = torch.randn(2, 19, HEADS, dim, device="cuda", dtype=dtype)
    k = torch.randn(2, 19, KV_HEADS, dim, device="cuda", dtype=dtype)
    rotary = _production_rotary(2, 19, dim, dtype) if reason != "no_rope" else None
    if reason == "support_query":
        monkeypatch.setattr(mammoth_dit, "_fused_cuda_supported", lambda *args, **kwargs: False)
    elif reason == "no_q_norm":
        attn.norm_q = None
    elif reason == "no_k_norm":
        attn.norm_k = None
    elif reason == "epsilon":
        attn.norm_k.variance_epsilon = 1e-3
    monkeypatch.setenv("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", "39" if reason == "token_gate" else "0")

    def reject(*args, **kwargs):
        raise AssertionError(f"Unsupported {reason} must keep the native chain")

    monkeypatch.setattr(mammoth_dit, "fused_qk_norm_rope", reject)
    with torch.no_grad():
        got = mammoth_dit._apply_qk_norm_rope(attn, q, k, rotary, rope_repeats_pairs=True)
        expected_q = attn.norm_q(q) if attn.norm_q is not None else q
        expected_k = attn.norm_k(k) if attn.norm_k is not None else k
        if rotary is not None:
            expected_q = mammoth_dit.apply_real_rotary_emb(expected_q, *rotary)
            expected_k = mammoth_dit.apply_real_rotary_emb(expected_k, *rotary)
    torch.testing.assert_close(got[0], expected_q, atol=0, rtol=0)
    torch.testing.assert_close(got[1], expected_k, atol=0, rtol=0)


def test_attention_benchmark_single_token_and_restores_gate(monkeypatch):
    from benchmarks.diffusion import benchmark_mammoth_moda2_qk_norm_rope as benchmark

    env = "VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS"
    monkeypatch.setenv(env, "123")
    result = benchmark._run_attention(1, warmup=0, iters=1)
    assert result["sequence"] == 1
    assert benchmark.os.environ[env] == "123"


@pytest.mark.parametrize("packed", [False, True], ids=["gated_native", "reused_packed"])
def test_prepared_rope_table_skips_per_layer_gate_and_packing(monkeypatch, packed):
    torch.manual_seed(7)
    batch, seq, dim = 2, 19, 120
    attn = _norm_attention(dim, torch.bfloat16)
    q = torch.randn(batch, seq, HEADS, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, seq, KV_HEADS, dim, device="cuda", dtype=torch.bfloat16)
    cos, sin = _production_rotary(batch, seq, dim, torch.float32)
    table = torch.cat((cos[..., 0::2], sin[..., 0::2]), dim=-1).reshape(batch * seq, dim) if packed else None
    rotary = cos, sin, table
    calls = []
    fused = mammoth_dit.fused_qk_norm_rope

    def record(*args, **kwargs):
        calls.append(args[4])
        return fused(*args, **kwargs)

    def forbid_gate(*args, **kwargs):
        raise AssertionError("Prepared RoPE must not resolve the token gate per attention layer")

    monkeypatch.setattr(mammoth_dit, "fused_qk_norm_rope", record)
    monkeypatch.setattr(mammoth_dit, "fused_qk_norm_rope_min_tokens", forbid_gate)
    with torch.no_grad():
        for _ in range(2):
            got = mammoth_dit._apply_qk_norm_rope(attn, q, k, rotary, rope_repeats_pairs=True)
            expected = (
                mammoth_dit.apply_real_rotary_emb(attn.norm_q(q), cos, sin),
                mammoth_dit.apply_real_rotary_emb(attn.norm_k(k), cos, sin),
            )
            for actual, want in zip(got, expected):
                torch.testing.assert_close(
                    actual, want.to(actual.dtype), atol=0.0625 if packed else 0, rtol=0.02 if packed else 0
                )
    supported = packed and mammoth_dit._fused_cuda_supported(q, k, dim, dim, interleaved=True)
    assert calls == ([table, table] if supported else [])


def test_model_prepares_three_rope_tables_with_one_threshold_lookup(monkeypatch):
    from vllm_omni.diffusion.models.mammoth_moda2.mammothmoda2_dit_model import Transformer2DModel
    from vllm_omni.diffusion.models.mammoth_moda2.rope_real import RotaryPosEmbedReal

    torch.manual_seed(7)
    model = (
        Transformer2DModel(
            patch_size=2,
            in_channels=4,
            hidden_size=48,
            num_layers=1,
            num_refiner_layers=1,
            num_attention_heads=6,
            num_kv_heads=2,
            multiple_of=8,
            ffn_dim_multiplier=1.0,
            axes_dim_rope=(2, 2, 4),
            axes_lens=(8, 8, 8),
            text_feat_dim=32,
        )
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )
    hidden = torch.randn(2, 4, 4, 4, device="cuda", dtype=torch.bfloat16)
    text = torch.randn(2, 4, 32, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(2, 4, device="cuda", dtype=torch.bool)
    freqs = RotaryPosEmbedReal.get_freqs_real((2, 2, 4), (8, 8, 8), 10000)
    calls = []

    def threshold(default):
        calls.append(default)
        return 9

    monkeypatch.setattr(mammoth_dit, "fused_qk_norm_rope_min_tokens", threshold)
    with torch.no_grad():
        prepared = model._prepare_embeddings(hidden, torch.ones(2, device="cuda"), text, mask, freqs, 2, 4, 4)
    context, noise, joint = prepared[6:9]
    assert calls == [0]
    assert all(len(rotary) == 3 for rotary in (context, noise, joint))
    assert context[2] is None and noise[2] is None
    assert joint[2].shape == (2 * 8, 8)
    for rotary in (context, noise, joint):
        if rotary[2] is not None:
            expected = torch.cat((rotary[0][..., 0::2], rotary[1][..., 0::2]), dim=-1).reshape(-1, 8).float()
            torch.testing.assert_close(rotary[2], expected, atol=0, rtol=0)

    # Exercise the prepared tuple through every real refiner and joint block,
    # then compare the same model forward with the unsupported-device fallback.
    args = (hidden, torch.ones(2, device="cuda"), text, freqs, mask)
    with torch.no_grad():
        fused_output = model(*args)
        monkeypatch.setattr(mammoth_dit, "_fused_cuda_supported", lambda *args, **kwargs: False)
        native_output = model(*args)
    assert fused_output.shape == hidden.shape
    torch.testing.assert_close(fused_output, native_output, atol=0.0625, rtol=0.02)
