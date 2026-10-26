# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Drama 3 GDN core for steps with prefills vs vLLM's standard ``_forward_core`` (CUDA + FlashInfer).

Each path runs on its own copy of the conv / SSM state pools; the outputs and
every pool row must agree. The batch mixes decode requests, a fresh prefill
whose pool row holds a previous request's state, and a continued prefill.
"""

import types

import pytest
import torch

from tests.helpers.mark import hardware_marks, hardware_test

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "H100"}, num_cards=1)]

K_HEADS, V_HEADS, HEAD_K, HEAD_V, WIDTH = 16, 32, 128, 128, 4


def _layer(device, pools):
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
        fi_chunk_gated_delta_rule,
    )

    dim = 2 * K_HEADS * HEAD_K + V_HEADS * HEAD_V
    g = torch.Generator(device="cpu").manual_seed(0)
    layer = types.SimpleNamespace(
        kv_cache=pools,
        conv1d=types.SimpleNamespace(
            weight=(torch.randn(dim, 1, WIDTH, generator=g) * 0.3).to(device, torch.bfloat16),
            bias=(torch.randn(dim, generator=g) * 0.1).to(device, torch.bfloat16),
        ),
        activation="silu",
        A_log=torch.randn(V_HEADS, generator=g).to(device),
        dt_bias=torch.randn(V_HEADS, generator=g).to(device),
        num_k_heads=K_HEADS,
        num_v_heads=V_HEADS,
        tp_size=1,
        head_k_dim=HEAD_K,
        head_v_dim=HEAD_V,
        key_dim=K_HEADS * HEAD_K,
        value_dim=V_HEADS * HEAD_V,
        prefix="gdn",
        enable_packed_recurrent_decode=False,
    )

    def chunk(**kw):
        return fi_chunk_gated_delta_rule(
            q=kw["q"],
            k=kw["k"],
            v=kw["v"],
            g=kw["g"],
            beta=kw["beta"],
            initial_state=kw["initial_state"],
            output_final_state=kw["output_final_state"],
            cu_seqlens=kw["cu_seqlens"],
            use_qk_l2norm_in_kernel=kw["use_qk_l2norm_in_kernel"],
        )

    layer.chunk_gated_delta_rule = _CallableNamespace(chunk, gdn_prefill_backend="flashinfer")
    layer.rearrange_mixed_qkv = types.MethodType(QwenGatedDeltaNetAttention.rearrange_mixed_qkv, layer)
    return layer, dim


class _CallableNamespace:
    def __init__(self, fn, **attrs):
        self._fn = fn
        self.__dict__.update(attrs)

    def __call__(self, **kw):
        return self._fn(**kw)


def _metadata(device, decode_lens, prefill, slots):
    """``decode_lens`` decodes of one token, then ``prefill`` = [(length, has_initial_state)]."""
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    lengths = [1] * decode_lens + [n for n, _ in prefill]
    starts = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)), dtype=torch.int32)
    has_initial = torch.tensor([True] * decode_lens + [h for _, h in prefill], device=device)
    decode_tokens = decode_lens
    nums_dict, batch_ptr, offset_ptr = compute_causal_conv1d_metadata(starts, device=device)
    slots = torch.tensor(slots, dtype=torch.int32, device=device)
    starts_dev = starts.to(device)
    md = GDNAttentionMetadata(
        num_prefills=len(prefill),
        num_prefill_tokens=int(starts[-1]) - decode_tokens,
        num_decodes=decode_lens,
        num_decode_tokens=decode_tokens,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=int(starts[-1]),
        has_initial_state=has_initial,
        non_spec_query_start_loc=starts_dev,
        non_spec_state_indices_tensor=slots,
        prefill_query_start_loc=starts_dev[decode_lens:] - decode_tokens,
        prefill_state_indices=slots[decode_lens:],
        prefill_has_initial_state=has_initial[decode_lens:],
        nums_dict=nums_dict,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=offset_ptr,
    )
    return md


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
# FlashInfer prefills read / write the pool in place; another backend takes the
# gather / scatter fallback around the same chunk kernel.
@pytest.mark.parametrize("prefill_backend", ["flashinfer", "triton"])
@pytest.mark.parametrize("decodes", [3, 0])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_drama3_gdn_core_matches_standard_path(monkeypatch, prefill_backend, decodes, state_dtype):
    pytest.importorskip("flashinfer.gdn_prefill")
    from vllm.model_executor.layers.mamba import mamba_utils
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn_module
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention

    from vllm_omni.model_executor.models.fish_omni.gdn import mixed_forward_core

    device = torch.device("cuda")
    conv_shape, ssm_shape = mamba_utils.MambaStateShapeCalculator.gated_delta_net_state_shape(
        1, K_HEADS, V_HEADS, HEAD_K, HEAD_V, WIDTH
    )
    pool_rows = 9
    g = torch.Generator(device="cpu").manual_seed(1)
    conv_pool = (torch.randn(pool_rows, *conv_shape, generator=g) * 0.5).to(device, torch.bfloat16)
    ssm_pool = (torch.randn(pool_rows, *ssm_shape, generator=g) * 0.05).to(device, state_dtype)
    # Fresh prefill in slot 6 sits on stale state; continued prefill in slot 2.
    prefill = [(37, False), (70, True)]
    slots = [1, 4, 7][:decodes] + [6, 2]
    md = _metadata(device, decodes, prefill, slots)
    tokens = md.num_actual_tokens

    def run(fn):
        pools = (conv_pool.clone(), ssm_pool.clone())
        layer, dim = _layer(device, pools)
        layer.chunk_gated_delta_rule.gdn_prefill_backend = prefill_backend
        torch.manual_seed(2)
        mixed_qkv = (torch.randn(tokens, dim, device=device) * 0.5).to(torch.bfloat16)
        b = torch.randn(tokens, V_HEADS, device=device).to(torch.bfloat16)
        a = torch.randn(tokens, V_HEADS, device=device).to(torch.bfloat16)
        out = torch.zeros(tokens + 3, V_HEADS, HEAD_V, device=device, dtype=torch.bfloat16)
        fn(layer, mixed_qkv, b, a, out)
        return out, pools

    monkeypatch.setattr(gdn_module, "get_forward_context", lambda: types.SimpleNamespace(attn_metadata={"gdn": md}))
    ref_out, (ref_conv, ref_ssm) = run(lambda layer, *args: QwenGatedDeltaNetAttention._forward_core(layer, *args))
    new_out, (new_conv, new_ssm) = run(lambda layer, *args: mixed_forward_core(layer, *args, md))

    torch.testing.assert_close(new_out.float(), ref_out.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(new_conv, ref_conv)
    # A bf16 pool rounds both paths' final states; allow one bf16 ulp.
    tol = 2e-3 if state_dtype == torch.float32 else 1e-2
    torch.testing.assert_close(new_ssm.float(), ref_ssm.float(), atol=tol, rtol=tol)
    # Untouched slots keep their state bit for bit.
    used = set(slots)
    for row in range(pool_rows):
        if row not in used:
            assert torch.equal(new_ssm[row], ssm_pool[row])


@hardware_test(res={"rocm": "MI325"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_zero_fresh_state_rows_only_touches_fresh_slots():
    from vllm_omni.model_executor.models.fish_omni.gdn import zero_fresh_state_rows_

    pool = torch.randn(6, 4, 8, 8, device="cuda")
    # A padded first-dimension stride, like a view into a larger cache.
    padded = torch.randn(6, 2, 4, 8, 8, device="cuda")[:, 0]
    for target in (pool, padded):
        before = target.clone()
        zero_fresh_state_rows_(
            target,
            torch.tensor([3, 1, 5], dtype=torch.int32, device="cuda"),
            torch.tensor([False, True, False], device="cuda"),
        )
        assert torch.count_nonzero(target[3]) == 0 and torch.count_nonzero(target[5]) == 0
        for row in (0, 1, 2, 4):
            assert torch.equal(target[row], before[row])


@hardware_test(res={"rocm": "MI325"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("backend", ["flashinfer", "triton"])
def test_metadata_builder_skips_fla_chunk_metadata_only_for_flashinfer(backend):
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder

    from vllm_omni.model_executor.models.fish_omni.gdn import (
        Drama3GatedDeltaNet,
        Drama3GDNAttentionMetadataBuilder,
    )

    assert issubclass(Drama3GatedDeltaNet.get_attn_backend(None).get_builder_cls(), GDNAttentionMetadataBuilder)
    starts_cpu = torch.tensor([0, 37, 107, 300], dtype=torch.int32)
    builder = object.__new__(Drama3GDNAttentionMetadataBuilder)
    builder.gdn_prefill_backend = backend
    reference = object.__new__(GDNAttentionMetadataBuilder)
    reference.gdn_prefill_backend = backend
    args = (starts_cpu.cuda(), starts_cpu, torch.device("cuda"))
    got = builder._build_chunk_metadata(*args)
    if backend == "flashinfer":
        assert got == (None, None)
    else:
        for new, ref in zip(got, reference._build_chunk_metadata(*args)):
            assert torch.equal(new, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_flashinfer_prefill_warmup_compiles_before_serving():
    """The warmup compiles both FlashInfer prefill variants once per pool layout."""
    pytest.importorskip("flashinfer.gdn_prefill")
    from vllm.model_executor.layers.mamba import mamba_utils

    from vllm_omni.model_executor.models.fish_omni import gdn

    device = torch.device("cuda")
    _, ssm_shape = mamba_utils.MambaStateShapeCalculator.gated_delta_net_state_shape(
        1, K_HEADS, V_HEADS, HEAD_K, HEAD_V, WIDTH
    )
    pool = torch.zeros(4, *ssm_shape, device=device, dtype=torch.bfloat16)
    layer, _ = _layer(device, (None, pool))
    gdn._FLASHINFER_WARM.clear()
    gdn._flashinfer_prefill_warmup(layer, pool, torch.bfloat16)
    assert len(gdn._FLASHINFER_WARM) == 1
    gdn._flashinfer_prefill_warmup(layer, pool, torch.bfloat16)
    assert len(gdn._FLASHINFER_WARM) == 1
    # The pool itself is untouched (the warmup runs on a scratch pool).
    assert not pool.any()
