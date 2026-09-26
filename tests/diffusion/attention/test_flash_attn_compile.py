# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import functools

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionBackend, FlashAttentionImpl
from vllm_omni.diffusion.attention.backends.utils import fa
from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cuda]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize("value_head_size", [64, 48])
def test_fa4_dense_dispatch_is_opaque_to_dynamic_torch_compile(monkeypatch, tmp_path, noncontiguous, value_head_size):
    """Verify the compile boundary with a mocked FA4 dispatcher; real FA4 hardware is not required."""
    marker = tmp_path / "kernel"
    marker.write_text("loaded", encoding="utf-8")

    @functools.cache
    def cached_kernel_loader():
        with open(marker, encoding="utf-8") as handle:
            handle.read()

    def fake_attention(query, key, value, **_kwargs):
        cached_kernel_loader()
        return query.new_empty((*query.shape[:-1], value.shape[-1]))

    monkeypatch.setattr(fa, "HAS_FLASH_ATTN", True)
    monkeypatch.setattr(fa, "IS_FLASH_ATTN_4", True)
    monkeypatch.setattr(fa, "flash_attn_func", fake_attention)

    impl = FlashAttentionImpl(
        num_heads=8,
        head_size=64,
        softmax_scale=0.125,
        causal=False,
    )
    q = torch.randn(1, 16, 8, 64, device="cuda", dtype=torch.bfloat16)
    if noncontiguous:
        q = q.transpose(1, 2).contiguous().transpose(1, 2)
        assert not q.is_contiguous()
    value = torch.randn(1, 16, 8, value_head_size, device="cuda", dtype=torch.bfloat16)
    if noncontiguous:
        value = value.transpose(1, 2).contiguous().transpose(1, 2)
    compiled = torch.compile(
        lambda query, key, value: impl.forward_cuda(query, key, value),
        fullgraph=True,
        dynamic=True,
    )
    out = compiled(q, q, value)

    assert out.shape == (*q.shape[:-1], value_head_size)
    assert out.is_contiguous()


@pytest.fixture(autouse=True)
def isolated_compiler_cache():
    # Each case changes kernel/configuration on the same Attention.forward code
    # object. Keep unrelated cases from exhausting its Dynamo recompile limit;
    # retain the cache across all shapes and replays within an individual test.
    torch.compiler.reset()
    yield
    torch.compiler.reset()


def _make_attention_layer(monkeypatch, *, head_size=64):
    from vllm_omni.diffusion.attention import layer as attention_layer

    monkeypatch.setattr(
        attention_layer,
        "get_attn_backend_for_role",
        lambda **_kwargs: (FlashAttentionBackend, None),
    )
    monkeypatch.setattr(
        attention_layer,
        "build_parallel_attention_strategy",
        lambda **_kwargs: NoParallelAttention(),
    )
    return attention_layer.Attention(num_heads=8, head_size=head_size, softmax_scale=head_size**-0.5, causal=False)


@hardware_test(res={"cuda": "B200"}, num_cards=1)
@pytest.mark.parametrize("head_dims", [(32, 32), (64, 64), (80, 48), (192, 128), (256, 256)])
def test_real_fa4_fullgraph_matches_sdpa_across_shapes(monkeypatch, head_dims):
    """Validate real FA4 numerics through the complete compiled attention layer."""
    capability = current_omni_platform.get_device_capability() if current_omni_platform.is_cuda() else None
    if capability is None or capability.major < 10:
        pytest.skip("Requires Blackwell")
    if not fa.IS_FLASH_ATTN_4 or fa.flash_attn_func is None:
        pytest.skip("Requires CuTe FlashAttention-4")

    from torch.nn.attention import SDPBackend, sdpa_kernel

    head_size, value_head_size = head_dims
    layer = _make_attention_layer(monkeypatch, head_size=head_size)
    compiled = torch.compile(layer, fullgraph=True, dynamic=True)
    generator = torch.Generator(device="cuda").manual_seed(42)
    # Tile boundaries, cross-attention lengths, batching, and a longer sequence.
    for batch, q_length, kv_length in ((1, 16, 16), (1, 129, 257), (2, 257, 129), (1, 1024, 1024)):
        query, key, value = (
            torch.randn(batch, length, 8, dim, device="cuda", dtype=torch.bfloat16, generator=generator)
            for length, dim in ((q_length, head_size), (kv_length, head_size), (kv_length, value_head_size))
        )
        with sdpa_kernel(SDPBackend.MATH):
            reference = torch.nn.functional.scaled_dot_product_attention(
                query.transpose(1, 2).float(),
                key.transpose(1, 2).float(),
                value.transpose(1, 2).float(),
                scale=head_size**-0.5,
            ).transpose(1, 2)
        actual = compiled(query, key, value)
        assert actual.dtype == query.dtype
        assert actual.device == query.device
        torch.testing.assert_close(actual.float(), reference, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(actual, layer(query, key, value))


@hardware_test(res={"cuda": "B200"}, num_cards=1)
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_real_fa4_custom_op_unequal_value_dimension_schema(noncontiguous):
    if not current_omni_platform.is_cuda() or not fa.IS_FLASH_ATTN_4:
        pytest.skip("Requires CuTe FlashAttention-4")
    query = torch.randn(1, 17, 8, 80, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 25, 8, 80, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(1, 25, 8, 48, device="cuda", dtype=torch.bfloat16)
    if noncontiguous:
        query, key, value = (t.transpose(1, 2).contiguous().transpose(1, 2) for t in (query, key, value))
        assert not query.is_contiguous()
    torch.library.opcheck(
        torch.ops.vllm_omni.fa4_dense_attention.default,
        (query, key, value, 80**-0.5, False, False),
    )
