# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import functools
from dataclasses import dataclass

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
)
from vllm_omni.diffusion.attention.backends.utils import fa
from vllm_omni.diffusion.attention.capabilities import (
    CompilationMode,
    ExecutionContext,
    OuterBoundary,
    ParallelStrategy,
    SupportStatus,
)
from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import ForwardContext
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cuda]


@pytest.fixture(autouse=True)
def isolated_compiler_cache():
    # Each case changes kernel/configuration on the same Attention.forward code
    # object. Keep unrelated cases from exhausting its Dynamo recompile limit;
    # retain the cache across all shapes and replays within an individual test.
    torch.compiler.reset()
    yield
    torch.compiler.reset()


@dataclass
class _ParallelStrategyStub:
    name: str
    enabled: bool = True


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


@pytest.fixture
def fake_fa4(monkeypatch, tmp_path):
    marker = tmp_path / "kernel"
    marker.write_text("loaded", encoding="utf-8")

    @functools.cache
    def cached_kernel_loader():
        with open(marker, encoding="utf-8") as handle:
            handle.read()

    def fake_attention(query, key, value, **kwargs):
        cached_kernel_loader()
        return (
            torch.nn.functional.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                scale=kwargs["softmax_scale"],
                is_causal=kwargs["causal"],
            )
            .transpose(1, 2)
            .contiguous()
        )

    monkeypatch.setattr(fa, "HAS_FLASH_ATTN", True)
    monkeypatch.setattr(fa, "IS_FLASH_ATTN_4", True)
    monkeypatch.setattr(fa, "flash_attn_func", fake_attention)
    monkeypatch.setattr(fa, "validate_fa4_head_dims", lambda *_args: True)

    return FlashAttentionImpl(
        num_heads=8,
        head_size=64,
        softmax_scale=0.125,
        causal=False,
    )


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_fa4_dense_dispatch_is_opaque_to_dynamic_torch_compile(fake_fa4):
    """Fast Dynamo/custom-op regression; Inductor coverage is separate."""
    impl = fake_fa4
    q = torch.randn(1, 16, 8, 64, device="cuda", dtype=torch.bfloat16)
    path = impl.resolve_execution_path(
        ExecutionContext(
            platform="cuda",
            kernel_variant="fa4",
            dtype="bfloat16",
            require_fullgraph=True,
        ),
        q,
        q,
        q,
        None,
    )
    assert path.support.status is SupportStatus.SUPPORTED
    assert path.compilation_mode is CompilationMode.CUSTOM_OP

    compile_count = 0

    def counting_backend(graph_module, _example_inputs):
        nonlocal compile_count
        compile_count += 1
        return graph_module.forward

    compiled = torch.compile(
        lambda query, key, value: impl.forward_cuda(query, key, value),
        backend=counting_backend,
        fullgraph=True,
        dynamic=True,
    )
    out = compiled(q, q, q)
    q2 = torch.randn(1, 24, 8, 64, device="cuda", dtype=torch.bfloat16)
    out2 = compiled(q2, q2, q2)

    assert out.shape == q.shape
    assert out2.shape == q2.shape
    torch.testing.assert_close(out, impl.forward_cuda(q, q, q))
    torch.testing.assert_close(out2, impl.forward_cuda(q2, q2, q2))
    assert compile_count == 1


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_fa4_production_attention_entry_compiles_with_inductor(
    monkeypatch,
    fake_fa4,
):
    """Compile through Attention.forward with the default Inductor backend."""
    from vllm_omni.diffusion.attention import layer as attention_layer

    layer = _make_attention_layer(monkeypatch)
    layer.attention = fake_fa4
    q = torch.randn(1, 16, 8, 64, device="cuda", dtype=torch.bfloat16)

    resolved_contexts = []
    resolve_execution_path = fake_fa4.resolve_execution_path

    def record_execution_path(context, query, key, value, attn_metadata):
        resolved_contexts.append(context)
        return resolve_execution_path(
            context,
            query,
            key,
            value,
            attn_metadata,
        )

    monkeypatch.setattr(fake_fa4, "resolve_execution_path", record_execution_path)
    context = ExecutionContext(
        platform="cuda",
        kernel_variant="fa4",
        dtype="bfloat16",
        paged_kv=True,
        parallel_strategy=ParallelStrategy.ULYSSES,
        require_fullgraph=True,
    )
    path = layer.resolve_execution_path(context, q, q, q, None)
    assert path.support.status is SupportStatus.SUPPORTED
    assert path.compilation_mode is CompilationMode.CUSTOM_OP
    assert resolved_contexts[-1].paged_kv is False
    assert resolved_contexts[-1].parallel_strategy is ParallelStrategy.NONE

    compiled = torch.compile(layer, fullgraph=True, dynamic=True)
    out = compiled(q, q, q)

    assert out.shape == q.shape
    torch.testing.assert_close(out, layer(q, q, q))
    q2 = torch.randn(1, 24, 8, 64, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(compiled(q2, q2, q2), layer(q2, q2, q2))

    layer.paged_kv_cache_role = "self"
    monkeypatch.setattr(layer, "_active_paged_kv_adapter", lambda: object())
    paged_path = layer.resolve_execution_path(context, q, q, q, None)
    assert resolved_contexts[-1].paged_kv is True
    assert paged_path.support.status is SupportStatus.UNMIGRATED
    layer.paged_kv_cache_role = None

    layer.use_ring = True
    layer.parallel_strategy = _ParallelStrategyStub(name="ulysses")
    hybrid_path = layer.resolve_execution_path(context, q, q, q, None)
    assert resolved_contexts[-1].parallel_strategy is ParallelStrategy.HYBRID_ULYSSES_RING
    assert hybrid_path.support.status is SupportStatus.UNMIGRATED
    layer.use_ring = False
    layer.parallel_strategy = NoParallelAttention()

    layer._hsdp_compile_boundary_enabled = True
    hsdp_path = layer.resolve_execution_path(context, q, q, q, None)
    assert OuterBoundary.HSDP in resolved_contexts[-1].outer_boundaries
    assert hsdp_path.support.status is SupportStatus.UNMIGRATED
    assert hsdp_path.compilation_mode is CompilationMode.EAGER_ONLY

    layer._hsdp_compile_boundary_enabled = False
    runtime_context = ForwardContext(
        omni_diffusion_config=OmniDiffusionConfig(
            parallel_config=DiffusionParallelConfig(use_hsdp=True, hsdp_shard_size=1),
        ),
    )
    monkeypatch.setattr(attention_layer, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(attention_layer, "get_forward_context", lambda: runtime_context)
    layer.use_ring = True
    layer.parallel_strategy = _ParallelStrategyStub(name="ring")
    runtime_hsdp_path = layer.resolve_execution_path(context, q, q, q, None)
    assert OuterBoundary.HSDP in resolved_contexts[-1].outer_boundaries
    assert resolved_contexts[-1].parallel_strategy is ParallelStrategy.NONE
    assert runtime_hsdp_path.support.status is SupportStatus.UNMIGRATED


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
    context = ExecutionContext(platform="cuda", require_fullgraph=True)
    # Tile boundaries, cross-attention lengths, batching, and a longer sequence.
    for batch, q_length, kv_length in ((1, 16, 16), (1, 129, 257), (2, 257, 129), (1, 1024, 1024)):
        query, key, value = (
            torch.randn(batch, length, 8, dim, device="cuda", dtype=torch.bfloat16, generator=generator)
            for length, dim in ((q_length, head_size), (kv_length, head_size), (kv_length, value_head_size))
        )
        path = layer.resolve_execution_path(context, query, key, value, None)
        assert path.requested_support(context).status is SupportStatus.SUPPORTED
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
@pytest.mark.parametrize("head_dims", [(192, 192), (65, 64), (64, 65)])
def test_real_fa4_dimension_rejection_matches_kernel(monkeypatch, head_dims):
    capability = current_omni_platform.get_device_capability() if current_omni_platform.is_cuda() else None
    if capability is None or capability.major not in (10, 11) or not fa.IS_FLASH_ATTN_4:
        pytest.skip("Requires SM100/SM110 CuTe FlashAttention-4")
    head_size, value_head_size = head_dims
    layer = _make_attention_layer(monkeypatch, head_size=head_size)
    query = torch.randn(1, 16, 8, head_size, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(1, 16, 8, value_head_size, device="cuda", dtype=torch.bfloat16)
    context = ExecutionContext(platform="cuda", require_fullgraph=True)
    path = layer.resolve_execution_path(context, query, query, value, None)
    assert path.requested_support(context).status is SupportStatus.UNSUPPORTED
    with pytest.raises(AssertionError) as error:
        layer(query, query, value)
    assert str(error.value) in path.support.reason


@hardware_test(res={"cuda": "B200"}, num_cards=1)
def test_real_fa4_custom_op_unequal_value_dimension_schema():
    if not current_omni_platform.is_cuda() or not fa.IS_FLASH_ATTN_4:
        pytest.skip("Requires CuTe FlashAttention-4")
    query = torch.randn(1, 17, 8, 80, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 25, 8, 80, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(1, 25, 8, 48, device="cuda", dtype=torch.bfloat16)
    torch.library.opcheck(
        torch.ops.vllm_omni.fa4_dense_attention.default,
        (query, key, value, 80**-0.5, False, False),
    )
