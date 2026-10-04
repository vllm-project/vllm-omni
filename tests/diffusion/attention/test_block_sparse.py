# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import math

import pytest
import torch

from tests.helpers.block_sparse import make_attention_inputs, selected_attention_reference
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, PackedPaddingMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionBackend
from vllm_omni.diffusion.attention.capabilities import (
    CompilationMode,
    ExecutionContext,
    PackingMode,
    ParallelStrategy,
    SupportStatus,
)
from vllm_omni.diffusion.data import (
    AttentionConfig,
    BlockSparseAttentionSpec,
    build_attention_config,
    parse_attention_config,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


def test_inline_configuration(monkeypatch):
    monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")
    config = build_attention_config(
        {
            "per_role": {
                "self": {
                    "name": "block_sparse",
                    "config": {
                        "selection": {
                            "name": "block_topk",
                            "config": {"target_sparsity": 0.5},
                        },
                        "backend": {"require": "FLASH_ATTN", "implementation": "auto"},
                    },
                },
                "cosmos3.gen": {"backend": "TORCH_SDPA"},
            }
        }
    )
    assert config.resolve_with_source("cosmos3.gen", "self")[0].backend == "TORCH_SDPA"
    spec = config.resolve_with_source("another", "self")[0]
    assert isinstance(spec, BlockSparseAttentionSpec)
    assert spec.selection == {
        "name": "block_topk",
        "config": {"target_sparsity": 0.5},
    }
    assert spec.block_size == (64, 64)
    assert spec.implementation == "auto"
    assert config.default.backend == "TORCH_SDPA"
    with pytest.raises(ValueError, match="mutually exclusive"):
        parse_attention_config({"default": {"name": "block_sparse"}}, attention_backend="FLASH_ATTN")
    with pytest.raises(ValueError, match="Cannot mix"):
        AttentionConfig(default={"name": "block_sparse", "backend": "FLASH_ATTN"})


@pytest.mark.parametrize(
    "config",
    [
        {"block_size": [0, 64]},
        {"block_size": [64]},
        {"block_size": [True, 64]},
        {"unknown_option": 1},
        {"selection": {"name": "unknown"}},
        {"backend": {"require": "UNKNOWN"}},
        {"backend": {}},
        {"backend": {"require": ""}},
        {"backend": {"require": ["FLASH_ATTN"]}},
        {"backend": {"require": True}},
        {"backend": {"prefer": []}},
        {"backend": {"prefer": ["FLASH_ATTN"], "implementation": "unknown"}},
        {"backend": {"require": "FLASH_ATTN", "prefer": ["FLASH_ATTN"]}},
        {"selection": {"name": "block_topk", "config": {"target_sparsity": float("nan")}}},
        {"selection": {"name": "block_topk", "config": {"target_sparsity": True}}},
        {"selection": {"name": "block_topk", "config": {"target_sparsity": 1}}},
    ],
)
def test_unsupported_configuration(config):
    with pytest.raises((TypeError, ValueError)):
        BlockSparseAttentionSpec(name="block_sparse", config=config)


@pytest.fixture
def hopper():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Requires Hopper SM90")
    pytest.importorskip("flash_attn.cute")


def make_impl(heads=4, kv_heads=2, sparsity=0.75, block_size=(64, 64), head_size=128):
    spec = BlockSparseAttentionSpec(
        name="block_sparse",
        config={
            "block_size": block_size,
            "selection": {"name": "block_topk", "config": {"target_sparsity": sparsity}},
        },
    )
    from vllm_omni.diffusion.attention.block_sparse import BlockSparseAttention

    return BlockSparseAttention(
        heads,
        kv_heads,
        head_size,
        head_size**-0.5,
        False,
        None,
        spec,
        adapter=FlashAttentionBackend.get_block_sparse_adapter()(),
    )


@pytest.mark.cuda
@torch.inference_mode()
def test_selected_key_numerics_and_compile(hopper):
    kv_heads = 2
    impl = make_impl(kv_heads=kv_heads)
    q, k, v = make_attention_inputs(kv_heads=kv_heads)
    metadata = AttentionMetadata(extra={"protected_kv_prefix": 65})
    selection = impl.selector.select(q, k, impl.scale, 65)
    indices = selection.indices
    assert torch.all(indices[..., :2] == torch.tensor([0, 1], device=q.device))
    assert torch.all(indices[..., 1:] > indices[..., :-1])
    expected = selected_attention_reference(q, k, v, selection, impl.scale, impl.block_size)
    impl.prepare_request(q, k, v, metadata)
    eager = impl.forward_cuda(q, k, v, metadata)
    torch.testing.assert_close(eager.float(), expected, atol=0.004, rtol=0.02)
    compiled = torch.compile(impl.forward_cuda, fullgraph=True)
    actual = compiled(q, k, v, metadata)
    torch.testing.assert_close(actual, eager, atol=0, rtol=0)
    # Same prepared layer, different values: intermediates must not retain request data.
    torch.testing.assert_close(compiled(q, k, -v, metadata), -eager, atol=0, rtol=0)
    result = impl.resolve_execution_path(ExecutionContext(platform="cuda", require_fullgraph=True), q, k, v, metadata)
    assert result.path == "fa4_block_sparse" and result.support.status is SupportStatus.SUPPORTED
    assert result.compilation_mode is CompilationMode.CUSTOM_OP


@pytest.mark.cuda
@torch.inference_mode()
def test_capability_rejections(hopper):
    impl = make_impl()
    q, k, v = make_attention_inputs()
    for metadata in [
        AttentionMetadata(attn_mask=torch.ones(1, device=q.device, dtype=torch.bool)),
        AttentionMetadata(extra={"cu_seqlens_q": torch.tensor([0, 129], device=q.device)}),
        AttentionMetadata(extra={"protected_kv_prefix": 1090}),
        AttentionMetadata(extra={"kv_cache_dtype": "fp8"}),
    ]:
        with pytest.raises(ValueError):
            impl.forward_cuda(q, k, v, metadata)
        result = impl.resolve_execution_path(ExecutionContext(platform="cuda"), q, k, v, metadata)
        assert result.support.status is SupportStatus.UNSUPPORTED and result.support.reason
    with pytest.raises(ValueError, match="matching Q/K/V dtypes"):
        impl.forward_cuda(q.float(), k, v)
    result = impl.resolve_execution_path(
        ExecutionContext(platform="cuda", parallel_strategy=ParallelStrategy.RING),
        q,
        k,
        v,
        None,
    )
    assert result.support.status is SupportStatus.UNSUPPORTED
    with pytest.raises(ValueError, match="native head mapping"):
        impl.forward_cuda(q, k.repeat_interleave(2, 2), v.repeat_interleave(2, 2))


@pytest.mark.cuda
@torch.inference_mode()
def test_full_attention_layer_prepares_inline_spec(hopper, monkeypatch):
    from types import SimpleNamespace

    from vllm_omni.diffusion.attention import layer as layer_mod
    from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
    from vllm_omni.diffusion.config import set_current_diffusion_config

    monkeypatch.setattr(layer_mod, "build_parallel_attention_strategy", lambda **kwargs: NoParallelAttention())
    cfg = SimpleNamespace(
        diffusion_attention_config=AttentionConfig(per_role={"cosmos3.gen": {"name": "block_sparse"}}),
        parallel_config=SimpleNamespace(ring_degree=1),
        dtype=torch.bfloat16,
        diffusion_kv_cache_dtype="float",
        diffusion_kv_cache_skip_step_indices=None,
        diffusion_kv_cache_skip_layer_indices=None,
    )
    with set_current_diffusion_config(cfg):
        layer = layer_mod.Attention(4, 128, False, 0.1, num_kv_heads=2, role="cosmos3.gen", role_category="cross")
    from vllm_omni.diffusion.attention.block_sparse import BlockSparseBackend

    assert layer.backend_pref == "FLASH_ATTN"
    assert layer.attn_backend is BlockSparseBackend
    assert not layer.attn_backend.supports_piecewise_spans
    assert not layer.attn_backend.supports_paged_kv
    assert layer.attn_backend.supports_packed_mask_free()
    assert not layer.attn_backend.supports_multi_doc_packed_varlen()
    assert layer.attention.adapter.provider == "FLASH_ATTN"
    assert layer.attention.selector.sparsity == 0.75
    q = torch.randn(1, 65, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 130, 2, 128, device="cuda", dtype=q.dtype)
    context = ExecutionContext(platform="cuda", require_fullgraph=True, kv_cache_dtype="float")
    pending = layer.resolve_execution_path(context, q, k, k, None)
    assert pending.support.status is SupportStatus.UNSUPPORTED
    assert "not prepared" in pending.support.reason
    actual = layer(q, k, k, None)
    assert layer.resolve_execution_path(context, q, k, k, None).support.status is SupportStatus.SUPPORTED
    assert actual.shape == q.shape
    torch.testing.assert_close(torch.compile(layer, fullgraph=True)(q, k, k, None), actual, atol=0, rtol=0)


def test_checkpoint_calibration_preserves_block_sparse_spec():
    from types import SimpleNamespace

    from vllm_omni.diffusion.data import OmniDiffusionConfig

    cfg = OmniDiffusionConfig(diffusion_attention_config={"default": {"name": "block_sparse"}})
    cfg._propagate_skip_softmax_calibration(SimpleNamespace(get=lambda _: None))
    assert isinstance(cfg.diffusion_attention_config.default, BlockSparseAttentionSpec)


def test_provider_neutral_config_and_missing_adapter():
    from vllm_omni.diffusion.attention.selector import get_attn_backend_for_role

    cfg = AttentionConfig(
        default={
            "name": "block_sparse",
            "config": {
                "backend": {"require": "FLASHINFER_ATTN", "implementation": "future-kernel-id"},
            },
        }
    )
    assert cfg.default.implementation == "future-kernel-id"
    with pytest.raises(ValueError, match="FLASHINFER_ATTN: no adapter"):
        get_attn_backend_for_role("self", 128, cfg)
    # IDs are validated by an adapter, not by the common schema.
    cfg = AttentionConfig(
        default={
            "name": "block_sparse",
            "config": {
                "backend": {"require": "FLASH_ATTN", "implementation": "future-kernel-id"},
            },
        }
    )
    with pytest.raises(ValueError, match="FA4 has no kernel-ID"):
        get_attn_backend_for_role("self", 128, cfg)


def test_new_provider_adapter_receives_opaque_id(monkeypatch):
    from vllm_omni.diffusion.attention import selector
    from vllm_omni.diffusion.attention.backends.flashinfer_attn import FlashInferAttentionBackend

    accepted = []

    class FutureAdapter:
        @staticmethod
        def validate_selection(implementation, head_size):
            accepted.append((implementation, head_size))
            if implementation == "unsupported-id":
                raise ValueError("unsupported kernel ID")

    monkeypatch.setattr(FlashInferAttentionBackend, "get_block_sparse_adapter", classmethod(lambda cls: FutureAdapter))

    def reject_dense_resolution(*args):
        pytest.fail("Sparse selection must not use dense backend support checks")

    monkeypatch.setattr(selector, "_cached_get_backend_cls", reject_dense_resolution)
    cfg = AttentionConfig(
        default={
            "name": "block_sparse",
            "config": {
                "backend": {"require": "FLASHINFER_ATTN", "implementation": "future-kernel-id"},
            },
        }
    )
    backend, resolved = selector.get_attn_backend_for_role("self", 128, cfg)
    assert backend is FlashInferAttentionBackend
    assert accepted == [("future-kernel-id", 128)]
    assert resolved is not None
    assert resolved.implementation == "future-kernel-id"
    cfg = AttentionConfig(
        default={
            "name": "block_sparse",
            "config": {
                "backend": {"require": "FLASHINFER_ATTN", "implementation": "unsupported-id"},
            },
        }
    )
    with pytest.raises(ValueError, match="unsupported kernel ID"):
        selector.get_attn_backend_for_role("self", 128, cfg)


@pytest.mark.parametrize("providers", [[], ["FLASH_ATTN"], ["FLASH_ATTN", "FLASHINFER_ATTN"]])
@pytest.mark.parametrize("location", ["default", "per_role"])
def test_provider_preference_is_rejected_before_loading(monkeypatch, providers, location):
    from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum

    def unexpected_load(*args, **kwargs):
        pytest.fail("Rejected preference lists must not load a provider")

    monkeypatch.setattr(DiffusionAttentionBackendEnum, "get_class", unexpected_load)
    entry = {"name": "block_sparse", "config": {"backend": {"prefer": providers}}}
    config = {"default": entry} if location == "default" else {"per_role": {"cosmos3.gen": entry}}
    with pytest.raises(ValueError, match=r"backend.prefer is not supported; use backend.require"):
        AttentionConfig(**config)


@pytest.mark.parametrize("error_type", [ImportError, ValueError, RuntimeError])
def test_pinned_provider_errors_propagate(monkeypatch, error_type):
    from vllm_omni.diffusion.attention.backends.flash_attn import FA4SparseAdapter
    from vllm_omni.diffusion.attention.selector import get_attn_backend_for_role

    error = error_type("provider failure")

    def fail(*args):
        raise error

    monkeypatch.setattr(FA4SparseAdapter, "validate_selection", fail)
    config = AttentionConfig(default={"name": "block_sparse", "config": {"backend": {"require": "FLASH_ATTN"}}})
    with pytest.raises(error_type) as caught:
        get_attn_backend_for_role("self", 128, config)
    assert caught.value is error


@pytest.mark.cuda
@pytest.mark.parametrize("block_size,kv_heads", [((64, 128), 2), ((128, 64), 4), ((128, 128), 2), ((256, 128), 4)])
@torch.inference_mode()
def test_direct_larger_geometry(hopper, monkeypatch, block_size, kv_heads):
    from vllm_omni.diffusion.attention.backends import flash_attn
    from vllm_omni.diffusion.attention.block_selection.subblock_topk import SubBlockTopK

    impl = make_impl(kv_heads=kv_heads, block_size=block_size)
    q, k, v = make_attention_inputs(kv_heads=kv_heads, q_len=257, kv_len=2177)
    metadata = AttentionMetadata(extra={"protected_kv_prefix": 129})
    selection = impl.selector.select(q, k, impl.scale, 129)
    indices = selection.indices
    block_q, block_kv = block_size
    prefix_blocks = math.ceil(129 / block_kv)
    assert indices.shape == (
        2,
        4,
        math.ceil(257 / block_q),
        prefix_blocks + SubBlockTopK.block_budget(math.ceil(2177 / block_kv) - prefix_blocks, 0.75),
    )
    torch.testing.assert_close(
        indices[..., :prefix_blocks],
        torch.arange(prefix_blocks, device=q.device, dtype=torch.int32).expand_as(indices[..., :prefix_blocks]),
    )
    kernel, sparse_type, dependency_version = flash_attn.FA4SparseAdapter._load_api()
    observed = []

    def checked_kernel(*args, **kwargs):
        pattern = kwargs["block_sparse_tensors"]
        observed.append(pattern.block_size)
        # The kernel receives the configured geometry and unchanged logical pattern.
        assert pattern.block_size == block_size
        torch.testing.assert_close(pattern.mask_block_idx, indices, atol=0, rtol=0)
        return kernel(*args, **kwargs)

    monkeypatch.setattr(
        flash_attn.FA4SparseAdapter,
        "_load_api",
        staticmethod(lambda: (checked_kernel, sparse_type, dependency_version)),
    )
    impl.prepare_request(q, k, v, metadata)
    eager = impl.forward(q, k, v, metadata)
    torch.testing.assert_close(
        eager.float(), selected_attention_reference(q, k, v, selection, impl.scale, block_size), atol=0.004, rtol=0.02
    )
    if block_size == (64, 128):
        compiled = torch.compile(impl.forward, fullgraph=True, dynamic=False)
        torch.testing.assert_close(compiled(q, k, v, metadata), eager, atol=0, rtol=0)
    assert observed


@pytest.mark.cuda
@torch.inference_mode()
def test_geometry_rejection_is_owned_by_strategy(hopper):
    # Positive geometry is accepted by the schema; constraints belong to execution.
    spec = BlockSparseAttentionSpec(name="block_sparse", config={"block_size": [96, 128]})
    assert spec.config["block_size"] == [96, 128]
    with pytest.raises(ValueError, match="SubBlock scorer"):
        make_impl(block_size=(96, 128))


def test_adapter_preparation_does_not_probe_synthetic_shapes(monkeypatch):
    from vllm_omni.diffusion.attention.backends.flash_attn import FA4SparseAdapter

    def unexpected_probe(*args, **kwargs):
        pytest.fail("Preparation must defer hardware and shape support to actual execution")

    monkeypatch.setattr(FA4SparseAdapter, "_load_api", staticmethod(lambda: (None, None, "test")))
    monkeypatch.setattr(FA4SparseAdapter, "_run", unexpected_probe)
    monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected_probe)
    adapter = FA4SparseAdapter()
    adapter.prepare("auto", 192, 12, 3, torch.device("cuda:0"), (128, 128))
    assert adapter.dependency_version == "test"


@pytest.fixture
def isolated_compile_cache():
    # Independent model geometries should not exhaust one shared Dynamo frame cache.
    torch.compiler.reset()
    yield
    torch.compiler.reset()


@pytest.mark.cuda
@pytest.mark.parametrize(
    "head_size,value_size,dtype,kv_heads",
    [
        (64, 32, torch.float16, 1),
        (96, 64, torch.bfloat16, 2),
        (192, 128, torch.float16, 4),
    ],
)
@torch.inference_mode()
def test_actual_request_dimensions_and_dtype(hopper, isolated_compile_cache, head_size, value_size, dtype, kv_heads):
    impl = make_impl(head_size=head_size, kv_heads=kv_heads, block_size=(128, 128))
    q, k, v = make_attention_inputs(
        kv_heads=kv_heads, kv_len=2177, head_size=head_size, value_size=value_size, dtype=dtype
    )
    # Exercise scoring with a noncontiguous feature dimension, including padded Triton head tiles.
    q, k, v = (torch.stack((t, t), dim=-1)[..., 0] for t in (q, k, v))
    metadata = AttentionMetadata(extra={"protected_kv_prefix": 129})
    selection = impl.selector.select(q, k, impl.scale, 129)
    expected = selected_attention_reference(q, k, v, selection, impl.scale, impl.block_size)
    impl.prepare_request(q, k, v, metadata)
    actual = impl.forward(q, k, v, metadata)
    assert actual.shape == (*q.shape[:-1], value_size)
    torch.testing.assert_close(actual.float(), expected, atol=0.004, rtol=0.02)
    if head_size == 96:
        compiled = torch.compile(impl.forward, fullgraph=True)
        torch.testing.assert_close(compiled(q, k, v, metadata), actual, atol=0, rtol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("compiled", [False, True])
@torch.inference_mode()
def test_provider_rejection_propagates_during_preparation(hopper, isolated_compile_cache, monkeypatch, compiled):
    impl = make_impl()
    q, k, v = make_attention_inputs()

    def reject(*args, **kwargs):
        raise ValueError("provider rejected actual request")

    monkeypatch.setattr(impl.adapter, "execute", reject)
    context = ExecutionContext(platform="cuda")
    pending = impl.resolve_execution_path(context, q, k, v, None)
    assert pending.support.status is SupportStatus.UNSUPPORTED
    assert "not prepared" in pending.support.reason
    run = (
        torch.compile(impl.forward, backend="eager", fullgraph=True, dynamic=True) if compiled else impl.prepare_request
    )
    with pytest.raises(ValueError, match="provider rejected actual request"):
        run(q, k, v)
    failed = impl.resolve_execution_path(context, q, k, v, None)
    assert failed.support.status is SupportStatus.UNSUPPORTED
    assert "provider rejected actual request" in failed.support.reason


@pytest.mark.cuda
@pytest.mark.parametrize("change", ["length", "strides", "dtype", "value_size", "prefix"])
@torch.inference_mode()
def test_eager_preparation_is_specific_to_request(hopper, change):
    impl = make_impl()
    q, k, v = make_attention_inputs()
    context = ExecutionContext(platform="cuda")
    impl.forward(q, k, v)
    assert impl.resolve_execution_path(context, q, k, v, None).support.status is SupportStatus.SUPPORTED
    metadata = None
    if change == "length":
        q = q[:, :-1]
    elif change == "strides":
        q = torch.stack((q, q), dim=-1)[..., 0]
    elif change == "dtype":
        q, k, v = (t.to(torch.float16) for t in (q, k, v))
    elif change == "value_size":
        v = v[..., :64]
    else:
        metadata = AttentionMetadata(extra={"protected_kv_prefix": 65})
    pending = impl.resolve_execution_path(context, q, k, v, metadata)
    assert pending.support.status is SupportStatus.UNSUPPORTED
    assert "not prepared" in pending.support.reason
    actual = impl.forward(q, k, v, metadata)
    assert actual.shape == (*q.shape[:-1], v.shape[-1])
    assert impl.resolve_execution_path(context, q, k, v, metadata).support.status is SupportStatus.SUPPORTED


@pytest.mark.cuda
@pytest.mark.parametrize("failure_stage", ["execute", "synchronize"])
@torch.inference_mode()
def test_preparation_failure_is_recorded_and_retryable(hopper, monkeypatch, failure_stage):
    impl = make_impl()
    q, k, v = make_attention_inputs()
    context = ExecutionContext(platform="cuda")

    def fail(*args, **kwargs):
        raise RuntimeError("provider preparation failure")

    with monkeypatch.context() as patch:
        target = impl.adapter if failure_stage == "execute" else torch.accelerator
        patch.setattr(target, failure_stage, fail)
        # Resolution must never trial-execute a kernel.
        assert "not prepared" in impl.resolve_execution_path(context, q, k, v, None).support.reason
        with pytest.raises(RuntimeError, match="provider preparation failure"):
            impl.forward(q, k, v)
        failed = impl.resolve_execution_path(context, q, k, v, None)
        assert failed.support.status is SupportStatus.UNSUPPORTED
        assert "provider preparation failure" in failed.support.reason
    impl.prepare_request(q, k, v)
    assert impl.resolve_execution_path(context, q, k, v, None).support.status is SupportStatus.SUPPORTED


@pytest.mark.cuda
@torch.inference_mode()
def test_dynamic_compilation_does_not_specialize_prepared_geometry(hopper, isolated_compile_cache):
    from torch._dynamo.testing import CompileCounterWithBackend

    impl = make_impl()
    # Exceed Dynamo's default recompilation limit without resetting its cache.
    inputs = [make_attention_inputs(q_len=65 + 64 * index, kv_len=1089 + 64 * index) for index in range(12)]
    expected = [impl.forward(*tensors) for tensors in inputs]
    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(impl.forward, backend=counter, fullgraph=True, dynamic=True)
    for tensors, eager in zip(inputs, expected):
        torch.testing.assert_close(compiled(*tensors), eager, atol=0, rtol=0)
    assert counter.frame_count == 1
    # A new geometry is prepared by the runtime handler, without graph fallback
    # or another graph. Capability queries remain read-only until that execution.
    tensors = make_attention_inputs(q_len=833, kv_len=1857)
    context = ExecutionContext(platform="cuda")
    assert impl.resolve_execution_path(context, *tensors, None).support.status is SupportStatus.UNSUPPORTED
    actual = compiled(*tensors)
    assert impl.resolve_execution_path(context, *tensors, None).support.status is SupportStatus.SUPPORTED
    torch.testing.assert_close(actual, impl.forward(*tensors), atol=0, rtol=0)
    assert counter.frame_count == 1


@pytest.mark.cuda
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@torch.inference_mode()
def test_fa4_dynamic_adapter_preserves_changing_selections(hopper, isolated_compile_cache, kv_heads):
    from torch._dynamo.testing import CompileCounterWithBackend

    from vllm_omni.diffusion.attention.block_selection.abstract import BlockSelection

    adapter = FlashAttentionBackend.get_block_sparse_adapter()()
    block_size = (64, 64)
    scale = 128**-0.5
    adapter.prepare("auto", 128, 4, kv_heads, torch.device("cuda"), block_size)

    # Scale and block geometry are fixed configuration, not dynamic inputs.
    # Passing scale as a float argument makes Dynamo try symbolic scalar
    # compilation before retrying with a specialized constant.
    def execute(q, k, v, selection):
        return adapter.execute(q, k, v, selection, 128**-0.5, (64, 64))

    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(execute, backend=counter, fullgraph=True, dynamic=True)
    for q_len, kv_len in ((129, 321), (257, 449), (385, 577), (129, 321)):
        q, k, v = make_attention_inputs(kv_heads=kv_heads, q_len=q_len, kv_len=kv_len)
        rows = (q.shape[0], q.shape[2], math.ceil(q_len / block_size[0]))
        first = (torch.arange(math.prod(rows), device=q.device).reshape(rows) % 2).int()
        counts = first + 1
        # Include the partial last KV block and poison inactive storage.
        last = torch.where(counts == 2, math.ceil(kv_len / block_size[1]) - 1, -999).int()
        indices = torch.stack((first, last), dim=-1)
        initial_result = None
        for active_first in (first, 1 - first, first):
            # Same shapes/counts, different per-batch/head/query-row selections.
            selection = BlockSelection(torch.stack((active_first, last), dim=-1), counts)
            expected = selected_attention_reference(q, k, v, selection, scale, block_size)
            eager = adapter.execute(q, k, v, selection, scale, block_size)
            actual = compiled(q, k, v, selection)
            torch.testing.assert_close(actual.float(), expected, atol=0.004, rtol=0.02)
            torch.testing.assert_close(actual, eager, atol=0, rtol=0)
            assert actual.is_contiguous() and actual.dtype == q.dtype and actual.device == q.device
            if initial_result is None:
                initial_result = actual.clone()
            elif torch.equal(selection.indices, indices):
                torch.testing.assert_close(actual, initial_result, atol=0, rtol=0)
            else:
                assert not torch.equal(actual, initial_result)
    # No compiler retries or recompilation across lengths and selections.
    assert counter.frame_count == 1


@pytest.mark.cuda
@pytest.mark.parametrize("fullgraph", [False, True])
@torch.inference_mode()
def test_compiled_first_request_prepares_without_graph_break(hopper, isolated_compile_cache, fullgraph):
    from torch._dynamo.testing import CompileCounterWithBackend

    impl = make_impl()
    tensors = make_attention_inputs()
    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(impl.forward, backend=counter, fullgraph=fullgraph, dynamic=True)
    context = ExecutionContext(platform="cuda")
    assert impl.resolve_execution_path(context, *tensors, None).support.status is SupportStatus.UNSUPPORTED
    output = compiled(*tensors)
    assert impl.resolve_execution_path(context, *tensors, None).support.status is SupportStatus.SUPPORTED
    torch.testing.assert_close(output, impl.forward(*tensors), atol=0, rtol=0)
    assert counter.frame_count == 1


def test_sparse_preflight_does_not_inherit_dense_provider_capabilities():
    from vllm_omni.diffusion.attention.block_sparse import BlockSparseAttention, BlockSparseBackend
    from vllm_omni.diffusion.attention.selector import get_attn_backend_for_capability

    config = AttentionConfig(per_role={"self": {"name": "block_sparse"}})
    backend = get_attn_backend_for_capability("self", config)
    assert backend is BlockSparseBackend
    assert backend.get_impl_cls() is BlockSparseAttention
    assert not backend.supports_attention_mask()
    assert backend.supports_packed_mask_free()
    assert not backend.supports_multi_doc_packed_varlen()
    assert not backend.supports_piecewise_spans
    assert not backend.supports_prefix_kv_slicing
    assert not backend.supports_paged_kv
    assert not backend.accept_output_buffer
    result = backend.resolve_capabilities(ExecutionContext(platform="cuda", require_fullgraph=True))
    assert result.support.status is SupportStatus.UNSUPPORTED
    assert "preparation" in result.support.reason
    # Dense selection still exposes the provider's own declarations.
    dense = get_attn_backend_for_capability("self", AttentionConfig(default={"backend": "FLASH_ATTN"}))
    assert dense is FlashAttentionBackend
    assert dense.supports_piecewise_spans and dense.supports_paged_kv


@pytest.mark.cuda
@pytest.mark.parametrize("fullgraph", [False, True])
@torch.inference_mode()
def test_regional_compilation_reuses_graph_across_sparse_owners(hopper, isolated_compile_cache, fullgraph):
    from torch._dynamo.testing import CompileCounterWithBackend

    from vllm_omni.diffusion.compile import regionally_compile

    calls = []

    class SparseTestBlock(torch.nn.Module):
        def __init__(self, index):
            super().__init__()
            self.attention = make_impl()
            execute = self.attention.adapter.execute

            def tracked_execute(*args, **kwargs):
                calls.append(index)
                # Give each owner a distinct result as well as a dispatch trace.
                return execute(*args, **kwargs) * (index + 1)

            self.attention.adapter.execute = tracked_execute

        def forward(self, q, k, v):
            return self.attention.forward(q, k, v)

    model = torch.nn.Module()
    model._repeated_blocks = ["SparseTestBlock"]
    model.blocks = torch.nn.ModuleList(SparseTestBlock(i) for i in range(12))
    # Cross both Q and KV block boundaries, then revisit the first geometry.
    inputs = [make_attention_inputs(q_len=q_len, kv_len=kv_len) for q_len, kv_len in ((65, 1089), (193, 1217))]
    inputs.append(inputs[0])
    expected = [[block(*tensors) for block in model.blocks] for tensors in inputs]
    counter = CompileCounterWithBackend("inductor")
    # Exactly the production regional-compilation helper: separate torch.compile
    # wrappers for the same forward code on distinct block instances.
    regionally_compile(model, backend=counter, fullgraph=fullgraph, dynamic=True)
    for tensors, outputs in zip(inputs, expected):
        calls.clear()
        for block, reference in zip(model.blocks, outputs):
            torch.testing.assert_close(block(*tensors), reference, atol=0, rtol=0)
        assert calls == list(range(12))
    # No cache resets or raised recompilation limit between owners or lengths.
    assert counter.frame_count == 1


def make_padded_request(q_length=65, kv_length=1089, prefix=65):
    q, k, v = (
        tensor[:1] for tensor in make_attention_inputs(q_len=q_length + 31, kv_len=kv_length + 47, value_size=64)
    )
    q[:, q_length:] = float("nan")
    k[:, kv_length:] = float("nan")
    v[:, kv_length:] = float("nan")
    cu_q = torch.tensor([0, q_length, q.shape[1]], device=q.device, dtype=torch.int32)
    cu_k = torch.tensor([0, kv_length, k.shape[1]], device=k.device, dtype=torch.int32)
    metadata = AttentionMetadata(
        packed_padding=PackedPaddingMetadata(q_length, kv_length, cu_q[:2], cu_k[:2]),
        extra={
            "cu_seqlens_q": cu_q,
            "cu_seqlens_k": cu_k,
            "max_seqlen_q": q_length,
            "max_seqlen_k": kv_length,
            "valid_kv_length": kv_length,
            "protected_kv_prefix": prefix,
        },
    )
    return q, k, v, metadata


@pytest.mark.cuda
@pytest.mark.parametrize("q_length,kv_length,prefix", [(65, 1089, 65), (97, 2051, 0)])
@torch.inference_mode()
def test_single_document_padding_preserves_routing_and_zeroes_tail(hopper, q_length, kv_length, prefix):
    impl = make_impl()
    q, k, v, metadata = make_padded_request(q_length, kv_length, prefix)
    real = (q[:, :q_length], k[:, :kv_length], v[:, :kv_length])
    plain_metadata = AttentionMetadata(extra={"protected_kv_prefix": prefix})
    expected = impl.forward(*real, plain_metadata)
    context = ExecutionContext(platform="cuda", packing_mode=PackingMode.PACKED_PADDING)
    # Preparing the executed views also prepares the equivalent typed request.
    assert impl.resolve_execution_path(context, q, k, v, metadata).support.status is SupportStatus.SUPPORTED
    actual = impl.forward(q, k, v, metadata)
    torch.testing.assert_close(actual[:, :q_length], expected, atol=0, rtol=0)
    assert actual.shape == (1, q.shape[1], q.shape[2], v.shape[-1]) and actual.is_contiguous()
    assert torch.count_nonzero(actual[:, q_length:]) == 0
    # Values in either padding region must have no influence, including on the
    # last partially filled real query block's pooled routing scores.
    q[:, q_length:] = 10000
    k[:, kv_length:] = -10000
    v[:, kv_length:] = 10000
    torch.testing.assert_close(impl.forward(q, k, v, metadata), actual, atol=0, rtol=0)
    # The typed contract alone is sufficient; redundant extras are optional.
    typed_only = AttentionMetadata(packed_padding=metadata.packed_padding, extra=plain_metadata.extra)
    torch.testing.assert_close(impl.forward(q, k, v, typed_only), actual, atol=0, rtol=0)
    larger = tuple(torch.cat((t, t.new_full((1, 17, *t.shape[2:]), float("nan"))), dim=1) for t in (q, k, v))
    extended = impl.forward(*larger, typed_only)
    torch.testing.assert_close(extended[:, :q_length], expected, atol=0, rtol=0)
    assert torch.count_nonzero(extended[:, q_length:]) == 0


@pytest.mark.cuda
@torch.inference_mode()
def test_padding_preparation_uses_executed_views(hopper):
    impl = make_impl()
    q, k, v, metadata = make_padded_request()
    context = ExecutionContext(platform="cuda", packing_mode=PackingMode.PACKED_PADDING)
    assert impl.resolve_execution_path(context, q, k, v, metadata).support.status is SupportStatus.UNSUPPORTED
    impl.prepare_request(q, k, v, metadata)
    plain = AttentionMetadata(extra={"protected_kv_prefix": 65})
    assert (
        impl.resolve_execution_path(
            ExecutionContext(platform="cuda"), q[:, :65], k[:, :1089], v[:, :1089], plain
        ).support.status
        is SupportStatus.SUPPORTED
    )
    # The padded storage was never prepared as a real attention request.
    assert (
        impl.resolve_execution_path(ExecutionContext(platform="cuda"), q, k, v, plain).support.status
        is SupportStatus.UNSUPPORTED
    )
    assert impl.resolve_execution_path(context, q, k, v, metadata).support.status is SupportStatus.SUPPORTED
    assert impl.resolve_execution_path(context, q, k, v, plain).support.status is SupportStatus.UNSUPPORTED
    assert (
        impl.resolve_execution_path(
            ExecutionContext(platform="cuda", packing_mode=PackingMode.MULTI_DOCUMENT), q, k, v, metadata
        ).support.status
        is SupportStatus.UNSUPPORTED
    )
    assert (
        impl.resolve_execution_path(
            ExecutionContext(
                platform="cuda", packing_mode=PackingMode.PACKED_PADDING, parallel_strategy=ParallelStrategy.RING
            ),
            q,
            k,
            v,
            metadata,
        ).support.status
        is SupportStatus.UNSUPPORTED
    )


@pytest.mark.cuda
@pytest.mark.parametrize("field", ["q_length", "kv_length"])
@pytest.mark.parametrize("bad_length", [True, 1.5, 0, -1, None, "cpu_scalar", "cuda_scalar", "overflow"])
@torch.inference_mode()
def test_padding_rejects_invalid_host_lengths(hopper, field, bad_length, monkeypatch):
    from dataclasses import replace

    impl = make_impl()
    q, k, v, metadata = make_padded_request()
    if bad_length == "cpu_scalar":
        bad_length = torch.tensor(1)
    elif bad_length == "cuda_scalar":
        bad_length = torch.tensor(1, device=q.device)
    elif bad_length == "overflow":
        bad_length = max(q.shape[1], k.shape[1]) + 1
    metadata.packed_padding = replace(metadata.packed_padding, **{field: bad_length})
    monkeypatch.setattr(impl, "_execute", lambda *args: pytest.fail("Invalid padding must not execute"))
    for run in (impl.forward, impl.prepare_request):
        with pytest.raises(ValueError, match=field):
            run(q, k, v, metadata)
    result = impl.resolve_execution_path(
        ExecutionContext(platform="cuda", packing_mode=PackingMode.PACKED_PADDING), q, k, v, metadata
    )
    assert result.support.status is SupportStatus.UNSUPPORTED and field in result.support.reason
    assert not impl._request_preparation


@pytest.mark.cuda
@pytest.mark.parametrize(
    "case",
    [
        "untyped",
        "wrong_type",
        "batch",
        "prefix",
        "cu_shape",
        "cu_dtype",
        "raw_multi_document",
        "extra_length",
        "extra_scalar",
        "mask",
        "piecewise",
    ],
)
@torch.inference_mode()
def test_padding_rejects_inconsistent_or_unsupported_metadata(hopper, case, monkeypatch):
    from dataclasses import replace

    impl = make_impl()
    q, k, v, metadata = make_padded_request()
    if case == "untyped":
        metadata.packed_padding = None
    elif case == "wrong_type":
        metadata.packed_padding = {"q_length": 65, "kv_length": 1089}
    elif case == "batch":
        q, k, v = (t.expand(2, *t.shape[1:]) for t in (q, k, v))
    elif case == "prefix":
        metadata.extra["protected_kv_prefix"] = 1090  # Within storage, beyond real K/V.
    elif case == "cu_shape":
        metadata.packed_padding = replace(metadata.packed_padding, cu_seqlens_q=metadata.extra["cu_seqlens_q"])
    elif case == "cu_dtype":
        metadata.packed_padding = replace(
            metadata.packed_padding, cu_seqlens_k=metadata.packed_padding.cu_seqlens_k.float()
        )
    elif case == "raw_multi_document":
        metadata.extra["cu_seqlens_q"] = torch.tensor([0, 20, 65, q.shape[1]], device=q.device, dtype=torch.int32)
    elif case == "extra_length":
        metadata.extra["max_seqlen_q"] = 64
    elif case == "extra_scalar":
        metadata.extra["valid_kv_length"] = torch.tensor(1089, device=q.device)
    elif case == "mask":
        metadata.attn_mask = torch.ones(1, k.shape[1], device=q.device, dtype=torch.bool)
    else:
        metadata.full_attn_spans = [[[0, 65]]]
    monkeypatch.setattr(impl, "_execute", lambda *args: pytest.fail("Invalid padding must not execute"))
    for run in (impl.forward, impl.prepare_request):
        with pytest.raises(ValueError):
            run(q, k, v, metadata)
    result = impl.resolve_execution_path(
        ExecutionContext(platform="cuda", packing_mode=PackingMode.PACKED_PADDING), q, k, v, metadata
    )
    assert result.support.status is SupportStatus.UNSUPPORTED and result.support.reason
    assert not impl._request_preparation


@pytest.mark.cuda
@torch.inference_mode()
def test_padding_compile(hopper, isolated_compile_cache):
    impl = make_impl()
    q, k, v, metadata = make_padded_request()
    impl.prepare_request(q, k, v, metadata)
    expected = impl.forward(q, k, v, metadata)
    compiled = torch.compile(impl.forward, fullgraph=True, dynamic=True)
    torch.testing.assert_close(compiled(q, k, v, metadata), expected, atol=0, rtol=0)
    # A different real length uses a different executed request despite having
    # the same padded storage size. Compiled dispatch must prepare that request.
    from dataclasses import replace

    changed = AttentionMetadata(
        packed_padding=replace(
            metadata.packed_padding,
            q_length=64,
            cu_seqlens_q=torch.tensor([0, 64], device=q.device, dtype=torch.int32),
        ),
        extra={"protected_kv_prefix": 65},
    )
    context = ExecutionContext(platform="cuda", packing_mode=PackingMode.PACKED_PADDING)
    assert impl.resolve_execution_path(context, q, k, v, changed).support.status is SupportStatus.UNSUPPORTED
    actual = compiled(q, k, v, changed)
    assert impl.resolve_execution_path(context, q, k, v, changed).support.status is SupportStatus.SUPPORTED
    plain = AttentionMetadata(extra={"protected_kv_prefix": 65})
    reference = impl.forward(q[:, :64], k[:, :1089], v[:, :1089], plain)
    torch.testing.assert_close(actual[:, :64], reference, atol=0, rtol=0)
    assert torch.count_nonzero(actual[:, 64:]) == 0


@pytest.mark.cuda
@torch.inference_mode()
def test_cloned_sparse_layer_requires_own_preparation(hopper):
    import copy
    import gc
    import weakref

    original = make_impl()
    q, k, v = make_attention_inputs()
    expected = original.forward(q, k, v)
    cloned = copy.deepcopy(original)
    assert cloned.adapter is not original.adapter and cloned.selector is not original.selector
    context = ExecutionContext(platform="cuda")
    assert original.resolve_execution_path(context, q, k, v, None).support.status is SupportStatus.SUPPORTED
    assert cloned.resolve_execution_path(context, q, k, v, None).support.status is SupportStatus.UNSUPPORTED
    original_ref = weakref.ref(original)
    del original
    gc.collect()
    assert original_ref() is None
    torch.testing.assert_close(cloned.forward(q, k, v), expected, atol=0, rtol=0)
    assert cloned.resolve_execution_path(context, q, k, v, None).support.status is SupportStatus.SUPPORTED


@pytest.mark.parametrize("adapter_mode", [CompilationMode.EAGER_ONLY, CompilationMode.TRACEABLE])
def test_shared_compilation_boundary_is_custom_op(adapter_mode):
    from types import SimpleNamespace

    from vllm_omni.diffusion.attention.block_sparse import BlockSparseAttention

    impl = object.__new__(BlockSparseAttention)
    impl.adapter = SimpleNamespace(compilation_mode=adapter_mode, provider="test", kernel_variant="test")
    impl._normalize_request = lambda q, k, v, metadata: (q, k, v, 0, 1)
    impl._request_signature = lambda *args: ()
    impl._request_preparation = {(): None}
    result = impl.resolve_execution_path(
        ExecutionContext(platform="cuda", require_fullgraph=True), None, None, None, None
    )
    assert result.support.status is SupportStatus.SUPPORTED
    assert result.compilation_mode is CompilationMode.CUSTOM_OP
