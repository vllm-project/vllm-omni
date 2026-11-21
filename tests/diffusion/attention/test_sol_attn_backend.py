# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
import types
from dataclasses import replace

import pytest
import torch
from vllm.platforms.interface import DeviceCapability

import vllm_omni.diffusion.attention.backends.sol_attn as sol_module
from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionMetadata,
    ConditioningBlockMask,
    PackedPaddingMetadata,
)
from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.diffusion.attention.backends.sol_attn import SolAttnBackend, SolAttnImpl
from vllm_omni.diffusion.attention.capabilities import (
    ExecutionContext,
    MaskMode,
    OuterBoundary,
    PackingMode,
    ParallelStrategy,
    SupportStatus,
)
from vllm_omni.diffusion.attention.sol_config import sol_block_selected
from vllm_omni.diffusion.data import (
    AttentionConfig,
    AttentionSpec,
    DiffusionParallelConfig,
    OmniDiffusionConfig,
    SolAttnSpec,
)
from vllm_omni.diffusion.forward_context import set_forward_context

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def platform(monkeypatch):
    monkeypatch.setattr(sol_module.current_omni_platform, "get_device_capability", lambda: DeviceCapability(8, 0))
    monkeypatch.setattr(sol_module, "get_current_diffusion_config_or_none", lambda: None)


@pytest.fixture
def kernel(monkeypatch):
    calls = []
    module = types.ModuleType("sol_attn")

    def sparse(query, key, value, *, exact_block_mask=None, **kwargs):
        kwargs["exact_block_mask"] = exact_block_mask
        calls.append((query, key, value, kwargs))
        return torch.full_like(query, 37)

    monkeypatch.setattr(module, "sol_attn", sparse, raising=False)
    monkeypatch.setitem(sys.modules, "sol_attn", module)
    return calls


def make_impl(*, requires_conditioning_mask=False, **options):
    settings = {"sink_conditioning": "off", **options}
    return SolAttnImpl(
        2,
        128,
        0.125,
        prefix="blocks.4.attention",
        requires_conditioning_mask=requires_conditioning_mask,
        backend_kwargs={"sol_attn": settings},
    )


def tensors():
    generator = torch.Generator().manual_seed(17)
    return tuple(torch.randn(1, 12, 2, 128, generator=generator, dtype=torch.bfloat16) for _ in range(3))


def packed(used=9):
    lengths = torch.tensor([0, used], dtype=torch.int32)
    return AttentionMetadata(
        packed_padding=PackedPaddingMetadata(used, used, lengths, lengths),
        extra={"cu_seqlens_q": torch.tensor([0, used, 12], dtype=torch.int32), "valid_kv_length": used},
    )


def dense(query, key, value, scale=0.125):
    return torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), scale=scale
    ).transpose(1, 2)


def test_registry_and_dependency_preflight(kernel, monkeypatch):
    assert DiffusionAttentionBackendEnum.SOL_ATTN.get_class() is SolAttnBackend
    assert SolAttnBackend.supported_platforms == ("cuda",)
    assert SolAttnBackend.supports_packed_mask_free()
    assert not SolAttnBackend.supports_multi_doc_packed_varlen()
    SolAttnBackend.validate_available()
    monkeypatch.setitem(sys.modules, "sol_attn", None)
    with pytest.raises(ImportError, match="upstream sol_attn"):
        SolAttnBackend.validate_available()


def test_old_upstream_package_fails_dependency_preflight(kernel, monkeypatch):
    monkeypatch.setattr(sys.modules["sol_attn"], "sol_attn", lambda q, k, v: q)
    with pytest.raises(ImportError, match="exact_block_mask"):
        SolAttnBackend.validate_available()


def test_per_role_configuration_round_trip(kernel):
    config = AttentionConfig(
        per_role={"self": {"backend": "SOL_ATTN", "sol_attn": {"tau": 1.5, "dense_blocks": "0-2,4"}}}
    )
    spec, _ = config.resolve_with_source("self")
    restored = SolAttnSpec(**spec.backend_kwargs()["sol_attn"])
    assert restored.tau == 1.5 and restored.sink_conditioning == "exact_kv_and_rows"
    assert make_impl(**spec.backend_kwargs()["sol_attn"]).dense_block
    assert set(spec.backend_kwargs()["sol_attn"]) == {
        "tau",
        "tau_profile",
        "dense_blocks",
        "min_tokens",
        "sink_conditioning",
    }
    nested = AttentionConfig(
        per_role={"minimax_h3": {"token_refiner": {"backend": "SOL_ATTN", "sol_attn": {"dense_blocks": "0"}}}}
    )
    assert nested.per_role["minimax_h3.token_refiner"].sol_attn.dense_blocks == "0"
    with pytest.raises(ValueError, match="only supported"):
        AttentionSpec(backend="FLASH_ATTN", sol_attn={})


@pytest.mark.parametrize(
    "options",
    [
        {"kv_splits": 3},
        {"kv_splits": True},
        {"kv_splits": 1.5},
        {"sink_tokens": -1},
        {"sink_tokens": 1.2},
        {"sink_start": -1},
        {"dense_steps": -1},
        {"dense_steps": True},
        {"dense_steps": 0.5},
        {"dense_layers": "4-2"},
        {"dense_layers": -1},
        {"dense_layers": True},
        {"dense_layers": "bad"},
        {"dense_blocks": [0, 2]},
        {"dense_blocks": "bad"},
        {"dense_blocks": "0", "dense_layers": "1"},
        {"min_tokens": -1},
        {"min_tokens": True},
        {"min_tokens": 1.5},
        {"tau_profile": "0-2"},
        {"tau_profile": "=1.0"},
        {"tau_profile": "0=nan"},
        {"tau_profile": "0=inf"},
        {"tau_profile": "0=-0.1"},
        {"tau_profile": "0=word"},
        {"tau_profile": {"0": 1.0}},
        {"thresh_type": "unknown"},
        {"tau": float("nan")},
        {"tau": float("inf")},
        {"prefix_sink": "false"},
        {"sink_mode": "video"},
        {"sink_mode": "text_audio", "sink_tokens": 1},
        {"sink_mode": "text_audio", "sink_start": 1},
        {"provider": "auto"},
        {"provider": "native"},
        {"provider": "cudnn"},
        {"provider": "flashinfer"},
        {"provider": True},
    ],
)
def test_invalid_options_fail_at_parse_time(options):
    with pytest.raises((TypeError, ValueError)):
        AttentionSpec(backend="SOL_ATTN", sol_attn=options)


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        ("0-2,-1", {0, 1, 2, 4}),
        ("-3--1", {2, 3, 4}),
        ("3-1", {1, 2, 3}),
        (" 0 - 2, , -1 ", {0, 1, 2, 4}),
        ("-99-1", {0, 1}),
        ("3-9999999999", {3, 4}),
        ("9", set()),
    ],
)
def test_relative_block_selectors_resolve_against_model_stack(selector, expected):
    assert {index for index in range(5) if sol_block_selected(selector, index, 5)} == expected


def test_layer_tuning_round_trip_and_last_rule_wins(kernel):
    settings = {
        "tau": 1.1,
        "min_tokens": 10,
        "dense_blocks": "0-2,-1",
        "tau_profile": "0-4=2.0;3-4=0.8 # sensitive layers\n4=0.6",
        "sink_conditioning": "off",
    }
    spec = AttentionSpec(backend="SOL_ATTN", sol_attn=settings)
    restored = SolAttnSpec(**spec.backend_kwargs()["sol_attn"])
    assert all(getattr(restored, name) == value for name, value in settings.items())
    impl = SolAttnImpl(2, 128, 0.125, prefix="blocks.4.attention", layer_count=6, backend_kwargs=spec.backend_kwargs())
    assert not impl.dense_block
    impl.forward_cuda(*tensors())
    assert kernel[-1][3]["tau"] == 0.6
    last = SolAttnImpl(2, 128, 0.125, prefix="blocks.5.attention", layer_count=6, backend_kwargs=spec.backend_kwargs())
    query, key, value = tensors()
    torch.testing.assert_close(last.forward_cuda(query, key, value), dense(query, key, value))
    assert last.tau == 1.1
    assert len(kernel) == 1


@pytest.mark.parametrize("options", [{"dense_blocks": "-1"}, {"tau_profile": "-1=0.8"}])
def test_relative_selectors_require_model_depth(options):
    with pytest.raises(ValueError, match="model-owned layer_count"):
        make_impl(**options)


@pytest.mark.parametrize("options", [{"dense_blocks": "0"}, {"tau_profile": "0=0.8"}])
def test_layer_tuning_requires_known_block_index(options):
    with pytest.raises(ValueError, match="model-owned block index"):
        SolAttnImpl(2, 128, 0.125, backend_kwargs={"sol_attn": options})
    with pytest.raises(ValueError, match="model-owned block index"):
        SolAttnImpl(2, 128, 0.125, prefix="blocks.5.attention", layer_count=5, backend_kwargs={"sol_attn": options})


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("min_tokens", [9, 10])
def test_min_tokens_uses_unpadded_length_and_reports_actual_path(kernel, min_tokens, compiled):
    torch.compiler.reset()
    impl = make_impl(min_tokens=min_tokens, tau_profile="0-5=0.7")
    query, key, value = tensors()
    metadata = packed(used=9)
    result = impl.resolve_execution_path(ExecutionContext(platform="cuda"), query, key, value, metadata)
    assert result.path == ("dense_guard" if min_tokens > 9 else "sol_sparse")
    forward = torch.compile(impl.forward_cuda, backend="aot_eager", fullgraph=True) if compiled else impl.forward_cuda
    output = forward(query, key, value, metadata)
    if min_tokens > 9:
        torch.testing.assert_close(output[:, :9], dense(query[:, :9], key[:, :9], value[:, :9]))
        assert not kernel
    else:
        assert torch.all(output[:, :9] == 37)
        assert kernel[-1][3]["tau"] == 0.7
    assert torch.count_nonzero(output[:, 9:]) == 0
    torch.compiler.reset()


@pytest.mark.parametrize(
    "options",
    [{"causal": True}, {"num_kv_heads": 1}, {"head_size": 64}, {"qkv_layout": "BNSD"}, {"computation": "cross"}],
)
def test_invalid_geometry_fails_at_construction(options):
    with pytest.raises(ValueError, match="SOL_ATTN"):
        SolAttnImpl(**{"num_heads": 2, "head_size": 128, "softmax_scale": 0.125, **options})


@pytest.mark.parametrize("attribute", ["ring_degree", "allgather_degree"])
def test_partial_sequence_parallelism_fails_at_construction(monkeypatch, attribute):
    config = OmniDiffusionConfig.__new__(OmniDiffusionConfig)
    parallel = DiffusionParallelConfig.__new__(DiffusionParallelConfig)
    parallel.ring_degree = 1
    parallel.allgather_degree = 1
    setattr(parallel, attribute, 2)
    config.parallel_config = parallel
    monkeypatch.setattr(sol_module, "get_current_diffusion_config_or_none", lambda: config)
    with pytest.raises(ValueError, match="full Q/K/V"):
        make_impl()


@pytest.mark.parametrize(
    "capability", [DeviceCapability(8, 0), DeviceCapability(9, 0), DeviceCapability(10, 0), DeviceCapability(12, 0)]
)
def test_missing_cute_or_non_hopper_uses_one_split(monkeypatch, capability):
    monkeypatch.setattr(sol_module.current_omni_platform, "get_device_capability", lambda: capability)
    monkeypatch.setitem(sys.modules, "cutlass.cute", None)
    impl = make_impl()
    assert impl._resolve_kv_splits(65535) == impl._resolve_kv_splits(65536) == 1
    expected_dense = sol_module.CuDNNAttentionImpl if capability.major == 12 else sol_module.SDPAImpl
    assert isinstance(impl.dense_attention, expected_dense)


def test_hopper_split_resolution_is_automatic(monkeypatch, kernel):
    monkeypatch.setattr(sol_module.current_omni_platform, "get_device_capability", lambda: DeviceCapability(9, 0))
    for name in ("cuda", "cuda.bindings", "cuda.bindings.driver", "cutlass", "cutlass.cute"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    impl = make_impl()
    for used in (65535, 65536):
        expected = 4 if used >= 65536 else 1
        assert impl._resolve_kv_splits(used) == expected
        query, key, value = (torch.zeros(1, used, 1, 128, dtype=torch.bfloat16) for _ in range(3))
        impl.forward_cuda(query, key, value)
        assert kernel[-1][3]["kv_splits"] == expected


def test_sparse_forward_trims_padding_and_uses_internal_kernel_policy(kernel):
    query, key, value = tensors()
    output = make_impl().forward_cuda(query, key, value, packed())
    assert len(kernel) == 1
    called_query, called_key, called_value, options = kernel[0]
    assert called_query.shape == (1, 9, 2, 128)
    torch.testing.assert_close(called_key, key[:, :9])
    torch.testing.assert_close(called_value, value[:, :9])
    assert options == {
        "scale": 0.125,
        "tau": 1.0,
        "thresh_type": "diag",
        "kv_splits": 1,
        "exact_block_mask": None,
    }
    assert torch.all(output[:, :9] == 37)
    assert torch.count_nonzero(output[:, 9:]) == 0


@pytest.mark.parametrize("min_tokens", [0, 100])
def test_batch_dimension_is_never_flattened_into_one_document(kernel, min_tokens):
    query, key, value = (tensor.repeat(2, 1, 1, 1) for tensor in tensors())
    with pytest.raises(ValueError, match="single nonempty sequence"):
        make_impl(min_tokens=min_tokens).forward_cuda(query, key, value)
    assert kernel == []


def test_nondefault_scale_changes_numeric_kernel_output(kernel, monkeypatch):
    def exact_kernel(query, key, value, **options):
        return dense(query, key, value, scale=options["scale"])

    monkeypatch.setattr(sys.modules["sol_attn"], "sol_attn", exact_kernel)
    query = torch.zeros(1, 64, 2, 128, dtype=torch.bfloat16)
    query[..., 0] = 1
    key = torch.zeros_like(query)
    key[:, :32, :, 0] = 4
    key[:, 32:, :, 0] = -4
    value = torch.zeros_like(query)
    value[:, 32:] = 2
    output = make_impl().forward_cuda(query, key, value)
    torch.testing.assert_close(output, dense(query, key, value, scale=0.125))
    assert not torch.equal(output, dense(query, key, value, scale=128**-0.5))


def test_padding_values_cannot_change_valid_sparse_output(kernel, monkeypatch):
    def exact_kernel(query, key, value, **options):
        return dense(query, key, value, scale=options["scale"])

    monkeypatch.setattr(sys.modules["sol_attn"], "sol_attn", exact_kernel)
    query, key, value = tensors()
    impl = make_impl()
    expected = impl.forward_cuda(query, key, value, packed())
    query[:, 9:] = 100
    key[:, 9:] = 100
    value[:, 9:] = 100
    actual = impl.forward_cuda(query, key, value, packed())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[:, :9], dense(query[:, :9], key[:, :9], value[:, :9]))
    assert torch.count_nonzero(actual[:, 9:]) == 0


def test_incompatible_kernel_cannot_silently_drop_sink_options(kernel, monkeypatch):
    calls = []

    def legacy_kernel(query, key, value, *, scale, tau, thresh_type, kv_splits):
        calls.append(query)
        return torch.empty_like(query)

    monkeypatch.setattr(sys.modules["sol_attn"], "sol_attn", legacy_kernel)
    with pytest.raises(TypeError, match="exact_block_mask"):
        make_impl().forward_cuda(*tensors())
    assert calls == []


@pytest.mark.parametrize("metadata", [None, AttentionMetadata()])
def test_compile_dynamic_sequence_without_explicit_length(kernel, metadata):
    torch.compiler.reset()
    impl = make_impl()
    forward = torch.compile(impl.forward_cuda, backend="aot_eager", fullgraph=True, dynamic=True)
    for length in (12, 17):
        query, key, value = (torch.randn(1, length, 2, 128, dtype=torch.bfloat16) for _ in range(3))
        output = forward(query, key, value, metadata)
        assert output.shape == query.shape
        assert torch.all(output == 37)
    torch.compiler.reset()


@pytest.mark.parametrize("options", [{"min_tokens": 10}, {"dense_blocks": "4"}])
def test_dense_guard_uses_only_valid_rows(kernel, options):
    query, key, value = tensors()
    output = make_impl(**options).forward_cuda(query, key, value, packed())
    torch.testing.assert_close(output[:, :9], dense(query[:, :9], key[:, :9], value[:, :9]))
    assert torch.count_nonzero(output[:, 9:]) == 0
    assert kernel == []


@pytest.mark.parametrize("compiled", [False, True])
def test_shared_schedule_switches_sol_without_backend_step_knobs(kernel, monkeypatch, compiled):
    import vllm_omni.diffusion.attention.layer as layer_module
    from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.forward_context import bind_attention_schedule, get_forward_context

    monkeypatch.setattr(layer_module, "build_parallel_attention_strategy", lambda **kwargs: NoParallelAttention())
    config = OmniDiffusionConfig(
        diffusion_attention_config={"default": {"backend": "TORCH_SDPA"}},
        diffusion_attention_schedule={
            "profiles": {"sparse": {"default": {"backend": "SOL_ATTN", "sol_attn": {"sink_conditioning": "off"}}}},
            "default": [{"start": 2, "end": None, "profile": "sparse"}],
        },
    )
    torch.compiler.reset()
    with set_current_diffusion_config(config):
        attention = layer_module.Attention(2, 128, False, 0.125, prefix="blocks.4.attention")
    forward = torch.compile(attention, backend="aot_eager", fullgraph=True) if compiled else attention
    query, key, value = tensors()
    with (
        set_forward_context(omni_diffusion_config=config),
        bind_attention_schedule(config.diffusion_attention_schedule.default),
    ):
        context = get_forward_context()
        context.total_denoise_steps = 3
        for step in (1, 2, 0):
            context.denoise_step_idx = step
            output = forward(query, key, value)
            if step < 2:
                torch.testing.assert_close(output, dense(query, key, value))
            else:
                assert torch.all(output == 37)
        assert len(kernel) == 1
    torch.compiler.reset()


@pytest.mark.parametrize(
    "metadata",
    [
        AttentionMetadata(attn_mask=torch.ones(1, 12, dtype=torch.bool)),
        AttentionMetadata(extra={"cu_seqlens_q": torch.tensor([0, 4, 12])}),
        AttentionMetadata(extra={"valid_kv_length": 0}),
        AttentionMetadata(extra={"valid_kv_length": 13}),
        AttentionMetadata(extra={"valid_kv_length": torch.tensor(9)}),
        AttentionMetadata(extra={"kv_cache_dtype": "fp8"}),
        replace(packed(), extra={"valid_kv_length": 8}),
        replace(packed(), packed_padding=replace(packed().packed_padding, q_length=8)),
    ],
)
def test_invalid_metadata_fails_before_dense_or_sparse(kernel, metadata):
    with pytest.raises(ValueError, match="SOL_ATTN"):
        make_impl(min_tokens=100).forward_cuda(*tensors(), metadata)
    assert kernel == []


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_wrong_dtype_is_rejected_during_dense_guard(dtype):
    query, key, value = (tensor.to(dtype) for tensor in tensors())
    with pytest.raises(TypeError, match="BF16"):
        make_impl(min_tokens=100).forward_cuda(query, key, value)


@pytest.mark.parametrize("strategy", list(ParallelStrategy))
def test_capability_reports_parallel_restrictions(strategy):
    query, key, value = tensors()
    context = ExecutionContext(platform="cuda", parallel_strategy=strategy, packing_mode=PackingMode.PACKED_PADDING)
    result = make_impl().resolve_execution_path(context, query, key, value, packed())
    assert (result.support.status is SupportStatus.SUPPORTED) == (
        strategy in (ParallelStrategy.NONE, ParallelStrategy.ULYSSES)
    )


def test_capability_rejects_masked_and_multidocument_contexts():
    query, key, value = tensors()
    for context in (
        ExecutionContext(platform="rocm"),
        ExecutionContext(platform="cuda", mask_mode=MaskMode.ARBITRARY),
        ExecutionContext(platform="cuda", packing_mode=PackingMode.MULTI_DOCUMENT),
    ):
        assert (
            make_impl().resolve_execution_path(context, query, key, value, None).support.status
            is SupportStatus.UNSUPPORTED
        )


def test_compile_captures_sparse_op_and_padding(kernel):
    torch.compiler.reset()
    query, key, value = tensors()
    impl = make_impl()
    expected = impl.forward_cuda(query, key, value, packed())
    compiled = torch.compile(impl.forward_cuda, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(query, key, value, packed()), expected)
    assert len(kernel) == 2
    torch.compiler.reset()


def test_hsdp_boundary_does_not_claim_guaranteed_fullgraph():
    query, key, value = tensors()
    context = ExecutionContext(
        platform="cuda", outer_boundaries=frozenset({OuterBoundary.HSDP}), require_fullgraph=True
    )
    result = make_impl().resolve_execution_path(context, query, key, value, None)
    assert result.support.status is SupportStatus.UNMIGRATED
    assert result.requested_support(context).status is SupportStatus.UNSUPPORTED


@pytest.mark.parametrize("options", [None, {}])
def test_sol_defaults_use_upstream_text_audio(kernel, options):
    config = AttentionConfig(
        per_role={
            "self": {"backend": "SOL_ATTN", "sol_attn": options},
            "minimax_h3.token_refiner": {"backend": "TORCH_SDPA"},
        }
    )
    spec, _ = config.resolve_with_source("self")
    impl = SolAttnImpl(2, 128, 0.125, backend_kwargs=spec.backend_kwargs())
    assert "provider" not in vars(impl.config)
    assert impl.config.sink_conditioning == "exact_kv_and_rows"
    assert impl.config.tau == 1.0
    assert impl.config.min_tokens == 0 and not impl.dense_block
    assert impl._resolve_kv_splits(65536) == 1
    refiner, _ = config.resolve_with_source("minimax_h3.token_refiner")
    assert refiner.backend == "TORCH_SDPA"
    output = impl.forward_cuda(*tensors(), packed())
    assert torch.all(output[:, :9] == 37)
    assert kernel[-1][3]["exact_block_mask"] is None
    required = SolAttnImpl(2, 128, 0.125, requires_conditioning_mask=True, backend_kwargs=spec.backend_kwargs())
    with pytest.raises(ValueError, match="conditioning_block_mask"):
        required.forward_cuda(*tensors(), packed())


@pytest.mark.parametrize(
    "option,value",
    [
        ("thresh_type", "diag"),
        ("kv_splits", "auto"),
        ("dense_steps", 0),
        ("dense_layers", "0,1"),
        ("sink_mode", "text_audio"),
        ("sink_tokens", 0),
        ("sink_start", 0),
        ("prefix_sink", True),
    ],
)
def test_removed_public_options_are_rejected(option, value):
    with pytest.raises(TypeError, match=option):
        AttentionSpec(backend="SOL_ATTN", sol_attn={option: value})


@pytest.mark.parametrize("mode", ["exact_kv", "text_audio", "prefix", True, None])
def test_unsupported_conditioning_modes_fail_at_startup(mode):
    with pytest.raises(ValueError, match="sink_conditioning"):
        AttentionSpec(backend="SOL_ATTN", sol_attn={"sink_conditioning": mode})


@pytest.mark.parametrize("compiled", [False, True])
def test_conditioning_off_does_not_forward_model_protection(kernel, compiled):
    torch.compiler.reset()
    metadata = packed()
    metadata.conditioning_block_mask = ConditioningBlockMask(9, 64, torch.ones(1, dtype=torch.bool))
    impl = make_impl(sink_conditioning="off")
    forward = torch.compile(impl.forward_cuda, backend="aot_eager", fullgraph=True) if compiled else impl.forward_cuda
    result = forward(*tensors(), metadata)
    assert torch.all(result[:, :9] == 37)
    assert kernel[-1][3]["exact_block_mask"] is None
    assert torch.count_nonzero(result[:, 9:]) == 0
    torch.compiler.reset()


@pytest.mark.parametrize("min_tokens", [0, 100])
@pytest.mark.parametrize(
    "protection",
    [
        None,
        ConditioningBlockMask(8, 64, torch.ones(1, dtype=torch.bool)),
        ConditioningBlockMask(9, 32, torch.ones(1, dtype=torch.bool)),
        ConditioningBlockMask(9, 64, torch.ones(2, dtype=torch.bool)),
        ConditioningBlockMask(9, 64, torch.ones(1)),
        ConditioningBlockMask(9, 64, torch.ones(1, 1, dtype=torch.bool)),
        ConditioningBlockMask(9, 64, torch.ones(1, dtype=torch.bool, device="meta")),
        ConditioningBlockMask(9.0, 64, torch.ones(1, dtype=torch.bool)),
        ConditioningBlockMask(9, 64.0, torch.ones(1, dtype=torch.bool)),
    ],
)
def test_text_audio_rejects_missing_or_stale_metadata(kernel, protection, min_tokens):
    metadata = packed()
    metadata.conditioning_block_mask = protection
    impl = make_impl(sink_conditioning="exact_kv_and_rows", min_tokens=min_tokens, requires_conditioning_mask=True)
    result = impl.resolve_execution_path(ExecutionContext(platform="cuda"), *tensors(), metadata)
    assert result.support.status is SupportStatus.UNSUPPORTED
    with pytest.raises(ValueError, match="conditioning"):
        impl.forward_cuda(*tensors(), metadata)
    assert kernel == []


@pytest.mark.parametrize("layer_index", [0, 1, 4, 47])
def test_text_audio_runs_upstream_for_all_fifty_steps(kernel, monkeypatch, layer_index):
    impl = SolAttnImpl(
        2,
        128,
        0.125,
        prefix=f"blocks.{layer_index}.attention",
    )
    metadata = packed()
    metadata.conditioning_block_mask = ConditioningBlockMask(9, 64, torch.ones(1, dtype=torch.bool))

    def unexpected_dense(*args):
        pytest.fail("Explicit all-step/all-layer profile must not invoke a dense guard")

    monkeypatch.setattr(impl.dense_attention, "forward_cuda", unexpected_dense)
    config = OmniDiffusionConfig()
    for step in range(50):
        with set_forward_context(omni_diffusion_config=config, denoise_step_idx=step):
            result = impl.resolve_execution_path(ExecutionContext(platform="cuda"), *tensors(), metadata)
            assert result.path == "sol_sparse"
            impl.forward_cuda(*tensors(), metadata)
    assert len(kernel) == 50
    assert all(call[3]["exact_block_mask"] is metadata.conditioning_block_mask.blocks for call in kernel)


def test_text_audio_rejects_noncontiguous_block_mask(kernel):
    query = torch.zeros(1, 129, 2, 128, dtype=torch.bfloat16)
    metadata = packed(129)
    metadata.conditioning_block_mask = ConditioningBlockMask(129, 64, torch.ones(6, dtype=torch.bool)[::2])
    impl = make_impl(sink_conditioning="exact_kv_and_rows")
    with pytest.raises(ValueError, match="contiguous"):
        impl.forward_cuda(query, query, query, metadata)


@pytest.mark.parametrize("compiled", [False, True])
def test_disjoint_text_audio_mask_is_forwarded_without_dense_recomputation(kernel, monkeypatch, compiled):
    torch.compiler.reset()
    used = 513
    query, key, value = (torch.randn(1, used + 17, 2, 128, dtype=torch.bfloat16) for _ in range(3))
    blocks = torch.tensor([True, False, False, False, True, False, False, False, True])
    metadata = replace(packed(used), conditioning_block_mask=ConditioningBlockMask(used, 64, blocks))
    impl = SolAttnImpl(2, 128, 0.125)

    def unexpected_dense(*args):
        pytest.fail("Upstream must protect text/audio on both axes inside the fused kernel")

    monkeypatch.setattr(impl.dense_attention, "forward_cuda", unexpected_dense)
    forward = torch.compile(impl.forward_cuda, backend="aot_eager", fullgraph=True) if compiled else impl.forward_cuda
    for changed in (False, True):
        if changed:
            blocks.logical_not_()
        actual = forward(query, key, value, metadata)
        assert torch.all(actual[:, :used] == 37)
        assert actual[:, used:].count_nonzero() == 0
        options = kernel[-1][3]
        assert options["exact_block_mask"] is blocks
        assert "sink_start" not in options and "sink_tokens" not in options
        assert kernel[-1][0].shape[1] == used
    torch.compiler.reset()


@pytest.mark.parametrize(
    "positions,expected",
    [([], [False, False, False]), ([0, 63], [True, False, False]), ([64, 128], [False, True, True])],
)
def test_generic_conditioning_builder_handles_disjoint_and_partial_blocks(positions, expected):
    tokens = torch.zeros(129, dtype=torch.bool)
    tokens[positions] = True
    protection = ConditioningBlockMask.from_token_mask(tokens)
    assert protection.used_len == 129 and protection.block_size == 64
    assert protection.blocks.tolist() == expected
    assert protection.blocks.is_contiguous()
    assert protection.blocks.device == tokens.device


@pytest.mark.parametrize(
    "tokens", [torch.zeros(0, dtype=torch.bool), torch.zeros(2), torch.zeros(1, 2, dtype=torch.bool)]
)
def test_generic_conditioning_builder_rejects_invalid_token_masks(tokens):
    with pytest.raises(ValueError, match="token mask"):
        ConditioningBlockMask.from_token_mask(tokens)


@pytest.mark.parametrize("block_size", [0, -1, True, 1.5])
def test_generic_conditioning_builder_rejects_invalid_block_size(block_size):
    with pytest.raises(ValueError, match="block_size"):
        ConditioningBlockMask.from_token_mask(torch.zeros(2, dtype=torch.bool), block_size)
