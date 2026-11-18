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
    PackedPaddingMetadata,
    TextAudioBlockMask,
    VideoTokenLayout,
    VideoTokenSpan,
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


def make_impl(**options):
    settings = {"sink_mode": "prefix", "dense_steps": 0, "dense_layers": "", **options}
    return SolAttnImpl(2, 128, 0.125, prefix="blocks.4.attention", backend_kwargs={"sol_attn": settings})


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
        per_role={"self": {"backend": "SOL_ATTN", "sol_attn": {"tau": 1.5, "dense_layers": "0-2,5"}}}
    )
    spec, _ = config.resolve_with_source("self")
    restored = SolAttnSpec(**spec.backend_kwargs()["sol_attn"])
    assert restored.tau == 1.5 and restored.prefix_sink
    assert make_impl(**spec.backend_kwargs()["sol_attn"]).dense_layers == {0, 1, 2, 5}
    nested = AttentionConfig(
        per_role={"minimax_h3": {"token_refiner": {"backend": "SOL_ATTN", "sol_attn": {"dense_layers": 0}}}}
    )
    assert nested.per_role["minimax_h3.token_refiner"].sol_attn.dense_layers == [0]
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


@pytest.mark.parametrize("capability", [DeviceCapability(8, 0), DeviceCapability(10, 0), DeviceCapability(12, 0)])
@pytest.mark.parametrize("kv_splits", [2, 4])
def test_split_kv_is_rejected_outside_hopper(monkeypatch, capability, kv_splits):
    monkeypatch.setattr(sol_module.current_omni_platform, "get_device_capability", lambda: capability)
    with pytest.raises(ValueError, match="SM90"):
        make_impl(kv_splits=kv_splits)


@pytest.mark.parametrize("capability", [DeviceCapability(9, 0), DeviceCapability(10, 0), DeviceCapability(12, 0)])
@pytest.mark.parametrize("kv_splits", ["auto", 1])
def test_hopper_and_blackwell_single_split_configuration(monkeypatch, capability, kv_splits):
    monkeypatch.setattr(sol_module.current_omni_platform, "get_device_capability", lambda: capability)
    monkeypatch.setitem(sys.modules, "cutlass.cute", None)
    impl = make_impl(kv_splits=kv_splits)
    assert impl.kv_splits == 1
    assert impl._resolve_kv_splits(65536) == 1
    expected_dense = sol_module.CuDNNAttentionImpl if capability.major == 12 else sol_module.SDPAImpl
    assert isinstance(impl.dense_attention, expected_dense)


@pytest.mark.parametrize("kv_splits", ["auto", 1, 2, 4])
def test_hopper_split_resolution_matches_upstream_policy(monkeypatch, kernel, kv_splits):
    monkeypatch.setattr(sol_module.current_omni_platform, "get_device_capability", lambda: DeviceCapability(9, 0))
    for name in ("cuda", "cuda.bindings", "cuda.bindings.driver", "cutlass", "cutlass.cute"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    impl = make_impl(kv_splits=kv_splits)
    for used in (65535, 65536):
        expected = (4 if used >= 65536 else 1) if kv_splits == "auto" else kv_splits
        assert impl._resolve_kv_splits(used) == expected
        query, key, value = (torch.zeros(1, used, 1, 128, dtype=torch.bfloat16) for _ in range(3))
        impl.forward_cuda(query, key, value)
        assert kernel[-1][3]["kv_splits"] == expected
    if kv_splits in (2, 4):
        with pytest.raises(ValueError, match="route group"):
            impl._resolve_kv_splits(64)


@pytest.mark.parametrize("kv_splits", [2, 4])
def test_explicit_hopper_splits_require_cute(monkeypatch, kv_splits):
    monkeypatch.setattr(sol_module.current_omni_platform, "get_device_capability", lambda: DeviceCapability(9, 0))
    monkeypatch.setitem(sys.modules, "cutlass.cute", None)
    with pytest.raises(ImportError, match="CuTe"):
        make_impl(kv_splits=kv_splits)


def test_sparse_forward_trims_padding_and_keeps_sink_queries_dense(kernel):
    query, key, value = tensors()
    output = make_impl(sink_tokens=3, sink_start=2).forward_cuda(query, key, value, packed())
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
        "sink_start": 2,
        "sink_tokens": 3,
        "exact_block_mask": None,
    }
    torch.testing.assert_close(output[:, 2:5], dense(query[:, 2:5], key[:, :9], value[:, :9]))
    assert torch.all(output[:, :2] == 37) and torch.all(output[:, 5:9] == 37)
    assert torch.count_nonzero(output[:, 9:]) == 0


@pytest.mark.parametrize("dense_steps", [0, 10])
def test_batch_dimension_is_never_flattened_into_one_document(kernel, dense_steps):
    query, key, value = (tensor.repeat(2, 1, 1, 1) for tensor in tensors())
    with pytest.raises(ValueError, match="single nonempty sequence"):
        make_impl(dense_steps=dense_steps).forward_cuda(query, key, value)
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
    with pytest.raises(TypeError, match="sink_start"):
        make_impl(sink_tokens=3).forward_cuda(*tensors())
    assert calls == []


@pytest.mark.parametrize(
    "video_layout,prefix",
    [
        (VideoTokenLayout(prefix_len=3, latent_grid=(1, 2, 3)), 3),
        (
            VideoTokenLayout(
                used_len=9,
                video_spans=(VideoTokenSpan(2, (1, 1, 2), "reference"), VideoTokenSpan(6, (1, 1, 3), "target")),
            ),
            6,
        ),
    ],
)
def test_model_owned_prefix_protects_audio_and_references(kernel, video_layout, prefix):
    query, key, value = tensors()
    metadata = replace(packed(), video_layout=video_layout)
    output = make_impl().forward_cuda(query, key, value, metadata)
    assert kernel[0][3]["sink_tokens"] == prefix
    torch.testing.assert_close(output[:, :prefix], dense(query[:, :prefix], key[:, :9], value[:, :9]))


@pytest.mark.parametrize("start,tokens,expected", [(None, 3, (6, 3)), (99, 4, (9, 0)), (7, 9, (7, 2))])
def test_sink_range_clamps_without_changing_tail_semantics(kernel, start, tokens, expected):
    query, key, value = tensors()
    make_impl(sink_start=start, sink_tokens=tokens).forward_cuda(query, key, value, packed())
    options = kernel[0][3]
    assert (options["sink_start"], options["sink_tokens"]) == expected


def test_out_of_bounds_sink_does_not_expand_video_prefix(kernel):
    query, key, value = tensors()
    metadata = replace(packed(), video_layout=VideoTokenLayout(prefix_len=3, latent_grid=(1, 2, 3)))
    impl = make_impl(sink_start=999, sink_tokens=4)
    result = impl.resolve_execution_path(ExecutionContext(platform="cuda"), query, key, value, metadata)
    assert result.path == "sol_sparse"
    output = impl.forward_cuda(query, key, value, metadata)
    assert kernel[0][3]["sink_start"] == 0
    assert kernel[0][3]["sink_tokens"] == 3
    torch.testing.assert_close(output[:, :3], dense(query[:, :3], key[:, :9], value[:, :9]))
    assert torch.all(output[:, 3:9] == 37)
    assert torch.count_nonzero(output[:, 9:]) == 0


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


@pytest.mark.parametrize("options", [{"dense_steps": 10}, {"dense_layers": "4"}, {"sink_tokens": 999}])
def test_dense_guard_uses_only_valid_rows(kernel, options):
    query, key, value = tensors()
    output = make_impl(**options).forward_cuda(query, key, value, packed())
    torch.testing.assert_close(output[:, :9], dense(query[:, :9], key[:, :9], value[:, :9]))
    assert torch.count_nonzero(output[:, 9:]) == 0
    assert kernel == []


@pytest.mark.parametrize("compiled", [False, True])
def test_denoise_step_switches_actual_execution(kernel, compiled):
    torch.compiler.reset()
    query, key, value = tensors()
    impl = make_impl(dense_steps=2)
    forward = torch.compile(impl.forward_cuda, backend="aot_eager", fullgraph=True) if compiled else impl.forward_cuda
    with set_forward_context(omni_diffusion_config=None):
        context = sol_module.get_forward_context()
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
        make_impl(dense_steps=10).forward_cuda(*tensors(), metadata)
    assert kernel == []


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_wrong_dtype_is_rejected_during_dense_warmup(dtype):
    query, key, value = (tensor.to(dtype) for tensor in tensors())
    with pytest.raises(TypeError, match="BF16"):
        make_impl(dense_steps=10).forward_cuda(query, key, value)


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


def test_compile_captures_sparse_op_padding_and_dense_sink(kernel):
    torch.compiler.reset()
    query, key, value = tensors()
    impl = make_impl(sink_tokens=3)
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
    assert impl.config.sink_mode == "text_audio"
    assert impl.config.tau == 1.0 and impl.config.thresh_type == "diag"
    assert impl.config.dense_steps == 0 and impl.dense_layers == set()
    assert impl._resolve_kv_splits(65536) == 1
    refiner, _ = config.resolve_with_source("minimax_h3.token_refiner")
    assert refiner.backend == "TORCH_SDPA"
    with pytest.raises(ValueError, match="text_audio_block_mask"):
        impl.forward_cuda(*tensors(), packed())


def test_sol_default_layer_selectors_are_independent():
    first, second = SolAttnSpec(), SolAttnSpec()
    assert isinstance(first.dense_layers, list)
    first.dense_layers.append(1)
    assert second.dense_layers == []


@pytest.mark.parametrize("dense_steps", [0, 10])
@pytest.mark.parametrize(
    "protection",
    [
        None,
        TextAudioBlockMask(8, 64, torch.ones(1, dtype=torch.bool)),
        TextAudioBlockMask(9, 32, torch.ones(1, dtype=torch.bool)),
        TextAudioBlockMask(9, 64, torch.ones(2, dtype=torch.bool)),
        TextAudioBlockMask(9, 64, torch.ones(1)),
        TextAudioBlockMask(9, 64, torch.ones(1, 1, dtype=torch.bool)),
        TextAudioBlockMask(9, 64, torch.ones(1, dtype=torch.bool, device="meta")),
        TextAudioBlockMask(9.0, 64, torch.ones(1, dtype=torch.bool)),
        TextAudioBlockMask(9, 64.0, torch.ones(1, dtype=torch.bool)),
    ],
)
def test_text_audio_rejects_missing_or_stale_metadata(kernel, protection, dense_steps):
    metadata = packed()
    metadata.text_audio_block_mask = protection
    impl = make_impl(sink_mode="text_audio", dense_steps=dense_steps)
    result = impl.resolve_execution_path(ExecutionContext(platform="cuda"), *tensors(), metadata)
    assert result.support.status is SupportStatus.UNSUPPORTED
    with pytest.raises(ValueError, match="text_audio"):
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
    metadata.text_audio_block_mask = TextAudioBlockMask(9, 64, torch.ones(1, dtype=torch.bool))

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
    assert all(call[3]["exact_block_mask"] is metadata.text_audio_block_mask.blocks for call in kernel)


def test_text_audio_rejects_noncontiguous_block_mask(kernel):
    query = torch.zeros(1, 129, 2, 128, dtype=torch.bfloat16)
    metadata = packed(129)
    metadata.text_audio_block_mask = TextAudioBlockMask(129, 64, torch.ones(6, dtype=torch.bool)[::2])
    impl = make_impl(sink_mode="text_audio")
    with pytest.raises(ValueError, match="contiguous"):
        impl.forward_cuda(query, query, query, metadata)


@pytest.mark.parametrize("compiled", [False, True])
def test_disjoint_text_audio_mask_is_forwarded_without_dense_recomputation(kernel, monkeypatch, compiled):
    torch.compiler.reset()
    used = 513
    query, key, value = (torch.randn(1, used + 17, 2, 128, dtype=torch.bfloat16) for _ in range(3))
    blocks = torch.tensor([True, False, False, False, True, False, False, False, True])
    metadata = replace(packed(used), text_audio_block_mask=TextAudioBlockMask(used, 64, blocks))
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
        assert options["sink_start"] == options["sink_tokens"] == 0
        assert kernel[-1][0].shape[1] == used
    torch.compiler.reset()
