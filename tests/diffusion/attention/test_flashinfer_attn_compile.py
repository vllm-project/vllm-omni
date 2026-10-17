# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import functools

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.attention.backends import flashinfer_attn
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import set_forward_context

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

# The suite compiles the same forward_cuda frame once per test with a fresh
# impl object (distinct handle/guards); Dynamo's default per-frame cache of 8
# entries would fail later tests once it fills.
torch._dynamo.config.cache_size_limit = 64


def _inputs(*, value_head_dim: int = 128):
    query = torch.randn(6, 2, 128, dtype=torch.bfloat16)
    key = torch.randn(10, 2, 128, dtype=torch.bfloat16)
    value = torch.randn(10, 2, value_head_dim, dtype=torch.bfloat16)
    workspace = torch.empty(0, dtype=torch.uint8)
    seq_lens = torch.tensor([5, 5], dtype=torch.int32)
    qo_indptr = torch.tensor([0, 3, 6], dtype=torch.int32)
    kv_indptr = torch.tensor([0, 5, 10], dtype=torch.int32)
    return (
        query,
        key,
        value,
        workspace,
        seq_lens,
        qo_indptr,
        kv_indptr,
        3,
        5,
        128**-0.5,
        2,
    )


def _fa2_inputs():
    query = torch.randn(2, 3, 2, 128, dtype=torch.bfloat16)
    key = torch.randn(2, 5, 2, 128, dtype=torch.bfloat16)
    value = torch.randn(2, 5, 2, 128, dtype=torch.bfloat16)
    return query, key, value, None, 128**-0.5


@pytest.fixture
def fake_fa2_single_prefill_kernel(monkeypatch):
    calls = []

    def kernel(query, key, value, **kwargs):
        calls.append((query, key, value, kwargs))
        return query.clone()

    monkeypatch.setattr(
        flashinfer_attn,
        "single_prefill_with_kv_cache",
        kernel,
    )
    return calls


def test_flashinfer_fa2_custom_op_schema_and_fake(fake_fa2_single_prefill_kernel):
    op = torch.ops.vllm_omni.flashinfer_fa2_attention.default

    result = torch.library.opcheck(
        op,
        _fa2_inputs(),
        test_utils=("test_schema", "test_faketensor"),
    )

    assert result == {
        "test_schema": "SUCCESS",
        "test_faketensor": "SUCCESS",
    }


def test_flashinfer_fa2_custom_op_preserves_inputs(fake_fa2_single_prefill_kernel):
    args = _fa2_inputs()
    snapshots = [tensor.clone() for tensor in args[:3]]

    first = flashinfer_attn._flashinfer_fa2_attention_op(*args)
    flashinfer_attn._flashinfer_fa2_attention_op(*args)

    assert len(fake_fa2_single_prefill_kernel) == 4
    assert first.shape == (2, 3, 2, 128)
    assert first.dtype == torch.bfloat16
    assert first.is_contiguous()
    for tensor, snapshot in zip(args[:3], snapshots, strict=True):
        torch.testing.assert_close(tensor, snapshot)
    assert all(call[3]["backend"] == "fa2" for call in fake_fa2_single_prefill_kernel)


@pytest.fixture
def fake_cute_dsl_kernel(monkeypatch):
    calls = []

    def kernel(**kwargs):
        calls.append(kwargs)
        query = kwargs["query"]
        value = kwargs["value"]
        return query.new_empty((*query.shape[:-1], value.shape[-1]))

    monkeypatch.setattr(
        flashinfer_attn,
        "trtllm_ragged_attention_deepseek",
        kernel,
    )
    return calls


def test_flashinfer_cute_dsl_custom_op_schema_and_fake(fake_cute_dsl_kernel):
    op = torch.ops.vllm_omni.flashinfer_cute_dsl_attention.default

    for value_head_dim in (128, 64):
        result = torch.library.opcheck(
            op,
            _inputs(value_head_dim=value_head_dim),
            test_utils=("test_schema", "test_faketensor"),
        )

        assert result == {
            "test_schema": "SUCCESS",
            "test_faketensor": "SUCCESS",
        }


def test_flashinfer_cute_dsl_custom_op_preserves_inputs(fake_cute_dsl_kernel):
    args = _inputs()
    tensor_snapshots = [tensor.clone() for tensor in args[:7]]

    first = flashinfer_attn._flashinfer_cute_dsl_attention_op(*args)
    second = flashinfer_attn._flashinfer_cute_dsl_attention_op(*args)

    assert len(fake_cute_dsl_kernel) == 2
    assert first.shape == (6, 2, 128)
    assert first.dtype == torch.bfloat16
    assert first.device == args[0].device
    assert first.is_contiguous()
    assert first.data_ptr() != args[0].data_ptr()
    assert second.data_ptr() != first.data_ptr()
    for tensor, snapshot in zip(args[:7], tensor_snapshots, strict=True):
        torch.testing.assert_close(tensor, snapshot)

    call = fake_cute_dsl_kernel[0]
    assert call["workspace_buffer"] is args[3]
    assert call["seq_lens"] is args[4]
    assert call["cum_seq_lens_q"] is args[5]
    assert call["cum_seq_lens_kv"] is args[6]
    assert call["backend"] == "cute-dsl"
    assert call["is_causal"] is False
    assert call["return_lse"] is False


def test_flashinfer_cute_dsl_custom_op_reports_missing_kernel(monkeypatch):
    monkeypatch.setattr(
        flashinfer_attn,
        "trtllm_ragged_attention_deepseek",
        None,
    )

    with pytest.raises(RuntimeError, match="FlashInfer cute-dsl kernel is unavailable"):
        flashinfer_attn._flashinfer_cute_dsl_attention_op(*_inputs())


@pytest.fixture
def executable_fa2_single_prefill_kernel(monkeypatch, tmp_path):
    marker = tmp_path / "flashinfer-fa2-kernel"
    marker.write_text("loaded", encoding="utf-8")
    calls = []

    @functools.cache
    def load_kernel():
        with open(marker, encoding="utf-8") as handle:
            return handle.read()

    def kernel(query, key, value, **kwargs):
        calls.append((query, key, value))
        assert load_kernel() == "loaded"
        return query.clone()

    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    monkeypatch.setattr(flashinfer_attn, "_is_cuda_execution_path", lambda *_tensors: True)
    monkeypatch.setattr(
        flashinfer_attn,
        "single_prefill_with_kv_cache",
        kernel,
    )
    return calls


class _UnexpectedWrapper:
    def plan(self, *args, **kwargs):
        pytest.fail("candidate path unexpectedly planned the stateful wrapper")

    def run(self, *args, **kwargs):
        pytest.fail("candidate path unexpectedly ran the stateful wrapper")


class _RecordingWrapper:
    def __init__(self):
        self.plan_calls = 0
        self.run_calls = 0
        self.plan_custom_masks = []

    def plan(self, *args, **kwargs):
        self.plan_calls += 1
        custom_mask = kwargs.get("custom_mask")
        self.plan_custom_masks.append(None if custom_mask is None else tuple(custom_mask.shape))

    def run(self, query, key, value, **kwargs):
        self.run_calls += 1
        return query.clone()


def _candidate_impl():
    impl = flashinfer_attn.FlashInferAttentionImpl.__new__(flashinfer_attn.FlashInferAttentionImpl)
    impl.causal = False
    impl.softmax_scale = 128**-0.5
    impl.device = torch.device("cpu")
    impl.dtype_qk = None
    impl.dtype_vo = None
    impl.flashinfer_backend = "cute-dsl"
    impl.backend_explicit = True
    impl._workspace = torch.empty(0, dtype=torch.uint8)
    impl._wrapper = _UnexpectedWrapper()
    impl._qo_indptr = None
    impl._kv_indptr = None
    impl._plan_key = None
    impl._sdpa_fallback = None
    impl.device_capability = (10, 0)
    # __new__ bypasses __init__, so register the boundary-op handle explicitly.
    impl._op_handle = flashinfer_attn._register_impl_for_boundary_op(impl)
    return impl


def _fa2_candidate_impl():
    impl = _candidate_impl()
    impl.flashinfer_backend = "fa2"
    return impl


@pytest.fixture
def executable_cute_dsl_kernel(monkeypatch, tmp_path):
    marker = tmp_path / "flashinfer-kernel"
    marker.write_text("loaded", encoding="utf-8")
    calls = []

    @functools.cache
    def load_kernel():
        with open(marker, encoding="utf-8") as handle:
            return handle.read()

    def kernel(**kwargs):
        calls.append(kwargs)
        assert load_kernel() == "loaded"
        batch_size = kwargs["batch_size"]
        query = kwargs["query"].reshape(batch_size, kwargs["max_q_len"], 2, 128)
        key = kwargs["key"].reshape(batch_size, kwargs["max_kv_len"], 2, 128)
        value = kwargs["value"].reshape(batch_size, kwargs["max_kv_len"], 2, 128)
        return (
            F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                scale=kwargs["bmm1_scale"],
            )
            .transpose(1, 2)
            .reshape_as(query)
            .reshape(-1, 2, 128)
            .contiguous()
        )

    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    monkeypatch.setattr(flashinfer_attn, "_is_cuda_execution_path", lambda *_tensors: True)
    monkeypatch.setattr(
        flashinfer_attn,
        "trtllm_ragged_attention_deepseek",
        kernel,
    )
    return calls


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_dense_cute_dsl_candidate_is_opaque_to_dynamic_fullgraph(
    executable_cute_dsl_kernel,
    noncontiguous,
):
    impl = _candidate_impl()
    compile_count = 0

    def counting_backend(graph_module, _example_inputs):
        nonlocal compile_count
        compile_count += 1
        return graph_module.forward

    compiled = torch.compile(
        impl.forward_cuda,
        backend=counting_backend,
        fullgraph=True,
        dynamic=True,
    )

    for batch_size, query_length, kv_length in ((1, 3, 5), (2, 7, 4)):
        tensors = [
            torch.randn(batch_size, 2, length, 128, dtype=torch.bfloat16).transpose(1, 2)
            for length in (query_length, kv_length, kv_length)
        ]
        if not noncontiguous:
            tensors = [tensor.contiguous() for tensor in tensors]
        else:
            assert all(not tensor.is_contiguous() for tensor in tensors)
        query, key, value = tensors
        snapshots = [tensor.clone() for tensor in tensors]

        expected = impl.forward_cuda(query, key, value)
        actual = compiled(query, key, value)

        torch.testing.assert_close(actual, expected)
        for tensor, snapshot in zip(tensors, snapshots, strict=True):
            torch.testing.assert_close(tensor, snapshot)

    # Dynamo specializes the first batch-size-1 input, so the batch-size-2
    # case may produce a second full graph even with dynamic=True.
    assert 1 <= compile_count <= 2
    assert len(executable_cute_dsl_kernel) == 4


def test_dense_fa2_candidate_is_opaque_to_dynamic_fullgraph(
    executable_fa2_single_prefill_kernel,
):
    impl = _fa2_candidate_impl()
    compiled = torch.compile(
        impl.forward_cuda,
        backend="eager",
        fullgraph=True,
        dynamic=True,
    )

    for batch_size, query_length, kv_length in ((1, 3, 5), (2, 4, 7)):
        query = torch.randn(batch_size, query_length, 2, 128, dtype=torch.bfloat16)
        key = torch.randn(batch_size, kv_length, 2, 128, dtype=torch.bfloat16)
        value = torch.randn(batch_size, kv_length, 2, 128, dtype=torch.bfloat16)

        expected = impl.forward_cuda(query, key, value)
        actual = compiled(query, key, value)

        torch.testing.assert_close(actual, expected)

    assert len(executable_fa2_single_prefill_kernel) == 6


@pytest.mark.parametrize(
    ("sequence_parallel_size", "use_hsdp"),
    [(2, False), (1, True)],
)
def test_active_parallel_context_keeps_stateful_wrapper(
    monkeypatch,
    sequence_parallel_size,
    use_hsdp,
):
    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    monkeypatch.setattr(flashinfer_attn, "_is_cuda_execution_path", lambda *_tensors: True)
    monkeypatch.setattr(flashinfer_attn, "trtllm_ragged_attention_deepseek", lambda **_kwargs: object())
    impl = _candidate_impl()
    wrapper = _RecordingWrapper()
    impl._wrapper = wrapper
    query = torch.randn(1, 4, 2, 128, dtype=torch.bfloat16)
    parallel_config_kwargs: dict[str, int | bool] = {
        "sequence_parallel_size": sequence_parallel_size,
        # The config validator requires sequence_parallel_size to equal
        # ulysses_degree * ring_degree.
        "ulysses_degree": sequence_parallel_size,
        "use_hsdp": use_hsdp,
    }
    if use_hsdp:
        # Standalone HSDP needs an explicit shard size; auto-derivation raises
        # when every other parallel dimension is 1.
        parallel_config_kwargs["hsdp_shard_size"] = 1
    runtime_config = OmniDiffusionConfig(
        parallel_config=DiffusionParallelConfig(**parallel_config_kwargs),
    )

    with set_forward_context(omni_diffusion_config=runtime_config):
        output = impl.forward_cuda(query, query, query)

    torch.testing.assert_close(output, query)
    assert wrapper.plan_calls == 1
    assert wrapper.run_calls == 1


@pytest.mark.gpu
@pytest.mark.cuda
def test_flashinfer_real_backend_eager_fullgraph_agreement():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for real FlashInfer execution")
    if not flashinfer_attn.HAS_FLASHINFER:
        pytest.skip("FlashInfer is not installed")

    capability = torch.cuda.get_device_capability()
    if capability == (8, 0):
        expected_backend = "fa2"
        if flashinfer_attn.single_prefill_with_kv_cache is None:
            pytest.skip("FlashInfer single-prefill API is unavailable")
    elif capability[0] >= 10 and flashinfer_attn._cute_dsl_dense_artifacts_published(capability):
        expected_backend = "cute-dsl"
        if flashinfer_attn.trtllm_ragged_attention_deepseek is None:
            pytest.skip("FlashInfer cute-dsl API is unavailable")
    else:
        pytest.skip("This test covers the original A100 FA2 path and the real cute-dsl path")

    impl = flashinfer_attn.FlashInferAttentionImpl(
        num_heads=2,
        head_size=128,
        softmax_scale=128**-0.5,
        backend_kwargs={"quant": {"flashinfer_backend": "auto"}},
    )
    assert impl.flashinfer_backend == expected_backend
    query = torch.randn(1, 4, 2, 128, device=impl.device, dtype=torch.bfloat16)
    key = torch.randn(1, 5, 2, 128, device=impl.device, dtype=torch.bfloat16)
    value = torch.randn(1, 5, 2, 128, device=impl.device, dtype=torch.bfloat16)

    from vllm_omni.diffusion.attention.capabilities import ExecutionContext, SupportStatus

    context = ExecutionContext(platform="cuda", require_fullgraph=True)
    path = impl.resolve_execution_path(context, query, key, value, None)
    expected = impl.forward_cuda(query, key, value)
    compiled = torch.compile(impl.forward_cuda, backend="eager", fullgraph=True)
    actual = compiled(query, key, value)

    assert path.support.status is SupportStatus.SUPPORTED
    assert path.compilation_mode.value == "custom_op"
    torch.testing.assert_close(actual, expected)


@pytest.mark.gpu
@pytest.mark.cuda
def test_flashinfer_cute_dsl_custom_op_matches_wrapper_and_sdpa():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for real FlashInfer execution")
    if not flashinfer_attn.HAS_FLASHINFER:
        pytest.skip("FlashInfer is not installed")
    capability = torch.cuda.get_device_capability()
    if not flashinfer_attn._cute_dsl_dense_artifacts_published(capability):
        pytest.skip("The cute-dsl dense artifacts are not published for this architecture")
    if flashinfer_attn.trtllm_ragged_attention_deepseek is None:
        pytest.skip("FlashInfer cute-dsl API is unavailable")

    impl = flashinfer_attn.FlashInferAttentionImpl(
        num_heads=2,
        head_size=128,
        softmax_scale=128**-0.5,
        backend_kwargs={"quant": {"flashinfer_backend": "auto"}},
    )

    from vllm_omni.diffusion.attention.capabilities import ExecutionContext, SupportStatus

    context = ExecutionContext(platform="cuda", require_fullgraph=True)
    for batch_size, query_length, kv_length in ((1, 4, 5), (2, 7, 4)):
        query = torch.randn(batch_size, query_length, 2, 128, device=impl.device, dtype=torch.bfloat16)
        key = torch.randn(batch_size, kv_length, 2, 128, device=impl.device, dtype=torch.bfloat16)
        value = torch.randn(batch_size, kv_length, 2, 128, device=impl.device, dtype=torch.bfloat16)

        path = impl.resolve_execution_path(context, query, key, value, None)
        assert path.support.status is SupportStatus.SUPPORTED
        assert path.compilation_mode.value == "custom_op"

        direct = impl.forward_cuda(query, key, value)
        wrapper = impl._run_batch_prefill(query, key, value, None)
        sdpa = F.scaled_dot_product_attention(
            query.transpose(1, 2).float(),
            key.transpose(1, 2).float(),
            value.transpose(1, 2).float(),
            scale=impl.softmax_scale,
        ).transpose(1, 2)

        torch.testing.assert_close(direct, wrapper)
        torch.testing.assert_close(direct.float(), sdpa, rtol=2e-2, atol=2e-2)


def test_neighboring_unverified_path_keeps_stateful_wrapper(monkeypatch):
    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    monkeypatch.setattr(
        flashinfer_attn,
        "trtllm_ragged_attention_deepseek",
        lambda **_kwargs: pytest.fail("non-cute-dsl path unexpectedly used the custom op"),
    )
    impl = _candidate_impl()
    impl.flashinfer_backend = "fa2"
    impl._wrapper = _RecordingWrapper()
    query = torch.randn(1, 4, 2, 128, dtype=torch.bfloat16)

    output = impl.forward_cuda(query, query, query)

    torch.testing.assert_close(output, query)
    assert impl._wrapper.plan_calls == 1
    assert impl._wrapper.run_calls == 1


@pytest.mark.parametrize(
    ("attribute", "value", "tensor_dtype", "head_dim", "metadata"),
    [
        ("flashinfer_backend", "fa2", torch.bfloat16, 128, None),
        ("causal", True, torch.bfloat16, 128, None),
        ("dtype_qk", torch.float8_e4m3fn, torch.bfloat16, 128, None),
        ("flashinfer_backend", "cute-dsl", torch.float16, 128, None),
        ("flashinfer_backend", "cute-dsl", torch.bfloat16, 64, None),
        (
            "flashinfer_backend",
            "cute-dsl",
            torch.bfloat16,
            128,
            AttentionMetadata(attn_mask=torch.ones(4, 4, dtype=torch.bool)),
        ),
        (
            "flashinfer_backend",
            "cute-dsl",
            torch.bfloat16,
            128,
            AttentionMetadata(
                extra={
                    "cu_seqlens_q": torch.tensor([0, 4], dtype=torch.int32),
                    "cu_seqlens_k": torch.tensor([0, 4], dtype=torch.int32),
                    "max_seqlen_q": 4,
                    "max_seqlen_k": 4,
                }
            ),
        ),
        (
            "flashinfer_backend",
            "cute-dsl",
            torch.bfloat16,
            128,
            AttentionMetadata(full_attn_spans=[[(0, 4)]]),
        ),
    ],
)
def test_only_exact_dense_cute_dsl_path_is_custom_op_candidate(
    attribute,
    value,
    tensor_dtype,
    head_dim,
    metadata,
):
    impl = _candidate_impl()
    setattr(impl, attribute, value)
    query = torch.empty(1, 4, 2, head_dim, dtype=tensor_dtype)

    assert not impl._is_cute_dsl_custom_op_candidate(query, query, query, metadata)


def test_flashinfer_attention_boundary_op_schema_and_fake():
    impl = _candidate_impl()
    impl.flashinfer_backend = "fa2"
    impl._wrapper = _RecordingWrapper()
    query = torch.randn(2, 3, 2, 128, dtype=torch.bfloat16)
    key = torch.randn(2, 5, 2, 128, dtype=torch.bfloat16)
    value = torch.randn(2, 5, 2, 128, dtype=torch.bfloat16)
    mask = torch.ones(3, 5, dtype=torch.bool)
    mask[:, -1] = False

    op = torch.ops.vllm_omni.flashinfer_attention.default
    result = torch.library.opcheck(
        op,
        (query, key, value, mask, impl._op_handle, False, False),
        test_utils=("test_schema", "test_faketensor"),
    )

    assert result == {
        "test_schema": "SUCCESS",
        "test_faketensor": "SUCCESS",
    }


_WRAPPER_ROUTE_CASES = [
    # Every configuration the review bot's P1 named outside the two dense
    # special-case envelopes, plus the packed-metadata entry: all must reach
    # the stateful wrapper through the opaque boundary op under fullgraph.
    ("fa3_backend", "fa3", False, torch.bfloat16, 128, "none"),
    ("causal", "fa2", True, torch.bfloat16, 128, "none"),
    ("fp16", "fa2", False, torch.float16, 128, "none"),
    ("head_dim_64", "fa2", False, torch.bfloat16, 64, "none"),
    ("masked", "fa2", False, torch.bfloat16, 128, "partial"),
    ("packed_metadata", "fa2", False, torch.bfloat16, 128, "packed"),
]


@pytest.mark.parametrize(
    ("case_id", "backend", "causal", "dtype", "head_dim", "mask_kind"),
    _WRAPPER_ROUTE_CASES,
)
def test_wrapper_route_configs_are_opaque_to_fullgraph(
    monkeypatch,
    case_id,
    backend,
    causal,
    dtype,
    head_dim,
    mask_kind,
):
    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    monkeypatch.setattr(flashinfer_attn, "_is_cuda_execution_path", lambda *_tensors: True)
    monkeypatch.setattr(
        flashinfer_attn,
        "single_prefill_with_kv_cache",
        lambda *_args, **_kwargs: pytest.fail("wrapper-route case unexpectedly used the fa2 custom op"),
    )
    monkeypatch.setattr(
        flashinfer_attn,
        "trtllm_ragged_attention_deepseek",
        lambda **_kwargs: pytest.fail("wrapper-route case unexpectedly used the cute-dsl custom op"),
    )
    impl = _candidate_impl()
    impl.flashinfer_backend = backend
    impl.causal = causal
    impl._wrapper = _RecordingWrapper()

    query = torch.randn(2, 3, 2, head_dim, dtype=dtype)
    key = torch.randn(2, 5, 2, head_dim, dtype=dtype)
    value = torch.randn(2, 5, 2, head_dim, dtype=dtype)
    metadata: AttentionMetadata | None
    if mask_kind == "partial":
        mask = torch.ones(3, 5, dtype=torch.bool)
        mask[:, -1] = False
        metadata = AttentionMetadata(attn_mask=mask)
    elif mask_kind == "packed":
        metadata = AttentionMetadata(
            extra={
                "cu_seqlens_q": torch.tensor([0, 3, 6], dtype=torch.int32),
                "cu_seqlens_k": torch.tensor([0, 5, 10], dtype=torch.int32),
                "max_seqlen_q": 3,
                "max_seqlen_k": 5,
            }
        )
    else:
        metadata = None

    assert not impl._is_fa2_custom_op_candidate(query, key, value, metadata)
    assert not impl._is_cute_dsl_custom_op_candidate(query, key, value, metadata)

    compiled = torch.compile(impl.forward_cuda, backend="eager", fullgraph=True)
    expected = impl.forward_cuda(query, key, value, metadata)
    actual = compiled(query, key, value, metadata)

    torch.testing.assert_close(actual, expected)
    # One eager plus one compiled invocation, both executing plan/run inside
    # the boundary op body.
    assert impl._wrapper.run_calls == 2
    if mask_kind == "partial":
        # The boundary op packed and flattened the mask before planning.
        assert impl._wrapper.plan_custom_masks[-1] == (2 * 3 * 5,)


def test_all_keep_mask_with_causal_is_rejected_before_the_boundary_op(monkeypatch):
    # Locked-in behavior change of the unified boundary op: rejection is
    # decided on the raw mask's presence (statically traceable), so an
    # all-keep mask no longer elides past the causal check into the wrapper.
    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    query = torch.randn(1, 4, 2, 8)
    metadata = AttentionMetadata(attn_mask=torch.ones(4, 4, dtype=torch.bool))

    auto_impl = _candidate_impl()
    auto_impl.flashinfer_backend = "fa2"
    auto_impl.causal = True
    auto_impl.backend_explicit = False
    auto_impl._wrapper = _UnexpectedWrapper()

    expected = auto_impl.forward_cuda(query, query, query, metadata)
    compiled = torch.compile(auto_impl.forward_cuda, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(query, query, query, metadata), expected)

    explicit_impl = _candidate_impl()
    explicit_impl.flashinfer_backend = "fa2"
    explicit_impl.causal = True
    explicit_impl.backend_explicit = True
    with pytest.raises(ValueError, match="causal=True"):
        explicit_impl.forward_cuda(query, query, query, metadata)


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.parametrize(
    ("causal", "dtype", "head_dim", "query_gt_kv", "masked"),
    [
        (False, torch.bfloat16, 128, False, False),
        (True, torch.bfloat16, 128, False, False),
        (False, torch.float16, 128, False, False),
        (False, torch.bfloat16, 64, False, False),
        (False, torch.bfloat16, 256, False, False),
        (False, torch.bfloat16, 128, True, False),
        (False, torch.bfloat16, 128, False, True),
    ],
)
def test_flashinfer_fa2_configs_fullgraph_eager_agreement(
    causal,
    dtype,
    head_dim,
    query_gt_kv,
    masked,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for real FlashInfer execution")
    if not flashinfer_attn.HAS_FLASHINFER:
        pytest.skip("FlashInfer is not installed")

    impl = flashinfer_attn.FlashInferAttentionImpl(
        num_heads=2,
        head_size=head_dim,
        softmax_scale=head_dim**-0.5,
        causal=causal,
        backend_kwargs={"quant": {"flashinfer_backend": "fa2"}},
    )
    batch_size, qo_len, kv_len = (2, 5, 3) if query_gt_kv else (2, 4, 7)
    query = torch.randn(batch_size, qo_len, 2, head_dim, device=impl.device, dtype=dtype)
    key = torch.randn(batch_size, kv_len, 2, head_dim, device=impl.device, dtype=dtype)
    value = torch.randn(batch_size, kv_len, 2, head_dim, device=impl.device, dtype=dtype)
    metadata: AttentionMetadata | None = None
    if masked:
        mask = torch.ones(batch_size, qo_len, kv_len, device=impl.device, dtype=torch.bool)
        mask[..., -1] = False
        metadata = AttentionMetadata(attn_mask=mask)

    compiled = torch.compile(impl.forward_cuda, backend="eager", fullgraph=True)
    expected = impl.forward_cuda(query, key, value, metadata)
    actual = compiled(query, key, value, metadata)

    torch.testing.assert_close(actual, expected)
