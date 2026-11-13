# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn as nn

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.backends import flashinfer_attn
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import set_forward_context

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.fixture(autouse=True)
def _fresh_dynamo_cache():
    # Each test compiles forward_cuda for freshly built impls; start every
    # test with an empty Dynamo frame cache so the default per-frame limit
    # applies and per-instance specialization cannot hide behind it.
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


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


def _boundary_impl():
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
    # __new__ bypasses __init__, so register the boundary-op handle explicitly
    # and mirror the CPU token __init__ would have built for it.
    impl._op_handle = flashinfer_attn._register_impl_for_boundary_op(impl)
    impl._op_token = torch.tensor(impl._op_handle, dtype=torch.int64)
    return impl


@pytest.mark.parametrize(
    ("sequence_parallel_size", "use_hsdp"),
    [(2, False), (1, True)],
)
@pytest.mark.cpu
def test_active_parallel_context_keeps_stateful_wrapper(
    monkeypatch,
    sequence_parallel_size,
    use_hsdp,
):
    # Kernel selection must not depend on the parallel strategy: after the
    # Ulysses all-to-all the impl just computes local attention, so an active
    # SP/HSDP context still routes through the boundary op to the stateful
    # wrapper.
    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    impl = _boundary_impl()
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


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_impl_token_stays_cpu_when_initialized_in_device_context():
    # The diffusers loader initializes models inside a target-device context,
    # so an unspecified device would put the token on CUDA and turn every
    # in-op handle lookup into a device synchronization.
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to exercise target-device-context initialization")
    if not flashinfer_attn.HAS_FLASHINFER:
        pytest.skip("FlashInfer is not installed")

    with torch.device("cuda", index=torch.accelerator.current_device_index()):
        impl = flashinfer_attn.FlashInferAttentionImpl(
            num_heads=2,
            head_size=128,
            softmax_scale=128**-0.5,
            backend_kwargs={"quant": {"flashinfer_backend": "fa2"}},
        )

    assert impl._op_token.device.type == "cpu"


@pytest.mark.cpu
def test_flashinfer_attention_boundary_op_schema_and_fake():
    impl = _boundary_impl()
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
        (query, key, value, mask, impl._op_token),
        test_utils=("test_schema", "test_faketensor"),
    )

    assert result == {
        "test_schema": "SUCCESS",
        "test_faketensor": "SUCCESS",
    }


_WRAPPER_ROUTE_CASES = [
    # Configurations outside the plain dense call, plus the packed-metadata
    # entry: all must reach the stateful wrapper through the opaque boundary
    # op under fullgraph.
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
@pytest.mark.cpu
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
    impl = _boundary_impl()
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


@pytest.mark.parametrize(
    ("backend", "causal", "backend_explicit"),
    [
        ("fa2", True, False),
        ("fa2", True, True),
        ("cute-dsl", False, False),
        ("cute-dsl", False, True),
    ],
)
@pytest.mark.cpu
def test_all_keep_mask_elides_before_the_rejections(monkeypatch, backend, causal, backend_explicit):
    # Pre-PR semantics: an all-keep mask packs to None *before* the causal and
    # cute-dsl rejections, so those calls run exactly like a call without any
    # mask — through the stateful wrapper here. The unified boundary op must
    # preserve that.
    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    query = torch.randn(1, 4, 2, 128, dtype=torch.bfloat16)
    metadata = AttentionMetadata(attn_mask=torch.ones(4, 4, dtype=torch.bool))

    impl = _boundary_impl()
    impl.flashinfer_backend = backend
    impl.causal = causal
    impl.backend_explicit = backend_explicit
    wrapper = _RecordingWrapper()
    impl._wrapper = wrapper

    expected = impl.forward_cuda(query, query, query, metadata)
    compiled = torch.compile(impl.forward_cuda, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(query, query, query, metadata), expected)

    # The elided mask skipped the rejections; the wrapper ran like a plain
    # dense call (plan cached across the eager + compiled invocations).
    assert wrapper.plan_calls == 1
    assert wrapper.run_calls == 2


@pytest.mark.parametrize(
    ("backend", "causal"),
    [("fa2", True), ("cute-dsl", False)],
)
@pytest.mark.cpu
def test_partial_mask_rejections_are_preserved(monkeypatch, backend, causal):
    # A mask with real masked-out positions is NOT elided: explicit backends
    # keep raising for causal/cute-dsl, exactly like before the PR.
    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    query = torch.randn(1, 4, 2, 128, dtype=torch.bfloat16)
    mask = torch.ones(4, 4, dtype=torch.bool)
    mask[:, -1] = False
    metadata = AttentionMetadata(attn_mask=mask)

    impl = _boundary_impl()
    impl.flashinfer_backend = backend
    impl.causal = causal
    impl.backend_explicit = True

    with pytest.raises(ValueError, match="causal=True|cute-dsl backend does not support custom masks"):
        impl.forward_cuda(query, query, query, metadata)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_auto_sdpa_fallback_layout_survives_inductor_fullgraph(monkeypatch):
    # The boundary op's fake promises a contiguous [B, Q, H, Dv] output, but
    # SDPAImpl returns a permuted (noncontiguous) tensor with the SDPA math
    # backend. Inductor only asserts the custom op's output layout when one
    # of its kernels consumes it, so the compiled block carries a downstream
    # consumer. Without the fallback materializing its return, this fails
    # with "expected size ... stride ..." against the fake's layout.
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Inductor output-layout verification")
    from torch.nn.attention import SDPBackend, sdpa_kernel

    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    device = torch.device("cuda", torch.accelerator.current_device_index())
    query = torch.randn(1, 4, 2, 128, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, 4, 2, 128, device=device, dtype=torch.bfloat16)
    value = torch.randn(1, 4, 2, 128, device=device, dtype=torch.bfloat16)
    mask = torch.ones(4, 4, device=device, dtype=torch.bool)
    mask[:, -1] = False
    metadata = AttentionMetadata(attn_mask=mask)

    impl = _boundary_impl()
    impl.flashinfer_backend = "cute-dsl"
    impl.causal = False
    impl.backend_explicit = False  # automatic: rejection routes to the SDPA fallback

    def block(q, k, v, md):
        return impl.forward_cuda(q, k, v, md) * 2

    with sdpa_kernel([SDPBackend.MATH]):
        expected = block(query, key, value, metadata)
        compiled = torch.compile(block, backend="inductor", fullgraph=True)
        actual = compiled(query, key, value, metadata)

    torch.testing.assert_close(actual, expected)


class _AttentionBlock(nn.Module):
    """A transformer-style block owning one attention impl, mirroring how
    DreamZero compiles blocks individually."""

    def __init__(self, impl):
        super().__init__()
        self.impl = impl

    def forward(self, query, key, value):
        return self.impl.forward_cuda(query, key, value)


@pytest.mark.cpu
def test_equivalent_instances_share_one_compiled_graph(monkeypatch):
    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    num_blocks = 10  # Above Dynamo's default per-frame cache of 8 entries.
    blocks = []
    for _ in range(num_blocks):
        impl = _boundary_impl()
        impl.flashinfer_backend = "fa2"
        impl._wrapper = _RecordingWrapper()
        blocks.append(_AttentionBlock(impl))

    query = torch.randn(2, 3, 2, 128, dtype=torch.bfloat16)
    key = torch.randn(2, 5, 2, 128, dtype=torch.bfloat16)
    value = torch.randn(2, 5, 2, 128, dtype=torch.bfloat16)
    eager_outputs = [block(query, key, value) for block in blocks]

    compile_count = 0

    def counting_backend(graph_module, _example_inputs):
        nonlocal compile_count
        compile_count += 1
        return graph_module.forward

    torch._dynamo.reset()
    with torch._dynamo.config.patch(cache_size_limit=8):
        compiled_outputs = []
        for block in blocks:
            compiled = torch.compile(
                block,
                backend=counting_backend,
                fullgraph=True,
                dynamic=True,
            )
            compiled_outputs.append(compiled(query, key, value))

    for actual, expected in zip(compiled_outputs, eager_outputs, strict=True):
        torch.testing.assert_close(actual, expected)
    # One graph shared by every equivalent instance, and each instance's
    # wrapper executed its own plan/run inside the boundary op.
    assert compile_count == 1
    for block in blocks:
        assert block.impl._wrapper.plan_calls == 1
        assert block.impl._wrapper.run_calls == 2  # eager + compiled calls


@hardware_test(res={"cuda": "L4"}, num_cards=1)
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
