# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Numerical integration tests for FlashInfer PR #4859 on SM120."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flashinfer_attn import FlashInferAttentionImpl
from vllm_omni.diffusion.data import AttentionSpec

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cuda]


@pytest.fixture(autouse=True)
def require_sm120():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    pytest.importorskip("flashinfer.attention.cute_dsl.sm120_fmha")


def _impl(head_dim, threshold=None, causal=False, heads=4, kv_heads=2):
    spec = AttentionSpec(
        backend="FLASHINFER_ATTN",
        quant={"flashinfer_backend": "cute-dsl-prims", "dtype_qk": "fp8_e4m3", "dtype_vo": "fp8_e4m3"},
        skip_softmax={"threshold": threshold} if threshold is not None else None,
    )
    return FlashInferAttentionImpl(
        heads, head_dim, head_dim**-0.5, causal=causal, num_kv_heads=kv_heads, backend_kwargs=spec.backend_kwargs()
    )


@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_sm120_dense_and_zero_match_quantized_sdpa(head_dim, dtype, causal):
    gen = torch.Generator(device="cuda").manual_seed(42)
    q = torch.randn(2, 129, 4, head_dim, device="cuda", dtype=dtype, generator=gen)
    k = torch.randn(2, 257, 2, head_dim, device="cuda", dtype=dtype, generator=gen)
    v = torch.randn(k.shape, device="cuda", dtype=dtype, generator=gen)
    dense = _impl(head_dim, causal=causal).forward_cuda(q, k, v)
    zero = _impl(head_dim, threshold=0.0, causal=causal).forward_cuda(q, k, v)
    from flashinfer.attention.cute_dsl.sm120_fmha import sm120_fmha_fp8_ragged_prefill

    direct = torch.empty_like(dense)
    sm120_fmha_fp8_ragged_prefill(
        q.flatten(0, 1).to(torch.float8_e4m3fn),
        k.flatten(0, 1).to(torch.float8_e4m3fn),
        v.flatten(0, 1).to(torch.float8_e4m3fn),
        direct.flatten(0, 1),
        torch.tensor([0, 129, 258], device="cuda", dtype=torch.int32),
        torch.tensor([0, 257, 514], device="cuda", dtype=torch.int32),
        max_seqlen_q=129,
        is_causal=causal,
        sm_scale=head_dim**-0.5,
    )
    torch.testing.assert_close(dense, direct, rtol=0, atol=0)
    # Use the same quantized inputs to isolate the attention kernel's error.
    q8, k8, v8 = (t.to(torch.float8_e4m3fn).float().transpose(1, 2) for t in (q, k, v))
    mask = None
    if causal:
        mask = torch.arange(257, device="cuda")[None, :] <= torch.arange(129, device="cuda")[:, None] + 128
    ref = F.scaled_dot_product_attention(q8, k8, v8, attn_mask=mask, enable_gqa=True).transpose(1, 2)
    torch.testing.assert_close(zero, dense, rtol=0, atol=0)
    assert torch.isfinite(dense).all()
    relative_l2 = (dense.float() - ref).norm() / ref.norm()
    # PRIMS quantizes softmax probabilities to E4M3 for PV as well as Q/K/V.
    # Upstream uses atol=rtol=0.2; keep a tighter aggregate error bound here.
    assert relative_l2 < 0.035


@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_sm120_real_skip_is_observable(head_dim):
    # Equal scores and tile-specific V prove skipping actually occurs. The
    # noncausal kernel visits the right tile first. Threshold 2 is diagnostic,
    # deliberately aggressive, and is not a recommended inference setting.
    q = torch.zeros(2, 128, 4, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(2, 256, 2, head_dim, device="cuda", dtype=q.dtype)
    v = torch.empty_like(k)
    v[:, :128] = -1
    v[:, 128:] = 1
    dense = _impl(head_dim).forward_cuda(q, k, v)
    skipped = _impl(head_dim, threshold=2.0).forward_cuda(q, k, v)
    torch.testing.assert_close(dense, torch.zeros_like(dense), atol=1e-3, rtol=0)
    torch.testing.assert_close(skipped, torch.ones_like(skipped), atol=1e-3, rtol=0)


def test_sm120_timestep_gate_changes_real_kernel_output():
    from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context

    # Low-score left tile still contributes to dense attention. Threshold 0.5
    # drops it after the right tile initializes the running maximum.
    q = torch.ones(1, 128, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(1, 256, 2, 128, device="cuda", dtype=q.dtype)
    k[:, :128] = -0.125
    v = torch.empty_like(k)
    v[:, :128] = -1
    v[:, 128:] = 1
    spec = AttentionSpec(
        backend="FLASHINFER_ATTN",
        quant={"flashinfer_backend": "cute-dsl-prims", "dtype_qk": "fp8_e4m3", "dtype_vo": "fp8_e4m3"},
        skip_softmax={"threshold": 0.5, "disabled_until_timestep": 0.94},
    )
    impl = FlashInferAttentionImpl(4, 128, 128**-0.5, num_kv_heads=2, backend_kwargs=spec.backend_kwargs())
    dense = _impl(128).forward_cuda(q, k, v)
    skipped = _impl(128, threshold=0.5).forward_cuda(q, k, v)
    assert not torch.equal(dense, skipped)
    for timestep, expected in [(1.0, dense), (0.94, skipped), (0.1, skipped), (1.0, dense), (None, dense)]:
        with override_forward_context(ForwardContext(denoise_timestep=timestep)):
            torch.testing.assert_close(impl.forward_cuda(q, k, v), expected, rtol=0, atol=0)


@pytest.mark.parametrize("threshold", [None, 0.0, 2.0])
def test_sm120_cuda_graph_replay_reads_new_inputs(threshold):
    impl = _impl(128, threshold=threshold)
    q = torch.zeros(1, 128, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(1, 256, 2, 128, device="cuda", dtype=q.dtype)
    v = torch.ones_like(k)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            impl.forward_cuda(q, k, v)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = impl.forward_cuda(q, k, v)
    v.fill_(2)
    graph.replay()
    torch.testing.assert_close(out, torch.full_like(q, 2), atol=1e-3, rtol=0)


def test_sm120_per_request_threshold_updates_between_graph_replays():
    impl = _impl(128)
    q = torch.zeros(2, 128, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(2, 256, 2, 128, device="cuda", dtype=q.dtype)
    v = torch.empty_like(k)
    v[:, :128] = -1
    v[:, 128:] = 1
    threshold = torch.tensor([0.0, 2.0], device="cuda", dtype=torch.float32)
    metadata = AttentionMetadata(extra={"skip_softmax_threshold": threshold})
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            impl.forward_cuda(q, k, v, metadata)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = impl.forward_cuda(q, k, v, metadata)
    graph.replay()
    torch.testing.assert_close(out[0], torch.zeros_like(out[0]), atol=1e-3, rtol=0)
    torch.testing.assert_close(out[1], torch.ones_like(out[1]), atol=1e-3, rtol=0)
    ptr = threshold.data_ptr()
    threshold.copy_(torch.tensor([2.0, 0.0], device="cuda"))
    graph.replay()
    assert threshold.data_ptr() == ptr
    torch.testing.assert_close(out[0], torch.ones_like(out[0]), atol=1e-3, rtol=0)
    torch.testing.assert_close(out[1], torch.zeros_like(out[1]), atol=1e-3, rtol=0)


def test_sm120_selected_through_attention_layer():
    from vllm_omni.diffusion.attention.layer import Attention
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    config = OmniDiffusionConfig(
        diffusion_attention_config={
            "per_role": {
                "self": {
                    "backend": "FLASHINFER_ATTN",
                    "quant": {
                        "flashinfer_backend": "cute-dsl-prims",
                        "dtype_qk": "fp8_e4m3",
                        "dtype_vo": "fp8_e4m3",
                    },
                    "skip_softmax": {"threshold": 2.0},
                }
            }
        }
    )
    with set_current_diffusion_config(config):
        layer = Attention(4, 128, False, 128**-0.5, num_kv_heads=2, role="self")
    assert isinstance(layer.attention, FlashInferAttentionImpl)
    q = torch.zeros(1, 128, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(1, 256, 2, 128, device="cuda", dtype=q.dtype)
    v = torch.empty_like(k)
    v[:, :128] = -1
    v[:, 128:] = 1
    torch.testing.assert_close(layer(q, k, v), torch.ones_like(q), atol=1e-3, rtol=0)
    metadata = AttentionMetadata(extra={"skip_softmax_threshold": None})
    torch.testing.assert_close(layer(q, k, v, metadata), torch.zeros_like(q), atol=1e-3, rtol=0)


@pytest.mark.parametrize("invalid", ["dtype", "shape", "cpu", "noncontiguous"])
def test_sm120_rejects_invalid_runtime_threshold(invalid):
    q = torch.ones(2, 128, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.ones(2, 256, 2, 128, device="cuda", dtype=q.dtype)
    thresholds = {
        "dtype": torch.ones(2, device="cuda", dtype=torch.float16),
        "shape": torch.ones(1, device="cuda"),
        "cpu": torch.ones(2),
        "noncontiguous": torch.ones(4, device="cuda")[::2],
    }
    metadata = AttentionMetadata(extra={"skip_softmax_threshold": thresholds[invalid]})
    with pytest.raises(ValueError, match="skip_softmax_threshold"):
        _impl(128).forward_cuda(q, k, k, metadata)


@pytest.mark.parametrize("num_requests", [1, 2])
@pytest.mark.parametrize("threshold", [None, 0.0, 0.001])
def test_minimax_h3_packed_attention_excludes_padding_and_isolates_requests(num_requests, threshold):
    from vllm_omni.diffusion.attention.backends.flashinfer_attn import FlashInferSM120AttentionBackend
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3Attention

    impl = _impl(128, threshold=threshold, heads=56, kv_heads=56)

    class Layer:
        attn_backend = FlashInferSM120AttentionBackend
        use_ring = False

        def __call__(self, q, k, v, metadata):
            assert metadata.attn_mask is None
            return impl.forward_cuda(q, k, v, metadata)

    owner = SimpleNamespace(attention=Layer(), vsa_sparsity=None)
    boundaries = [0, 129, 192] if num_requests == 1 else [0, 129, 322, 384]
    cu = torch.tensor(boundaries, device="cuda", dtype=torch.int32)
    q = torch.zeros(boundaries[-1], 56, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.full_like(q, 256)  # Poison padding: a leak would dominate valid output.
    v[:129] = 1
    if num_requests == 2:
        v[129:322] = -1
    out = MiniMaxH3Attention._run_packed_attention(
        owner,
        q,
        k,
        v,
        cu_seqlens=cu,
        max_seqlen=129 if num_requests == 1 else 193,
        packed_total=boundaries[-1],
        num_requests=num_requests,
    )
    torch.testing.assert_close(out[:129], torch.ones_like(out[:129]), atol=1e-3, rtol=0)
    if num_requests == 1:
        torch.testing.assert_close(out[129:], torch.zeros_like(out[129:]), atol=0, rtol=0)
    else:
        torch.testing.assert_close(out[129:322], -torch.ones_like(out[129:322]), atol=1e-3, rtol=0)


def test_packed_thresholds_index_logical_requests():
    impl = _impl(128)
    q = torch.zeros(1, 512, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(1, 512, 2, 128, device="cuda", dtype=q.dtype)
    v = torch.ones_like(k)
    v[:, :128] = -1
    v[:, 256:384] = -1
    cu = torch.tensor([0, 256, 512], device="cuda", dtype=torch.int32)
    thresholds = torch.tensor([0.0, 2.0], device="cuda", dtype=torch.float32)
    metadata = AttentionMetadata(
        extra={
            "cu_seqlens_q": cu,
            "cu_seqlens_k": cu,
            "max_seqlen_q": 256,
            "skip_softmax_threshold": thresholds,
        }
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            impl.forward_cuda(q, k, v, metadata)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = impl.forward_cuda(q, k, v, metadata)
    graph.replay()
    torch.testing.assert_close(out[:, :256], torch.zeros_like(out[:, :256]), atol=1e-3, rtol=0)
    torch.testing.assert_close(out[:, 256:], torch.ones_like(out[:, 256:]), atol=1e-3, rtol=0)
    thresholds.copy_(torch.tensor([2.0, 0.0], device="cuda"))
    graph.replay()
    torch.testing.assert_close(out[:, :256], torch.ones_like(out[:, :256]), atol=1e-3, rtol=0)
    torch.testing.assert_close(out[:, 256:], torch.zeros_like(out[:, 256:]), atol=1e-3, rtol=0)


@pytest.mark.parametrize("heads", [24, 40], ids=["wan22-ti2v-5b", "wan22-a14b"])
@pytest.mark.parametrize("threshold", [None, 0.0, 1e-4])
def test_wan22_self_attention_forward(heads, threshold):
    from vllm_omni.diffusion.attention.layer import Attention
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanSelfAttention

    # Exercise Wan's split/normalize/reshape/output path at checkpoint head
    # geometry, with synthetic projection output rather than model weights.
    dim = heads * 128
    gen = torch.Generator(device="cuda").manual_seed(42)
    qkv = torch.randn(1, 257, 3 * dim, device="cuda", dtype=torch.bfloat16, generator=gen)
    spec = AttentionSpec(
        backend="FLASHINFER_ATTN",
        quant={"flashinfer_backend": "cute-dsl-prims", "dtype_qk": "fp8_e4m3", "dtype_vo": "fp8_e4m3"},
        skip_softmax={"threshold": threshold} if threshold is not None else None,
    )
    config = OmniDiffusionConfig(diffusion_attention_config={"per_role": {"self": spec}})
    with set_current_diffusion_config(config):
        attention = Attention(heads, 128, False, 128**-0.5, role="self", qkv_layout="BSND")
    norm = torch.nn.RMSNorm(dim, eps=1e-5, elementwise_affine=False)
    owner = SimpleNamespace(
        to_qkv=lambda x: (x, None),
        num_heads=heads,
        num_kv_heads=heads,
        head_dim=128,
        norm_q=norm,
        norm_k=norm,
        to_gate_compress=None,
        attn=attention,
        to_out=torch.nn.Identity(),
        dropout=torch.nn.Identity(),
    )
    out = WanSelfAttention.forward(owner, qkv)
    q, k, v = qkv.chunk(3, dim=-1)
    q, k = norm(q), norm(k)
    q, k, v = (t.reshape(1, 257, heads, 128).float().transpose(1, 2) for t in (q, k, v))
    reference = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2)
    assert out.shape == (1, 257, dim) and out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
    assert (out.float() - reference).norm() / reference.norm() < 0.07
