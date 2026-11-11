# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MAGI-2 packed attention: Ascend operator parity and fallback selection."""

import pytest
import torch

pytest.importorskip("torch_npu")

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.magi2 import attention
from vllm_omni.diffusion.models.magi2.attention import VarlenHandler
from vllm_omni.diffusion.models.magi2.layers.native.attention_ops import torch_varlen_attention_with_sink
from vllm_omni.diffusion.models.magi2.layers.npu import attention_ops as npu_attention
from vllm_omni.diffusion.models.magi2.preview_data_proxy import (
    Magi2DataProxy,
    Magi2PreviewDataProxyConfig,
    ModelInput,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


@pytest.fixture
def npu_runtime():
    torch_npu = pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an available NPU")
    return torch_npu


@hardware_test(res={"npu": "A5"}, num_cards=1)
@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires NPU")
def test_npu_data_proxy_keeps_varlen_bounds_on_cpu(npu_runtime):
    proxy = Magi2DataProxy(Magi2PreviewDataProxyConfig(time_channel_dim=0))
    packed = proxy.process_input(
        ModelInput(
            x_t=torch.zeros(1, 2, 1, 1, 2, device="npu"),
            audio_x_t=torch.zeros(1, 1, 2, device="npu"),
            audio_feat_len=[1],
            txt_feat=torch.zeros(1, 1, 2, device="npu"),
            txt_feat_len=[1],
            t=torch.tensor([0.5], device="npu"),
        )
    )

    bounds = packed.varlen_handler.cu_seqlens_q
    assert bounds is packed.varlen_handler.cu_seqlens_k
    assert bounds is not None
    assert bounds.device.type == "cpu"
    assert bounds.dtype == torch.int32
    assert packed.token_sequence.device.type == "npu"
    assert packed.token_sequence.shape[0] == 4
    assert bounds.tolist() == [0, 4]


def _inputs(head_dim=64, kv_heads=2, sink_count=2, *, cu_q=(0, 3, 9), cu_k=(0, 4, 11)):
    generator = torch.Generator().manual_seed(19)
    q = torch.randn(cu_q[-1], 4, head_dim, generator=generator).to(torch.bfloat16)
    k = torch.randn(cu_k[-1], kv_heads, head_dim, generator=generator).to(torch.bfloat16)
    v = torch.randn(cu_k[-1], kv_heads, head_dim, generator=generator).to(torch.bfloat16)
    sink = None
    if sink_count is not None:
        sink = torch.linspace(-1.98765, 2.12345, sink_count * 4).reshape(sink_count, 4)
    bounds_q = torch.tensor(cu_q, dtype=torch.int64)
    bounds_k = torch.tensor(cu_k, dtype=torch.int64)
    varlen = VarlenHandler(bounds_q, bounds_k, torch.diff(bounds_q).max().item(), torch.diff(bounds_k).max().item())
    return q, k, v, sink, varlen


@pytest.mark.cpu
@pytest.mark.parametrize("softcap", [-1.0, 2.0])
def test_npu_attention_cpu_inputs_use_native(mocker, softcap):
    q, k, v, sink, varlen = _inputs()
    expected = torch_varlen_attention_with_sink(
        q, k, v, cu_seqlens_q=varlen.cu_seqlens_q, cu_seqlens_k=varlen.cu_seqlens_k, sink=sink, softcap=softcap
    )
    fallback = mocker.spy(npu_attention, "native_varlen_attention_with_sink")

    actual = npu_attention.varlen_attention_with_sink(
        q,
        k,
        v,
        cu_seqlens_q=varlen.cu_seqlens_q,
        cu_seqlens_k=varlen.cu_seqlens_k,
        max_seqlen_q=varlen.max_seqlen_q,
        max_seqlen_k=varlen.max_seqlen_k,
        sink=sink,
        softcap=softcap,
    )

    fallback.assert_called_once()
    assert fallback.call_args.kwargs["softcap"] == softcap
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@hardware_test(res={"npu": "A5"}, num_cards=1)
@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires NPU")
@pytest.mark.parametrize(
    "head_dim,kv_heads,sink_count,noncontiguous,bounds_device",
    [
        pytest.param(64, 4, None, False, "cpu", id="mha-no-sink"),
        pytest.param(128, 2, 1, False, "npu", id="gqa-one-sink-device-bounds"),
        pytest.param(128, 2, 2, True, "cpu", id="gqa-multiple-sinks-strided"),
        pytest.param(64, 2, 0, False, "cpu", id="gqa-empty-sink"),
    ],
)
def test_real_npu_attention_matches_sink_oracle(
    mocker, npu_runtime, head_dim, kv_heads, sink_count, noncontiguous, bounds_device
):
    q, k, v, sink, varlen = _inputs(head_dim, kv_heads, sink_count)
    expected = torch_varlen_attention_with_sink(
        q, k, v, cu_seqlens_q=varlen.cu_seqlens_q, cu_seqlens_k=varlen.cu_seqlens_k, sink=sink
    )
    device_inputs = tuple(tensor.to("npu") for tensor in (q, k, v))
    if noncontiguous:
        device_inputs = tuple(tensor.transpose(0, 1).contiguous().transpose(0, 1) for tensor in device_inputs)
        assert all(not tensor.is_contiguous() for tensor in device_inputs)
    device_sink = None if sink is None else sink.to("npu")
    device_varlen = VarlenHandler(
        varlen.cu_seqlens_q.to(bounds_device),
        varlen.cu_seqlens_k.to(bounds_device),
        varlen.max_seqlen_q,
        varlen.max_seqlen_k,
    )
    provider = mocker.spy(npu_runtime, "npu_fused_infer_attention_score_v2")
    fallback = mocker.spy(npu_attention, "native_varlen_attention_with_sink")
    assert attention.varlen_attention_with_sink is npu_attention.varlen_attention_with_sink

    with torch.inference_mode():
        actual = attention.packed_attention_with_sink(*device_inputs, device_varlen, sink=device_sink)

    provider.assert_called_once()
    fallback.assert_not_called()
    args, kwargs = provider.call_args
    assert all(tensor.is_contiguous() for tensor in args[:3])
    assert kwargs["actual_seq_qlen"] == [3, 9]
    assert kwargs["actual_seq_kvlen"] == [4, 11]
    assert kwargs["num_query_heads"] == 4 and kwargs["num_key_value_heads"] == kv_heads
    assert kwargs["softmax_scale"] == head_dim**-0.5
    assert kwargs["input_layout"] == "TND" and kwargs["sparse_mode"] == 0
    assert kwargs["return_softmax_lse"] is False
    if sink_count:
        folded = kwargs["learnable_sink"]
        assert folded.dtype == torch.bfloat16 and folded.device.type == "npu" and folded.is_contiguous()
        torch.testing.assert_close(folded.cpu(), torch.logsumexp(sink, dim=0).to(torch.bfloat16), rtol=0, atol=0)
    else:
        assert kwargs["learnable_sink"] is None
    assert actual.shape == q.shape and actual.dtype == q.dtype and actual.device.type == "npu"
    assert torch.isfinite(actual).all()
    # Check the FP32 oracle, including the adapter's extra BF16 sink-logit rounding.
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-2, atol=3e-3)
    for device_tensor, original in zip(device_inputs, (q, k, v), strict=True):
        torch.testing.assert_close(device_tensor.cpu(), original, rtol=0, atol=0)
    if sink is not None:
        assert device_sink is not None
        torch.testing.assert_close(device_sink.cpu(), sink, rtol=0, atol=0)


@hardware_test(res={"npu": "A5"}, num_cards=1)
@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires NPU")
def test_real_npu_attention_matches_sink_oracle_across_tiles(mocker, npu_runtime):
    q, k, v, sink, varlen = _inputs(128, 2, 2, cu_q=(0, 37, 300), cu_k=(0, 41, 330))
    expected = torch_varlen_attention_with_sink(
        q, k, v, cu_seqlens_q=varlen.cu_seqlens_q, cu_seqlens_k=varlen.cu_seqlens_k, sink=sink
    )
    provider = mocker.spy(npu_runtime, "npu_fused_infer_attention_score_v2")
    fallback = mocker.spy(npu_attention, "native_varlen_attention_with_sink")

    with torch.inference_mode():
        actual = attention.packed_attention_with_sink(
            q.to("npu"), k.to("npu"), v.to("npu"), varlen, sink=sink.to("npu")
        )

    provider.assert_called_once()
    fallback.assert_not_called()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-2, atol=3e-3)


@hardware_test(res={"npu": "A5"}, num_cards=1)
@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires NPU")
def test_npu_attention_positive_softcap_uses_native(mocker, npu_runtime):
    q, k, v, sink, varlen = _inputs()
    expected = torch_varlen_attention_with_sink(
        q,
        k,
        v,
        cu_seqlens_q=varlen.cu_seqlens_q,
        cu_seqlens_k=varlen.cu_seqlens_k,
        sink=sink,
        softcap=2.0,
    )
    device_varlen = VarlenHandler(
        varlen.cu_seqlens_q.to("npu"), varlen.cu_seqlens_k.to("npu"), varlen.max_seqlen_q, varlen.max_seqlen_k
    )
    provider = mocker.spy(npu_runtime, "npu_fused_infer_attention_score_v2")
    fallback = mocker.spy(npu_attention, "native_varlen_attention_with_sink")

    with torch.inference_mode():
        actual = attention.packed_attention_with_sink(
            q.to("npu"), k.to("npu"), v.to("npu"), device_varlen, sink=sink.to("npu"), softcap=2.0
        )

    provider.assert_not_called()
    fallback.assert_called_once()
    assert fallback.call_args.kwargs["softcap"] == 2.0
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-2, atol=3e-3)
