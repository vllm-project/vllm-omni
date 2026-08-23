# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flashinfer_sm120_attn import (
    FlashInferSM120AttentionImpl,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


def _impl(**quant_overrides):
    quant = {
        "dtype_qk": "fp8_e4m3",
        "flashinfer_backend": "cute-dsl-prims",
        **quant_overrides,
    }
    return FlashInferSM120AttentionImpl(
        num_heads=2,
        num_kv_heads=2,
        head_size=32,
        softmax_scale=1.0 / math.sqrt(32),
        backend_kwargs={"quant": quant},
        prefix="transformer.blocks.0.attn",
    )


def test_requires_explicit_prims_fp8_config():
    with pytest.raises(ValueError, match="dtype_qk"):
        FlashInferSM120AttentionImpl(2, 32, 1.0, backend_kwargs={})
    with pytest.raises(ValueError, match="flashinfer_backend"):
        FlashInferSM120AttentionImpl(
            2,
            32,
            1.0,
            backend_kwargs={"quant": {"dtype_qk": "fp8_e4m3"}},
        )


def test_static_scale_validation_and_quantization():
    with pytest.raises(ValueError, match="q_scale"):
        _impl(q_scale=0.0)
    value = torch.tensor([-2.0, 0.5, 3.0], dtype=torch.bfloat16)
    actual = _impl(q_scale=0.5, k_scale=0.5, v_scale=0.5)._quantize(value, 0.5)
    torch.testing.assert_close(actual.float(), torch.tensor([-4.0, 1.0, 6.0]), atol=0, rtol=0)


def test_first_call_calibration_is_cached_with_headroom():
    impl = _impl()
    q = torch.tensor([1.0, -2.0])
    k = torch.tensor([3.0, -1.0])
    v = torch.tensor([4.0, -2.0])
    first = impl._resolve_scales(q, k, v)
    assert first == pytest.approx((2.0 / 224.0, 3.0 / 224.0, 4.0 / 224.0))
    second = impl._resolve_scales(q * 100, k * 100, v * 100)
    assert second == first


def test_ragged_wrapper_is_planned_once_and_receives_scales(monkeypatch):
    import vllm_omni.diffusion.attention.backends.flashinfer_sm120_attn as mod

    observed = {"plans": 0, "runs": 0}

    class FakeWrapper:
        def __init__(self, workspace, layout, backend):
            observed.update(workspace=workspace, layout=layout, backend=backend)

        def plan(self, qo, kv, hq, hkv, dim, **kwargs):
            observed["plans"] += 1
            observed.update(qo=qo.clone(), kv=kv.clone(), hq=hq, hkv=hkv, dim=dim, plan_kwargs=kwargs)

        def run(self, q, k, v, *, out, q_scale, k_scale, v_scale):
            observed["runs"] += 1
            observed.update(
                q_dtype=q.dtype,
                k_dtype=k.dtype,
                v_dtype=v.dtype,
                scales=(q_scale, k_scale, v_scale),
            )
            out.zero_()
            return out

    monkeypatch.setattr(mod, "_ragged_wrapper_cls", lambda: FakeWrapper)
    monkeypatch.setattr(mod, "_get_sm120_workspace", lambda device: torch.empty(1, dtype=torch.uint8))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 0))

    impl = _impl(q_scale=0.5, k_scale=0.25, v_scale=0.125)
    q = torch.randn(1, 4, 2, 32, dtype=torch.bfloat16)
    cu = torch.tensor([0, 3, 4], dtype=torch.int32)
    metadata = AttentionMetadata(extra={"cu_seqlens_q": cu, "cu_seqlens_k": cu})

    first = impl.forward_cuda(q, q, q, metadata)
    second = impl.forward_cuda(q, q, q, metadata)

    assert first.shape == q.shape and first.dtype == q.dtype
    assert second.shape == q.shape
    assert observed["layout"] == "NHD"
    assert observed["backend"] == "cute-dsl-prims"
    assert observed["plans"] == 1 and observed["runs"] == 2
    assert observed["q_dtype"] == torch.float8_e4m3fn
    assert observed["scales"] == (0.5, 0.25, 0.125)
    assert observed["qo"].tolist() == [0, 3, 4]
