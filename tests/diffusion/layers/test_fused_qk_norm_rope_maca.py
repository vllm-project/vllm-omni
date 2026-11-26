# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_omni.diffusion.layers import fused_qk_norm_rope as ops

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.cpu
@pytest.mark.parametrize(
    "platform, compiling, interleaved, tokens, dtype, use_custom_op",
    [
        pytest.param("maca", False, False, 17, torch.bfloat16, True, id="maca-eager"),
        pytest.param("maca", True, False, 17, torch.bfloat16, False, id="maca-compiled"),
        pytest.param("cuda", True, False, 17, torch.bfloat16, True, id="cuda-compiled"),
        pytest.param("rocm", False, False, 17, torch.bfloat16, False, id="other-platform"),
        pytest.param("maca", False, True, 17, torch.bfloat16, False, id="maca-interleaved"),
        pytest.param("maca", False, False, 0, torch.bfloat16, True, id="maca-zero-tokens"),
        pytest.param("maca", False, False, 17, torch.float32, False, id="maca-fp32"),
    ],
)
def test_half_split_qk_rope_dispatch(monkeypatch, platform, compiling, interleaved, tokens, dtype, use_custom_op):
    monkeypatch.setattr(ops, "HAS_TRITON", True)
    monkeypatch.setattr(
        ops, "current_platform", SimpleNamespace(device_name=platform, is_cuda=lambda: platform == "cuda")
    )
    monkeypatch.setattr(ops, "current_omni_platform", SimpleNamespace(is_npu=lambda: False))
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: compiling)

    custom_op = Mock(side_effect=ops._fused_qk_norm_rope_impl)
    launch = Mock(side_effect=lambda x, *_, **__: x)
    eager = Mock(side_effect=lambda q, k, *_: (q, k))
    monkeypatch.setattr(torch.ops.vllm_omni, "fused_qk_norm_rope", custom_op)
    monkeypatch.setattr(ops, "_launch_fused_rms_norm_rope", launch)
    monkeypatch.setattr(ops, "_eager_qk_norm_rope", eager)

    with FakeTensorMode(), torch.inference_mode():
        qkv = torch.empty(tokens, 21, 128, dtype=dtype, device="cuda")
        q, k, _ = qkv.split(7, dim=1)
        q_weight = qkv.new_empty(128)
        k_weight = qkv.new_empty(128)
        rope_table = qkv.new_empty((tokens, 96))
        out_q, out_k = ops.fused_qk_norm_rope(q, k, q_weight, k_weight, rope_table, 1e-6, interleaved=interleaved)

    assert out_q is q and out_k is k
    if use_custom_op:
        custom_op.assert_called_once()
        assert launch.call_count == 2
        assert launch.call_args_list[0].args[0] is q
        assert launch.call_args_list[1].args[0] is k
        eager.assert_not_called()
    else:
        custom_op.assert_not_called()
        launch.assert_not_called()
        eager.assert_called_once()
