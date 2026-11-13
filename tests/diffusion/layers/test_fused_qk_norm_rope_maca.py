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
    "platform, grad_enabled, interleaved, compiling, uses_maca",
    [
        ("maca", False, False, False, True),
        ("maca", True, False, False, False),
        ("cuda", False, False, False, False),
        ("maca", False, True, False, False),
        ("maca", False, False, True, False),
    ],
)
def test_maca_qk_rope_dispatch(monkeypatch, platform, grad_enabled, interleaved, compiling, uses_maca):
    monkeypatch.setattr(ops, "HAS_TRITON", True)
    monkeypatch.setattr(ops, "current_platform", SimpleNamespace(device_name=platform, is_cuda=lambda: False))
    monkeypatch.setattr(ops, "current_omni_platform", SimpleNamespace(is_npu=lambda: False))
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: compiling)

    launch = Mock(side_effect=lambda x, *_, **__: x)
    eager = Mock(side_effect=lambda q, k, *_: (q, k))
    monkeypatch.setattr(ops, "_launch_fused_rms_norm_rope", launch)
    monkeypatch.setattr(ops, "_eager_qk_norm_rope", eager)

    with FakeTensorMode(), torch.set_grad_enabled(grad_enabled):
        qkv = torch.empty(17, 21, 128, dtype=torch.bfloat16, device="cuda")
        q, k, _ = qkv.split(7, dim=1)
        q_weight = qkv.new_empty(128)
        k_weight = qkv.new_empty(128)
        rope_table = qkv.new_empty((17, 96))
        out_q, out_k = ops.fused_qk_norm_rope(q, k, q_weight, k_weight, rope_table, 1e-6, interleaved=interleaved)

    assert out_q is q and out_k is k
    if uses_maca:
        assert launch.call_count == 2
        assert launch.call_args_list[0].args[0] is q
        assert launch.call_args_list[1].args[0] is k
        assert all(not call.kwargs for call in launch.call_args_list)
        eager.assert_not_called()
    else:
        launch.assert_not_called()
        eager.assert_called_once()
