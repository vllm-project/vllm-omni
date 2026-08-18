# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.layers.swiglu import SwiGLU
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3MLP
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def test_dit_swiglu_native_matches_packed_reference() -> None:
    packed = torch.randn(257, 256, dtype=torch.float32)
    gate, up = packed.chunk(2, dim=-1)

    torch.testing.assert_close(SwiGLU().forward_native(packed), F.silu(gate) * up)


def test_dit_mlp_uses_packed_swiglu_activation() -> None:
    class PackedFC1:
        def __init__(self, packed: torch.Tensor) -> None:
            self.packed = packed

        def __call__(self, _: torch.Tensor) -> tuple[torch.Tensor, None]:
            return self.packed, None

    class IdentityFC2:
        def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
            return x, None

    tensor_kwargs: dict[str, object] = {"dtype": torch.float32}
    if current_omni_platform.is_npu():
        tensor_kwargs = {"device": "npu", "dtype": torch.bfloat16}
    packed = torch.randn(4, 128, **tensor_kwargs)
    mlp = SimpleNamespace(
        fc1=PackedFC1(packed),
        act_fn=SwiGLU(),
        fc2=IdentityFC2(),
    )

    output = MiniMaxH3MLP.forward(mlp, torch.empty(4, 1, device=packed.device))
    gate, up = packed.chunk(2, dim=-1)

    torch.testing.assert_close(
        output,
        F.silu(gate) * up,
        **({"atol": 2e-2, "rtol": 2e-2} if current_omni_platform.is_npu() else {}),
    )


@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires Ascend NPU")
def test_dit_swiglu_npu_matches_packed_reference() -> None:
    """DiT feeds a packed two-dimensional [gate, up] tensor to SwiGLU."""

    packed = torch.randn(257, 256, device="npu", dtype=torch.bfloat16)
    gate, up = packed.chunk(2, dim=-1)

    torch.testing.assert_close(
        SwiGLU()(packed),
        F.silu(gate) * up,
        atol=2e-2,
        rtol=2e-2,
    )
