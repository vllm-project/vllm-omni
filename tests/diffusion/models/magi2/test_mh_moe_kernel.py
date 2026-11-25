# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.magi2.layers.cuda import fused_moe_kernels as kernels

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _expert_inputs(tokens):
    torch.manual_seed(0)
    d_head, d_expert = 256, 1280
    x = torch.randn(tokens, 1, d_head, device="cuda", dtype=torch.bfloat16)
    gather_ids = torch.arange(tokens, device="cuda", dtype=torch.int32)
    probs = torch.ones(tokens, device="cuda", dtype=torch.float32)
    offsets = torch.tensor([0, tokens], device="cuda", dtype=torch.int64)
    gate = torch.randn(1, d_head, d_expert, device="cuda", dtype=x.dtype) / d_head**0.5
    up = torch.randn_like(gate) / d_head**0.5
    down = torch.randn(1, d_expert, d_head, device="cuda", dtype=x.dtype) / d_expert**0.5
    return x, gather_ids, probs, offsets, gate, up, down


@hardware_test(res={"cuda": ["B200", "H100", "H200", "L4"]}, num_cards=1)
@pytest.mark.parametrize("tokens", [1, 65, 129])
@pytest.mark.parametrize("deterministic", [False, True])
def test_released_expert_shape_matches_torch(monkeypatch, tokens, deterministic):
    args = _expert_inputs(tokens)
    x = args[0]
    monkeypatch.setattr(kernels, "_RESOLVED_BLOCK_CONFIG", {})
    with torch.inference_mode():
        actual = kernels.triton_mh_moe_forward(*args, deterministic=deterministic)
        expected = kernels.torch_mh_moe_forward(*args)
        repeated = kernels.triton_mh_moe_forward(*args, deterministic=deterministic)

    key = (x.device, x.dtype, 256, 1280, deterministic)
    assert kernels._RESOLVED_BLOCK_CONFIG[key] in kernels._block_config_candidates(x.device.index)
    # Torch rounds projections to BF16 before activation, while the fused
    # reference kernel retains FP32 accumulators through the activation.
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(repeated, expected, atol=2e-2, rtol=2e-2)


@hardware_test(res={"cuda": ["B200", "H100", "H200", "L4"]}, num_cards=1)
@pytest.mark.parametrize("deterministic", [False, True])
def test_resource_rejection_retries_real_kernel_and_caches(monkeypatch, deterministic):
    args = _expert_inputs(65)
    original_kernel = kernels._mh_moe_kernel
    attempts = []

    class RejectLargeTile:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                attempts.append(args[29])
                if args[29] == 128:
                    raise kernels.triton.runtime.errors.OutOfResources(528, 512, "tensor memory")
                return original_kernel[grid](*args, **kwargs)

            return launch

    monkeypatch.setattr(kernels, "_mh_moe_kernel", RejectLargeTile())
    monkeypatch.setattr(kernels, "_RESOLVED_BLOCK_CONFIG", {})
    monkeypatch.setattr(
        kernels, "_block_config_candidates", lambda device_id: ((128, 64, 32, 2, 8), (64, 64, 32, 2, 4))
    )
    with torch.inference_mode():
        actual = kernels.triton_mh_moe_forward(*args, deterministic=deterministic)
        repeated = kernels.triton_mh_moe_forward(*args, deterministic=deterministic)
        expected = kernels.torch_mh_moe_forward(*args)

    assert attempts == [128, 64, 64]
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(repeated, expected, atol=2e-2, rtol=2e-2)
