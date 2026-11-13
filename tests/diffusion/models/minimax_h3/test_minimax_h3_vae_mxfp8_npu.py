# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""NPU numerical smoke tests for MiniMax H3 VAE MXFP8."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_test
from vllm_omni.platforms import current_omni_platform
from vllm_omni.platforms.npu import is_a5

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.fixture
def a5_device():
    if not current_omni_platform.is_npu():
        pytest.skip("requires an NPU")
    device = current_omni_platform.get_torch_device()
    if not is_a5(device):
        pytest.skip("requires an Ascend A5")
    return device


@hardware_test(res={"npu": "A5"}, num_cards=1)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mxfp8_attention_unaligned_batch_matches_reference(a5_device, dtype):
    pytest.importorskip("mindiesd")
    from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
    from vllm_omni.diffusion.models.minimax_h3.vae_mxfp8 import _run_mxfp8_fa

    torch.manual_seed(37)
    query, key, value = (torch.randn(2, 195, 32, 64, device=a5_device, dtype=dtype) for _ in range(3))
    backend = FlashAttentionImpl(num_heads=32, head_size=64, softmax_scale=64**-0.5, qkv_layout="BSND")
    with torch.inference_mode():
        actual = _run_mxfp8_fa(backend, query, key, value)
        expected = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        ).transpose(1, 2)
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    error = (actual.float() - expected.float()).abs()
    # A smoke bound for this fixed random input, not a full-video quality limit.
    assert error.max().item() < 0.3
    assert error.mean().item() < 0.03


@hardware_test(res={"npu": "A5"}, num_cards=1)
def test_mxfp8_projection_survives_cpu_offload(a5_device, monkeypatch):
    import torch_npu
    from vllm_ascend import utils as ascend_utils

    from vllm_omni.diffusion.models.minimax_h3.vae_mxfp8 import _NPUMxfp8VideoVAELinear
    from vllm_omni.diffusion.offloader.module_residency import PinnedModuleStager
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    torch.manual_seed(37)
    # The native H3 decoder's QKV projection includes a bias.
    linear = torch.nn.Linear(2048, 6144, device="cpu")
    config = DiffusionMXFP8Config()
    cpu_device = torch.device("cpu")
    with monkeypatch.context() as nz_config:
        nz_config.setattr(ascend_utils, "get_ascend_config", lambda: SimpleNamespace(weight_nz_mode=1))
        resident = _NPUMxfp8VideoVAELinear(deepcopy(linear), config, device=a5_device, load_device=a5_device)
    assert torch_npu.get_npu_format(resident.weight) == ascend_utils.ACL_FORMAT_FRACTAL_NZ
    staged = _NPUMxfp8VideoVAELinear(linear, config, a5_device, cpu_device)
    stager = PinnedModuleStager(staged, a5_device)
    hidden_states = torch.randn(1, 195, 2048, device=a5_device, dtype=torch.bfloat16)

    with torch.inference_mode():
        expected = resident(hidden_states)

    for _ in range(2):
        assert all(parameter.device.type == "cpu" for parameter in staged.parameters())
        stager.load()
        try:
            assert all(parameter.device.type == "npu" for parameter in staged.parameters())
            assert torch_npu.get_npu_format(staged.weight) == ascend_utils.ACL_FORMAT_FRACTAL_ND
            with torch.inference_mode():
                actual = staged(hidden_states)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finally:
            stager.offload()
    assert all(parameter.device.type == "cpu" for parameter in staged.parameters())
