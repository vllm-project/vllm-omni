# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Matrix QKV uses the existing online FP8 loader and platform-selected kernel."""

import pytest
import torch
import torch.nn.functional as F
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.linear import QKVParallelLinear, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerTensorOnlineLinearMethod
from vllm.model_executor.model_loader.reload.layerwise import finalize_layerwise_processing
from vllm.utils.torch_utils import set_default_torch_dtype

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.matrix_game4.transformer import MatrixQKV
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]


@torch.inference_mode()
@pytest.mark.parametrize("ignored", [False, True])
def test_qkv_loads_separate_checkpoint_shards_with_standard_fp8(monkeypatch, ignored):
    for module in ("vllm.model_executor.parameter", "vllm.model_executor.layers.linear"):
        monkeypatch.setattr(f"{module}.get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(f"{module}.get_tensor_model_parallel_world_size", lambda: 1)
    prefix = "blocks.0.attn1.to_qkv"
    config = OmniDiffusionConfig(
        model="matrix-resources",
        quantization_config={"method": "fp8", "ignored_layers": [prefix] if ignored else []},
    )
    device = torch.device("cuda")
    vllm_config = create_diffusion_vllm_config(device, config)
    with set_current_vllm_config(vllm_config), set_default_torch_dtype(torch.bfloat16), device:
        torch.manual_seed(42)
        layers = [
            cls(
                64,
                64,
                1,
                bias=True,
                params_dtype=torch.bfloat16,
                disable_tp=True,
                quant_config=config.quantization_config,
                prefix=prefix,
            )
            for cls in (MatrixQKV, QKVParallelLinear)
        ]
        weights = torch.randn(3, 64, 64, dtype=torch.bfloat16) * 0.02
        biases = torch.randn(3, 64, dtype=torch.bfloat16) * 0.01
        for layer in layers:
            for index, shard in enumerate(("q", "k", "v")):
                layer.bias.weight_loader(layer.bias, biases[index], shard)
                layer.weight.weight_loader(layer.weight, weights[index], shard)
        if not ignored:
            finalize_layerwise_processing(torch.nn.ModuleList(layers), vllm_config.model_config)
        inputs = torch.randn(1, 128, 64, dtype=torch.bfloat16)
        actual, _ = layers[0](inputs)
        expected = torch.cat([F.linear(inputs, weight, bias) for weight, bias in zip(weights, biases)], dim=-1)
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        if ignored:
            assert isinstance(layers[0].quant_method, UnquantizedLinearMethod)
            assert torch.equal(actual, expected)
        else:
            assert isinstance(layers[0].quant_method, Fp8PerTensorOnlineLinearMethod)
            assert layers[0]._already_called_process_weights_after_loading
            assert torch.equal(actual, layers[1](inputs)[0])
            assert (actual.float() - expected.float()).norm() / expected.float().norm() < 0.06
