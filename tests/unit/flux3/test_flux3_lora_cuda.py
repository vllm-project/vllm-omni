# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA smoke test for Flux3 raw LoRA activation."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import safetensors.torch
import torch
import torch.nn as nn
from vllm.lora.request import LoRARequest

from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.models.flux3.flux3_transformer import Flux3Transformer2DModel
from vllm_omni.diffusion.models.flux3.pipeline_flux3 import Flux3Pipeline

DEVICE = torch.device("cuda")
require_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")

_TINY_DIT_KWARGS = {
    "hidden_size": 128,
    "num_heads": 4,
    "depth": 1,
    "depth_single_blocks": 2,
    "depth_late_blocks": 0,
    "context_in_dim": 32,
    "vec_in_dim": 16,
    "mlp_ratio": 2.0,
    "axes_dim": (8, 8, 8, 8),
    "theta": 100,
}


@pytest.fixture(autouse=True, scope="module")
def mock_tp() -> Generator[None, None, None]:
    """Mock tensor parallelism as a one-rank CUDA deployment."""
    from vllm_omni.diffusion.attention import selector as attn_selector

    attn_selector._cached_get_backend_cls.cache_clear()
    attn_selector._log_backend_resolution.cache_clear()
    patches = [
        mock.patch(
            "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        mock.patch(
            "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        mock.patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        mock.patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        mock.patch(
            "vllm.model_executor.layers.linear.tensor_model_parallel_all_reduce",
            side_effect=lambda x: x,
        ),
        mock.patch(
            "vllm.model_executor.layers.linear.tensor_model_parallel_all_gather",
            side_effect=lambda x, dim=-1: x,
        ),
    ]
    for patch in patches:
        patch.start()
    yield
    for patch in patches:
        patch.stop()
    attn_selector._cached_get_backend_cls.cache_clear()
    attn_selector._log_backend_resolution.cache_clear()


def _minimal_pipeline() -> Flux3Pipeline:
    """Build a pipeline containing only the CUDA transformer LoRA target."""
    pipeline = object.__new__(Flux3Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.transformer = Flux3Transformer2DModel(**_TINY_DIT_KWARGS).to(DEVICE)
    pipeline.od_config = SimpleNamespace(enable_distributed_layerwise_offload=False)
    pipeline._raw_lora_adapter_ids = set()
    return pipeline


def _layer_output(layer: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Return the tensor portion of a vLLM parallel linear layer output."""
    output = layer(inputs)
    return output[0] if isinstance(output, tuple) else output


@pytest.mark.core_model
@pytest.mark.cuda
@require_gpu
def test_flux3_raw_lora_activates_and_deactivates_image_input_projection(tmp_path: Path) -> None:
    """CUDA single-rank deployment, image latent input, projected-token output, raw Flux3 LoRA activation, synthetic data."""
    pipeline = _minimal_pipeline()
    projection_inputs = torch.randn(2, 128, device=DEVICE)
    baseline = _layer_output(pipeline.transformer.emb_in["image"], projection_inputs)

    rank = 1
    adapter_path = tmp_path / "flux3-image-input.safetensors"
    safetensors.torch.save_file(
        {
            "diffusion_model.emb_in.image.lora_A.weight": torch.ones(rank, 128),
            "diffusion_model.emb_in.image.lora_B.weight": torch.ones(128, rank),
        },
        str(adapter_path),
        metadata={"rank": str(rank), "alpha": str(rank), "target_modules": '["image"]'},
    )
    request = LoRARequest(lora_name="flux3-image-input", lora_int_id=1, lora_path=str(adapter_path))
    manager = DiffusionLoRAManager(pipeline, device=DEVICE, dtype=torch.float32)

    manager.set_active_adapter(request)
    adapted = _layer_output(pipeline.transformer.emb_in["image"], projection_inputs)
    assert not torch.allclose(adapted, baseline)

    manager.set_active_adapter(None)
    restored = _layer_output(pipeline.transformer.emb_in["image"], projection_inputs)
    torch.testing.assert_close(restored, baseline)
