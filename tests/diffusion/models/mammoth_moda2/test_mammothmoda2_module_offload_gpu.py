# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GPU lifecycle tests for MammothModa2 module-level (component) CPU offload.

These run the real offload backend (no mocks) and assert that enabling offload
moves the DiT to CPU, while disabling restores weights bit-exactly.
"""

import pytest
import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit import (
    MammothModa2DiTPipeline,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _tiny_config() -> OmniDiffusionConfig:
    raw = {
        "model_type": "mammothmoda2",
        "llm_config": {
            "model_type": "mammothmoda2_qwen2_5_vl",
            "text_config": {
                "model_type": "mammothmoda2_qwen2_5_vl_text",
                "hidden_size": 32,
                "gen_vocab_start_index": 100,
            },
        },
        "gen_vae_config": {"block_out_channels": [32, 32]},
        "gen_dit_config": {
            "hidden_size": 32,
            "in_channels": 4,
            "num_attention_heads": 4,
            "num_kv_heads": 2,
            "axes_dim_rope": (4, 2, 2),
            "axes_lens": (16, 16, 16),
            "num_layers": 2,
            "multiple_of": 8,
            "text_feat_dim": 32,
        },
        "gen_axes_dim_rope": [8, 8, 8],
        "gen_axes_lens": [16, 16, 16],
    }
    return OmniDiffusionConfig(
        model="/models/MammothModa2-Preview",
        model_class_name="MammothModa2DiTPipeline",
        tf_model_config=TransformerConfig.from_dict(raw),
    )


def _build_pipeline() -> MammothModa2DiTPipeline:
    torch.manual_seed(42)
    return MammothModa2DiTPipeline(od_config=_tiny_config()).to("cuda:0").eval()


def _requires_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires an NVIDIA CUDA device")


def test_module_offload_enable_moves_dit_to_cpu() -> None:
    _requires_cuda()
    pipeline = _build_pipeline()
    pipeline.enable_omni_model_cpu_offload(device=torch.device("cuda:0"), pin_memory=True, use_hsdp=False)
    assert pipeline._model_cpu_offload_modules
    assert next(pipeline.gen_transformer.parameters()).device.type == "cpu"
    assert next(pipeline.gen_vae.parameters()).device.type == "cuda"


def test_module_offload_disable_restores_weights() -> None:
    _requires_cuda()
    pipeline = _build_pipeline()
    master = {name: tensor.detach().cpu().clone() for name, tensor in pipeline.state_dict().items()}
    pipeline.enable_omni_model_cpu_offload(device=torch.device("cuda:0"), pin_memory=True, use_hsdp=False)
    pipeline.disable_omni_model_cpu_offload()
    assert pipeline._model_cpu_offload_modules == []
    for name, tensor in pipeline.state_dict().items():
        torch.testing.assert_close(tensor.cpu(), master[name], rtol=0, atol=0)
