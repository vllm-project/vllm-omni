# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch.nn as nn
from vllm.model_executor.layers.linear import UnquantizedLinearMethod


class _RecordingQuantConfig:
    def __init__(self) -> None:
        self.prefixes: list[str] = []

    def get_quant_method(self, layer: nn.Module, prefix: str) -> UnquantizedLinearMethod:
        self.prefixes.append(prefix)
        return UnquantizedLinearMethod()


def test_qwen_image_transformer_block_uses_checkpoint_aligned_quant_prefixes(monkeypatch):
    import vllm.model_executor.layers.linear as linear_module
    import vllm.model_executor.parameter as parameter_module

    import vllm_omni.diffusion.models.qwen_image.qwen_image_transformer as qwen_module

    monkeypatch.setattr(linear_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(linear_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(qwen_module, "RMSNorm", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(qwen_module, "Attention", lambda *args, **kwargs: nn.Identity())

    quant_config = _RecordingQuantConfig()

    qwen_module.QwenImageTransformerBlock(
        dim=128,
        num_attention_heads=1,
        attention_head_dim=128,
        quant_config=quant_config,
        prefix="transformer_blocks.0",
    )

    assert quant_config.prefixes == [
        "transformer_blocks.0.attn.to_qkv",
        "transformer_blocks.0.attn.add_kv_proj",
        "transformer_blocks.0.attn.to_add_out",
        "transformer_blocks.0.attn.to_out",
        "transformer_blocks.0.img_mlp.net.0.proj",
        "transformer_blocks.0.img_mlp.net.2",
        "transformer_blocks.0.txt_mlp.net.0.proj",
        "transformer_blocks.0.txt_mlp.net.2",
    ]
