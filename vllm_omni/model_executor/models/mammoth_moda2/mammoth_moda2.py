"""顶层入口：MammothModa2ForConditionalGeneration

仿照 Qwen2_5OmniForConditionalGeneration，根据 model_stage 选择子模块：
- ar  : 自定义 MoE 语言 + 视觉塔
- dit : 生成 DiT 阶段
- vae : 预留（暂不实现）
"""

from __future__ import annotations

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm.multimodal import MULTIMODAL_REGISTRY
from .mammoth_moda2_ar import (
    MammothModa2ARMultiModalProcessor,
    MammothModa2ARProcessingInfo,
    MammothModa2ARDummyInputsBuilder,
)


@MULTIMODAL_REGISTRY.register_processor(
    MammothModa2ARMultiModalProcessor,
    info=MammothModa2ARProcessingInfo,
    dummy_inputs=MammothModa2ARDummyInputsBuilder,
)
class MammothModa2ForConditionalGeneration(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        cfg = vllm_config.model_config.hf_config
        self.model_stage = vllm_config.model_config.model_stage

        if self.model_stage == "ar":
            # AR 阶段：多模态 + MoE 文本
            self.model = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "ar"),
                hf_config=cfg.llm_config if hasattr(cfg, "llm_config") else cfg.text_config,
                architectures=["MammothModa2ARForConditionalGeneration"],
            )
        elif self.model_stage == "dit":
            self.model = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "dit"),
                hf_config=cfg.gen_dit_config if hasattr(cfg, "gen_dit_config") else cfg,
                architectures=["MammothModa2DiTForConditionalGeneration"],
            )
        elif self.model_stage == "vae":
            # 预留：目前未实现 VAEs，给出明确报错
            raise NotImplementedError("MammothModa2 VAE stage not implemented yet.")
        else:
            raise ValueError(f"Unsupported model_stage: {self.model_stage}")

        # 暴露中间张量创建器供 PP 使用（若子模块提供）
        self.make_empty_intermediate_tensors = getattr(
            self.model, "make_empty_intermediate_tensors", lambda: None
        )

    def forward(self, *args, **kwargs) -> OmniOutput | torch.Tensor:
        out = self.model(*args, **kwargs)
        # 子模块可能直接返回 OmniOutput / tensor；保持向后兼容
        if isinstance(out, OmniOutput):
            return out
        return OmniOutput(text_hidden_states=out, multimodal_outputs=None, intermediate_tensors=None)

    def compute_logits(self, hidden_states: torch.Tensor):
        if hasattr(self.model, "compute_logits"):
            return self.model.compute_logits(hidden_states)
        return None

    def load_weights(self, weights):
        if hasattr(self.model, "load_weights"):
            return self.model.load_weights(weights)
        return set()
