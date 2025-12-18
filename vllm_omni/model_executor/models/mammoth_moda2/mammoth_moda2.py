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
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.utils import add_prefix_to_loaded_weights
from vllm.multimodal import MULTIMODAL_REGISTRY
from .mammoth_moda2_ar import (
    MammothModa2ARMultiModalProcessor,
    MammothModa2ARProcessingInfo,
    MammothModa2ARDummyInputsBuilder,
)
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration


@MULTIMODAL_REGISTRY.register_processor(
    MammothModa2ARMultiModalProcessor,
    info=MammothModa2ARProcessingInfo,
    dummy_inputs=MammothModa2ARDummyInputsBuilder,
)
class MammothModa2ForConditionalGeneration(nn.Module, SupportsMultiModal,
                                           SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        cfg = vllm_config.model_config.hf_config
        self.model_stage = vllm_config.model_config.model_stage
        self.multimodal_config = vllm_config.model_config.multimodal_config

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

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int):  # noqa: ARG003
        return Qwen2_5_VLForConditionalGeneration.get_placeholder_str(modality,
                                                                      i)

    def get_language_model(self) -> nn.Module:
        if hasattr(self.model, "get_language_model"):
            return self.model.get_language_model()
        return self.model

    def get_multimodal_embeddings(self, **kwargs: object):
        if hasattr(self.model, "get_multimodal_embeddings"):
            return self.model.get_multimodal_embeddings(**kwargs)
        return []

    def get_input_embeddings(self,
                             input_ids: torch.Tensor,
                             multimodal_embeddings=None) -> torch.Tensor:
        if hasattr(self.model, "get_input_embeddings"):
            return self.model.get_input_embeddings(
                input_ids, multimodal_embeddings=multimodal_embeddings)
        raise NotImplementedError(
            "Underlying model does not implement get_input_embeddings")

    def forward(self, *args, **kwargs) -> OmniOutput | torch.Tensor:
        out = self.model(*args, **kwargs)
        # 子模块可能直接返回 OmniOutput / tensor；保持向后兼容
        if isinstance(out, OmniOutput):
            return out
        return OmniOutput(text_hidden_states=out, multimodal_outputs=None, intermediate_tensors=None)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput):
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hasattr(self.model, "compute_logits"):
            return self.model.compute_logits(hidden_states)
        return None

    def load_weights(self, weights):
        if hasattr(self.model, "load_weights"):
            loaded = self.model.load_weights(weights)
            # 本 wrapper 仅把子模型挂在 `self.model` 下，因此需要把返回的已加载参数名补上 "model." 前缀，
            # 才能与 DefaultModelLoader 的 strict check（基于本对象 named_parameters）对齐。
            return add_prefix_to_loaded_weights(loaded, "model")
        return set()
