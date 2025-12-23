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
    # 让 vllm_omni/worker/gpu_model_runner.py 的 `extract_multimodal_outputs`
    # 走 OmniOutput 分支，从而拿到纯 torch.Tensor 的 text_hidden_states，避免后续
    # `hidden_states[logit_indices]` 因类型不匹配（list/tuple）而报错。
    have_multimodal_outputs = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        # 与 Qwen2_5OmniForConditionalGeneration 保持一致：实例级标记
        self.have_multimodal_outputs = True
        self.vllm_config = vllm_config
        cfg = vllm_config.model_config.hf_config
        self.model_stage = vllm_config.model_config.model_stage
        self.multimodal_config = vllm_config.model_config.multimodal_config

        # 仅用于调试/对齐 qwen2.5-omni：未使用的 stage 显式置空
        self.ar = None
        self.dit = None
        self.vae = None

        if self.model_stage == "ar":
            # AR 阶段：多模态 + MoE 文本
            self.ar = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "ar"),
                hf_config=cfg.llm_config if hasattr(cfg, "llm_config") else cfg.text_config,
                architectures=["MammothModa2ARForConditionalGeneration"],
            )
            self.model = self.ar
        elif self.model_stage == "dit":
            self.dit = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "dit"),
                # NOTE: init_vllm_registered_model -> VllmConfig.with_hf_config 要求传入的是
                # transformers.PretrainedConfig；而 Mammothmoda2Config.gen_dit_config 是 dict（diffusers config）。
                # DiT stage 的 hf_config 仍然使用顶层组合配置 Mammothmoda2Config，由 DiT 模块自己读取
                # config.gen_dit_config / gen_vae_config 等 dict。
                hf_config=cfg,
                architectures=["MammothModa2DiTForConditionalGeneration"],
            )
            self.model = self.dit
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
        # DiT stage does not consume token embeddings from `input_ids`; it uses
        # condition embeddings passed via additional_information.
        # However, vLLM's generation runner may still request token embeddings
        # to populate `inputs_embeds` buffers, so we provide a dummy tensor.
        if self.model_stage == "dit":
            hidden_size = int(self.vllm_config.model_config.get_hidden_size())
            try:
                target_dtype = next(self.model.parameters()).dtype
            except StopIteration:
                target_dtype = self.vllm_config.model_config.dtype
            return torch.zeros(
                (input_ids.numel(), hidden_size),
                device=input_ids.device,
                dtype=target_dtype,
            )
        raise NotImplementedError("Underlying model does not implement get_input_embeddings")

    def forward(self, *args, **kwargs) -> OmniOutput | torch.Tensor:
        out = self.model(*args, **kwargs)
        if isinstance(out, OmniOutput):
            return out
        # 子模块可能直接返回 tensor / list；保持向后兼容
        if isinstance(out, list):
            out = out[0]
        return OmniOutput(text_hidden_states=out, multimodal_outputs={}, intermediate_tensors=None)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput):
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hasattr(self.model, "compute_logits"):
            return self.model.compute_logits(hidden_states)
        return None

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict[str, object]]:
        if self.model_stage != "dit":
            raise RuntimeError(f"get_dummy_runtime_additional_information only valid for dit stage, got {self.model_stage}")
        if self.dit is None:
            raise RuntimeError("dit stage model is not initialized")
        if not hasattr(self.dit, "get_dummy_runtime_additional_information"):
            raise AttributeError("dit model missing get_dummy_runtime_additional_information")
        return self.dit.get_dummy_runtime_additional_information(num_reqs)

    def load_weights(self, weights):
        # 参考 Qwen2_5OmniForConditionalGeneration：按 stage 把权重交给对应子模块加载，
        # 并将子模块返回的“已加载参数名集合”补上正确的前缀，以通过 DefaultModelLoader 的严格校验。
        if self.model_stage == "ar":
            if self.ar is None or not hasattr(self.ar, "load_weights"):
                return set()
            loaded = self.ar.load_weights(weights)
            return add_prefix_to_loaded_weights(loaded, "ar")
        if self.model_stage == "dit":
            if self.dit is None or not hasattr(self.dit, "load_weights"):
                return set()
            loaded = self.dit.load_weights(weights)
            return add_prefix_to_loaded_weights(loaded, "dit")
        if self.model_stage == "vae":
            return set()
        raise ValueError(f"Unsupported model_stage: {self.model_stage}")
