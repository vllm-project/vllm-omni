"""
Autoregressive (AR) stage scaffold for MammothModa2 in vLLM-Omni.

This class will wrap the HF Mammothmoda2 LLM component (Qwen3-VL based)
to produce AR tokens / hidden states that will be consumed by the DiT stage.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig
from vllm.config import VllmConfig
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix

from vllm_omni.model_executor.models.output_templates import OmniOutput


class MammothModa2ARForConditionalGeneration(nn.Module):
    """
    Autoregressive阶段封装：调用 Mammothmoda2Model 的 llm_model 进行文本/多模态 token 生成，
    返回隐状态供采样和后续 DiT 阶段使用。

    多模态处理沿用底层 Qwen2.5-VL 逻辑
    """

    have_multimodal_outputs = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.prefix = prefix
        self.model_stage = "ar"

        mc = vllm_config.model_config
        trust_remote_code = getattr(mc, "trust_remote_code", True)

        # 优先使用上层已加载好的 hf_config，避免重复下载；否则再回退拉取
        hf_combined = getattr(mc, "hf_config", None)
        if hf_combined is None:
            hf_combined = AutoConfig.from_pretrained(
                mc.model, trust_remote_code=trust_remote_code, revision=mc.revision
            )

        llm_config = getattr(hf_combined, "llm_config", None)
        if llm_config is None:
            raise ValueError("MammothModa2 config must contain llm_config for AR stage.")

        llm_vllm_config = vllm_config.with_hf_config(
            llm_config, architectures=["Qwen2_5_VLForConditionalGeneration"]
        )
        self.model = init_vllm_registered_model(
            vllm_config=llm_vllm_config,
            prefix=maybe_prefix(prefix, "ar"),
            hf_config=llm_config,
            architectures=["Qwen2_5_VLForConditionalGeneration"],
        )
        self.model.eval()

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> OmniOutput:
        # 单条也自动补 batch 维；其余情况保持原状
        if input_ids is not None and input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if positions is not None and positions.ndim == 1:
            positions = positions.unsqueeze(0)

        # 直接透传给 vLLM 模型；如果已是 OmniOutput 则直接返回
        outputs = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if isinstance(outputs, OmniOutput):
            return outputs

        # 兼容兜底：提取 text_hidden_states
        if hasattr(outputs, "text_hidden_states"):
            hidden = outputs.text_hidden_states
            mm_out = getattr(outputs, "multimodal_outputs", None)
        elif isinstance(outputs, torch.Tensor):
            hidden = outputs
            mm_out = None
        elif isinstance(outputs, (list, tuple)) and outputs and isinstance(outputs[0], torch.Tensor):
            hidden = outputs[0]
            mm_out = None
        else:
            raise RuntimeError(f"Unsupported output type from AR model: {type(outputs)}")

        return OmniOutput(text_hidden_states=hidden, multimodal_outputs=mm_out)

    @torch.inference_mode()
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "compute_logits"):
            logits = self.model.compute_logits(hidden_states)
            return logits.float()
        raise NotImplementedError("compute_logits not available for MammothModa2 AR model.")
