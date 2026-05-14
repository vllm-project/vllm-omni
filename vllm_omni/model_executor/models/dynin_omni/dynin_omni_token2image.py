from __future__ import annotations

import os
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .dynin_omni import DyninOmniStageBase
from .dynin_omni_common import (
    DetokTarget,
    _to_bool,
    coerce_token_ids_1d,
    get_dynin_magvit_attr,
    normalize_runtime_info,
    resolve_dynin_infer_sources,
    resolve_hidden_size,
    unwrap_first_value,
)

logger = init_logger(__name__)


class DyninOmniToken2Image(DyninOmniStageBase):
    """Stage-2: token detokenization to image (or pass-through)."""

    stage_name = "Dynin token2image"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        del prefix
        super().__init__()

        self.vllm_config = vllm_config
        self.have_multimodal_outputs = True
        self.requires_raw_input_tokens = True
        self.hidden_size = resolve_hidden_size(vllm_config=vllm_config)
        self._vq_model = None
        self._vq_model_path: str | None = None
        self._vq_local_files_only: bool | None = None

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        del positions, intermediate_tensors, inputs_embeds
        if input_ids is None:
            raise ValueError("token2image stage requires input_ids")
        runtime_info = normalize_runtime_info(kwargs.get("runtime_additional_information"))
        detok_id = int(unwrap_first_value(runtime_info.get("detok_id"), 0))
        tokens = coerce_token_ids_1d(input_ids)

        if detok_id != DetokTarget.IMAGE:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "token_ids": tokens,
                    "detok_id": torch.tensor([detok_id], dtype=torch.long, device=tokens.device),
                },
            )

        image = self._decode_image_tokens(tokens, runtime_info=runtime_info)
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "image": image,
                "detok_id": torch.tensor([detok_id], dtype=torch.long, device=image.device),
            },
        )

    def _decode_image_tokens(self, tokens: torch.Tensor, runtime_info: dict[str, Any]) -> torch.Tensor:
        # Follow DYNIN validation path:
        #   tokens -> clamp -> vq_model.decode_code -> (x+1)/2 -> [0,1].
        vq_model = self._ensure_vq_model(runtime_info=runtime_info, ref_device=tokens.device)
        codebook_size = int(unwrap_first_value(runtime_info.get("codebook_size"), 8192))
        image_vocab_offset = unwrap_first_value(runtime_info.get("image_vocab_offset"), None)
        if image_vocab_offset is None:
            text_vocab_size = unwrap_first_value(runtime_info.get("text_vocab_size"), None)
            num_new_special_tokens = int(unwrap_first_value(runtime_info.get("num_new_special_tokens"), 0))
            if text_vocab_size is not None:
                image_vocab_offset = int(text_vocab_size) + num_new_special_tokens

        token_ids = tokens.to(torch.long)
        if image_vocab_offset is not None:
            off = int(image_vocab_offset)
            token_ids = torch.where(token_ids >= off, token_ids - off, token_ids)
        token_ids = torch.clamp(token_ids, min=0, max=max(0, codebook_size - 1))
        token_ids = token_ids.unsqueeze(0)

        decoded = vq_model.decode_code(token_ids)
        decoded = torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0)
        if decoded.ndim != 4 or decoded.shape[0] == 0:
            raise RuntimeError(f"Unexpected MAGVIT decode output shape: {tuple(decoded.shape)}")
        return decoded[0].contiguous()

    def _ensure_vq_model(self, runtime_info: dict[str, Any], ref_device: torch.device) -> Any:
        sources = resolve_dynin_infer_sources(vllm_config=self.vllm_config, runtime_info=runtime_info)
        model_path = str(sources.vq_image_source)
        local_files_only = bool(sources.vq_image_local_files_only)
        if self._vq_model is None or self._vq_model_path != model_path or self._vq_local_files_only != local_files_only:
            disable_xet = unwrap_first_value(
                runtime_info.get("hf_hub_disable_xet"),
                unwrap_first_value(runtime_info.get("disable_hf_xet"), True),
            )
            if _to_bool(disable_xet, default=True):
                os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
            logger.info(
                "Loading DYNIN image detokenizer from %s (local_files_only=%s)",
                model_path,
                local_files_only,
            )
            try:
                MAGVITv2 = get_dynin_magvit_attr(
                    "MAGVITv2",
                    source=model_path,
                    local_files_only=local_files_only,
                )
                try:
                    self._vq_model = MAGVITv2.from_pretrained(
                        model_path,
                        local_files_only=local_files_only,
                    )
                except TypeError:
                    self._vq_model = MAGVITv2.from_pretrained(model_path)
            except Exception as e:
                raise RuntimeError(
                    "Failed to load MAGVITv2 from local DYNIN submodel implementation "
                    f"for model path '{model_path}'. "
                    "If your environment cannot access huggingface.co, set "
                    "additional_information.vq_model_image_path to a local MAGVITv2 directory "
                    "and set additional_information.vq_model_image_local_files_only=true."
                ) from e
            self._vq_model.eval()
            self._vq_model.requires_grad_(False)
            self._vq_model_path = model_path
            self._vq_local_files_only = local_files_only
        if hasattr(self._vq_model, "to"):
            self._vq_model = self._vq_model.to(ref_device)
        return self._vq_model

    def embed_multimodal(self, **kwargs: Any) -> Any:
        del kwargs
        return None
