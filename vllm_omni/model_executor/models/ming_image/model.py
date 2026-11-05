# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""vLLM-native MLLM stage for inclusionAI Ming-Image."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VisionTransformer
from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.parse import MultiModalDataParser

from vllm_omni.model_executor.models.ming_flash_omni.ming_flash_omni_thinker import (
    MingFlashOmniThinkerDummyInputsBuilder,
    MingFlashOmniThinkerForConditionalGeneration,
    MingFlashOmniThinkerMultiModalProcessor,
    MingFlashOmniThinkerProcessingInfo,
)
from vllm_omni.model_executor.models.ming_flash_omni.modeling_bailing_moe_v2 import (
    BailingMoeV2ForCausalLM,
)
from vllm_omni.model_executor.models.ming_flash_omni.projectors import VisionProjector
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.transformers_utils.configs.ming_flash_omni import BailingMM2Config
from vllm_omni.transformers_utils.processors.ming import MingImageProcessor


class _MingImageDataParser(MultiModalDataParser):
    def parse_mm_data(self, mm_data):
        normalized = dict(mm_data)
        if "img2img" in normalized:
            references = normalized.pop("img2img")
            references = references if isinstance(references, list) else [references]
            existing = normalized.get("image", [])
            existing = existing if isinstance(existing, list) else [existing]
            normalized["image"] = existing + references
        return super().parse_mm_data(normalized)


class MingImageProcessingInfo(MingFlashOmniThinkerProcessingInfo):
    def get_hf_config(self) -> BailingMM2Config:
        return self.ctx.get_hf_config(BailingMM2Config)

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(MingImageProcessor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "img2img": 1}

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        counts = dict(mm_counts or {})
        counts["image"] = counts.get("image", 0) + counts.pop("img2img", 0)
        return super(MingFlashOmniThinkerProcessingInfo, self).get_mm_max_tokens_per_item(
            seq_len=seq_len,
            mm_counts=counts,
        )

    def get_data_parser(self):
        return _MingImageDataParser()


class MingImageDummyInputsBuilder(MingFlashOmniThinkerDummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        count = mm_counts.get("image", 0) + mm_counts.get("img2img", 0)
        return self.info.get_hf_processor().image_token * count

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options=None,
    ) -> MultiModalDataDict:
        count = mm_counts.get("image", 0) + mm_counts.get("img2img", 0)
        return {
            "image": self._get_dummy_images(
                width=448,
                height=448,
                num_images=count,
            )
        }


class MingImageMultiModalProcessor(MingFlashOmniThinkerMultiModalProcessor):
    def apply(self, inputs, timing_ctx):
        modalities = list(inputs.hf_processor_mm_kwargs.get("modalities") or [])
        is_image_generation = "image" in modalities or "img2img" in modalities
        if is_image_generation:
            tokenizer = self.info.get_tokenizer()
            if isinstance(inputs.prompt, str):
                prompt_text = inputs.prompt
            else:
                prompt_text = tokenizer.decode(
                    inputs.prompt,
                    skip_special_tokens=False,
                )

            processor = self.info.get_hf_processor()
            # expects to place <IMAGE> inside the HUMAN message.
            formatted_prompt = processor._apply_image_generation_template(
                prompt_text,
                has_reference_image="img2img" in modalities,
            )
            prompt_ids = tokenizer.encode(
                formatted_prompt,
                add_special_tokens=False,
            )

            # Normalize img2img so the shared parent processor does't prepend a second placeholder outside that message
            processor_kwargs = dict(inputs.hf_processor_mm_kwargs)
            processor_kwargs["modalities"] = ["image" if modality == "img2img" else modality for modality in modalities]
            inputs = replace(
                inputs,
                prompt=prompt_ids,
                hf_processor_mm_kwargs=processor_kwargs,
            )

        return super().apply(inputs, timing_ctx)


@MULTIMODAL_REGISTRY.register_processor(
    MingImageMultiModalProcessor,
    info=MingImageProcessingInfo,
    dummy_inputs=MingImageDummyInputsBuilder,
)
class MingImageForConditionalGeneration(MingFlashOmniThinkerForConditionalGeneration):
    """Ming-Image MLLM stage with Qwen2.5-VL and direct-state capture."""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.": "language_model.",
            "linear_proj.": "linear_proj.proj.",
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        torch.nn.Module.__init__(self)
        thinker_config = vllm_config.model_config.hf_config
        if not isinstance(thinker_config, BailingMM2Config):
            raise TypeError(f"Expected BailingMM2Config, got {type(thinker_config).__name__}")
        if thinker_config.llm_config is None or thinker_config.vision_config is None:
            raise ValueError("Ming-Image requires llm_config and vision_config.")

        llm_config = thinker_config.llm_config
        self.config = llm_config
        self.thinker_config = thinker_config
        self.have_multimodal_outputs = True

        with self._mark_language_model(vllm_config):
            self.language_model = BailingMoeV2ForCausalLM(
                vllm_config=vllm_config.with_hf_config(llm_config),
                prefix=maybe_prefix(prefix, "llm"),
            )

        with self._mark_tower_model(vllm_config, "image"):
            self.vision = Qwen2_5_VisionTransformer(
                thinker_config.vision_config,
                norm_eps=llm_config.rms_norm_eps,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "vision"),
            )
            self.linear_proj = VisionProjector(
                vision_dim=thinker_config.vision_config.out_hidden_size,
                llm_dim=llm_config.hidden_size,
                mlp_depth=getattr(thinker_config, "mlp_depth", 2),
            )

        self.audio = None
        self.linear_proj_audio = None
        self.query_tokens_dict = torch.nn.ParameterDict()
        component_path = Path(vllm_config.model_config.model)
        model_root = component_path.parent if component_path.name == "mllm" else component_path
        self._load_image_gen_query_tokens(str(model_root))
        if not self.query_tokens_dict:
            raise ValueError(f"Ming-Image query tokens were not found under {model_root / 'mlp'}")

        self.capture_layers = (5, 12, llm_config.num_hidden_layers)
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    def extract_image_feature(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(pixel_values.device.type, dtype=torch.bfloat16):
            image_embeds = self.vision(pixel_values, grid_thw=grid_thw.tolist())
        return F.normalize(self.linear_proj(image_embeds), dim=-1)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor | None:
        # vLLM v1 calls this with hidden states only; the reused Bailing model
        # still accepts the legacy metadata argument (which it does not use).
        return self.language_model.compute_logits(hidden_states, sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_weights = super().load_weights(weights)
        # Query tokens live in the root-level mlp checkpoint and are loaded by
        # ``_load_image_gen_query_tokens`` during construction, outside vLLM's
        # main checkpoint iterator. Include them in the returned set so strict
        # post-load validation accounts for that checkpoint-owned component.
        loaded_weights.update(f"query_tokens_dict.{scale}" for scale in self.query_tokens_dict)
        return loaded_weights

    def _compute_modality_masks(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids is None:
            if inputs_embeds is None:
                raise ValueError("input_ids and inputs_embeds cannot both be None")
            empty_mask = torch.zeros(
                inputs_embeds.shape[:-1],
                device=inputs_embeds.device,
                dtype=torch.bool,
            )
            return empty_mask, empty_mask
        return input_ids == self.config.image_patch_token, torch.zeros_like(input_ids, dtype=torch.bool)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> OmniOutput:
        image_mask, audio_mask = self._compute_modality_masks(input_ids, inputs_embeds)
        result = self.language_model.forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            image_mask=image_mask,
            audio_mask=audio_mask,
            capture_layers=self.capture_layers,
        )
        if not isinstance(result, tuple):
            return OmniOutput(text_hidden_states=result, multimodal_outputs={"final_hidden_states": result})

        hidden_states, captured = result
        multimodal_outputs: dict[str, Any] = {"final_hidden_states": hidden_states}
        for layer_idx, layer_hidden in captured.items():
            multimodal_outputs[f"hidden_states_{layer_idx}"] = layer_hidden
        return OmniOutput(
            text_hidden_states=hidden_states,
            multimodal_outputs=multimodal_outputs,
        )


__all__ = ["MingImageForConditionalGeneration"]
