# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""One-stage entry points for the LTX model family."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.dmd2 import DMD2PipelineMixin

from .ltx2_components import LTX2_COMPONENT_PROFILE, LTX2_DISTILLED_ONE_STAGE_COMPONENT_PROFILE
from .ltx2_components import (
    get_ltx2_post_process_func as get_ltx2_post_process_func,  # noqa: F401
)
from .ltx2_conditioning import LTXI2VConditioningMixin
from .ltx2_recipes import LTX2_DISTILLED_ONE_STAGE_RECIPE, LTX2_ONE_STAGE_RECIPE, LTX_POSITIVE_ONLY_RECIPE
from .ltx2_runtime import LTXRuntime

if TYPE_CHECKING:
    from vllm.lora.lora_model import LoRAModel


class LTX2Pipeline(LTXI2VConditioningMixin, LTXRuntime):
    """LTX-2 family one-stage entry, configured from checkpoint metadata."""

    pipeline_kind = "one_stage"
    unified_text_image_entry = True
    component_profile = LTX2_COMPONENT_PROFILE
    pipeline_recipe = LTX2_ONE_STAGE_RECIPE
    _dit_modules: ClassVar[list[str]] = list(component_profile.dit_modules)
    _encoder_modules: ClassVar[list[str]] = list(component_profile.encoder_modules)
    _vae_modules: ClassVar[list[str]] = list(component_profile.vae_modules)
    _resident_modules: ClassVar[list[str]] = list(component_profile.resident_modules)
    supports_request_batch = True

    @property
    def supports_weight_reload(self) -> bool:
        """Admit the audited encoder path only for merged adapter sessions."""
        config = getattr(self, "od_config", None)
        transfer = getattr(config, "weight_transfer_config", None)
        backend = transfer.get("backend") if isinstance(transfer, dict) else getattr(transfer, "backend", None)
        return (
            getattr(self, "model_version", None) == "2.3"
            and self.pipeline_kind == "one_stage"
            and getattr(config, "lora_application", "runtime") == "merged"
            and backend == "omni_lora_ipc"
            and getattr(self, "_ltx23_text_encoder_lora_prepared", False) is True
        )

    def merged_lora_spec(self):
        """Keep the merged policy inside the Gemma language-model subtree."""
        from vllm_omni.lora.capability import MergedLoRACapabilityError, MergedLoRASpec

        from .ltx2_lora import ltx23_text_lora_aliases

        if (
            getattr(self, "model_version", None) != "2.3"
            or self.pipeline_kind != "one_stage"
            or getattr(self.od_config, "lora_application", "runtime") != "merged"
            or getattr(self, "_ltx23_text_encoder_lora_prepared", False) is not True
        ):
            raise MergedLoRACapabilityError(
                "LTX merged text-encoder LoRA requires a prepared LTX2.3 one-stage pipeline"
            )
        return MergedLoRASpec(
            components=("text_encoder.model.language_model",),
            name_aliases=ltx23_text_lora_aliases,
            validate_binding=self._validate_text_encoder_lora_binding,
        )

    def _validate_text_encoder_lora_binding(
        self,
        *,
        lora_model: LoRAModel,
        bound_lora_names: frozenset[str],
    ) -> None:
        # A bare suffix can otherwise broadcast one factor pair to every
        # matching decoder projection through the generic manager fallback.
        # The encoder slice accepts full HF or component-qualified names only.
        for name in bound_lora_names:
            if not name.removeprefix("text_encoder.").startswith("model.language_model.layers."):
                raise ValueError(f"LTX text LoRA requires a qualified language-model projection: {name}")


class LTX2DistilledOneStagePipeline(LTX2Pipeline):
    """Merged-distilled checkpoint at its native one-stage resolution."""

    pipeline_kind = "distilled_one_stage"
    component_profile = LTX2_DISTILLED_ONE_STAGE_COMPONENT_PROFILE
    pipeline_recipe = LTX2_DISTILLED_ONE_STAGE_RECIPE


class LTX2T2VDMD2Pipeline(DMD2PipelineMixin, LTX2Pipeline):
    """LTX2 T2V entry for FastGen DMD2-distilled models."""

    pipeline_kind = "dmd2"
    support_image_input = False
    pipeline_recipe = LTX_POSITIVE_ONLY_RECIPE

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.__init_dmd2__()


class LTX2I2VDMD2Pipeline(DMD2PipelineMixin, LTX2Pipeline):
    """LTX2 I2V entry for FastGen DMD2-distilled models."""

    pipeline_kind = "dmd2"
    unified_text_image_entry = False
    pipeline_recipe = LTX_POSITIVE_ONLY_RECIPE

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.__init_dmd2__()
