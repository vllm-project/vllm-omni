# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Two-stage entry points for the LTX model family."""

from collections.abc import Iterable

import torch
from vllm_omni.diffusion.data import OmniDiffusionConfig

from .ltx2_components import (
    LTX2_DISTILLED_COMPONENT_PROFILE,
    LTX2_TWO_STAGE_COMPONENT_PROFILE,
    resolve_ltx_artifact,
)
from .ltx2_components import (
    get_ltx2_post_process_func as get_ltx2_post_process_func,  # noqa: F401
)
from .ltx2_lora import LTXResidentLoRAController
from .ltx2_recipes import (
    LTX2_DISTILLED_TWO_STAGE_RECIPE,
    LTX2_TWO_STAGE_RECIPE,
    LTXPhaseRecipe,
)
from .pipeline_ltx2 import LTX2Pipeline


class LTX2TwoStagePipeline(LTX2Pipeline):
    """Regular checkpoint with low-resolution generation and LoRA refinement."""

    pipeline_kind = "two_stage"
    component_profile = LTX2_TWO_STAGE_COMPONENT_PROFILE
    pipeline_recipe = LTX2_TWO_STAGE_RECIPE
    supports_request_batch = False
    support_image_input = False

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        profile = self.component_profile
        if getattr(od_config, "lora_path", None) is not None:
            raise ValueError(
                f"{self.__class__.__name__} reserves LoRA execution for its stage-2 distilled adapter; "
                "request or static LoRA composition is not supported yet."
            )
        if profile.artifact_repo_id is None or profile.distilled_lora_filename is None:
            raise ValueError(f"{profile.name} does not declare a stage-2 adapter.")
        adapter_path = resolve_ltx_artifact(
            od_config.model,
            profile.artifact_repo_id,
            profile.distilled_lora_filename,
        )
        self._phase_lora_controller = LTXResidentLoRAController(self, adapter_path)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = self._phase_lora_controller.merge_stage2_weights(weights)
        return super().load_weights(weights)

    def _enter_phase(self, phase: LTXPhaseRecipe) -> None:
        self._phase_lora_controller.enter(phase.transformer_phase)


class LTX2DistilledPipeline(LTX2Pipeline):
    """Unified LTX-2 distilled two-stage T2V/I2V entry."""

    pipeline_kind = "distilled_two_stage"
    component_profile = LTX2_DISTILLED_COMPONENT_PROFILE
    pipeline_recipe = LTX2_DISTILLED_TWO_STAGE_RECIPE
    supports_request_batch = False
    support_image_input = True
    unified_text_image_entry = True

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
