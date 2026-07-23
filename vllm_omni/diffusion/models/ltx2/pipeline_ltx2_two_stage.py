# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Two-stage entry points for the LTX model family."""

from collections.abc import Iterable

import torch
from vllm_omni.diffusion.data import OmniDiffusionConfig

from .ltx2_adapter_parser import LTXAdapterParser
from .ltx2_components import (
    LTX2_DISTILLED_COMPONENT_PROFILE,
    LTX2_TWO_STAGE_COMPONENT_PROFILE,
    resolve_ltx_artifact,
)
from .ltx2_components import (
    get_ltx2_post_process_func as get_ltx2_post_process_func,  # noqa: F401
)
from .ltx2_lora import LTXResidentLoRAController
from .ltx2_phase_adapter import LTXPhaseAdapterRuntime
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
        self._phase_adapter: LTXResidentLoRAController | LTXPhaseAdapterRuntime | None = None
        self._resident_lora_controller: LTXResidentLoRAController | None = None
        profile = self.component_profile
        if getattr(od_config, "lora_path", None) is not None:
            raise ValueError(
                f"{self.__class__.__name__} reserves LoRA execution for its stage-2 distilled adapter; "
                "request or static LoRA composition is not supported yet."
            )
        if profile.artifact_repo_id is None or profile.distilled_lora_filename is None:
            raise ValueError(f"{profile.name} does not declare a stage-2 adapter.")
        model_config = getattr(od_config, "model_config", {}) or {}
        lora_mode = model_config.get("ltx_two_stage_lora_mode", "resident")
        if lora_mode not in {"resident", "dynamic"}:
            raise ValueError(
                "model_config.ltx_two_stage_lora_mode must be either 'resident' or 'dynamic', "
                f"got {lora_mode!r}."
            )
        model_paths = getattr(od_config, "model_paths", {}) or {}
        adapter_path = resolve_ltx_artifact(
            od_config.model,
            profile.artifact_repo_id,
            profile.distilled_lora_filename,
            explicit_path=model_paths.get("distilled_lora"),
        )
        if lora_mode == "resident":
            self._resident_lora_controller = LTXResidentLoRAController(self, adapter_path)
            self._phase_adapter = self._resident_lora_controller
        else:
            manifest = LTXAdapterParser(self.transformer).parse(adapter_path, name="ltx_distilled")
            self._phase_adapter = LTXPhaseAdapterRuntime(self.transformer, manifest, dtype=od_config.dtype)
            self._phase_adapter.install_structure()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if self._resident_lora_controller is not None:
            weights = self._resident_lora_controller.merge_stage2_weights(weights)
        return super().load_weights(weights)

    def eval(self):
        """Materialize dynamic adapter buffers after loading the base weights."""
        result = super().eval()
        if isinstance(self._phase_adapter, LTXPhaseAdapterRuntime):
            self._phase_adapter.finalize_adapter_data()
        return result

    def _enter_phase(self, phase: LTXPhaseRecipe) -> None:
        adapter = self._phase_adapter
        if adapter is None:
            raise RuntimeError(f"Transformer phase {phase.transformer_phase!r} requires a stage adapter controller.")
        if phase.transformer_phase == "base":
            adapter.set_active(None)
        elif phase.transformer_phase == "distilled_lora":
            adapter.set_active("ltx_distilled")
        else:
            raise ValueError(f"Unsupported LTX Transformer phase: {phase.transformer_phase!r}.")


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
