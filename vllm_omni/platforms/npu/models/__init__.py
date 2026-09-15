# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

_VOXCPM2_TALKER_ARCH = "VoxCPM2TalkerForConditionalGeneration"


def apply_post_load_model_patches(model: object, model_config: Any) -> None:
    """Apply model-specific Ascend setup after weights are loaded."""
    if getattr(model_config, "model_arch", None) != _VOXCPM2_TALKER_ARCH:
        return

    from vllm_omni.platforms.npu.models.voxcpm2_talker import (
        setup_voxcpm2_loc_dit_npu_graph,
    )

    setup_voxcpm2_loc_dit_npu_graph(model)
