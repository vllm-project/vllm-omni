# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Registry of mid-way interaction handlers during a diffusion generation.
A pipeline only declares the modalities that it supports.
And a pipeline can choose which handler for each modality (there can be more than one handler for a modality).

Outer key: pipeline architecture name (``od_config.model_class_name``).
Inner key: modality name (e.g. ``\"prompt\"``).
"""

from __future__ import annotations

from vllm_omni.diffusion.interaction.modality_handlers.base import InteractionHandler
from vllm_omni.diffusion.interaction.modality_handlers.prompt import PromptInteractionHandler

STRUCTURED_HANDLER_REGISTRY: dict[str, dict[str, type[InteractionHandler]]] = {
    # Pipeline class name -> modality -> handler class.
    "HeliosPipeline": {
        "prompt": PromptInteractionHandler,
    },
    "HeliosPyramidPipeline": {
        "prompt": PromptInteractionHandler,
    },
}
