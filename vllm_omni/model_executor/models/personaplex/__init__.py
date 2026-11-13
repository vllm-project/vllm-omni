# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vllm-omni integration for PersonaPlex (a Moshi finetune, full-duplex S2S).

Offline/batch runs go through a two-stage audio->audio pipeline (talker ->
code2wav); real-time conversation is served over the duplex WebSocket path.
"""

from vllm_omni.model_executor.models.personaplex.configuration_personaplex import (
    PersonaPlexConfig,
    PersonaPlexDepformerConfig,
    PersonaPlexMimiConfig,
)

__all__ = [
    "PersonaPlexConfig",
    "PersonaPlexDepformerConfig",
    "PersonaPlexMimiConfig",
]
