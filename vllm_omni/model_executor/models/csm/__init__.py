# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM-Omni integration for Sesame's CSM-1B (2-stage dual-AR TTS).

Stage 0 = backbone AR + inline 31-step depth decoder (LLM_AR);
Stage 1 = Mimi vocoder / code2wav (LLM_GENERATION).
"""

from vllm_omni.model_executor.models.csm.configuration_csm import CsmConfig

__all__ = ["CsmConfig"]
