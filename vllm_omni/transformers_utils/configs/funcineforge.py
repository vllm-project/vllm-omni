# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FunCineForge config registration shim.

Re-exports ``FunCineForgeConfig`` from the model package and registers it
with ``AutoConfig`` so that models with ``model_type = "funcineforge"`` in
their ``config.json`` (or injected via ``hf_overrides``) are resolved
automatically.
"""

from transformers import AutoConfig

from vllm_omni.model_executor.models.funcineforge.config import FunCineForgeConfig

__all__ = ["FunCineForgeConfig"]

AutoConfig.register("funcineforge", FunCineForgeConfig)
