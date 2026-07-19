# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Alpamayo VLA (Vision-Language-Action) model package for vLLM-Omni.

Port of NVIDIA's Alpamayo-R1 / Alpamayo-1.5 autonomous-driving VLA models.
See ``feature_list.md`` at the repo root for the integration plan.
"""

from vllm_omni.model_executor.models.alpamayo.configuration_alpamayo import (
    Alpamayo15Config,
    AlpamayoR1Config,
)

__all__ = ["Alpamayo15Config", "AlpamayoR1Config"]
