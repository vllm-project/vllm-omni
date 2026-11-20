# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Kyutai AURA: speech-driven streaming talking-head video."""

from .config import AURA_1_3B, KyutaiAuraConfig, resolve_kyutai_aura_config
from .pipeline_kyutai_aura import KyutaiAuraPipeline, get_kyutai_aura_post_process_func
from .transformer import KyutaiAuraTransformer

__all__ = [
    "AURA_1_3B",
    "KyutaiAuraConfig",
    "KyutaiAuraPipeline",
    "KyutaiAuraTransformer",
    "get_kyutai_aura_post_process_func",
    "resolve_kyutai_aura_config",
]
