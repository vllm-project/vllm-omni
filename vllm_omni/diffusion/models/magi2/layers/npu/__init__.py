# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os

from ..native.rope import apply_rotary_emb
from .attention_ops import varlen_attention_with_sink
from .mh_moe import Magi2MultiHeadMoE
from .normalization import MultiModalityRMSNorm

if os.environ.get("MAGI2_USE_NPU_ROTARY_MUL", "1") != "0":
    from .rope import apply_rotary_emb

__all__ = ["Magi2MultiHeadMoE", "MultiModalityRMSNorm", "apply_rotary_emb", "varlen_attention_with_sink"]
