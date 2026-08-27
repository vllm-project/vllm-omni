# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Diffusion Host Weight Runtime producers."""

from .final_layout_bf16 import (
    FINAL_LAYOUT_BF16_MANIFEST_SCHEMA,
    FINAL_LAYOUT_BF16_POLICY,
    FINAL_LAYOUT_BF16_PRODUCER_ID,
    FINAL_LAYOUT_BF16_REPRESENTATION,
    FINAL_LAYOUT_BF16_SPEC,
    FINAL_LAYOUT_BF16_VERSION,
    FinalLayoutBF16Policy,
    FinalLayoutBF16Producer,
)
from .final_layout_fp8 import (
    DEFAULT_FP8_QUANT_CHUNK_BYTES,
    DEFAULT_FP8_SHARD_SIZE_BYTES,
    FinalLayoutFP8Producer,
)

__all__ = [
    "FINAL_LAYOUT_BF16_MANIFEST_SCHEMA",
    "FINAL_LAYOUT_BF16_POLICY",
    "FINAL_LAYOUT_BF16_PRODUCER_ID",
    "FINAL_LAYOUT_BF16_REPRESENTATION",
    "FINAL_LAYOUT_BF16_SPEC",
    "FINAL_LAYOUT_BF16_VERSION",
    "FinalLayoutBF16Policy",
    "FinalLayoutBF16Producer",
    "DEFAULT_FP8_QUANT_CHUNK_BYTES",
    "DEFAULT_FP8_SHARD_SIZE_BYTES",
    "FinalLayoutFP8Producer",
]
