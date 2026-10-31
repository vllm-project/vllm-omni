# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax H3 LiteVAE checkpoint loading and installation."""

from .checkpoint import (
    H3LiteDecoderCheckpoint,
    _validate_config,
    convert_h3_lite_decoder_state_dict,
    read_h3_lite_decoder_checkpoint,
    resolve_h3_lightvae_decoder_path,
)
from .native import install_h3_lite_decoder

__all__ = [
    "H3LiteDecoderCheckpoint",
    "_validate_config",
    "convert_h3_lite_decoder_state_dict",
    "read_h3_lite_decoder_checkpoint",
    "resolve_h3_lightvae_decoder_path",
    "install_h3_lite_decoder",
]
