# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Vendored Audex causal speech decoder (Apache-2.0).

Verbatim copy of the ``audex_causal_speech_decoder`` remote-code package from
nvidia/Nemotron-Labs-Audex-2B. Vendored because transformers' dynamic module
loader resolves HuggingFace-cache blob symlinks to their real paths, which
breaks the package's relative imports when loading with trust_remote_code
directly from a hub snapshot.
"""

from .configuration_audex_causal_speech_decoder import AudexCausalSpeechDecoderConfig
from .modeling_audex_causal_speech_decoder import (
    AudexCausalSpeechDecoderModel,
    AudexCausalSpeechDecoderSession,
)

__all__ = [
    "AudexCausalSpeechDecoderConfig",
    "AudexCausalSpeechDecoderModel",
    "AudexCausalSpeechDecoderSession",
]
