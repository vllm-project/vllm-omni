# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Load-time registration hook for the DRVEXO config."""

from transformers import AutoConfig

from .drvexo_speech_8b import DRVEXOSpeech8BConfig


def register_drvexo_speech_8b() -> None:
    try:
        AutoConfig.register(DRVEXOSpeech8BConfig.model_type, DRVEXOSpeech8BConfig)
    except ValueError as exc:
        if "already registered" not in str(exc):
            raise
