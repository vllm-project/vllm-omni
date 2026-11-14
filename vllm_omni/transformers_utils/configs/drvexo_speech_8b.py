# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Transformers config for the DRVEXO-speech-8b Omni model.

The top-level config deliberately carries the component contracts.  This is
needed by an Omni loader because stage 0, stage 1 and stage 2 are constructed
from one model directory and must not rely on a developer's absolute paths.
"""

from __future__ import annotations

from transformers import PretrainedConfig, Qwen3Config


class DRVEXOSpeech8BConfig(Qwen3Config):
    model_type = "drvexo_speech_8b"

    def __init__(self, **kwargs):
        # The Thinker reply is the exact text sent to this pipeline's Talker.
        self.combine_text_audio_response = kwargs.pop("combine_text_audio_response", True)
        self.audio_encoder_path = kwargs.pop("audio_encoder_path", None)
        self.audio_encoder_config = kwargs.pop("audio_encoder_config", None)
        self.audio_feature_dim = int(kwargs.pop("audio_feature_dim", 2048))
        self.audio_pad_token_id = kwargs.pop("audio_pad_token_id", None)
        self.tts_model_path = kwargs.pop("tts_model_path", None)
        self.connector_checkpoint = kwargs.pop("connector_checkpoint", None)
        self.enable_audio_output = kwargs.pop("enable_audio_output", True)
        self.speech_bridge_layers = tuple(kwargs.pop("speech_bridge_layers", (7, 15, 23, 27)))
        talker_config = kwargs.pop("talker_config", None)
        code2wav_config = kwargs.pop("code2wav_config", None)
        self.talker_config = PretrainedConfig(**talker_config) if isinstance(talker_config, dict) else talker_config
        self.code2wav_config = (
            PretrainedConfig(**code2wav_config) if isinstance(code2wav_config, dict) else code2wav_config
        )
        super().__init__(**kwargs)


__all__ = ["DRVEXOSpeech8BConfig"]
