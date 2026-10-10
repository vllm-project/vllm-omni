# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Register Lychee-FD checkpoint configs with Hugging Face AutoConfig."""

from transformers import AutoConfig

from vllm_omni.model_executor.models.lychee_fd.configuration_lychee import (
    LycheeAudioEncoderConfig,
    LycheeFDConfig,
)

for _model_type, _config_cls in (
    (LycheeFDConfig.model_type, LycheeFDConfig),
    (LycheeAudioEncoderConfig.model_type, LycheeAudioEncoderConfig),
):
    try:
        AutoConfig.register(_model_type, _config_cls)
    except ValueError:
        pass

__all__ = ["LycheeAudioEncoderConfig", "LycheeFDConfig"]
