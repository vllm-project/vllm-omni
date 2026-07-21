# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SenseNova-U1 HF config adapters for vLLM-Omni two-stage pipeline."""

from typing import Any, ClassVar

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig


def _build_sub_config(config: PretrainedConfig | dict[str, Any] | None) -> PretrainedConfig | None:
    if config is None or isinstance(config, PretrainedConfig):
        return config
    if not isinstance(config, dict):
        return None

    config_dict = dict(config)
    model_type = config_dict.pop("model_type", None)
    if model_type:
        try:
            return AutoConfig.for_model(model_type, **config_dict)
        except Exception:
            config_dict["model_type"] = model_type

    return PretrainedConfig.from_dict(config_dict)


class NEOChatConfig(PretrainedConfig):
    model_type = "neo_chat"
    is_composition = True
    sub_configs: ClassVar = {"llm_config": AutoConfig}

    def __init__(
        self,
        llm_config: PretrainedConfig | dict[str, Any] | None = None,
        vision_config: PretrainedConfig | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        llm_config = _build_sub_config(llm_config)
        vision_config = _build_sub_config(vision_config)
        self.llm_config = llm_config
        self.vision_config = vision_config
        super().__init__(**kwargs)
        self.llm_config = llm_config
        self.vision_config = vision_config

    def get_text_config(self, decoder: bool = False) -> PretrainedConfig | None:  # noqa: ARG002
        return getattr(self, "llm_config", None)


try:
    AutoConfig.register(NEOChatConfig.model_type, NEOChatConfig)
except ValueError:
    pass

__all__ = ["NEOChatConfig"]
