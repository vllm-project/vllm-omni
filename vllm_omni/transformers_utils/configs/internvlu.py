# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local Hugging Face configurations for InternVL-U."""

from __future__ import annotations

import os
from typing import Any, ClassVar

from transformers import AutoConfig, PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class InternVisionConfig(PretrainedConfig):
    """Configuration of the InternVL-U InternVision encoder."""

    model_type = "intern_vit_6b"
    base_config_key = "vision_config"

    def __init__(
        self,
        num_channels: int = 3,
        patch_size: int = 14,
        image_size: int = 224,
        qkv_bias: bool = False,
        hidden_size: int = 3200,
        num_attention_heads: int = 25,
        intermediate_size: int = 12800,
        qk_normalization: bool = True,
        num_hidden_layers: int = 48,
        use_flash_attn: bool = True,
        hidden_act: str = "gelu",
        norm_type: str = "rms_norm",
        layer_norm_eps: float = 1e-6,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        initializer_factor: float = 0.1,
        downsample_ratio: float = 1.0,
        fullatt_block_indexes: list[int] | None = None,
        window_size: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.qkv_bias = qkv_bias
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.qk_normalization = qk_normalization
        self.num_hidden_layers = num_hidden_layers
        self.use_flash_attn = use_flash_attn
        self.hidden_act = hidden_act
        self.norm_type = norm_type
        self.layer_norm_eps = layer_norm_eps
        self.dropout = dropout
        self.drop_path_rate = drop_path_rate
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.downsample_ratio = downsample_ratio
        self.fullatt_block_indexes = fullatt_block_indexes
        self.window_size = window_size

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        **kwargs: Any,
    ) -> PretrainedConfig:
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if "vision_config" in config_dict:
            config_dict = config_dict["vision_config"]

        model_type = config_dict.get("model_type")
        if model_type is not None and model_type != cls.model_type:
            logger.warning(
                "You are using a model of type %s to instantiate a model of type %s. This may not be supported.",
                model_type,
                cls.model_type,
            )

        return cls.from_dict(config_dict, **kwargs)


class InternVLUChatConfig(PretrainedConfig):
    """Composite InternVision + Qwen configuration released with InternVL-U."""

    model_type = "internvlu_chat"
    is_composition = True
    sub_configs: ClassVar = {
        "vision_config": InternVisionConfig,
        "llm_config": AutoConfig,
    }

    def __init__(
        self,
        vision_config: dict[str, Any] | PretrainedConfig | None = None,
        llm_config: dict[str, Any] | PretrainedConfig | None = None,
        use_backbone_lora: int = 0,
        use_llm_lora: int = 0,
        pad2square: bool = False,
        select_layer: int = -1,
        force_image_size: int | None = None,
        downsample_ratio: float = 0.5,
        template: str | None = None,
        dynamic_image_size: bool = False,
        anyres_image_size: bool = False,
        use_thumbnail: bool = False,
        thumbnail_pos: str = "start",
        enable_multi_scale: bool = False,
        scale_downsample_ratio: float = 0.5,
        ps_version: str = "v1",
        min_dynamic_patch: int = 1,
        max_dynamic_patch: int = 6,
        extra_select_layer: int = -1,
        **kwargs: Any,
    ) -> None:
        if vision_config is None:
            vision_config = {"architectures": ["InternVisionModel"]}
        if isinstance(vision_config, dict):
            vision_config = InternVisionConfig(**vision_config)

        if llm_config is None:
            llm_config = {
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
            }
        if isinstance(llm_config, dict):
            if "model_type" not in llm_config:
                raise ValueError("InternVL-U llm_config must define model_type")
            llm_config = AutoConfig.for_model(**llm_config)

        # Transformers validators can call get_text_config() from
        # PretrainedConfig.__init__, so both nested configs must exist first.
        self.vision_config = vision_config
        self.llm_config = llm_config

        kwargs.setdefault("tie_word_embeddings", llm_config.tie_word_embeddings)
        kwargs.setdefault("architectures", ["InternVLUChatModel"])
        super().__init__(**kwargs)

        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.pad2square = pad2square
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.anyres_image_size = anyres_image_size
        self.use_thumbnail = use_thumbnail
        self.thumbnail_pos = thumbnail_pos
        self.enable_multi_scale = enable_multi_scale
        self.scale_downsample_ratio = scale_downsample_ratio
        self.ps_version = ps_version
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.extra_select_layer = extra_select_layer

    @property
    def text_config(self) -> PretrainedConfig:
        return self.llm_config

    def get_text_config(self, decoder: bool = False) -> PretrainedConfig:  # noqa: ARG002
        return self.llm_config


AutoConfig.register(InternVisionConfig.model_type, InternVisionConfig)
AutoConfig.register(InternVLUChatConfig.model_type, InternVLUChatConfig)


__all__ = ["InternVisionConfig", "InternVLUChatConfig"]
