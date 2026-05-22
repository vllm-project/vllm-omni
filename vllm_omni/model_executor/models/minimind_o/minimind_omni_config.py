# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from transformers import AutoConfig, PretrainedConfig


@dataclass
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 8,
        use_moe: bool = False,
        dropout: float = 0.0,
        vocab_size: int = 6400,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        flash_attn: bool = True,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        head_dim: int | None = None,
        hidden_act: str = "silu",
        intermediate_size: int | None = None,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1e6,
        tie_word_embeddings: bool = True,
        inference_rope_scaling: bool = False,
        rope_scaling: dict[str, Any] | None = None,
        num_experts: int = 4,
        num_experts_per_tok: int = 1,
        moe_intermediate_size: int | None = None,
        norm_topk_prob: bool = True,
        router_aux_loss_coef: float = 5e-4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.flash_attn = flash_attn
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim or self.hidden_size // self.num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size or math.ceil(hidden_size * math.pi / 64) * 64
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings
        self.inference_rope_scaling = inference_rope_scaling
        self.rope_scaling = (
            rope_scaling
            if rope_scaling is not None
            else {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

        # MoE specific configs (ignored if use_moe is False).
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size or self.intermediate_size
        self.norm_topk_prob = norm_topk_prob
        self.router_aux_loss_coef = router_aux_loss_coef

    def get_text_config(self, **kwargs: Any) -> MiniMindConfig:
        return self


class MiniMindOmniVisionConfig(PretrainedConfig):
    model_type = "minimind-o-vision"

    def __init__(
        self,
        hidden_size: int = 768,
        image_ids: list[int] | None = None,
        image_special_token: str = "<|image_pad|>",
        image_hidden_size: int = 768,
        image_token_len: int = 64,
        vision_model_path: str = "jingyaogong/siglip2-base-p32-256-ve",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.image_ids = image_ids or [12]
        self.image_special_token = image_special_token
        self.image_hidden_size = image_hidden_size
        self.image_token_len = image_token_len
        self.vision_model_path = vision_model_path


class MiniMindOmniAudioConfig(PretrainedConfig):
    model_type = "minimind-o-audio"

    def __init__(
        self,
        hidden_size: int = 768,
        audio_ids: list[int] | None = None,
        audio_special_token: str = "<|audio_pad|>",
        audio_hidden_size: int = 512,
        audio_encoder_path: str = "jingyaogong/SenseVoiceSmall",
        audio_vocab_size: int = 2112,
        audio_pad_token: int = 2049,
        audio_stop_token: int = 2050,
        audio_spk_token: int = 2051,
        audio_sample_rate: int = 16000,
        audio_target_channels: int = 1,
        max_audio_tokens: int = 3000,
        spk_emb_size: int = 192,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.audio_ids = audio_ids or [16]
        self.audio_special_token = audio_special_token
        self.audio_hidden_size = audio_hidden_size
        self.audio_encoder_path = audio_encoder_path
        self.audio_vocab_size = audio_vocab_size
        self.audio_pad_token = audio_pad_token
        self.audio_stop_token = audio_stop_token
        self.audio_spk_token = audio_spk_token
        self.audio_sample_rate = audio_sample_rate
        self.audio_target_channels = audio_target_channels
        self.max_audio_tokens = max_audio_tokens
        self.spk_emb_size = spk_emb_size


class MiniMindOmniCode2WavConfig(PretrainedConfig):
    model_type = "minimind-o-code2wav"

    def __init__(
        self,
        mimi_path: str | None = "jingyaogong/mimi",
        codec_num_code_layers: int = 8,
        codec_sample_rate: int = 24000,
        codec_pad_token: int = 2049,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.mimi_path = mimi_path
        self.codec_num_code_layers = codec_num_code_layers
        self.codec_sample_rate = codec_sample_rate
        self.codec_pad_token = codec_pad_token


class MiniMindOmniTalkerConfig(MiniMindConfig):
    model_type = "minimind-o-talker"

    def __init__(
        self,
        num_code_layers: int = 8,
        text_hidden_size: int = 768,
        audio_vocab_size: int = 2112,
        audio_pad_token: int = 2049,
        audio_stop_token: int = 2050,
        audio_spk_token: int = 2051,
        spk_emb_size: int = 192,
        **kwargs: Any,
    ) -> None:
        vocab_size = kwargs.pop("vocab_size", audio_vocab_size)
        eos_token_id = kwargs.pop("eos_token_id", audio_stop_token)
        tie_word_embeddings = kwargs.pop("tie_word_embeddings", False)
        super().__init__(
            vocab_size=vocab_size,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.num_code_layers = num_code_layers
        self.text_hidden_size = text_hidden_size
        self.audio_vocab_size = audio_vocab_size
        self.audio_pad_token = audio_pad_token
        self.audio_stop_token = audio_stop_token
        self.audio_spk_token = audio_spk_token
        self.spk_emb_size = spk_emb_size


class MiniMindOmniConfig(PretrainedConfig):
    model_type = "minimind-o"
    sub_configs = {
        "text_config": MiniMindConfig,
        "talker_config": MiniMindOmniTalkerConfig,
        "vision_config": MiniMindOmniVisionConfig,
        "audio_config": MiniMindOmniAudioConfig,
        "code2wav_config": MiniMindOmniCode2WavConfig,
    }

    def __init__(
        self,
        text_config: dict[str, Any] | PretrainedConfig | None = None,
        talker_config: dict[str, Any] | PretrainedConfig | None = None,
        vision_config: dict[str, Any] | PretrainedConfig | None = None,
        audio_config: dict[str, Any] | PretrainedConfig | None = None,
        code2wav_config: dict[str, Any] | PretrainedConfig | None = None,
        hidden_size: int = 768,
        num_hidden_layers: int = 8,
        use_moe: bool = False,
        dropout: float = 0.0,
        vocab_size: int = 6400,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        flash_attn: bool = True,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        head_dim: int | None = None,
        hidden_act: str = "silu",
        intermediate_size: int | None = None,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1e6,
        tie_word_embeddings: bool = True,
        inference_rope_scaling: bool = False,
        rope_scaling: dict[str, Any] | None = None,
        num_experts: int = 4,
        num_experts_per_tok: int = 1,
        moe_intermediate_size: int | None = None,
        norm_topk_prob: bool = True,
        router_aux_loss_coef: float = 5e-4,
        num_talker_hidden_layers: int = 4,
        talker_hidden_size: int = 768,
        audio_ids: list[int] | None = None,
        audio_special_token: str = "<|audio_pad|>",
        audio_hidden_size: int = 512,
        audio_encoder_path: str = "jingyaogong/SenseVoiceSmall",
        audio_vocab_size: int = 2112,
        audio_pad_token: int = 2049,
        audio_stop_token: int = 2050,
        audio_spk_token: int = 2051,
        audio_sample_rate: int = 16000,
        audio_target_channels: int = 1,
        max_audio_tokens: int = 3000,
        spk_emb_size: int = 192,
        think_end_ids: list[int] | None = None,
        image_ids: list[int] | None = None,
        image_special_token: str = "<|image_pad|>",
        image_hidden_size: int = 768,
        image_token_len: int = 64,
        vision_model_path: str = "jingyaogong/siglip2-base-p32-256-ve",
        bridge_layer: int | None = None,
        **kwargs: Any,
    ) -> None:
        text_defaults = {
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "use_moe": use_moe,
            "dropout": dropout,
            "vocab_size": vocab_size,
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
            "flash_attn": flash_attn,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "head_dim": head_dim,
            "hidden_act": hidden_act,
            "intermediate_size": intermediate_size,
            "max_position_embeddings": max_position_embeddings,
            "rms_norm_eps": rms_norm_eps,
            "rope_theta": rope_theta,
            "tie_word_embeddings": tie_word_embeddings,
            "inference_rope_scaling": inference_rope_scaling,
            "rope_scaling": rope_scaling,
            "num_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "moe_intermediate_size": moe_intermediate_size,
            "norm_topk_prob": norm_topk_prob,
            "router_aux_loss_coef": router_aux_loss_coef,
        }
        if isinstance(text_config, dict):
            text_config = MiniMindConfig(**{**text_defaults, **text_config})
        elif text_config is None:
            text_config = MiniMindConfig(**text_defaults)
        self.text_config = text_config

        super().__init__(**kwargs)

        # init vision config
        vision_defaults = {
            "hidden_size": hidden_size,
            "image_ids": image_ids,
            "image_special_token": image_special_token,
            "image_hidden_size": image_hidden_size,
            "image_token_len": image_token_len,
            "vision_model_path": vision_model_path,
        }
        if isinstance(vision_config, dict):
            vision_config = MiniMindOmniVisionConfig(**{**vision_defaults, **vision_config})
        elif vision_config is None:
            vision_config = MiniMindOmniVisionConfig(**vision_defaults)
        self.vision_config = vision_config

        # init audio config
        audio_defaults = {
            "hidden_size": hidden_size,
            "audio_ids": audio_ids,
            "audio_special_token": audio_special_token,
            "audio_hidden_size": audio_hidden_size,
            "audio_encoder_path": audio_encoder_path,
            "audio_vocab_size": audio_vocab_size,
            "audio_pad_token": audio_pad_token,
            "audio_stop_token": audio_stop_token,
            "audio_spk_token": audio_spk_token,
            "audio_sample_rate": audio_sample_rate,
            "audio_target_channels": audio_target_channels,
            "max_audio_tokens": max_audio_tokens,
            "spk_emb_size": spk_emb_size,
        }
        if isinstance(audio_config, dict):
            audio_config = MiniMindOmniAudioConfig(**{**audio_defaults, **audio_config})
        elif audio_config is None:
            audio_config = MiniMindOmniAudioConfig(**audio_defaults)
        self.audio_config = audio_config

        # init code2wav config
        if isinstance(code2wav_config, dict):
            code2wav_config = MiniMindOmniCode2WavConfig(**code2wav_config)
        elif code2wav_config is None:
            code2wav_config = MiniMindOmniCode2WavConfig()
        self.code2wav_config = code2wav_config

        talker_defaults = {
            "hidden_size": talker_hidden_size,
            "num_hidden_layers": num_talker_hidden_layers,
            "use_moe": self.text_config.use_moe,
            "dropout": self.text_config.dropout,
            "vocab_size": self.text_config.vocab_size,
            "bos_token_id": self.text_config.bos_token_id,
            "eos_token_id": self.audio_config.audio_stop_token,
            "flash_attn": self.text_config.flash_attn,
            "num_attention_heads": self.text_config.num_attention_heads,
            "num_key_value_heads": self.text_config.num_key_value_heads,
            "head_dim": None,
            "hidden_act": self.text_config.hidden_act,
            "intermediate_size": None,
            "max_position_embeddings": self.text_config.max_position_embeddings,
            "rms_norm_eps": self.text_config.rms_norm_eps,
            "rope_theta": self.text_config.rope_theta,
            "tie_word_embeddings": False,
            "inference_rope_scaling": self.text_config.inference_rope_scaling,
            "rope_scaling": self.text_config.rope_scaling,
            "num_experts": self.text_config.num_experts,
            "num_experts_per_tok": self.text_config.num_experts_per_tok,
            "moe_intermediate_size": None,
            "norm_topk_prob": self.text_config.norm_topk_prob,
            "router_aux_loss_coef": self.text_config.router_aux_loss_coef,
            "num_code_layers": self.code2wav_config.codec_num_code_layers,
            "text_hidden_size": self.text_config.hidden_size,
            "audio_vocab_size": self.audio_config.audio_vocab_size,
            "audio_pad_token": self.audio_config.audio_pad_token,
            "audio_stop_token": self.audio_config.audio_stop_token,
            "audio_spk_token": self.audio_config.audio_spk_token,
            "spk_emb_size": self.audio_config.spk_emb_size,
        }
        if isinstance(talker_config, dict):
            talker_config = MiniMindOmniTalkerConfig(**{**talker_defaults, **talker_config})
        elif talker_config is None:
            talker_config = MiniMindOmniTalkerConfig(**talker_defaults)
        self.talker_config = talker_config

        self.num_talker_hidden_layers = self.talker_config.num_hidden_layers
        self.talker_hidden_size = self.talker_config.hidden_size
        self.think_end_ids = think_end_ids or [26, 234, 234]  # </think>\n\n

        self.audio_ids = self.audio_config.audio_ids
        self.audio_special_token = self.audio_config.audio_special_token
        self.audio_hidden_size = self.audio_config.audio_hidden_size
        self.audio_encoder_path = self.audio_config.audio_encoder_path
        self.audio_vocab_size = self.audio_config.audio_vocab_size
        self.audio_pad_token = self.audio_config.audio_pad_token
        self.audio_stop_token = self.audio_config.audio_stop_token
        self.audio_spk_token = self.audio_config.audio_spk_token
        self.audio_sample_rate = self.audio_config.audio_sample_rate
        self.audio_target_channels = self.audio_config.audio_target_channels
        self.max_audio_tokens = self.audio_config.max_audio_tokens
        self.spk_emb_size = self.audio_config.spk_emb_size

        self.mimi_path = self.code2wav_config.mimi_path
        self.codec_num_code_layers = self.code2wav_config.codec_num_code_layers
        self.codec_sample_rate = self.code2wav_config.codec_sample_rate
        self.codec_pad_token = self.code2wav_config.codec_pad_token

        self.image_ids = self.vision_config.image_ids
        self.image_special_token = self.vision_config.image_special_token
        self.image_hidden_size = self.vision_config.image_hidden_size
        self.image_token_len = self.vision_config.image_token_len
        self.vision_model_path = self.vision_config.vision_model_path
        self.bridge_layer = (
            bridge_layer
            if bridge_layer is not None
            else int(getattr(self.text_config, "num_hidden_layers", num_hidden_layers)) // 2 - 1
        )

    def get_text_config(self, **kwargs: Any) -> PretrainedConfig:
        return self.text_config


try:
    AutoConfig.register(MiniMindConfig.model_type, MiniMindConfig)
    AutoConfig.register(MiniMindOmniTalkerConfig.model_type, MiniMindOmniTalkerConfig)
    AutoConfig.register(MiniMindOmniCode2WavConfig.model_type, MiniMindOmniCode2WavConfig)
    AutoConfig.register(MiniMindOmniConfig.model_type, MiniMindOmniConfig)
except ValueError:
    pass

__all__ = [
    "MiniMindConfig",
    "MiniMindOmniAudioConfig",
    "MiniMindOmniCode2WavConfig",
    "MiniMindOmniConfig",
    "MiniMindOmniTalkerConfig",
    "MiniMindOmniVisionConfig",
]
