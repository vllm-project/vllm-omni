# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MOSS-TTS model configuration."""

from __future__ import annotations

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.qwen3 import Qwen3Config


class MossTTSDelayConfig(PretrainedConfig):
    """Config for MossTTSDelayModel (8B: MOSS-TTS, MOSS-TTSD, MOSS-SoundEffect;
    1.7B: MOSS-VoiceGenerator).

    The HuggingFace checkpoint stores a nested ``language_config`` which is a
    Qwen3 config.  We unwrap it here so that vLLM can use ``get_text_config()``
    to size KV caches and allocate memory correctly.
    """

    model_type = "moss_tts_delay"

    def __init__(
        self,
        language_config: dict | None = None,
        n_vq: int = 32,
        audio_vocab_size: int = 1024,
        sampling_rate: int = 24000,
        audio_pad_code: int = 1024,
        audio_start_token_id: int = 151652,
        audio_end_token_id: int = 151653,
        audio_user_slot_token_id: int = 151654,
        audio_assistant_gen_slot_token_id: int = 151656,
        audio_assistant_delay_slot_token_id: int = 151662,
        pad_token_id: int = 151643,
        im_start_token_id: int = 151644,
        im_end_token_id: int = 151645,
        codec_model_name_or_path: str = "OpenMOSS-Team/MOSS-Audio-Tokenizer",
        **kwargs: object,
    ) -> None:
        if language_config is None:
            language_config = {}
        if isinstance(language_config, dict):
            language_config.pop("model_type", None)
            self.language_config = Qwen3Config(**language_config)
        else:
            self.language_config = language_config

        super().__init__(**kwargs)

        self.n_vq = n_vq
        self.audio_vocab_size = audio_vocab_size
        self.sampling_rate = sampling_rate
        self.audio_pad_code = audio_pad_code
        self.audio_start_token_id = audio_start_token_id
        self.audio_end_token_id = audio_end_token_id
        self.audio_user_slot_token_id = audio_user_slot_token_id
        self.audio_assistant_gen_slot_token_id = audio_assistant_gen_slot_token_id
        self.audio_assistant_delay_slot_token_id = audio_assistant_delay_slot_token_id
        self.pad_token_id = pad_token_id
        self.im_start_token_id = im_start_token_id
        self.im_end_token_id = im_end_token_id
        self.codec_model_name_or_path = codec_model_name_or_path

    def get_text_config(self, **_: object) -> Qwen3Config:
        return self.language_config


class MossTTSLocalTransformerConfig(PretrainedConfig):
    """Config for the lightweight local depth transformer in MossTTSRealtime.

    Mirrors upstream ``MossTTSRealtimeLocalTransformerConfig`` so the realtime
    checkpoint loads via ``AutoConfig`` without dropping fields silently.
    """

    model_type = "moss_tts_realtime_local_transformer"

    def __init__(
        self,
        head_dim: int = 128,
        hidden_size: int = 2048,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        intermediate_size: int = 6144,
        max_position_embeddings: int = 33,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1_000_000.0,
        rope_type: str = "linear",
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        pad_token_id: int = 1024,
        audio_pad_token: int = 1024,
        audio_vocab_size: int = 1027,
        rvq: int = 16,
        **kwargs: object,
    ) -> None:
        super().__init__(pad_token_id=pad_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_type = rope_type
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.audio_pad_token = audio_pad_token
        self.audio_vocab_size = audio_vocab_size
        self.rvq = rvq
        # The shared ``CodePredictorBaseModel`` (reused from Qwen3-TTS /
        # Qwen3-Omni) reads ``num_code_groups`` and ``vocab_size`` to size its
        # codebook embeddings and causal mask. MossTTS names the same
        # quantities ``rvq`` and ``audio_vocab_size``; mirror them so the
        # common transformer body can be driven from this config unchanged.
        self.num_code_groups = rvq
        self.vocab_size = audio_vocab_size
        # rope_parameters used by transformers.modeling_rope_utils
        self.rope_parameters = {
            "rope_type": rope_type,
            "rope_theta": rope_theta,
            "factor": 1.0,
        }


class MossTTSRealtimeConfig(PretrainedConfig):
    """Config for MossTTSRealtime (1.7B, TTFB ~180 ms streaming model).

    Mirrors the upstream layout: a nested ``language_config`` (Qwen3) plus a
    ``local_config`` for the per-step depth transformer that decodes the RVQ
    codebooks. The flat fields here are the *outer* model's attributes.
    """

    model_type = "moss_tts_realtime"

    def __init__(
        self,
        language_config: dict | None = None,
        local_config: dict | None = None,
        rvq: int = 16,
        audio_vocab_size: int = 1027,
        audio_pad_token: int = 1024,
        sampling_rate: int = 24000,
        reference_audio_pad: int = 151654,
        text_pad: int = 151655,
        vocab_size: int = 151936,
        bos_token_id: int = 151643,
        eos_token_id: int = 151645,
        initializer_range: float = 0.02,
        codec_model_name_or_path: str = "OpenMOSS-Team/MOSS-Audio-Tokenizer",
        **kwargs: object,
    ) -> None:
        if language_config is None:
            language_config = {}
        if isinstance(language_config, dict):
            language_config.pop("model_type", None)
            self.language_config = Qwen3Config(**language_config)
        else:
            self.language_config = language_config

        if local_config is None:
            local_config = {}
        if isinstance(local_config, dict):
            local_config.pop("model_type", None)
            self.local_config = MossTTSLocalTransformerConfig(**local_config)
        else:
            self.local_config = local_config

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.rvq = rvq
        self.n_vq = rvq  # alias for uniform access from talker code
        self.audio_vocab_size = audio_vocab_size
        self.audio_pad_token = audio_pad_token
        self.sampling_rate = sampling_rate
        self.reference_audio_pad = reference_audio_pad
        self.text_pad = text_pad
        self.vocab_size = vocab_size
        self.initializer_range = initializer_range
        self.codec_model_name_or_path = codec_model_name_or_path

    # The talker still reads ``hidden_size`` etc. directly from the config in
    # places — proxy them through to ``language_config`` for backward compat.
    @property
    def hidden_size(self) -> int:
        return int(self.language_config.hidden_size)

    @property
    def num_hidden_layers(self) -> int:
        return int(self.language_config.num_hidden_layers)

    @property
    def num_attention_heads(self) -> int:
        return int(self.language_config.num_attention_heads)

    @property
    def intermediate_size(self) -> int:
        return int(self.language_config.intermediate_size)

    @property
    def max_position_embeddings(self) -> int:
        return int(self.language_config.max_position_embeddings)

    @property
    def rms_norm_eps(self) -> float:
        return float(self.language_config.rms_norm_eps)

    @property
    def rope_theta(self) -> float:
        return float(self.language_config.rope_theta)

    # vLLM sizes KV cache from get_text_config(); return the inner Qwen3 cfg.
    def get_text_config(self, **_: object) -> Qwen3Config:
        return self.language_config


class MossTTSLocalConfig(PretrainedConfig):
    """Config for MossTTSLocalModel (MOSS-TTS-Local-Transformer-v1.5).

    Like ``MossTTSDelayConfig``, the HF checkpoint stores a nested Qwen3 config
    (``qwen3_config``) which we unwrap via ``get_text_config()``. It also stores
    a nested GPT2-style config (``gpt2_config``) describing the single-layer
    local depth transformer that decodes the ``n_vq`` audio codebooks per frame
    (see ``modeling_moss_tts_local_depth.py``).
    """

    model_type = "moss_tts_local"

    def __init__(
        self,
        qwen3_config: dict | None = None,
        gpt2_config: dict | None = None,
        n_vq: int = 12,
        audio_vocab_size: int = 1024,
        audio_pad_token_id: int = 1024,
        pad_token_id: int = 151643,
        im_start_token_id: int = 151644,
        im_end_token_id: int = 151645,
        audio_start_token_id: int = 151669,
        audio_end_token_id: int = 151670,
        audio_user_slot_token_id: int = 151654,
        audio_assistant_slot_token_id: int = 151656,
        sampling_rate: int = 48000,
        audio_tokenizer_name_or_path: str = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2",
        local_text_head_mode: str = "binary",
        **kwargs: object,
    ) -> None:
        if qwen3_config is None:
            qwen3_config = {}
        if isinstance(qwen3_config, dict):
            qwen3_config = dict(qwen3_config)
            qwen3_config.pop("model_type", None)
            self.qwen3_config = Qwen3Config(**qwen3_config)
        else:
            self.qwen3_config = qwen3_config

        if isinstance(gpt2_config, dict):
            gpt2_config = dict(gpt2_config)
            gpt2_config.pop("model_type", None)
            self.gpt2_config = GPT2Config(**gpt2_config)
        else:
            self.gpt2_config = gpt2_config

        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.n_vq = n_vq
        self.audio_vocab_size = audio_vocab_size
        self.audio_pad_token_id = audio_pad_token_id
        self.audio_pad_code = audio_pad_token_id
        self.im_start_token_id = im_start_token_id
        self.im_end_token_id = im_end_token_id
        self.audio_start_token_id = audio_start_token_id
        self.audio_end_token_id = audio_end_token_id
        self.audio_user_slot_token_id = audio_user_slot_token_id
        self.audio_assistant_slot_token_id = audio_assistant_slot_token_id
        self.sampling_rate = sampling_rate
        self.audio_tokenizer_name_or_path = audio_tokenizer_name_or_path
        self.local_text_head_mode = local_text_head_mode

    def get_text_config(self, **_: object) -> Qwen3Config:
        return self.qwen3_config


AutoConfig.register("moss_tts_delay", MossTTSDelayConfig)
AutoConfig.register("moss_tts_realtime", MossTTSRealtimeConfig)
AutoConfig.register("moss_tts_local", MossTTSLocalConfig)

__all__ = [
    "MossTTSDelayConfig",
    "MossTTSRealtimeConfig",
    "MossTTSLocalTransformerConfig",
    "MossTTSLocalConfig",
]
