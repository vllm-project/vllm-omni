# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Configs for Sophon-Realtime checkpoints.

A Sophon-Realtime checkpoint is a directory with one subdirectory per stage::

    config.json        SophonRealtimeConfig (layout + which talker it uses)
    thinker/           DuplexJev: Qwen3-ASR audio encoder + Ultravox-style projector
                       + Qwen3 (or Qwen3-VL) language model
    talker/            CosyVoice3 (Tag) or Qwen3-TTS (Flash) talker and vocoder
    voices/            voice prompts

The thinker's audio encoder is the Qwen3-ASR encoder without its output
projection; ``Qwen3ASREncoderConfig`` describes it.
"""

from transformers import AutoConfig, PretrainedConfig


class SophonRealtimeConfig(PretrainedConfig):
    """Root config. It only describes the layout; each stage loads its own
    config and weights from its subdirectory."""

    model_type = "sophon_realtime"

    def __init__(
        self,
        thinker_subdir: str = "thinker",
        talker_subdir: str = "talker",
        voices_subdir: str = "voices",
        default_voice: str = "chelsie",
        default_language: str = "zh",
        talker_kind: str | None = None,
        **kwargs,
    ):
        self.thinker_subdir = thinker_subdir
        self.talker_subdir = talker_subdir
        self.voices_subdir = voices_subdir
        self.default_voice = default_voice
        self.default_language = default_language
        self.talker_kind = talker_kind
        super().__init__(**kwargs)


class Qwen3ASREncoderConfig(PretrainedConfig):
    """Qwen3-ASR audio encoder (conv front end + windowed transformer), as used
    by the DuplexJev thinker. Field names follow the Qwen3-ASR release."""

    model_type = "qwen3_asr_encoder_portable"

    def __init__(
        self,
        d_model: int = 896,
        downsample_hidden_size: int = 480,
        encoder_attention_heads: int = 14,
        encoder_ffn_dim: int = 3584,
        encoder_layers: int = 18,
        num_mel_bins: int = 128,
        n_window: int = 50,
        n_window_infer: int = 800,
        max_position_embeddings: int = 13,
        activation_function: str = "gelu",
        activation_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
        scale_embedding: bool = False,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.d_model = d_model
        self.downsample_hidden_size = downsample_hidden_size
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.num_mel_bins = num_mel_bins
        self.n_window = n_window
        self.n_window_infer = n_window_infer
        self.max_position_embeddings = max_position_embeddings
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.scale_embedding = scale_embedding
        self.initializer_range = initializer_range
        # Names used by vLLM's Qwen3-Omni audio encoder. DuplexJev reads the
        # encoder output before Qwen3-ASR's output projection, so output_dim
        # only sizes a layer that the thinker replaces.
        self.max_source_positions = kwargs.pop("max_source_positions", max_position_embeddings)
        self.output_dim = kwargs.pop("output_dim", d_model)
        self.conv_chunksize = kwargs.pop("conv_chunksize", 500)
        super().__init__(**kwargs)

    @property
    def hidden_size(self) -> int:
        return self.d_model

    @property
    def encoder_ds_factor(self) -> int:
        # three stride-2 convolutions: 8x from mel frames to encoder frames
        return 8


AutoConfig.register(SophonRealtimeConfig.model_type, SophonRealtimeConfig)
AutoConfig.register(Qwen3ASREncoderConfig.model_type, Qwen3ASREncoderConfig)
