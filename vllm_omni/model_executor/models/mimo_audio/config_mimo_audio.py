# Copyright 2025 Xiaomi Corporation.
import copy
from dataclasses import dataclass

from transformers import PretrainedConfig, Qwen2Config

SPAN_CODEC_START_TOKEN_ID = 151670
SPAN_CODEC_END_TOKEN_ID = 151672
TALKER_CODEC_PAD_TOKEN_ID = 151667
TEXT_GROUP_SIZE = 5
PAD_GROUP_SIZE = 5
NO_INTERLEAVE_NEXT_TOKEN_ID = 151671


@dataclass
class MiMoAudioConfig(Qwen2Config):
    def __init__(
        self,
        *,
        speech_vocab_size: str | int = "1025-1025-129-129-129-129-129-129",
        speech_zeroemb_idx: str | int = "1024-1024-128-128-128-128-128-128",
        delay_pattern: str = "0-1-2-3-4-5-6-7",
        head_dim: int = 128,
        group_size: int = 4,
        audio_channels: int = 8,
        local_dim: int = 1024,
        local_layers: int = 16,
        local_attn_heads: int = 64,
        local_ffn_dim: int = 4096,
        local_attn_dropout: float = 0.1,
        input_local_layers: int = 6,
        input_local_dim: int | None = None,
        input_full_attention: bool | None = None,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.speech_vocab_size = speech_vocab_size
        self.speech_zeroemb_idx = speech_zeroemb_idx
        self.delay_pattern = delay_pattern

        self.head_dim = head_dim

        self.group_size = group_size
        self.audio_channels = audio_channels

        self.local_dim = local_dim
        self.local_layers = local_layers
        self.local_attn_heads = local_attn_heads
        self.local_ffn_dim = local_ffn_dim
        self.local_attn_dropout = local_attn_dropout

        self.input_local_layers = input_local_layers
        self.input_local_dim = input_local_dim or local_dim

        self.input_full_attention = input_full_attention

    def _parse_maybe_list(self, value: str | int, length: int) -> list[int]:
        if isinstance(value, str) and "-" in value:
            return [int(s) for s in value.split("-")]
        return [int(value)] * length

    def parsed_speech_empty_ids(self):
        return self._parse_maybe_list(self.speech_zeroemb_idx, self.audio_channels)

    def parsed_speech_vocab_sizes(self):
        return self._parse_maybe_list(self.speech_vocab_size, self.audio_channels)

    def parsed_delay_pattern(self):
        return self._parse_maybe_list(self.delay_pattern, self.audio_channels)

    def local_config(self):
        config = copy.deepcopy(self)

        config.hidden_size = self.local_dim
        config.num_hidden_layers = self.local_layers
        config.num_attention_heads = self.local_attn_heads
        config.num_key_value_heads = self.local_attn_heads
        config.head_dim = config.hidden_size // self.local_attn_heads
        config.intermediate_size = self.local_ffn_dim
        config.attention_dropout = self.local_attn_dropout

        return config

    def input_local_config(self):
        config = copy.deepcopy(self)

        config.hidden_size = self.input_local_dim
        config.num_hidden_layers = self.input_local_layers
        config.num_attention_heads = self.local_attn_heads
        config.num_key_value_heads = self.local_attn_heads
        config.head_dim = config.hidden_size // self.local_attn_heads
        config.intermediate_size = config.hidden_size * 4
        config.attention_dropout = self.local_attn_dropout

        return config

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"speech_vocab_size={self.speech_vocab_size!r}, "
            f"speech_zeroemb_idx={self.speech_zeroemb_idx!r}, "
            f"delay_pattern={self.delay_pattern!r}, "
            f"head_dim={self.head_dim}, "
            f"group_size={self.group_size}, "
            f"audio_channels={self.audio_channels}, "
            f"local_dim={self.local_dim}, "
            f"local_layers={self.local_layers}, "
            f"local_attn_heads={self.local_attn_heads}, "
            f"local_ffn_dim={self.local_ffn_dim}, "
            f"local_attn_dropout={self.local_attn_dropout}, "
            f"input_local_layers={self.input_local_layers}, "
            f"input_local_dim={self.input_local_dim}, "
            f"input_full_attention={self.input_full_attention})"
        )


class MiMoAudioTokenizerConfig(PretrainedConfig):
    model_type = "mimo_audio_tokenizer"

    def __init__(
        self,
        max_audio_seconds: int = 1800,
        stride_size: int = 2,
        avg_pooler: int = 1,
        d_model: int = 768,
        scale_embedding: bool = True,
        kernel_size: int = 3,
        activation_function: str = "gelu",
        encoder_layers: int = 8,
        encoder_skip_layer_id: int = None,
        encoder_attention_heads: int = 12,
        encoder_ffn_dim: int = 3072,
        encoder_causal: bool = False,
        encoder_attn_window_size: list[int] = None,
        decoder_layers: int = 8,
        decoder_attention_heads: int = 12,
        decoder_ffn_dim: int = 3072,
        decoder_kernel_size: int = 3,
        decoder_stride_size: int = 2,
        decoder_causal: bool = True,
        decoder_attn_window_size: list[int] = None,
        nfft: int = 1024,
        vocoder_dim: int = 512,
        vocoder_intermediate_dim: int = 4096,
        vocoder_num_layers: int = 30,
        n_mels: int = 80,
        sampling_rate: int = 24000,
        hop_length: int = 240,
        window_size: int = 1024,
        vocoder_padding: str = "same",
        fmin: int = 0,
        fmax: int = None,
        num_quantizers: int = 12,
        codebook_size: list[int] = None,
        threshold_ema_dead_code: int = 10,
        position_embedding_type: str = "rope",
        rope_theta: int = 10000,
        rope_type: str = "default",
        ln_type: str = "LayerNorm",
        vocoder_attention_heads: int = 4,
        vocoder_attn_window_size: list[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_audio_seconds = max_audio_seconds
        self.stride_size = stride_size
        self.avg_pooler = avg_pooler
        self.d_model = d_model
        self.scale_embedding = scale_embedding
        self.kernel_size = kernel_size
        self.activation_function = activation_function
        self.encoder_layers = encoder_layers
        self.encoder_skip_layer_id = encoder_skip_layer_id
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_causal = encoder_causal
        self.encoder_attn_window_size = encoder_attn_window_size if encoder_attn_window_size is not None else [-1, -1]
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_kernel_size = decoder_kernel_size
        self.decoder_stride_size = decoder_stride_size
        self.decoder_causal = decoder_causal
        self.decoder_attn_window_size = decoder_attn_window_size if decoder_attn_window_size is not None else [-1, -1]
        self.nfft = nfft
        self.vocoder_dim = vocoder_dim
        self.vocoder_intermediate_dim = vocoder_intermediate_dim
        self.vocoder_num_layers = vocoder_num_layers
        self.n_mels = n_mels
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.window_size = window_size
        self.vocoder_padding = vocoder_padding
        self.fmin = fmin
        self.fmax = fmax
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size if codebook_size is not None else [1024]
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.position_embedding_type = position_embedding_type
        self.rope_theta = rope_theta
        self.rope_type = rope_type
        self.ln_type = ln_type
        self.vocoder_attention_heads = vocoder_attention_heads
        self.vocoder_attn_window_size = vocoder_attn_window_size if vocoder_attn_window_size is not None else [40, 10]
