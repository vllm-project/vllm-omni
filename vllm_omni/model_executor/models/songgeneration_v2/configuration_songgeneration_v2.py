# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Configuration classes for SongGeneration v2.

These classes preserve the upstream config.yaml fields in a HF-style config
so the AR LeLM stage and Flow1dVAE renderer stage can be loaded independently
by vLLM-Omni.
"""

from __future__ import annotations

from typing import Any, ClassVar

from transformers.configuration_utils import PretrainedConfig


class SongGenerationV2LeLMConfig(PretrainedConfig):
    """LeLM autoregressive token generator config.

    Defaults match the upstream ``songgeneration_v2_large/config.yaml``.
    The model emits 3 parallel token streams: mixed/song, vocal, and bgm.

    Two-transformer layout:
    - ``num_layers`` is the first transformer (stream 0: mixed/song).
    - ``num_layers_sub`` is the second transformer (streams 1+2: vocal, bgm).

    Conditioner naming caveat: the YAML key ``description`` actually receives
    the lyrics text, and ``type_info`` receives the style description.
    """

    model_type = "songgeneration_v2_lelm"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        lm_type: str = "Llama",
        dim: int = 2048,
        intermediate_size: int = 11008,
        num_heads: int = 16,
        num_key_value_heads: int | None = None,
        num_layers: int = 36,
        num_layers_sub: int = 12,
        code_depth: int = 3,
        code_size: int = 16384,
        max_position_embeddings: int = 10000,
        max_position_embeddings_sub: int = 10000,
        rope_theta: float = 500000.0,
        rope_theta_sub: float = 500000.0,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
        norm: str = "layer_norm",
        rms_norm_eps: float = 1e-5,
        use_flash_attn_2: bool = True,
        kv_repeat: int = 1,
        codebooks_pattern: dict[str, Any] | None = None,
        classifier_free_guidance: dict[str, Any] | None = None,
        attribute_dropout: dict[str, Any] | None = None,
        conditioners: dict[str, Any] | None = None,
        fuser: dict[str, Any] | None = None,
        prompt_len: int = 10,
        audio_tokenizer_frame_rate: int = 25,
        sample_rate: int = 48000,
        eos_token_id: int | None = None,
        special_token_id: int | None = None,
        null_token_id: int | None = None,
        generated_stream_names: list[str] | tuple[str, ...] | None = None,
        decoder_stream_indices: list[int] | tuple[int, ...] | None = None,
        reference_generation_params: dict[str, Any] | None = None,
        top_k_per_stream: list[int] | tuple[int, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        eos_token_id = code_size if eos_token_id is None else eos_token_id
        special_token_id = code_size + 1 if special_token_id is None else special_token_id
        null_token_id = special_token_id if null_token_id is None else null_token_id
        kwargs.setdefault("eos_token_id", eos_token_id)
        kwargs.setdefault("pad_token_id", special_token_id)
        super().__init__(**kwargs)

        self.lm_type = lm_type
        self.dim = dim
        self.hidden_size = dim
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.num_attention_heads = num_heads
        if num_key_value_heads is None:
            num_key_value_heads = max(1, num_heads // max(1, kv_repeat))
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = dim // num_heads
        self.num_layers = num_layers
        self.num_hidden_layers = num_layers
        self.num_layers_sub = num_layers_sub
        self.code_depth = code_depth
        self.code_size = code_size
        self.vocab_size = code_size + 1
        self.input_vocab_size = code_size + 2
        self.max_position_embeddings = max_position_embeddings
        self.max_position_embeddings_sub = max_position_embeddings_sub
        self.rope_theta = rope_theta
        self.rope_theta_sub = rope_theta_sub
        self.dropout = dropout
        self.activation = activation
        self.hidden_act = activation
        self.norm_first = norm_first
        self.norm = norm
        self.rms_norm_eps = rms_norm_eps
        self.use_flash_attn_2 = use_flash_attn_2
        self.kv_repeat = kv_repeat

        self.prompt_len = prompt_len
        self.audio_tokenizer_frame_rate = audio_tokenizer_frame_rate
        self.prompt_frames = prompt_len * audio_tokenizer_frame_rate
        self.sample_rate = sample_rate
        self.hop_length = sample_rate // audio_tokenizer_frame_rate

        self.eos_token_id = eos_token_id
        self.special_token_id = special_token_id
        self.null_token_id = null_token_id
        self.generated_stream_names = list(generated_stream_names or ["mixed", "vocal", "bgm"])
        self.decoder_stream_indices = list(decoder_stream_indices or [1, 2])
        self.reference_generation_params = reference_generation_params or {
            "use_sampling": True,
            "temperature": 0.8,
            "top_k": 5000,
            "top_p": 0.0,
            "cfg_coef": 1.5,
            "seed": 42,
        }
        # Per-stream top_k: stream 0 (mixed/song) samples with top_k=5000;
        # streams 1 (vocal) and 2 (bgm) are greedy (top_k=1) in the reference
        # ``codeclm/models/lm_levo.py`` AR loop. Stage 0 must apply these
        # per-stream when sampling — using a single global top_k will sample
        # the dual-track streams stochastically and produce wrong tokens.
        self.top_k_per_stream = list(top_k_per_stream or [5000, 1, 1])

        self.codebooks_pattern = codebooks_pattern or {
            "modeling": "delay",
            "delay": {
                "delays": [0, self.prompt_frames, self.prompt_frames],
                "flatten_first": 0,
                "empty_initial": 0,
            },
        }
        self.classifier_free_guidance = classifier_free_guidance or {
            "training_dropout": 0.15,
            "inference_coef": 1.5,
        }
        self.attribute_dropout = attribute_dropout or {
            "args": {
                "active_on_eval": False,
            },
            "text": {
                "description": 0.0,
                "type_info": 0.2,
            },
            "audio": {
                "prompt_audio": 0.5,
            },
        }
        self.fuser = fuser or {
            "sum": [],
            "prepend": ["description", "prompt_audio", "type_info"],
        }
        self.conditioners = conditioners or {
            "prompt_audio": {
                "model": "qt_embedding",
                "qt_embedding": {
                    "code_size": code_size,
                    "code_depth": code_depth,
                    "max_len": self.prompt_frames + 2,
                },
            },
            "description": {
                "model": "QwTokenizer",
                "QwTokenizer": {
                    "token_path": "third_party/Qwen2-7B",
                    "max_len": 600,
                },
            },
            "type_info": {
                "model": "QwTextTokenizer",
                "QwTextTokenizer": {
                    "token_path": "third_party/Qwen2-7B",
                    "max_len": 100,
                },
            },
        }

    def get_text_config(self, **kwargs: Any) -> SongGenerationV2LeLMConfig:
        return self


class SongGenerationV2Flow1dVAESeparateConfig(PretrainedConfig):
    """Flow1dVAE separate vocal/bgm renderer config."""

    model_type = "songgeneration_v2_flow1dvae"

    def __init__(
        self,
        audio_tokenizer_checkpoint_sep: str = "Flow1dVAESeparate_./ckpt/model_septoken/model_2.safetensors",
        vae_config: str = "./ckpt/vae/stable_audio_1920_vae.json",
        vae_model: str = "./ckpt/vae/autoencoder_music_1320k.ckpt",
        sample_rate: int = 48000,
        audio_tokenizer_frame_rate_sep: int = 25,
        audio_tokenizer_code_depth_sep: int = 2,
        code_size: int = 16384,
        prompt_len: int = 10,
        guidance_scale: float = 1.5,
        num_steps: int = 10,
        decode_window_frames: int = 1000,
        decode_hop_frames: int = 750,
        decode_overlap_frames: int = 250,
        output_channels: int = 2,
        decoder_stream_indices: list[int] | tuple[int, ...] | None = None,
        vocal_replacement_token_id: int = 3142,
        bgm_replacement_token_id: int = 9670,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.audio_tokenizer_checkpoint_sep = audio_tokenizer_checkpoint_sep
        self.vae_config = vae_config
        self.vae_model = vae_model
        self.sample_rate = sample_rate
        self.audio_tokenizer_frame_rate_sep = audio_tokenizer_frame_rate_sep
        self.frame_rate = audio_tokenizer_frame_rate_sep
        self.audio_tokenizer_code_depth_sep = audio_tokenizer_code_depth_sep
        self.code_size = code_size
        self.prompt_len = prompt_len
        self.prompt_frames = prompt_len * audio_tokenizer_frame_rate_sep
        self.hop_length = sample_rate // audio_tokenizer_frame_rate_sep
        self.guidance_scale = guidance_scale
        self.num_steps = num_steps
        self.decode_window_frames = decode_window_frames
        self.decode_hop_frames = decode_hop_frames
        self.decode_overlap_frames = decode_overlap_frames
        self.output_channels = output_channels
        self.decoder_stream_indices = list(decoder_stream_indices or [1, 2])
        self.vocal_replacement_token_id = vocal_replacement_token_id
        self.bgm_replacement_token_id = bgm_replacement_token_id

    def get_text_config(self, **kwargs: Any) -> SongGenerationV2Flow1dVAESeparateConfig:
        return self


class SongGenerationV2Config(PretrainedConfig):
    """Top-level SongGeneration v2 composition config."""

    model_type = "songgeneration_v2"
    is_composition = True
    sub_configs: ClassVar = {
        "lelm_config": SongGenerationV2LeLMConfig,
        "decoder_config": SongGenerationV2Flow1dVAESeparateConfig,
    }

    def __init__(
        self,
        lelm_config: dict[str, Any] | SongGenerationV2LeLMConfig | None = None,
        decoder_config: dict[str, Any] | SongGenerationV2Flow1dVAESeparateConfig | None = None,
        max_dur: int = 270,
        min_dur: int = 30,
        prompt_len: int = 10,
        pad_to_max: bool = True,
        audio_tokenizer_checkpoint: str = "Flow1dVAE1rvq_./ckpt/model_1rvq/model_2_fixed.safetensors",
        audio_tokenizer_frame_rate: int = 25,
        audio_tokenizer_code_depth: int = 1,
        sample_rate: int = 48000,
        **kwargs: Any,
    ) -> None:
        if isinstance(lelm_config, dict):
            lelm_config = SongGenerationV2LeLMConfig(**lelm_config)
        if isinstance(decoder_config, dict):
            decoder_config = SongGenerationV2Flow1dVAESeparateConfig(**decoder_config)

        self.lelm_config = lelm_config or SongGenerationV2LeLMConfig(
            prompt_len=prompt_len,
            audio_tokenizer_frame_rate=audio_tokenizer_frame_rate,
            sample_rate=sample_rate,
        )
        self.decoder_config = decoder_config or SongGenerationV2Flow1dVAESeparateConfig(
            prompt_len=prompt_len,
            sample_rate=sample_rate,
        )

        # Sub-configs must exist before super().__init__ (HF validate_token_ids
        # calls get_text_config during init).
        super().__init__(**kwargs)

        self.max_dur = max_dur
        self.min_dur = min_dur
        self.prompt_len = prompt_len
        self.pad_to_max = pad_to_max
        self.audio_tokenizer_checkpoint = audio_tokenizer_checkpoint
        self.audio_tokenizer_frame_rate = audio_tokenizer_frame_rate
        self.audio_tokenizer_code_depth = audio_tokenizer_code_depth
        self.sample_rate = sample_rate
        self.prompt_frames = prompt_len * audio_tokenizer_frame_rate
        self.hop_length = sample_rate // audio_tokenizer_frame_rate

    def get_text_config(self, **kwargs: Any) -> SongGenerationV2LeLMConfig:
        return self.lelm_config


__all__ = [
    "SongGenerationV2Config",
    "SongGenerationV2Flow1dVAESeparateConfig",
    "SongGenerationV2LeLMConfig",
]
