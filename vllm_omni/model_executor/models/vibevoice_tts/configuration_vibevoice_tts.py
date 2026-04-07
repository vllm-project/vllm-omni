from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from vllm.logger import init_logger

logger = init_logger(__name__)


class VibeVoiceDiffusionHeadConfig(PretrainedConfig):
    model_type = "vibevoice_diffusion_head"

    def __init__(
        self,
        hidden_size: int = 1536,
        head_layers: int = 4,
        head_ffn_ratio: float = 3.0,
        rms_norm_eps: float = 1e-5,
        latent_size: int = 64,
        speech_vae_dim: int | None = None,
        prediction_type: str = "v_prediction",
        diffusion_type: str = "ddpm",
        ddpm_num_steps: int = 1000,
        ddpm_num_inference_steps: int = 20,
        ddpm_beta_schedule: str = "cosine",
        ddpm_batch_mul: int = 4,
        **kwargs: Any,
    ) -> None:
        self.hidden_size = hidden_size
        self.head_layers = head_layers
        self.head_ffn_ratio = head_ffn_ratio
        self.rms_norm_eps = rms_norm_eps
        self.latent_size = latent_size
        self.speech_vae_dim = speech_vae_dim
        self.prediction_type = prediction_type
        self.diffusion_type = diffusion_type
        self.ddpm_num_steps = ddpm_num_steps
        self.ddpm_num_inference_steps = ddpm_num_inference_steps
        self.ddpm_beta_schedule = ddpm_beta_schedule
        self.ddpm_batch_mul = ddpm_batch_mul
        super().__init__(**kwargs)


class VibeVoiceAcousticTokenizerConfig(PretrainedConfig):
    model_type = "vibevoice_acoustic_tokenizer"

    def __init__(
        self,
        channels: int = 1,
        corpus_normalize: float = 0.0,
        causal: bool = True,
        vae_dim: int = 64,
        fix_std: float = 0.5,
        std_dist_type: str = "gaussian",
        mixer_layer: str = "depthwise_conv",
        conv_norm: str = "none",
        pad_mode: str = "constant",
        disable_last_norm: bool = True,
        layernorm: str = "RMSNorm",
        layernorm_eps: float = 1e-5,
        layernorm_elementwise_affine: bool = True,
        conv_bias: bool = True,
        layer_scale_init_value: float = 1e-6,
        weight_init_value: float = 1e-2,
        encoder_n_filters: int = 32,
        encoder_ratios: list[int] | None = None,
        encoder_depths: str = "3-3-3-3-3-3-8",
        decoder_n_filters: int = 32,
        decoder_ratios: list[int] | None = None,
        decoder_depths: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.channels = channels
        self.corpus_normalize = corpus_normalize
        self.causal = causal
        self.vae_dim = vae_dim
        self.fix_std = fix_std
        self.std_dist_type = std_dist_type
        self.mixer_layer = mixer_layer
        self.conv_norm = conv_norm
        self.pad_mode = pad_mode
        self.disable_last_norm = disable_last_norm
        self.layernorm = layernorm
        self.layernorm_eps = layernorm_eps
        self.layernorm_elementwise_affine = layernorm_elementwise_affine
        self.conv_bias = conv_bias
        self.layer_scale_init_value = layer_scale_init_value
        self.weight_init_value = weight_init_value
        self.encoder_n_filters = encoder_n_filters
        self.encoder_ratios = encoder_ratios if encoder_ratios is not None else [8, 5, 5, 4, 2, 2]
        self.encoder_depths = encoder_depths
        self.decoder_n_filters = decoder_n_filters
        self.decoder_ratios = decoder_ratios if decoder_ratios is not None else list(self.encoder_ratios)
        self.decoder_depths = decoder_depths


class VibeVoiceSemanticTokenizerConfig(PretrainedConfig):
    model_type = "vibevoice_semantic_tokenizer"

    def __init__(
        self,
        channels: int = 1,
        corpus_normalize: float = 0.0,
        causal: bool = True,
        vae_dim: int = 128,
        fix_std: float = 0,
        std_dist_type: str = "none",
        mixer_layer: str = "depthwise_conv",
        conv_norm: str = "none",
        pad_mode: str = "constant",
        disable_last_norm: bool = True,
        layernorm: str = "RMSNorm",
        layernorm_eps: float = 1e-5,
        layernorm_elementwise_affine: bool = True,
        conv_bias: bool = True,
        layer_scale_init_value: float = 1e-6,
        weight_init_value: float = 1e-2,
        encoder_n_filters: int = 32,
        encoder_ratios: list[int] | None = None,
        encoder_depths: str = "3-3-3-3-3-3-8",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.channels = channels
        self.corpus_normalize = corpus_normalize
        self.causal = causal
        self.vae_dim = vae_dim
        self.fix_std = fix_std
        self.std_dist_type = std_dist_type
        self.mixer_layer = mixer_layer
        self.conv_norm = conv_norm
        self.pad_mode = pad_mode
        self.disable_last_norm = disable_last_norm
        self.layernorm = layernorm
        self.layernorm_eps = layernorm_eps
        self.layernorm_elementwise_affine = layernorm_elementwise_affine
        self.conv_bias = conv_bias
        self.layer_scale_init_value = layer_scale_init_value
        self.weight_init_value = weight_init_value
        self.encoder_n_filters = encoder_n_filters
        self.encoder_ratios = encoder_ratios if encoder_ratios is not None else [8, 5, 5, 4, 2, 2]
        self.encoder_depths = encoder_depths


class VibeVoiceTTSConfig(PretrainedConfig):
    """HuggingFace-style config for VibeVoice TTS models.

    Maps to the microsoft/VibeVoice-1.5B config.json which contains
    decoder_config (Qwen2), acoustic_tokenizer_config, semantic_tokenizer_config,
    and diffusion_head_config.
    """

    model_type = "vibevoice"

    def __init__(
        self,
        decoder_config: dict | Qwen2Config | None = None,
        acoustic_tokenizer_config: dict | VibeVoiceAcousticTokenizerConfig | None = None,
        semantic_tokenizer_config: dict | VibeVoiceSemanticTokenizerConfig | None = None,
        diffusion_head_config: dict | VibeVoiceDiffusionHeadConfig | None = None,
        acoustic_vae_dim: int = 64,
        semantic_vae_dim: int = 128,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if decoder_config is None:
            self.decoder_config = Qwen2Config()
        elif isinstance(decoder_config, dict):
            self.decoder_config = Qwen2Config(**decoder_config)
        else:
            self.decoder_config = decoder_config

        if acoustic_tokenizer_config is None:
            self.acoustic_tokenizer_config = VibeVoiceAcousticTokenizerConfig()
        elif isinstance(acoustic_tokenizer_config, dict):
            self.acoustic_tokenizer_config = VibeVoiceAcousticTokenizerConfig(**acoustic_tokenizer_config)
        else:
            self.acoustic_tokenizer_config = acoustic_tokenizer_config

        if semantic_tokenizer_config is None:
            self.semantic_tokenizer_config = VibeVoiceSemanticTokenizerConfig()
        elif isinstance(semantic_tokenizer_config, dict):
            self.semantic_tokenizer_config = VibeVoiceSemanticTokenizerConfig(**semantic_tokenizer_config)
        else:
            self.semantic_tokenizer_config = semantic_tokenizer_config

        if diffusion_head_config is None:
            self.diffusion_head_config = VibeVoiceDiffusionHeadConfig()
        elif isinstance(diffusion_head_config, dict):
            self.diffusion_head_config = VibeVoiceDiffusionHeadConfig(**diffusion_head_config)
        else:
            self.diffusion_head_config = diffusion_head_config

        self.acoustic_vae_dim = acoustic_vae_dim
        self.semantic_vae_dim = semantic_vae_dim

    def get_text_config(self, **kwargs: Any) -> Qwen2Config:
        """Required by vLLM for memory profiling."""
        return self.decoder_config

    @property
    def sampling_rate(self) -> int:
        return 24000

    @property
    def frame_rate(self) -> float:
        return 7.5
