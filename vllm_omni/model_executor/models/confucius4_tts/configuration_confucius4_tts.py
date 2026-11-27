# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from transformers import PretrainedConfig

DEFAULT_MAX_TEXT_TOKENS = 520
POSITIONAL_SEMANTIC_TOKENS = 1520
MAX_GENERATED_SEMANTIC_TOKENS = 1518
MAX_SEMANTIC_CODE = 8191
SEMANTIC_BOS_TOKEN_ID = 8192
SEMANTIC_EOS_TOKEN_ID = 8193
OUTPUT_SAMPLE_RATE = 22050
DEFAULT_MAX_REFERENCE_AUDIO_SECONDS = 30.0
# Rotary positions of the S2A DiT (reference mel frames plus generated frames). The
# released checkpoint stores a table of this size and was not trained past it, so it
# is the fixed table length and the upper bound of ``s2a_max_positions``.
S2A_MAX_POSITIONS = 4096


class Confucius4TTSConfig(PretrainedConfig):
    model_type = "confucius4_tts"

    def __init__(
        self,
        num_layers: int = 24,
        hidden_size: int = 1280,
        num_attention_heads: int = 20,
        max_text_tokens: int = DEFAULT_MAX_TEXT_TOKENS,
        max_semantic_tokens: int = POSITIONAL_SEMANTIC_TOKENS,
        text_vocab_size: int = 32000,
        vocab_size: int = 8194,
        text_embedding_dim: int = 4096,
        speaker_embedding_dim: int = 1024,
        bos_token_id: int = SEMANTIC_BOS_TOKEN_ID,
        eos_token_id: int = SEMANTIC_EOS_TOKEN_ID,
        t2s_checkpoint: str = "t2s_model.safetensors",
        s2a_checkpoint: str = "s2a_model.pt",
        w2v_stat: str = "wav2vec2bert_stats.pt",
        vocoder_name: str = "nvidia/bigvgan_v2_22khz_80band_256x",
        diffusion_steps: int = 25,
        inference_cfg_rate: float = 0.7,
        mel_length_ratio: float = 1.72,
        output_sample_rate: int = OUTPUT_SAMPLE_RATE,
        max_reference_audio_seconds: float = DEFAULT_MAX_REFERENCE_AUDIO_SECONDS,
        s2a_max_positions: int = S2A_MAX_POSITIONS,
        s2a_input_size: int = 512,
        s2a_output_size: int = 80,
        s2a_spk_embed_dim: int = 192,
        s2a_semantic_embed_dim: int = 1024,
        s2a_lm_latent_dim: int = 1280,
        s2a_semantic_codebook_size: int = 8192,
        s2a_semantic_codebook_dim: int = 8,
        s2a_semantic_output_dim: int = 1024,
        s2a_lr_channels: int = 512,
        s2a_lr_sampling_ratios: tuple[int, ...] = (1, 1, 1, 1),
        s2a_lr_out_channels: int = 512,
        s2a_lr_groups: int = 1,
        s2a_lr_in_channels: int = 1024,
        s2a_cfm_inference_cfg_rate: float = 0.7,
        s2a_cfm_t_scheduler: str = "linear",
        s2a_estimator_hidden_dim: int = 512,
        s2a_estimator_num_heads: int = 8,
        s2a_estimator_depth: int = 13,
        s2a_estimator_mel_dim: int = 80,
        s2a_estimator_cond_dim: int = 512,
        s2a_estimator_style_dim: int = 192,
        s2a_estimator_mlp_ratio: float = 3.0,
        s2a_estimator_dropout: float = 0.0,
        s2a_estimator_long_skip_connection: bool = True,
        s2a_estimator_final_layer: str = "wavenet",
        s2a_estimator_wavenet_hidden_dim: int = 512,
        s2a_estimator_wavenet_kernel_size: int = 5,
        s2a_estimator_wavenet_dilation_rate: int = 1,
        s2a_estimator_wavenet_num_layers: int = 8,
        s2a_estimator_wavenet_dropout: float = 0.0,
        **kwargs,
    ):
        kwargs.setdefault(
            "architectures",
            [
                "Confucius4TTSTalkerForConditionalGeneration",
                "Confucius4TTSS2ADecoder",
            ],
        )
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.max_text_tokens = max_text_tokens
        self.max_semantic_tokens = max_semantic_tokens
        self.text_vocab_size = text_vocab_size
        self.vocab_size = vocab_size
        self.text_embedding_dim = text_embedding_dim
        self.speaker_embedding_dim = speaker_embedding_dim
        self.t2s_checkpoint = t2s_checkpoint
        self.s2a_checkpoint = s2a_checkpoint
        self.w2v_stat = w2v_stat
        self.vocoder_name = vocoder_name
        self.diffusion_steps = diffusion_steps
        self.inference_cfg_rate = inference_cfg_rate
        self.mel_length_ratio = mel_length_ratio
        self.output_sample_rate = output_sample_rate
        self.max_reference_audio_seconds = max_reference_audio_seconds
        self.s2a_max_positions = s2a_max_positions
        self.s2a_input_size = s2a_input_size
        self.s2a_output_size = s2a_output_size
        self.s2a_spk_embed_dim = s2a_spk_embed_dim
        self.s2a_semantic_embed_dim = s2a_semantic_embed_dim
        self.s2a_lm_latent_dim = s2a_lm_latent_dim
        self.s2a_semantic_codebook_size = s2a_semantic_codebook_size
        self.s2a_semantic_codebook_dim = s2a_semantic_codebook_dim
        self.s2a_semantic_output_dim = s2a_semantic_output_dim
        self.s2a_lr_channels = s2a_lr_channels
        self.s2a_lr_sampling_ratios = tuple(s2a_lr_sampling_ratios)
        self.s2a_lr_out_channels = s2a_lr_out_channels
        self.s2a_lr_groups = s2a_lr_groups
        self.s2a_lr_in_channels = s2a_lr_in_channels
        self.s2a_cfm_inference_cfg_rate = s2a_cfm_inference_cfg_rate
        self.s2a_cfm_t_scheduler = s2a_cfm_t_scheduler
        self.s2a_estimator_hidden_dim = s2a_estimator_hidden_dim
        self.s2a_estimator_num_heads = s2a_estimator_num_heads
        self.s2a_estimator_depth = s2a_estimator_depth
        self.s2a_estimator_mel_dim = s2a_estimator_mel_dim
        self.s2a_estimator_cond_dim = s2a_estimator_cond_dim
        self.s2a_estimator_style_dim = s2a_estimator_style_dim
        self.s2a_estimator_mlp_ratio = s2a_estimator_mlp_ratio
        self.s2a_estimator_dropout = s2a_estimator_dropout
        self.s2a_estimator_long_skip_connection = s2a_estimator_long_skip_connection
        self.s2a_estimator_final_layer = s2a_estimator_final_layer
        self.s2a_estimator_wavenet_hidden_dim = s2a_estimator_wavenet_hidden_dim
        self.s2a_estimator_wavenet_kernel_size = s2a_estimator_wavenet_kernel_size
        self.s2a_estimator_wavenet_dilation_rate = s2a_estimator_wavenet_dilation_rate
        self.s2a_estimator_wavenet_num_layers = s2a_estimator_wavenet_num_layers
        self.s2a_estimator_wavenet_dropout = s2a_estimator_wavenet_dropout


class Confucius4Config(Confucius4TTSConfig):
    """Config alias matching the official checkpoint's model_type."""

    model_type = "confucius4"

    def __init__(self, **kwargs):
        # Transformers 5 generates an initializer for subclasses without one,
        # which would otherwise discard the inherited TTS-specific fields.
        super().__init__(**kwargs)
