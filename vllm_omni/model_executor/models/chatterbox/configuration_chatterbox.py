"""HuggingFace-compatible configs for Chatterbox TTS variants.

Chatterbox does not ship a standard ``config.json``.  These classes provide the
parameters that vLLM needs to instantiate the T3 backbone and S3Gen vocoder.
The stage YAML uses ``hf_overrides`` to force the architecture name.

Two variants:
- ``ChatterboxTurboConfig``: GPT-2-medium backbone (350M), meanflow S3Gen
- ``ChatterboxConfig``: LLaMA-520M backbone, standard S3Gen with CFG support
"""

from transformers import PretrainedConfig


def _validate_chatterbox_config(cfg: "PretrainedConfig") -> None:
    """Shared sanity checks for both Chatterbox variants.

    vLLM's sampler assumes ``logits.shape[-1] == config.vocab_size``.  T3's
    output head emits speech logits (``speech_vocab_size``), so we require
    ``vocab_size == speech_vocab_size``.  Special tokens must also land inside
    the speech vocab or sampling / EOS detection silently breaks.
    """
    if cfg.vocab_size != cfg.speech_vocab_size:
        raise ValueError(
            f"vocab_size ({cfg.vocab_size}) must equal speech_vocab_size "
            f"({cfg.speech_vocab_size}) so vLLM's sampler reads the full "
            f"speech-head logits."
        )
    for name in ("start_speech_token", "stop_speech_token"):
        tok = getattr(cfg, name)
        if not (0 <= tok < cfg.speech_vocab_size):
            raise ValueError(f"{name}={tok} is outside [0, speech_vocab_size={cfg.speech_vocab_size}).")


class ChatterboxTurboConfig(PretrainedConfig):
    """Configuration for Chatterbox Turbo (350M, GPT-2 backbone)."""

    model_type = "chatterbox_turbo"

    def __init__(
        self,
        # T3 GPT-2-medium backbone
        hidden_size: int = 1024,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        intermediate_size: int = 4096,
        # vocab_size must equal speech_vocab_size for vLLM sampler compatibility
        # (logits dimension must match config.vocab_size).
        vocab_size: int = 6563,
        text_vocab_size: int = 50276,  # GPT-2 text tokenizer vocab
        speech_vocab_size: int = 6563,
        max_position_embeddings: int = 8196,
        # Special tokens
        start_speech_token: int = 6561,
        stop_speech_token: int = 6562,
        # Conditioning
        speaker_embed_size: int = 256,
        speech_cond_prompt_len: int = 375,
        use_perceiver_resampler: bool = False,
        emotion_adv: bool = False,
        # S3Gen
        s3gen_sample_rate: int = 24000,
        s3_token_rate: int = 25,
        s3gen_meanflow: bool = True,  # Turbo uses meanflow (2 CFM steps)
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.text_vocab_size = text_vocab_size
        self.speech_vocab_size = speech_vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.start_speech_token = start_speech_token
        self.stop_speech_token = stop_speech_token
        self.speaker_embed_size = speaker_embed_size
        self.speech_cond_prompt_len = speech_cond_prompt_len
        self.use_perceiver_resampler = use_perceiver_resampler
        self.emotion_adv = emotion_adv
        self.s3gen_sample_rate = s3gen_sample_rate
        self.s3_token_rate = s3_token_rate
        self.s3gen_meanflow = s3gen_meanflow
        _validate_chatterbox_config(self)


class ChatterboxConfig(PretrainedConfig):
    """Configuration for Chatterbox Original (520M, LLaMA backbone)."""

    model_type = "chatterbox"

    def __init__(
        self,
        # T3 LLaMA-520M backbone
        hidden_size: int = 1024,
        num_hidden_layers: int = 30,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        intermediate_size: int = 4096,
        # vocab_size must equal speech_vocab_size for vLLM sampler compatibility
        # (logits dimension must match config.vocab_size).  The custom
        # EnTokenizer text vocab (704) is tracked separately via
        # ``text_vocab_size`` and consumed by ``text_emb`` in the T3 model.
        vocab_size: int = 8194,
        text_vocab_size: int = 704,  # Custom EnTokenizer vocab
        speech_vocab_size: int = 8194,
        max_position_embeddings: int = 131072,
        rope_theta: float = 500000.0,
        rms_norm_eps: float = 1e-5,
        # Special tokens
        start_speech_token: int = 6561,
        stop_speech_token: int = 6562,
        # Conditioning
        speaker_embed_size: int = 256,
        speech_cond_prompt_len: int = 150,
        use_perceiver_resampler: bool = True,
        emotion_adv: bool = True,
        # Learned position embeddings (Original uses these)
        input_pos_emb: str = "learned",
        max_text_tokens: int = 2048,
        max_speech_tokens: int = 4096,
        # S3Gen
        s3gen_sample_rate: int = 24000,
        s3_token_rate: int = 25,
        s3gen_meanflow: bool = False,  # Original uses standard CFM (10 steps)
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.text_vocab_size = text_vocab_size
        self.speech_vocab_size = speech_vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.start_speech_token = start_speech_token
        self.stop_speech_token = stop_speech_token
        self.speaker_embed_size = speaker_embed_size
        self.speech_cond_prompt_len = speech_cond_prompt_len
        self.use_perceiver_resampler = use_perceiver_resampler
        self.emotion_adv = emotion_adv
        self.input_pos_emb = input_pos_emb
        self.max_text_tokens = max_text_tokens
        self.max_speech_tokens = max_speech_tokens
        self.s3gen_sample_rate = s3gen_sample_rate
        self.s3_token_rate = s3_token_rate
        self.s3gen_meanflow = s3gen_meanflow
        _validate_chatterbox_config(self)
