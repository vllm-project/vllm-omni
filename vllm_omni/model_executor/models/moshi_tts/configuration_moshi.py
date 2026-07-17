from __future__ import annotations

from transformers import PretrainedConfig


class MoshiMainConfig(PretrainedConfig):
    """Main temporal transformer config, compatible with vLLM's LlamaModel."""

    model_type = "moshi_main"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        intermediate_size: int = 11264,
        max_position_embeddings: int = 3000,
        rms_norm_eps: float = 1e-8,
        rope_theta: float = 10000.0,
        hidden_act: str = "silu",
        head_dim: int | None = None,
        sliding_window: int | None = 3000,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        tie_word_embeddings: bool = False,
        audio_vocab_size: int = 2048,
        num_codebooks: int = 8,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.audio_vocab_size = audio_vocab_size
        self.num_codebooks = num_codebooks

        # RoPE: vLLM's LlamaModel reads both config.rope_theta and
        # config.rope_parameters depending on the code path.
        self.rope_theta = rope_theta
        self.rope_parameters = {
            "rope_type": "default",
            "rope_theta": rope_theta,
        }

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class MoshiDepthConfig(PretrainedConfig):
    """Depth decoder config.

    All linear layers use per-codebook weights (MoshiFlexibleLinear).
    No RoPE — positions are implicit via the autoregressive codebook order.
    Max sequence length = num_codebooks + 1 (text token + codebook tokens).
    """

    model_type = "moshi_depth"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 1024,
        input_size: int = 4096,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 16,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        ffn_dim: int = 5632,
        max_position_embeddings: int = 9,
        rms_norm_eps: float = 1e-8,
        hidden_act: str = "silu",
        sliding_window: int = 8,
        attention_dropout: float = 0.0,
        audio_vocab_size: int = 2048,
        num_codebooks: int = 8,
        norm_type: str = "rms_norm",
        weights_per_step_schedule: list[int] | None = None,
        use_kv_cache: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.ffn_dim = ffn_dim
        self.intermediate_size = ffn_dim // 2
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout
        self.audio_vocab_size = audio_vocab_size
        self.num_codebooks = num_codebooks
        self.norm_type = norm_type  # "rms_norm" (moshi) or "layer_norm" (hibiki)
        self.weights_per_step_schedule = weights_per_step_schedule  # e.g. [0,1,...,8,8,...,8]
        self.use_kv_cache = use_kv_cache
        super().__init__(**kwargs)


class MoshiVLLMConfig(PretrainedConfig):
    """Top-level config wrapping main + depth + audio encoder configs.

    Constructed from an HF MoshiConfig by extracting and remapping
    parameters for each sub-component.
    """

    model_type = "moshi_vllm"

    def __init__(
        self,
        main_config: dict | MoshiMainConfig | None = None,
        depth_config: dict | MoshiDepthConfig | None = None,
        audio_encoder_config: dict | None = None,
        num_codebooks: int = 8,
        audio_vocab_size: int = 2048,
        n_q: int | None = None,
        delays: list[int] | None = None,
        cross_attention: bool = False,
        cross_attention_dim: int = 512,
        **kwargs,
    ):
        if isinstance(main_config, dict):
            self.main_config = MoshiMainConfig(**main_config)
        else:
            self.main_config = main_config or MoshiMainConfig()

        if isinstance(depth_config, dict):
            self.depth_config = MoshiDepthConfig(**depth_config)
        else:
            self.depth_config = depth_config or MoshiDepthConfig()

        self.cross_attention = cross_attention
        self.cross_attention_dim = cross_attention_dim

        self.audio_encoder_config = audio_encoder_config or {}
        self.num_codebooks = num_codebooks
        self.audio_vocab_size = audio_vocab_size
        # n_q = total codebooks (dep_q moshi + user). Default: 2 * num_codebooks.
        self.n_q = n_q if n_q is not None else 2 * num_codebooks
        # Per-codebook delays. Default: Moshi's [0, 0, 1, ..., 1, 0, 1, ..., 1].
        self.delays = delays
        super().__init__(**kwargs)

    def get_text_config(self, **kwargs) -> MoshiMainConfig:
        """Return the main transformer config.

        Required by vLLM's LlamaModel which calls
        vllm_config.model_config.hf_config.get_text_config() to get
        model dimensions (hidden_size, num_layers, etc.).
        """
        return self.main_config

    @classmethod
    def from_hf_config(cls, hf_config) -> MoshiVLLMConfig:
        """Construct from an HF MoshiConfig.

        Works for both native HF checkpoints (moshiko) and converted
        checkpoints (Hibiki-Zero, TTS). Converted configs store extra fields
        (dep_q, n_q, delays, etc.) as kwargs attributes on MoshiConfig,
        which override the auto-created defaults.
        """
        num_codebooks = hf_config.num_codebooks
        audio_vocab_size = hf_config.audio_vocab_size

        dep_q = getattr(hf_config, "dep_q", None)
        if dep_q is not None:
            num_codebooks = dep_q

        rope_theta = getattr(hf_config, "rope_theta", 10000.0)
        if rope_theta == 10000.0 and hasattr(hf_config, "rope_parameters") and hf_config.rope_parameters:
            rope_theta = hf_config.rope_parameters.get("rope_theta", rope_theta)

        depth_hidden = getattr(hf_config, "depth_hidden_size", None)
        explicit_intermediate = getattr(hf_config, "intermediate_size", None)
        if explicit_intermediate is not None and depth_hidden is not None:
            intermediate_size = explicit_intermediate
        else:
            ffn_dim = getattr(hf_config, "ffn_dim", 22528)
            intermediate_size = ffn_dim // 2

        main_config = MoshiMainConfig(
            # +1 for padding token at index vocab_size
            vocab_size=hf_config.vocab_size + 1,
            hidden_size=hf_config.hidden_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads),
            intermediate_size=intermediate_size,
            max_position_embeddings=hf_config.max_position_embeddings,
            rms_norm_eps=getattr(hf_config, "rms_norm_eps", 1e-8),
            rope_theta=rope_theta,
            hidden_act=getattr(hf_config, "hidden_act", "silu"),
            head_dim=getattr(hf_config, "head_dim", None),
            sliding_window=getattr(hf_config, "sliding_window", 3000),
            audio_vocab_size=audio_vocab_size,
            num_codebooks=num_codebooks,
        )

        depth_hf = getattr(hf_config, "depth_decoder_config", None)

        if depth_hf is not None and depth_hidden is None:
            depth_config = MoshiDepthConfig(
                vocab_size=depth_hf.vocab_size,
                hidden_size=depth_hf.hidden_size,
                input_size=depth_hf.input_size,
                num_hidden_layers=depth_hf.num_hidden_layers,
                num_attention_heads=depth_hf.num_attention_heads,
                num_key_value_heads=getattr(depth_hf, "num_key_value_heads", depth_hf.num_attention_heads),
                head_dim=depth_hf.head_dim,
                ffn_dim=depth_hf.ffn_dim,
                max_position_embeddings=depth_hf.max_position_embeddings,
                rms_norm_eps=depth_hf.rms_norm_eps,
                hidden_act=depth_hf.hidden_act,
                sliding_window=depth_hf.sliding_window,
                audio_vocab_size=depth_hf.audio_vocab_size,
                num_codebooks=depth_hf.num_codebooks,
            )
        else:
            d_hidden = depth_hidden or 1024
            d_intermediate = getattr(hf_config, "depth_intermediate_size", d_hidden * 3)
            depth_config = MoshiDepthConfig(
                vocab_size=hf_config.vocab_size,
                hidden_size=d_hidden,
                input_size=hf_config.hidden_size,
                num_hidden_layers=getattr(hf_config, "depth_num_hidden_layers", 6),
                num_attention_heads=getattr(hf_config, "depth_num_attention_heads", 16),
                num_key_value_heads=getattr(hf_config, "depth_num_key_value_heads", 16),
                ffn_dim=d_intermediate * 2,
                max_position_embeddings=num_codebooks + 1,
                sliding_window=num_codebooks,
                audio_vocab_size=audio_vocab_size,
                num_codebooks=num_codebooks,
            )

        n_q = getattr(hf_config, "n_q", None)
        delays = getattr(hf_config, "delays", None)
        depth_schedule = getattr(hf_config, "depformer_weights_per_step_schedule", None)
        depth_norm = getattr(hf_config, "depformer_norm", None)
        if depth_norm and "layer_norm" in depth_norm:
            depth_config.norm_type = "layer_norm"
        if depth_schedule:
            depth_config.weights_per_step_schedule = depth_schedule

        aec = getattr(hf_config, "audio_encoder_config", None)
        audio_encoder_config = aec.to_dict() if aec and hasattr(aec, "to_dict") else {}

        cross_attention = bool(getattr(hf_config, "cross_attention", False))
        cond = getattr(hf_config, "conditioners", None) or {}
        if isinstance(cond, dict):
            cross_attention_dim = int((cond.get("speaker_wavs") or {}).get("tensor", {}).get("dim", 512))
        else:
            cross_attention_dim = 512

        return cls(
            main_config=main_config,
            depth_config=depth_config,
            audio_encoder_config=audio_encoder_config,
            num_codebooks=num_codebooks,
            audio_vocab_size=audio_vocab_size,
            n_q=n_q,
            delays=delays,
            cross_attention=cross_attention,
            cross_attention_dim=cross_attention_dim,
        )

    @classmethod
    def from_converted_config(cls, cfg: dict) -> MoshiVLLMConfig:
        """Construct from a converted config dict (e.g. from convert_moshi_tts_weights.py)."""
        num_codebooks = cfg["num_codebooks"]
        audio_vocab_size = cfg.get("audio_vocab_size", 2048)
        vocab_size = cfg.get("vocab_size", 32001)

        main_config = MoshiMainConfig(
            vocab_size=vocab_size,
            hidden_size=cfg["hidden_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
            intermediate_size=cfg["intermediate_size"],
            max_position_embeddings=cfg.get("max_position_embeddings", 3000),
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-8),
            rope_theta=cfg.get("rope_theta", 10000.0),
            hidden_act=cfg.get("hidden_act", "silu"),
            sliding_window=cfg.get("sliding_window", 3000),
            audio_vocab_size=audio_vocab_size,
            num_codebooks=num_codebooks,
        )

        depth_norm = cfg.get("depformer_norm", "rms_norm")
        norm_type = "layer_norm" if "layer_norm" in depth_norm else "rms_norm"
        depth_ffn = cfg.get("depth_intermediate_size", 2816)

        depth_config = MoshiDepthConfig(
            vocab_size=vocab_size,
            hidden_size=cfg.get("depth_hidden_size", 1024),
            input_size=cfg["hidden_size"],
            num_hidden_layers=cfg.get("depth_num_hidden_layers", 6),
            num_attention_heads=cfg.get("depth_num_attention_heads", 16),
            num_key_value_heads=cfg.get("depth_num_key_value_heads", cfg.get("depth_num_attention_heads", 16)),
            ffn_dim=depth_ffn * 2,
            max_position_embeddings=num_codebooks + 1,
            sliding_window=num_codebooks,
            audio_vocab_size=audio_vocab_size,
            num_codebooks=num_codebooks,
            norm_type=norm_type,
            weights_per_step_schedule=cfg.get("depformer_weights_per_step_schedule"),
        )

        cond = cfg.get("conditioners") or {}
        cross_attention_dim = int((cond.get("speaker_wavs") or {}).get("tensor", {}).get("dim", 512))

        return cls(
            main_config=main_config,
            depth_config=depth_config,
            audio_encoder_config={},
            num_codebooks=num_codebooks,
            audio_vocab_size=audio_vocab_size,
            n_q=cfg.get("n_q", 2 * num_codebooks),
            delays=cfg.get("delays"),
            cross_attention=bool(cfg.get("cross_attention", False)),
            cross_attention_dim=cross_attention_dim,
        )
