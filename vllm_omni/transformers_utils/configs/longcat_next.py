from transformers import AutoConfig, PretrainedConfig

LONGCAT_NEXT_MODEL_TYPE = "longcat_next"


class LongcatNextConfig(PretrainedConfig):
    model_type = LONGCAT_NEXT_MODEL_TYPE

    def __init__(
        self,
        vocab_size: int = 282624,
        hidden_size: int = 3072,
        intermediate_size: int = 6144,
        num_hidden_layers: int = 14,
        num_attention_heads: int = 32,
        max_position_embeddings: int = 131072,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000000.0,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings


AutoConfig.register(LONGCAT_NEXT_MODEL_TYPE, LongcatNextConfig)
