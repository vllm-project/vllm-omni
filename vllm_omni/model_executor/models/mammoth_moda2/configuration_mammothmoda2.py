"""本地配置实现，复刻 mammothmoda2 的 HF 配置并注册 AutoConfig."""

from typing import ClassVar, Literal

from transformers import AutoConfig, PretrainedConfig
from transformers.configuration_utils import layer_type_validation
from transformers.modeling_rope_utils import rope_config_validation

__all__ = [
    "Mammothmoda2Config",
    "Mammothmoda2Qwen2_5_VLConfig",
    "Mammothmoda2Qwen2_5_VLTextConfig",
    "Mammothmoda2Qwen2_5_VLVisionConfig",
]


class Mammothmoda2Qwen2_5_VLVisionConfig(PretrainedConfig):
    model_type = "mammothmoda2_qwen2_5_vl"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth: int = 32,
        hidden_size: int = 3584,
        hidden_act: str = "silu",
        intermediate_size: int = 3420,
        num_heads: int = 16,
        in_channels: int = 3,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        tokens_per_second: int = 4,
        window_size: int = 112,
        out_hidden_size: int = 3584,
        fullatt_block_indexes: list[int] | None = None,
        initializer_range: float = 0.02,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.tokens_per_second = tokens_per_second
        self.window_size = window_size
        self.fullatt_block_indexes = fullatt_block_indexes or [7, 15, 23, 31]
        self.out_hidden_size = out_hidden_size
        self.initializer_range = initializer_range


class Mammothmoda2Qwen2_5_VLTextConfig(PretrainedConfig):
    """文本子配置，保持与上游 mammothmoda2_qwen2_5_vl_text 完全一致。"""

    model_type = "mammothmoda2_qwen2_5_vl_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `Qwen2_5_VL`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: int = 152064,
        hidden_size: int = 8192,
        intermediate_size: int = 29568,
        num_hidden_layers: int = 80,
        num_attention_heads: int = 64,
        num_key_value_heads: int | None = 8,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-05,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 1000000.0,
        use_sliding_window: bool = False,
        sliding_window: int = 4096,
        max_window_layers: int = 80,
        layer_types: list[str] | None = None,
        attention_dropout: float = 0.0,
        rope_scaling: dict | None = None,
        image_token_id: int | None = None,
        video_token_id: int | None = None,
        extra_gen_vocab: bool = True,
        gen_vocab_size: int = 32800,
        gen_vocab_start_index: int | None = None,
        moe_type: str = "ffn",
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types)

        # custom config
        self.extra_gen_vocab = extra_gen_vocab
        self.gen_vocab_size = gen_vocab_size
        self.moe_type = moe_type
        if gen_vocab_start_index is None:
            if self.extra_gen_vocab:
                self.gen_vocab_start_index = self.vocab_size
            else:
                self.gen_vocab_start_index = self.vocab_size - self.gen_vocab_size
        else:
            self.gen_vocab_start_index = gen_vocab_start_index

        # RoPE 校验
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            if self.rope_scaling["type"] == "mrope":
                self.rope_scaling["type"] = "default"
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self, ignore_keys={"mrope_section"})
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class Mammothmoda2Qwen2_5_VLConfig(PretrainedConfig):
    """组合配置：text_config + vision_config。"""

    model_type = "mammothmoda2_qwen2_5_vl"
    sub_configs = {
        "vision_config": Mammothmoda2Qwen2_5_VLVisionConfig,
        "text_config": Mammothmoda2Qwen2_5_VLTextConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config: dict | PretrainedConfig | None = None,
        vision_config: dict | PretrainedConfig | None = None,
        image_token_id: int = 151655,
        video_token_id: int = 151656,
        extra_gen_vocab: bool = True,
        gen_vocab_size: int = 32800,
        gen_vocab_start_index: int | None = None,
        moe_type: str = "ffn",
        **kwargs,
    ) -> None:
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif isinstance(vision_config, PretrainedConfig):
            self.vision_config = vision_config
        else:
            self.vision_config = self.sub_configs["vision_config"]()

        text_extra_kwargs = {
            "extra_gen_vocab": extra_gen_vocab,
            "gen_vocab_size": gen_vocab_size,
            "moe_type": moe_type,
            "gen_vocab_start_index": gen_vocab_start_index,
        }
        if isinstance(text_config, dict):
            for key, val in text_extra_kwargs.items():
                text_config.setdefault(key, val)
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif isinstance(text_config, PretrainedConfig):
            self.text_config = text_config
        else:
            for key, val in text_extra_kwargs.items():
                kwargs.setdefault(key, val)
            self.text_config = self.sub_configs["text_config"](**kwargs)

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

        self.extra_gen_vocab = extra_gen_vocab
        self.gen_vocab_size = gen_vocab_size
        self.moe_type = moe_type
        self.gen_vocab_start_index = self.text_config.gen_vocab_start_index

        super().__init__(**kwargs)


class Mammothmoda2Config(PretrainedConfig):
    """顶层 mammothmoda2 组合配置，与上游保持一致。"""

    model_type = "mammothmoda2"
    is_composition = True
    sub_configs: ClassVar = {"llm_config": AutoConfig}

    def __init__(
        self,
        *,
        llm_config: dict | None = None,
        gen_vae_config: dict | None = None,
        gen_dit_config: dict | None = None,
        gen_condition_mode: Literal["text", "image", "text_image"] = "image",
        gen_image_condition_refiner_config: dict | None = None,
        gen_axes_dim_rope: list[int] | None = None,
        gen_axes_lens: list[int] | None = None,
        gen_transport_config: dict | None = None,
        initializer_range: float = 0.02,
        architectures: list[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.llm_config = AutoConfig.for_model(**llm_config) if llm_config is not None else None
        self.gen_vae_config = gen_vae_config
        self.gen_dit_config = gen_dit_config

        self.gen_condition_mode = gen_condition_mode
        self.gen_image_condition_refiner_config = gen_image_condition_refiner_config
        self.gen_axes_dim_rope = gen_axes_dim_rope or [40, 40, 40]
        self.gen_axes_lens = gen_axes_lens or [10000, 10000, 10000]
        self.gen_transport_config = gen_transport_config or {}
        self.initializer_range = initializer_range
        # HF 权重里 architectures = ["Mammothmoda2Model"]，此处保持该名称，若用户传入其它值则按用户为准
        if architectures is None:
            self.architectures = ["Mammothmoda2Model"]
        else:
            self.architectures = [
                "Mammothmoda2Model" if a.lower() == "mammothmoda2model" else a for a in architectures
            ]

    def get_text_config(self, decoder: bool = False) -> PretrainedConfig:  # noqa: ARG002
        return self.llm_config


# Register model_type -> config class for AutoConfig
AutoConfig.register(Mammothmoda2Config.model_type, Mammothmoda2Config)
AutoConfig.register(Mammothmoda2Qwen2_5_VLConfig.model_type, Mammothmoda2Qwen2_5_VLConfig)
AutoConfig.register(Mammothmoda2Qwen2_5_VLTextConfig.model_type, Mammothmoda2Qwen2_5_VLTextConfig)
AutoConfig.register(Mammothmoda2Qwen2_5_VLVisionConfig.model_type, Mammothmoda2Qwen2_5_VLVisionConfig)
