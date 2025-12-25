"""Configuration implementation for MammothModa2, replicating HF configuration and registering with AutoConfig."""

from typing import ClassVar, Literal

from transformers import AutoConfig, AutoTokenizer, PretrainedConfig
from transformers.configuration_utils import layer_type_validation
from transformers.modeling_rope_utils import rope_config_validation
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLTextConfig,
    Qwen2_5_VLVisionConfig,
)

from .tokenization_mammothmoda2_qwen2_5_vl import MammothUTokenizer

__all__ = [
    "Mammothmoda2Config",
    "Mammothmoda2Qwen2_5_VLConfig",
    "Mammothmoda2Qwen2_5_VLTextConfig",
    "Mammothmoda2Qwen2_5_VLVisionConfig",
]


class Mammothmoda2Qwen2_5_VLVisionConfig(Qwen2_5_VLVisionConfig):
    # NOTE: Must not conflict with `Mammothmoda2Qwen2_5_VLConfig.model_type`.
    # Otherwise, `AutoConfig.for_model(model_type="mammothmoda2_qwen2_5_vl")` might be 
    # incorrectly parsed as a VisionConfig, causing `llm_config` to become a 
    # "vision sub-config" where `vision_config` is just a dict, leading to 
    # attribute access errors like `vision_config.patch_size` in vLLM.
    model_type = "mammothmoda2_qwen2_5_vl_vision"
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
        super().__init__(
            depth=depth,
            hidden_size=hidden_size,
            hidden_act=hidden_act,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            in_channels=in_channels,
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            temporal_patch_size=temporal_patch_size,
            tokens_per_second=tokens_per_second,
            window_size=window_size,
            out_hidden_size=out_hidden_size,
            fullatt_block_indexes=fullatt_block_indexes or [7, 15, 23, 31],
            initializer_range=initializer_range,
            **kwargs,
        )


class Mammothmoda2Qwen2_5_VLTextConfig(Qwen2_5_VLTextConfig):
    """Text sub-configuration, consistent with upstream mammothmoda2_qwen2_5_vl_text."""

    model_type = "mammothmoda2_qwen2_5_vl_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = Qwen2_5_VLTextConfig.base_model_tp_plan
    base_model_pp_plan = Qwen2_5_VLTextConfig.base_model_pp_plan

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
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window,
            max_window_layers=max_window_layers,
            layer_types=layer_types,
            attention_dropout=attention_dropout,
            rope_scaling=rope_scaling,
            **kwargs,
        )

        self.extra_gen_vocab = extra_gen_vocab
        self.gen_vocab_size = gen_vocab_size
        self.moe_type = moe_type
        if gen_vocab_start_index is None:
            self.gen_vocab_start_index = (
                self.vocab_size if self.extra_gen_vocab else self.vocab_size - self.gen_vocab_size
            )
        else:
            self.gen_vocab_start_index = gen_vocab_start_index

        # NOTE: vLLM V1 uses `hf_text_config.vocab_size` for sampling parameter validation 
        # (e.g., allowed_token_ids). Although MammothModa2's gen vocab is implemented via 
        # independent gen_embed/gen_head, the overall vocab size should still cover the 
        # gen vocab token ID range from the perspective of "output logits dimension".
        if self.extra_gen_vocab:
            self.vocab_size = int(self.gen_vocab_start_index) + int(self.gen_vocab_size)

        # Extra token IDs for multi-modal placeholders.
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id


class Mammothmoda2Qwen2_5_VLConfig(Qwen2_5_VLConfig):
    """Combined configuration: text_config + vision_config."""

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
        vision_start_token_id: int = 151652,
        vision_end_token_id: int = 151653,
        extra_gen_vocab: bool = True,
        gen_vocab_size: int = 32800,
        gen_vocab_start_index: int | None = None,
        moe_type: str = "ffn",
        **kwargs,
    ) -> None:
        if isinstance(vision_config, dict):
            vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            vision_config = self.sub_configs["vision_config"]()

        text_extra_kwargs = {
            "extra_gen_vocab": extra_gen_vocab,
            "gen_vocab_size": gen_vocab_size,
            "moe_type": moe_type,
            "gen_vocab_start_index": gen_vocab_start_index,
        }
        if isinstance(text_config, dict):
            for key, val in text_extra_kwargs.items():
                text_config.setdefault(key, val)
            text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            text_config = self.sub_configs["text_config"](**{**text_extra_kwargs, **kwargs})
        elif isinstance(text_config, PretrainedConfig):
            text_config = text_config

        super().__init__(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            vision_start_token_id=vision_start_token_id,
            vision_end_token_id=vision_end_token_id,
            **kwargs,
        )

        if not hasattr(self, "text_config"):
            self.text_config = text_config
        if not hasattr(self, "vision_config"):
            self.vision_config = vision_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.extra_gen_vocab = getattr(self.text_config, "extra_gen_vocab", extra_gen_vocab)
        self.gen_vocab_size = getattr(self.text_config, "gen_vocab_size", gen_vocab_size)
        self.moe_type = getattr(self.text_config, "moe_type", moe_type)
        self.gen_vocab_start_index = getattr(
            self.text_config, "gen_vocab_start_index", gen_vocab_start_index
        )
        # Inherit tokenizer_class to prevent AutoTokenizer from falling back to Qwen2.
        self.tokenizer_class = "MammothUTokenizer"


class Mammothmoda2Config(PretrainedConfig):
    """Top-level MammothModa2 composition configuration, consistent with upstream."""

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
        # Ensure AutoTokenizer prioritizes MammothUTokenizer over the default Qwen2Tokenizer.
        self.tokenizer_class = "MammothUTokenizer"
        # HF weights use architectures = ["Mammothmoda2Model"]; maintain this name by default.
        if architectures is None:
            self.architectures = ["Mammothmoda2Model"]
        else:
            self.architectures = [
                "Mammothmoda2Model" if a.lower() == "mammothmoda2model" else a for a in architectures
            ]

    def get_text_config(self, decoder: bool = False) -> PretrainedConfig:  # noqa: ARG002
        return self.llm_config

    def _require_llm_config(self) -> PretrainedConfig:
        if self.llm_config is None:
            raise AttributeError("Mammothmoda2Config.llm_config is None")
        return self.llm_config

    # ---- Proxy attrs for vLLM multimodal/mrope helpers ----
    # vllm_omni/model_executor/layers/mrope.py expects these fields on `hf_config`.
    # In MammothModa2, they live in the nested `llm_config` (VL config), so we
    # expose them here to make the top-level composition config compatible.
    @property
    def vision_config(self):
        return self._require_llm_config().vision_config

    @property
    def image_token_id(self) -> int:
        return int(self._require_llm_config().image_token_id)

    @property
    def video_token_id(self) -> int:
        return int(self._require_llm_config().video_token_id)

    @property
    def vision_start_token_id(self) -> int:
        return int(self._require_llm_config().vision_start_token_id)

    @property
    def vision_end_token_id(self) -> int:
        return int(self._require_llm_config().vision_end_token_id)


# Register model_type -> config class for AutoConfig
AutoConfig.register(Mammothmoda2Config.model_type, Mammothmoda2Config)
AutoConfig.register(Mammothmoda2Qwen2_5_VLConfig.model_type, Mammothmoda2Qwen2_5_VLConfig)
AutoConfig.register(Mammothmoda2Qwen2_5_VLTextConfig.model_type, Mammothmoda2Qwen2_5_VLTextConfig)
AutoConfig.register(Mammothmoda2Qwen2_5_VLVisionConfig.model_type, Mammothmoda2Qwen2_5_VLVisionConfig)

AutoTokenizer.register(
    config_class=Mammothmoda2Config,
    slow_tokenizer_class=MammothUTokenizer
)
AutoTokenizer.register(
    config_class=Mammothmoda2Qwen2_5_VLConfig,
    slow_tokenizer_class=MammothUTokenizer
)
