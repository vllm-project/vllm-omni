# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Copyright 2024 ANT Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration for Ming-flash-omni-2.0 model"""

import os
from typing import Any, ClassVar

from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedTokenizerFast
from transformers.utils import logging

logger = logging.get_logger(__name__)


class BailingMoeV2Config(PretrainedConfig):
    model_type = "bailing_moe_v2"
    ignore_keys_at_rope_validation = {"mrope_section"}

    def __init__(
        self,
        vocab_size=30592,
        hidden_size=1024,
        intermediate_size=None,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=0,
        hidden_act="silu",
        use_qkv_bias=False,
        use_qk_norm=False,
        use_bias=True,
        rms_norm_eps=1e-05,
        norm_head=False,
        tie_word_embeddings=False,
        embedding_dropout=0.0,
        attention_dropout=0.0,
        output_dropout=0.0,
        initializer_range=0.02,
        max_position_embeddings=16384,
        rope_theta=10000.0,
        use_cache=True,
        use_sliding_window=False,
        sliding_window=81920,
        max_window_layers=28,
        rope_scaling=None,
        mrope_section=None,
        pad_token_id=126081,
        num_experts=16,
        num_shared_experts=1,
        num_experts_per_tok=2,
        n_group=8,
        topk_group=4,
        routed_scaling_factor=2.5,
        moe_intermediate_size=None,
        first_k_dense_replace=0,
        head_dim=None,
        output_router_logits=False,
        partial_rotary_factor=0.5,
        router_type="topN",
        _attn_implementation="flash_attention_2",
        use_interleaved_frame_timestamp=True,
        # Multimodal token IDs
        image_patch_token=157157,
        video_patch_token=157175,
        audio_patch_token=157168,
        image_start_token=157158,
        video_start_token=157160,
        audio_start_token=157169,
        image_end_token=157159,
        video_end_token=157161,
        audio_end_token=157170,
        # Position encoding parameters
        spatial_merge_size=2,
        tokens_per_second=2,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.use_qkv_bias = use_qkv_bias
        self.use_bias = use_bias
        self.norm_head = norm_head
        self.rms_norm_eps = rms_norm_eps
        self.embedding_dropout = embedding_dropout
        self.attention_dropout = attention_dropout
        self.output_dropout = output_dropout
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.use_cache = use_cache
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.head_dim = head_dim or self.hidden_size // self.num_attention_heads
        self.use_qk_norm = use_qk_norm  # arg unused; QK norm is always applied

        # By default, match the value of `mrope_section`
        # to `apply_3d_rotary_pos_emb` in Ming's repo:
        # https://github.com/inclusionAI/Ming/blob/3954fcb880ff5e61ff128bcf7f1ec344d46a6fe3/modeling_bailing_moe_v2.py
        if mrope_section is None:
            mrope_section = (rope_scaling or {}).get("mrope_section", [8, 12, 12])
        # Ensure mrope_section is stored inside rope_scaling
        if rope_scaling is not None and isinstance(rope_scaling, dict):
            rope_scaling = dict(rope_scaling)
            rope_scaling.setdefault("mrope_section", mrope_section)
        self.rope_scaling = rope_scaling

        # NOTE: Expose rope_parameters["mrope_section"]
        # This refers to the pattern used for GLM-Image in vllm_omni/patch.py
        rope_type = (rope_scaling or {}).get("type", (rope_scaling or {}).get("rope_type", ""))
        if rope_type in ("video_rope", "3D", "mrope"):
            self.rope_parameters = {"mrope_section": mrope_section}
        else:
            self.rope_parameters = None

        # MoE configs
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.output_router_logits = output_router_logits
        self.routed_scaling_factor = routed_scaling_factor
        self.partial_rotary_factor = partial_rotary_factor
        self.router_type = router_type
        self.use_interleaved_frame_timestamp = use_interleaved_frame_timestamp
        self._attn_implementation = _attn_implementation

        # Multimodal token IDs and position encoding
        self.image_patch_token = image_patch_token
        self.video_patch_token = video_patch_token
        self.audio_patch_token = audio_patch_token
        self.image_start_token = image_start_token
        self.video_start_token = video_start_token
        self.audio_start_token = audio_start_token
        self.image_end_token = image_end_token
        self.video_end_token = video_end_token
        self.audio_end_token = audio_end_token
        self.spatial_merge_size = spatial_merge_size
        self.tokens_per_second = tokens_per_second

        super().__init__(pad_token_id=pad_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)


class Qwen3VLMoeVisionConfig(PretrainedConfig):
    """Configuration class for Qwen3 MoE Vision Transformer"""

    model_type = "qwen3_moe_vit"

    def __init__(
        self,
        depth=27,
        hidden_size=1152,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=4304,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=3584,
        num_position_embeddings=2304,
        deepstack_visual_indexes=[8, 16, 24],
        initializer_range=0.02,
        **kwargs,
    ):
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
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range
        self.deepstack_visual_indexes = deepstack_visual_indexes

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if "vision_config" in config_dict:
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class WhisperEncoderConfig(PretrainedConfig):
    """Configuration class for Whisper audio encoder"""

    model_type = "whisper_encoder"

    def __init__(
        self,
        whisper_encoder_config: dict[str, Any] | None = None,
        ds_kernel_size=3,
        ds_stride=2,
        norm_query_embeds=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.whisper_encoder_config = whisper_encoder_config or {}
        self.ds_kernel_size = ds_kernel_size
        self.ds_stride = ds_stride
        self.norm_query_embeds = norm_query_embeds


class BailingMM2Config(PretrainedConfig):
    model_type = "bailingmm_moe_v2_lite"
    ignore_keys_at_rope_validation = {"mrope_section"}
    is_composition = True
    sub_configs: ClassVar = {"llm_config": AutoConfig}

    def __init__(
        self,
        mlp_depth=1,
        llm_config: BailingMoeV2Config | None = None,
        vision_config: Qwen3VLMoeVisionConfig | None = None,
        audio_config: WhisperEncoderConfig | None = None,
        **kwargs,
    ):
        self.audio_config = WhisperEncoderConfig(**audio_config) if isinstance(audio_config, dict) else audio_config
        self.vision_config = (
            Qwen3VLMoeVisionConfig(**vision_config) if isinstance(vision_config, dict) else vision_config
        )
        self.llm_config = BailingMoeV2Config(**llm_config) if isinstance(llm_config, dict) else llm_config
        self.mlp_depth = mlp_depth
        super().__init__(**kwargs)

    def get_text_config(self, decoder: bool = False) -> PretrainedConfig:  # noqa: ARG002
        return self.llm_config


class MingImageGenConfig(PretrainedConfig):
    """Configuration for Ming-flash-omni-2.0 image generation stage.

    Mirrors the layout of the HF checkpoint at
    https://huggingface.co/inclusionAI/Ming-flash-omni-2.0 where image-gen
    components live in sibling subfolders (``connector/``, ``transformer/``,
    ``vae/``, ``scheduler/``, ``mlp/``).
    """

    model_type = "ming_flash_omni_imagegen"

    def __init__(
        self,
        connector_subfolder: str = "connector",
        transformer_subfolder: str = "transformer",
        vae_subfolder: str = "vae",
        scheduler_subfolder: str = "scheduler",
        mlp_subfolder: str = "mlp",
        diffusion_c_input_dim: int = 2560,
        thinker_hidden_size: int = 4096,
        img_gen_scales: list[int] | None = None,
        text_encoder_norm: bool = True,
        # Default sampling parameters — match Ming's defaults in
        # ``modeling_bailingmm2.py::generate`` (``image_gen_steps=30``,
        # ``image_gen_cfg=2.0``).
        num_inference_steps: int = 30,
        guidance_scale: float = 2.0,
        default_height: int = 1024,
        default_width: int = 1024,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.connector_subfolder = connector_subfolder
        self.transformer_subfolder = transformer_subfolder
        self.vae_subfolder = vae_subfolder
        self.scheduler_subfolder = scheduler_subfolder
        self.mlp_subfolder = mlp_subfolder
        self.diffusion_c_input_dim = diffusion_c_input_dim
        self.thinker_hidden_size = thinker_hidden_size
        self.img_gen_scales = img_gen_scales if img_gen_scales is not None else [16]
        self.text_encoder_norm = text_encoder_norm
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.default_height = default_height
        self.default_width = default_width

    @property
    def num_query_tokens(self) -> int:
        """Total number of learnable query tokens appended to the thinker input.

        For ``img_gen_scales=[16]`` this yields 256 tokens (a single 16x16 grid).
        """
        return sum(s * s for s in self.img_gen_scales)


class MingFlashOmniTalkerConfig(PretrainedConfig):
    """Configuration class for Ming-flash-omni-2.0 talker (TTS) stage.

    The talker uses a Qwen2 LLM backbone with CFM (Conditional Flow Matching)
    via a DiT diffusion transformer, plus an Aggregator that maps generated
    audio latents back to the LLM embedding space for autoregressive generation.
    """

    model_type = "ming_flash_omni_talker"

    def __init__(
        self,
        llm_config: dict[str, Any] | None = None,
        flowmodel: dict[str, Any] | None = None,
        aggregator: dict[str, Any] | None = None,
        steps: int = 10,
        patch_size: int = 4,
        history_patch_size: int = 32,
        latent_dim: int = 64,
        cfg_strength: float = 2.0,
        audio_vae_path: str | None = None,
        campplus_model: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm_config = llm_config
        self.flowmodel = flowmodel or {}
        self.aggregator = aggregator or {}
        self.steps = steps
        self.patch_size = patch_size
        self.history_patch_size = history_patch_size
        self.latent_dim = latent_dim
        self.cfg_strength = cfg_strength
        self.audio_vae_path = audio_vae_path
        self.campplus_model = campplus_model

    def get_text_config(self, decoder: bool = False) -> PretrainedConfig:  # noqa: ARG002
        # NOTE: transformers v5 runs validators (e.g. validate_token_ids -> get_text_config)
        # during PretrainedConfig.__init__, before llm_config is assigned
        llm_config = getattr(self, "llm_config", None)
        if isinstance(llm_config, dict):
            return PretrainedConfig.from_dict(llm_config)

        return llm_config


class MingFlashOmniConfig(PretrainedConfig):
    """Configuration class for unified Ming-flash-omni-2.0 model"""

    model_type = "ming_flash_omni"
    is_composition = True
    sub_configs: ClassVar = {
        "thinker_config": BailingMM2Config,
        "image_gen_config": MingImageGenConfig,
        "talker_config": MingFlashOmniTalkerConfig,
    }

    def __init__(
        self,
        thinker_config: BailingMM2Config | dict[str, Any] | None = None,
        image_gen_config: MingImageGenConfig | dict[str, Any] | None = None,
        talker_config: MingFlashOmniTalkerConfig | dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(thinker_config, dict):
            self.thinker_config = BailingMM2Config(**thinker_config)
        else:
            self.thinker_config = thinker_config or BailingMM2Config()

        # Image generation config
        if isinstance(image_gen_config, dict):
            self.image_gen_config = MingImageGenConfig(**image_gen_config)
        elif image_gen_config is None:
            self.image_gen_config = MingImageGenConfig()
        else:
            self.image_gen_config = image_gen_config

        # Talker config
        if isinstance(talker_config, dict):
            self.talker_config = MingFlashOmniTalkerConfig(**talker_config)
        else:
            self.talker_config = talker_config

    def get_text_config(self, decoder: bool = False) -> PretrainedConfig:  # noqa: ARG002
        return self.thinker_config.get_text_config()


# Register model_type -> config class for AutoConfig
AutoConfig.register(BailingMoeV2Config.model_type, BailingMoeV2Config)
AutoConfig.register(BailingMM2Config.model_type, BailingMM2Config)
AutoConfig.register(MingImageGenConfig.model_type, MingImageGenConfig)
AutoConfig.register(MingFlashOmniTalkerConfig.model_type, MingFlashOmniTalkerConfig)
AutoConfig.register(MingFlashOmniConfig.model_type, MingFlashOmniConfig)

# Register tokenizer mapping for composition configs so that
# AutoTokenizer.from_pretrained can resolve the tokenizer class
AutoTokenizer.register(BailingMM2Config, fast_tokenizer_class=PreTrainedTokenizerFast)
AutoTokenizer.register(MingFlashOmniTalkerConfig, fast_tokenizer_class=PreTrainedTokenizerFast)
AutoTokenizer.register(MingFlashOmniConfig, fast_tokenizer_class=PreTrainedTokenizerFast)
