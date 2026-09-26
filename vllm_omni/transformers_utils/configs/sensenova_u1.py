# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""HuggingFace-style configuration classes for SenseNova-U1."""

from __future__ import annotations

from transformers import AutoConfig, PretrainedConfig, Qwen3Config, Qwen3MoeConfig

_SENSENOVA_U1_LLM_HIDDEN_DEFAULT = 4096


def _restore_legacy_rope_theta(config) -> None:
    """Expose the v4 rope attribute expected by the vendored model code."""
    if hasattr(config, "rope_theta"):
        return
    rope_parameters = getattr(config, "rope_parameters", None) or {}
    config.rope_theta = float(rope_parameters.get("rope_theta", 10000.0))


def _backfill_layer_types(config) -> None:
    """Qwen3MoeConfig does not always populate ``layer_types``; attention reads it."""
    existing = getattr(config, "layer_types", None)
    if existing and len(existing) == config.num_hidden_layers:
        return
    use_swa = bool(getattr(config, "use_sliding_window", False)) and getattr(config, "sliding_window", None) is not None
    max_window_layers = int(getattr(config, "max_window_layers", 0) or 0)
    config.layer_types = [
        "sliding_attention" if (use_swa and i >= max_window_layers) else "full_attention"
        for i in range(config.num_hidden_layers)
    ]


def _is_moe_llm_config(llm_config) -> bool:
    """Detect whether an ``llm_config`` (dict or object) targets a MoE backbone."""
    if isinstance(llm_config, dict):
        model_type = llm_config.get("model_type", "")
        archs = llm_config.get("architectures") or []
        num_experts = llm_config.get("num_experts", 0) or 0
    else:
        model_type = getattr(llm_config, "model_type", "")
        archs = getattr(llm_config, "architectures", None) or []
        num_experts = getattr(llm_config, "num_experts", 0) or 0

    if isinstance(model_type, str) and "moe" in model_type.lower():
        return True
    for arch in archs:
        arch_str = str(arch)
        if "Moe" in arch_str or "MoE" in arch_str:
            return True
    try:
        return int(num_experts) > 1
    except (TypeError, ValueError):
        return False


# Adapted from: https://github.com/OpenSenseNova/SenseNova-U1/blob/main/src/sensenova_u1/models/neo_unify/configuration_neo_chat.py
class SenseNovaU1LLMConfig(Qwen3Config):
    """Qwen3-based dense LLM backbone config with 3D RoPE extensions."""

    model_type = "sensenova_u1_llm"

    def __init__(
        self,
        rope_theta: float = 10000.0,
        rope_theta_hw: float = 10000.0,
        max_position_embeddings_hw: int = 10000,
        **kwargs,
    ):
        self.rope_theta = rope_theta
        self.rope_theta_hw = rope_theta_hw
        self.max_position_embeddings_hw = max_position_embeddings_hw
        super().__init__(**kwargs)


class SenseNovaU1MoELLMConfig(Qwen3MoeConfig):
    """Qwen3-MoE backbone used by SenseNova-U1-A3B.

    Every decoder layer carries two parallel sparse MoE blocks:

    * ``mlp`` — understanding path (``num_experts`` / ``num_experts_per_tok`` /
      ``moe_intermediate_size``)
    * ``mlp_mot_gen`` — image-generation path (``gen_num_experts`` etc.)

    Each gen-path knob falls back to its understanding-path counterpart when
    unset, so vanilla single-MoE configs keep working.
    """

    model_type = "sensenova_u1_moe_llm"

    def __init__(
        self,
        rope_theta_hw: float = 10000.0,
        max_position_embeddings_hw: int = 10000,
        gen_num_experts: int | None = None,
        gen_num_experts_per_tok: int | None = None,
        gen_moe_intermediate_size: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        _restore_legacy_rope_theta(self)
        self.rope_theta_hw = rope_theta_hw
        self.max_position_embeddings_hw = max_position_embeddings_hw
        self.gen_num_experts = int(gen_num_experts) if gen_num_experts is not None else int(self.num_experts)
        self.gen_num_experts_per_tok = (
            int(gen_num_experts_per_tok) if gen_num_experts_per_tok is not None else int(self.num_experts_per_tok)
        )
        self.gen_moe_intermediate_size = (
            int(gen_moe_intermediate_size) if gen_moe_intermediate_size is not None else int(self.moe_intermediate_size)
        )
        _backfill_layer_types(self)


def _build_llm_config(llm_config):
    """Instantiate the right LLM config object from a dict or pre-built config."""
    if isinstance(llm_config, dict):
        if _is_moe_llm_config(llm_config):
            return SenseNovaU1MoELLMConfig(**llm_config)
        return SenseNovaU1LLMConfig(**llm_config)
    return llm_config


# Adapted from https://github.com/OpenSenseNova/SenseNova-U1/blob/main/src/sensenova_u1/models/neo_unify/configuration_neo_vit.py#L10
class SenseNovaU1VisionConfig(PretrainedConfig):
    """Vision embedding config (2D RoPE + conv patch embed, no transformer)."""

    model_type = "sensenova_u1_vision"

    def __init__(
        self,
        num_channels: int = 3,
        patch_size: int = 16,
        hidden_size: int = 1024,
        llm_hidden_size: list[int] | int | None = None,
        downsample_ratio: list[float] | float | None = None,
        rope_theta_vision: float = 10000.0,
        max_position_embeddings_vision: int = 10000,
        **kwargs,
    ):
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.llm_hidden_size = llm_hidden_size if llm_hidden_size is not None else [_SENSENOVA_U1_LLM_HIDDEN_DEFAULT]
        self.downsample_ratio = downsample_ratio if downsample_ratio is not None else [0.5]
        self.rope_theta_vision = rope_theta_vision
        self.max_position_embeddings_vision = max_position_embeddings_vision
        super().__init__(**kwargs)


# Adapted from https://github.com/OpenSenseNova/SenseNova-U1/blob/main/training/sensenovavl/model/sensenovavl_moe_chat/configuration_sensenovavl_chat.py#L17
class SenseNovaU1Config(PretrainedConfig):
    """Top-level composite config for SenseNova-U1.

    Nests ``llm_config`` and ``vision_config`` sub-configs alongside the
    flow-matching / diffusion parameters.  When constructed from a dict
    (e.g. via ``from_pretrained``), sub-dicts are automatically promoted
    to their typed config objects. Dense 8B checkpoints become
    :class:`SenseNovaU1LLMConfig`; A3B MoE checkpoints become
    :class:`SenseNovaU1MoELLMConfig`.
    """

    model_type = "sensenova_u1"

    def __init__(
        self,
        llm_config: dict | SenseNovaU1LLMConfig | SenseNovaU1MoELLMConfig | None = None,
        vision_config: dict | SenseNovaU1VisionConfig | None = None,
        downsample_ratio: float = 0.5,
        template: str = "neo1_0",
        fm_head_layers: int = 2,
        fm_head_dim: int = 4096,
        fm_head_mlp_ratio: float = 1.0,
        use_pixel_head: bool = False,
        noise_scale: float = 1.0,
        noise_scale_mode: str = "none",
        noise_scale_base_image_seq_len: int = 256,
        noise_scale_max_value: float = 10.0,
        add_noise_scale_embedding: bool = False,
        time_schedule: str = "standard",
        time_shift_type: str = "exponential",
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        base_image_seq_len: int = 256,
        max_image_seq_len: int = 4096,
        concat_time_token_num: int = 0,
        t_eps: float = 0.02,
        **kwargs,
    ):
        self.llm_config = _build_llm_config(llm_config) if llm_config is not None else SenseNovaU1LLMConfig()

        if isinstance(vision_config, dict):
            vision_config = SenseNovaU1VisionConfig(**vision_config)
        self.vision_config = vision_config or SenseNovaU1VisionConfig()

        self.downsample_ratio = downsample_ratio
        self.template = template
        self.fm_head_layers = fm_head_layers
        self.fm_head_dim = fm_head_dim
        self.fm_head_mlp_ratio = fm_head_mlp_ratio
        self.use_pixel_head = use_pixel_head
        self.noise_scale = noise_scale
        self.noise_scale_mode = noise_scale_mode
        self.noise_scale_base_image_seq_len = noise_scale_base_image_seq_len
        self.noise_scale_max_value = noise_scale_max_value
        self.add_noise_scale_embedding = add_noise_scale_embedding
        self.time_schedule = time_schedule
        self.time_shift_type = time_shift_type
        self.base_shift = base_shift
        self.max_shift = max_shift
        self.base_image_seq_len = base_image_seq_len
        self.max_image_seq_len = max_image_seq_len
        self.concat_time_token_num = concat_time_token_num
        self.t_eps = t_eps
        super().__init__(**kwargs)


AutoConfig.register("sensenova_u1", SenseNovaU1Config)

__all__ = [
    "SenseNovaU1Config",
    "SenseNovaU1LLMConfig",
    "SenseNovaU1MoELLMConfig",
    "SenseNovaU1VisionConfig",
]
