# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import PretrainedConfig

from . import register_model_config


def _default_diffusion_model_cfg() -> dict:
    return {
        "positional_embeddings": None,
        "num_layers": 16,
        "num_attention_heads": 32,
        "attention_head_dim": 48,
        "norm_type": "ada_norm",
        "output_dim": 1024,
        "interleave_self_attention": True,
    }


class Gr00tN1d7Config(PretrainedConfig):
    model_type = "Gr00tN1d7"

    def __init__(
        self,
        model_dtype: str = "bfloat16",
        model_name: str = "nvidia/Cosmos-Reason2-2B",
        backbone_model_type: str = "qwen",
        model_revision: str | None = None,
        backbone_embedding_dim: int = 2048,
        select_layer: int = 12,
        reproject_vision: bool = False,
        use_flash_attention: bool = True,
        load_bf16: bool = False,
        image_crop_size: tuple | None = (230, 230),
        image_target_size: tuple | None = (256, 256),
        shortest_image_edge: int | None = None,
        crop_fraction: float | None = None,
        random_rotation_angle: int | None = None,
        color_jitter_params: dict | None = None,
        formalize_language: bool = True,
        apply_sincos_state_encoding: bool = False,
        use_percentiles: bool = True,
        use_relative_action: bool = False,
        max_state_dim: int = 132,
        max_action_dim: int = 132,
        action_horizon: int = 40,
        hidden_size: int = 1024,
        input_embedding_dim: int = 1536,
        state_history_length: int = 1,
        add_pos_embed: bool = True,
        attn_dropout: float = 0.2,
        use_vlln: bool = True,
        max_seq_len: int = 1024,
        use_alternate_vl_dit: bool = True,
        attend_text_every_n_blocks: int = 2,
        diffusion_model_cfg: dict | None = None,
        vl_self_attention_cfg: dict | None = None,
        num_inference_timesteps: int = 4,
        num_timestep_buckets: int = 1000,
        exclude_state: bool = False,
        use_mean_std: bool = False,
        max_num_embodiments: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_dtype = model_dtype
        self.model_name = model_name
        self.backbone_model_type = backbone_model_type
        self.model_revision = model_revision
        self.backbone_embedding_dim = backbone_embedding_dim
        self.select_layer = select_layer
        self.reproject_vision = reproject_vision
        self.use_flash_attention = use_flash_attention
        self.load_bf16 = load_bf16
        self.image_crop_size = image_crop_size
        self.image_target_size = image_target_size
        self.shortest_image_edge = shortest_image_edge
        self.crop_fraction = crop_fraction
        self.random_rotation_angle = random_rotation_angle
        self.color_jitter_params = color_jitter_params
        self.formalize_language = formalize_language
        self.apply_sincos_state_encoding = apply_sincos_state_encoding
        self.use_percentiles = use_percentiles
        self.use_relative_action = use_relative_action
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.action_horizon = action_horizon
        self.hidden_size = hidden_size
        self.input_embedding_dim = input_embedding_dim
        self.state_history_length = state_history_length
        self.add_pos_embed = add_pos_embed
        self.attn_dropout = attn_dropout
        self.use_vlln = use_vlln
        self.max_seq_len = max_seq_len
        self.use_alternate_vl_dit = use_alternate_vl_dit
        self.attend_text_every_n_blocks = attend_text_every_n_blocks
        self.diffusion_model_cfg = (
            diffusion_model_cfg if diffusion_model_cfg is not None else _default_diffusion_model_cfg()
        )
        self.vl_self_attention_cfg = vl_self_attention_cfg
        self.num_inference_timesteps = num_inference_timesteps
        self.num_timestep_buckets = num_timestep_buckets
        self.exclude_state = exclude_state
        self.use_mean_std = use_mean_std
        self.max_num_embodiments = max_num_embodiments


register_model_config("Gr00tN1d7", Gr00tN1d7Config)
