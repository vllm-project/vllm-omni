from typing import List, Optional

from dataclasses import dataclass


@dataclass
class FrequencyEmbedding:
    num_freqs: int
    max_freq_log2: float
    min_freq_log2: float = 0.
    log_sampling: bool = True
    include_input: bool = True


@dataclass
class PreBlock:
    input_dim: int
    ffn_hidden_dim: int
    model_dim: int
    out_dim: int
    time_embed_dim: int
    num_ffn_layers: int
    embed_args: FrequencyEmbedding


@dataclass
class PostBlock:
    input_dim: int
    ffn_hidden_dim: int
    model_dim: int
    time_embed_dim: int
    num_ffn_layers: int
    context_dim: int
    context_group_dim: int


@dataclass
class SeqUNet:
    in_channels: int
    model_channels: int
    out_channels: int
    num_res_blocks: int
    attention_resolutions: List[int]
    dropout: float = 0.0
    channel_mult: List[int] = (1, 2, 4, 8)
    conv_resample: bool = True
    use_checkpoint: bool = False
    use_fp16: bool = False
    num_heads: int = -1
    num_head_channels: int = -1
    num_heads_upsample: int = -1
    use_scale_shift_norm: bool = False
    resblock_updown: bool = False
    use_new_attention_order: bool = False
    use_spatial_transformer: bool = False    # custom transformer support
    transformer_depth: int = 1              # custom transformer support
    context_dim: Optional[int] = None                 # custom transformer support
    legacy: bool = True
    norm_type: str = 'gn'
    attention_norm_type: str = 'gn'
    gn_groups: int = 32
    ff_mult: int = 4
    gated_ff: bool = True
    trfm_proj_input: bool = True
    timestep_transform: Optional[str] = None
    pre_block_config: Optional[PreBlock] = None
    post_block_config: Optional[PostBlock] = None
    condition_mode: Optional[str] = None


@dataclass
class WaveGrad:
    signal_size: int
    cond_size: int
    hidden_size: int


@dataclass
class Diffusion:
    conditioning_key: Optional[str] = None
    condition_dropout: float = 0.0
    upsampling_mode: Optional[str] = None
    scale_factor: float = 1.0
    scale_by_std: bool = False
    beta_schedule: str = "linear"
    linear_start: float = 1e-4
    linear_end: float = 2e-2
    loss_type: str = "l2"
    parameterization: str = "eps"


@dataclass
class KarrasDiffusion:
    sigma_data: float = 1.
    sigma_min: float = 1e-2
    sigma_max: float = 80.0
    sigma_dist_type: str = 'lognormal'
    sigma_dist_mean: float = -1.2
    sigma_dist_std: float = 1.2
    parameterization: str = 'hybrid'
    loss_type: str = "l2"


@dataclass
class Sampler:
    type: str = 'plms'
    steps: int = 10
    guidance_scale: float = 1.0
