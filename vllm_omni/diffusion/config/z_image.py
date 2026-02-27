# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_omni.diffusion.config.base import BaseDiTConfig


class ZImageTransformer2DModelConfig(BaseDiTConfig):
    """
    A pretrainedConfig for z-image DiT models.

    Ref for default values: https://github.com/huggingface/diffusers/blob/v0.36.0/src/diffusers/models/transformers/transformer_z_image.py#L315
    """  # noqa: E501

    # Expected _class_name in Diffusers
    _class_name = "ZImageTransformer2DModel"

    def __init__(
        self,
        all_patch_size: tuple[int, ...] = (2,),
        all_f_patch_size: tuple[int, ...] = (1,),
        in_channels: int = 16,
        dim: int = 3840,
        n_layers: int = 30,
        n_refiner_layers: int = 2,
        n_heads: int = 30,
        n_kv_heads: int = 30,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
        cap_feat_dim: int = 2560,
        rope_theta: float = 256.0,
        t_scale: float = 1000.0,
        axes_dims: list[int] = [32, 48, 48],
        axes_lens: list[int] = [1024, 512, 512],
        **kwargs,
    ):
        super().__init__()
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.in_channels = in_channels
        self.dim = dim
        self.n_layers = n_layers
        self.n_refiner_layers = n_refiner_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.norm_eps = norm_eps
        self.qk_norm = qk_norm
        self.cap_feat_dim = cap_feat_dim
        self.rope_theta = rope_theta
        self.t_scale = t_scale
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
