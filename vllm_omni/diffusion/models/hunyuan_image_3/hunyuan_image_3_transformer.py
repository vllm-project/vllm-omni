# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import inspect
import logging
import math
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import regex as re
import torch
import torch.nn.functional as F
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils.outputs import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from PIL import Image
from safetensors.torch import load_file
from torch import nn
from torchvision import transforms
from transformers import PretrainedConfig, Siglip2ImageProcessorFast
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from vllm.config import CacheConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    is_pp_missing_parameter,
    make_layers,
)
from vllm.v1.attention.backend import AttentionType

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.distributed.parallel_state import get_pp_group
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.layers.rope import RotaryEmbedding

logger = logging.getLogger(__name__)


def _is_moe(config: PretrainedConfig) -> bool:
    num_experts = getattr(config, "num_experts", None)
    if isinstance(num_experts, int):
        return num_experts > 1
    if isinstance(num_experts, list) and num_experts:
        # Ensure all elements are integers before calling max.
        if all(isinstance(e, int) for e in num_experts):
            return max(num_experts) > 1
        else:
            return False
    return False


def _get_cla_factor(config: PretrainedConfig) -> int:
    if not getattr(config, "use_cla", False):
        return 1
    return getattr(config, "cla_share_factor", 1)


def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
) -> tuple[list[int], int]:
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def real_batched_index_select(t, dim, idx):
    """index_select for batched index and batched t"""
    assert t.ndim >= 2 and idx.ndim >= 2, f"{t.ndim=} {idx.ndim=}"
    assert len(t) == len(idx), f"{len(t)=} != {len(idx)=}"
    return torch.stack([torch.index_select(t[i], dim - 1, idx[i]) for i in range(len(t))])


def conv_nd(dims, *args, **kwargs):  # noqa: N802
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def normalization(channels, **kwargs):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: a nn.Module for normalization.
    """
    return nn.GroupNorm(32, channels, **kwargs)


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def _to_tuple(x, dim=2):
    if isinstance(x, int):
        return (x,) * dim
    elif len(x) == dim:
        return x
    else:
        raise ValueError(f"Expected length {dim} or int, but got {x}")


def get_meshgrid_nd(start, *args, dim=2):  # noqa: N802
    if len(args) == 0:
        # start is grid_size
        num = _to_tuple(start, dim=dim)
        start = (0,) * dim
        stop = num
    elif len(args) == 1:
        # start is start, args[0] is stop, step is 1
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = [stop[i] - start[i] for i in range(dim)]
        # assert num are all integers
        num_int = [int(x) for x in num]
        assert (torch.tensor(num) == torch.tensor(num_int)).all(), f"num should be int, but got {num}"
        num = num_int
    elif len(args) == 2:
        # start is start, args[0] is stop, args[1] is num
        start = _to_tuple(start, dim=dim)  # Left-Top       eg: 12,0
        stop = _to_tuple(args[0], dim=dim)  # Right-Bottom   eg: 20,32
        num = _to_tuple(args[1], dim=dim)  # Target Size    eg: 32,124
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    # PyTorch implement of np.linspace(start[i], stop[i], num[i], endpoint=False)
    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = torch.linspace(a, b, n + 1, dtype=torch.float32)[:n]
        axis_grid.append(g)
    grid = torch.meshgrid(*axis_grid, indexing="ij")  # dim x [H, W]
    grid = torch.stack(grid, dim=0)  # [dim, H, W]

    return grid


def build_2d_rope(
    seq_len: int,
    n_elem: int,
    image_infos: list[tuple[slice, tuple[int, int]]] | None = None,
    device: torch.device | None = None,
    base: int = 10000,
    base_rescale_factor: float = 1.0,
    return_all_pos: bool = False,
):
    assert n_elem % 4 == 0, f"n_elem must be divisible by 4, but got {n_elem}."

    # theta
    if base_rescale_factor != 1.0:
        base *= base_rescale_factor ** (n_elem / (n_elem - 2))
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))
    theta = theta.reshape(1, n_elem // 4, 2)  # [1, half_d, 2]

    # position indices
    if image_infos is None:
        image_infos = []

    image_infos_list = [image_infos]
    sample_seq_lens = [seq_len]

    # Prepare position indices for each sample
    x_sections = []
    y_sections = []
    for sample_id, sample_image_infos in enumerate(image_infos_list):
        last_pos = 0
        for sec_slice, (h, w) in sample_image_infos:
            L = sec_slice.start  # start from 0, so image_slice.start is just L
            # previous text
            if last_pos < L:
                y_sections.append(torch.arange(last_pos, L))
                x_sections.append(torch.arange(last_pos, L))
            elif h is None:
                # Interleave data has overlapped positions for <boi> <size> <ratio> <timestep> <eoi> tokens.
                y_sections.append(torch.arange(sec_slice.start, sec_slice.stop))
                x_sections.append(torch.arange(sec_slice.start, sec_slice.stop))
                continue
            else:
                # Interleave data has overlapped positions for noised image and the successive clean image,
                # leading to last_pos (= last text end L + noise w * h) > L (last text end L).
                pass
            # current image
            beta_y = L + (w * h - h) / 2
            beta_x = L + (w * h - w) / 2
            grid = get_meshgrid_nd((beta_y, beta_x), (beta_y + h, beta_x + w))  # [2, h, w] # noqa: N802
            grid = grid.reshape(2, -1)  # (y, x)
            y_sections.append(grid[0])
            x_sections.append(grid[1])
            # step
            last_pos = L + w * h
        # final text
        y_sections.append(torch.arange(last_pos, sample_seq_lens[sample_id]))
        x_sections.append(torch.arange(last_pos, sample_seq_lens[sample_id]))

    x_pos = torch.cat(x_sections).long()
    y_pos = torch.cat(y_sections).long()
    # If there are overlap positions, we need to remove them.
    x_pos = x_pos[:seq_len]
    y_pos = y_pos[:seq_len]
    all_pos = torch.stack((y_pos, x_pos), dim=1).unsqueeze(1).to(device)  # [seq_len, 1, 2]

    # calc rope
    idx_theta = (all_pos * theta).reshape(all_pos.shape[0], n_elem // 2).repeat(1, 1)

    cos = torch.cos(idx_theta)
    sin = torch.sin(idx_theta)

    if return_all_pos:
        return cos, sin, all_pos

    return cos, sin


def build_batch_2d_rope(
    seq_len: int,
    n_elem: int,
    image_infos: list[list[tuple[slice, tuple[int, int]]]] | None = None,
    device: torch.device | None = None,
    base: int = 10000,
    base_rescale_factor: float = 1.0,
    return_all_pos: bool = False,
):
    cos_list, sin_list, all_pos_list = [], [], []
    if image_infos is None:
        image_infos = [None]
    for i, image_info in enumerate(image_infos):
        res = build_2d_rope(
            seq_len,
            n_elem,
            image_infos=image_info,
            device=device,
            base=base,
            base_rescale_factor=base_rescale_factor,
            return_all_pos=return_all_pos,
        )
        if isinstance(res, tuple) and len(res) == 3:
            cos, sin, all_pos = res
        elif isinstance(res, tuple) and len(res) == 2:
            cos, sin = res
            all_pos = None
        else:
            raise ValueError(
                "build_2d_rope must return a tuple of length 2 or 3 "
                f"when return_all_pos={return_all_pos}, got: {type(res)} with length "
                f"{len(res) if isinstance(res, tuple) else 'N/A'}"
            )
        cos_list.append(cos)
        sin_list.append(sin)
        all_pos_list.append(all_pos)
    stacked_cos = torch.stack(cos_list, dim=0)
    stacked_sin = torch.stack(sin_list, dim=0)
    if return_all_pos:
        return stacked_cos, stacked_sin, all_pos_list

    return stacked_cos, stacked_sin


def get_full_state_dict(model_path):
    files = glob.glob(os.path.join(model_path, "*.safetensors"))
    full_sd = {}
    for f in files:
        full_sd.update(load_file(f))
    return full_sd


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=None, unsqueeze_dim=1, mla=False
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`):
            The query tensor.
        k (`torch.Tensor`):
            The key tensor.
        cos (`torch.Tensor`):
            The cosine part of the rotary embedding.
        sin (`torch.Tensor`):
            The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.

    Returns:
        `Tuple[torch.Tensor, torch.Tensor]`:
            A tuple comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    if position_ids is not None:
        cos = cos[position_ids]
        sin = sin[position_ids]

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    if mla:
        b, h, s, d = q.shape
        q = q.reshape(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

        b, h, s, d = k.shape
        k = k.reshape(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def default(value, default_value):
    return value if value is not None else default_value


class Resolution:
    def __init__(self, size, *args):
        if isinstance(size, str):
            if "x" in size:
                size = size.split("x")
                size = (int(size[0]), int(size[1]))
            else:
                size = int(size)
        if len(args) > 0:
            size = (size, args[0])
        if isinstance(size, int):
            size = (size, size)

        self.h = self.height = size[0]
        self.w = self.width = size[1]
        self.r = self.ratio = self.height / self.width

    def __getitem__(self, idx):
        if idx == 0:
            return self.h
        elif idx == 1:
            return self.w
        else:
            raise IndexError(f"Index {idx} out of range")

    def __str__(self):
        return f"{self.h}x{self.w}"


class ResolutionGroup:
    def __init__(self, base_size=None, step=None, align=1):
        self.align = align
        self.base_size = base_size
        assert base_size % align == 0, f"base_size {base_size} is not divisible by align {align}"
        if base_size is not None and not isinstance(base_size, int):
            raise ValueError(f"base_size must be None or int, but got {type(base_size)}")
        if step is None:
            step = base_size // 16
        if step is not None and step > base_size // 2:
            raise ValueError(f"step must be smaller than base_size // 2, but got {step} > {base_size // 2}")

        self.step = step
        self.data = self._calc_by_step()

        self.ratio = np.array([x.ratio for x in self.data])
        self.attr = ["" for _ in range(len(self.data))]
        self.prefix_space = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __repr__(self):
        prefix = self.prefix_space * " "
        prefix_close = (self.prefix_space - 4) * " "
        res_str = f"ResolutionGroup(base_size={self.base_size}, step={self.step}, data="
        attr_maxlen = max([len(x) for x in self.attr] + [5])
        res_str += (
            f"\n{prefix}ID: height width   ratio {' ' * max(0, attr_maxlen - 4)}count  h/16 w/16    tokens\n{prefix}"
        )
        res_str += ("\n" + prefix).join(
            [
                f"{i:2d}: ({x.h:4d}, {x.w:4d})  {self.ratio[i]:.4f}  {self.attr[i]:>{attr_maxlen}s}  "
                f"({x.h // 16:3d}, {x.w // 16:3d})  {x.h // 16 * x.w // 16:6d}"
                for i, x in enumerate(self.data)
            ]
        )
        res_str += f"\n{prefix_close})"
        return res_str

    def _calc_by_step(self):
        assert self.align <= self.step, f"align {self.align} must be smaller than step {self.step}"

        min_height = self.base_size // 2
        min_width = self.base_size // 2
        max_height = self.base_size * 2
        max_width = self.base_size * 2

        resolutions = [Resolution(self.base_size, self.base_size)]

        cur_height, cur_width = self.base_size, self.base_size
        while True:
            if cur_height >= max_height and cur_width <= min_width:
                break

            cur_height = min(cur_height + self.step, max_height)
            cur_width = max(cur_width - self.step, min_width)
            resolutions.append(Resolution(cur_height // self.align * self.align, cur_width // self.align * self.align))

        cur_height, cur_width = self.base_size, self.base_size
        while True:
            if cur_height <= min_height and cur_width >= max_width:
                break

            cur_height = max(cur_height - self.step, min_height)
            cur_width = min(cur_width + self.step, max_width)
            resolutions.append(Resolution(cur_height // self.align * self.align, cur_width // self.align * self.align))

        resolutions = sorted(resolutions, key=lambda x: x.ratio)

        return resolutions

    def get_target_size(self, width, height):
        ratio = height / width
        idx = np.argmin(np.abs(self.ratio - ratio))
        reso = self.data[idx]
        return reso.w, reso.h

    def get_base_size_and_ratio_index(self, width, height):
        ratio = height / width
        idx = np.argmin(np.abs(self.ratio - ratio))
        return self.base_size, idx


class ImageInfo:
    """Class to store image information for processing and generation."""

    def __init__(
        self,
        image_type: str = None,
        image_tensor: torch.Tensor = None,
        image_width: int = None,
        image_height: int = None,
        token_width: int = None,
        token_height: int = None,
        image_token_length: int = None,
        base_size: int = None,
        ratio_index: int = None,
        **kwargs,
    ):
        self.image_type = image_type
        self.image_tensor = image_tensor
        self.image_width = image_width
        self.w = image_width
        self.image_height = image_height
        self.h = image_height
        self.token_width = token_width
        self.tk_w = token_width
        self.token_height = token_height
        self.tk_h = token_height
        self.image_token_length = default(
            image_token_length,
            token_width * token_height if token_width is not None and token_height is not None else None,
        )
        self.base_size = base_size
        self.ratio_index = ratio_index

        self.add_timestep_token = kwargs.get("add_timestep_token", True)
        self.add_guidance_token = kwargs.get("add_guidance_token", False)
        self.use_front_boi_token = kwargs.get("use_front_boi_token", True)
        self.add_image_shape_token = kwargs.get("add_image_shape_token", True)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access to attributes."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key '{key}' not found in ImageInfo")

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-like assignment to attributes."""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"Key '{key}' not found in ImageInfo")

    def __contains__(self, key: str) -> bool:
        """Check if the key exists in the ImageInfo object."""
        return hasattr(self, key)

    def __repr__(self):
        return (
            f"ImageInfo(image_type={self.image_type}, image_tensor={self.image_tensor}, "
            f"image_width={self.image_width}, image_height={self.image_height}, "
            f"token_width={self.token_width}, token_height={self.token_height}, "
            f"image_token_length={self.image_token_length}, "
            f"base_size={self.base_size}, ratio_index={self.ratio_index}"
        )

    @property
    def meta_info(self):
        # Used for image sections of tkwrapper.encode_general()
        if self.image_type in ["vae", "gen_image"]:
            return dict(
                token_length=self.image_token_length,
                add_timestep_token=self.add_timestep_token,
                add_guidance_token=self.add_guidance_token,
                use_front_boi_token=self.use_front_boi_token,
                add_image_shape_token=self.add_image_shape_token,
                base_size=self.base_size,
                ratio_idx=self.ratio_index,
                # for rope 2d
                token_height=self.token_height,
                token_width=self.token_width,
                # for bc
                image_height=self.image_height,
                image_width=self.image_width,
            )
        else:
            raise ValueError(f"Unknown image type '{self.image_type}'")


class ImageTensor(torch.Tensor):
    # This class is just for type hinting purposes. Attribute `i` should be defined
    # as an instance attribute of the torch.Tensor instance, like: tensor.i = ImageInfo(...)
    i: ImageInfo
    vision_encoder_kwargs: dict


class JointImageInfo:
    def __init__(self, vae_image_info: ImageInfo, vision_image_info: ImageInfo, vision_encoder_kwargs: dict = None):
        self.vae_image_info = vae_image_info
        self.vision_image_info = vision_image_info
        self.vision_encoder_kwargs = vision_encoder_kwargs

        # Define key attributes to align with ImageInfo for uniformity
        self.image_type = "joint_image"
        self.image_token_length = vae_image_info.image_token_length + vision_image_info.image_token_length

        self.add_timestep_token = vae_image_info.add_timestep_token
        self.use_front_boi_token = vae_image_info.use_front_boi_token
        self.add_image_shape_token = vae_image_info.add_image_shape_token

    def __repr__(self):
        return f"JointImageInfo(vae_image={self.vae_image_info}, vision_image={self.vision_image_info})"

    @property
    def meta_info(self):
        # Used for image sections of tkwrapper.encode_general()
        return dict(
            token_length=[self.vae_image_info.image_token_length, self.vision_image_info.image_token_length],
            add_timestep_token=self.add_timestep_token,
            use_front_boi_token=self.use_front_boi_token,
            add_image_shape_token=self.add_image_shape_token,
            base_size=self.vae_image_info.base_size,
            ratio_idx=self.vae_image_info.ratio_index,
            # for rope 2d
            token_height=[self.vae_image_info.token_height, self.vision_image_info.token_height],
            token_width=[self.vae_image_info.token_width, self.vision_image_info.token_width],
            # for bc
            image_height=[self.vae_image_info.image_height, self.vision_image_info.image_height],
            image_width=[self.vae_image_info.image_width, self.vision_image_info.image_width],
        )

    @property
    def num_special_tokens(self):
        return (
            2  # <boi> + <eoi>
            + (1 if self.add_timestep_token else 0)
            + (2 if self.add_image_shape_token else 0)
            + 1  # <joint_image_sep>
        )

    def copy(self, copy_image_tensor=True):
        if copy_image_tensor and (
            self.vae_image_info.image_tensor is None or self.vision_image_info.image_tensor is None
        ):
            raise ValueError("image_tensor is None, cannot copy")
        return JointImageInfo(
            self.vae_image_info.copy(copy_image_tensor),
            self.vision_image_info.copy(copy_image_tensor),
            self.vision_encoder_kwargs,
        )

    def zeros_(self):
        self.vae_image_info.zeros_()
        self.vision_image_info.zeros_()


class JointImage:
    def __init__(self, vae_image: ImageTensor, vision_image: ImageTensor):
        self.vae_image = vae_image
        self.vision_image = vision_image
        self.i = JointImageInfo(vae_image.i, vision_image.i)


class Config:
    def __init__(self, config):
        if config is not None:
            for key, value in config.items():
                setattr(self, key, value)


class LightProjector(nn.Module):
    def __init__(self, config):
        config = Config(config)
        super().__init__()

        if config.projector_type == "linear":
            modules = nn.Linear(config.input_dim, config.n_embed)

        elif config.projector_type == "mlp_gelu":
            modules = [nn.Linear(config.input_dim, config.n_embed)]
            for _ in range(1, config.depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.n_embed, config.n_embed))
            modules = nn.Sequential(*modules)

        else:
            raise ValueError(f"Unknown projector type: {config.projector_type}")

        self.layers = modules


class HunYuanRotary2DEmbedder:
    r"""
    A RoPE wrapper specifically designed for HunYuan-Image attention.

    This class implements Rotary Position Embedding (RoPE) specifically optimized for
    the HunYuan-Image model's attention mechanism. It handles the application of rotary
    embeddings to query and key tensors with support for custom position embeddings
    and attention metadata.

    Example:
        ```python
        embedder = HunYuanRotaryEmbedder(num_heads=num_h, num_kv_heads=num_kv, head_dim=h_d)
        q, k = embedder(q, k, hidden_states, custom_pos_emb, first_step, attn_meta)
        ```

    Args:
        num_heads (`int`):
            Number of attention heads in the model.
        num_kv_heads (`int`):
            Number of key-value heads (for grouped-query attention).
        head_dim (`int`):
            Dimension of each attention head.

    Methods:
        __call__: Applies rotary position embedding to query and key tensors.
    """

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.custom_pos_emb: tuple[torch.Tensor, torch.Tensor] | None = None
        self.rope = RotaryEmbedding(is_neox_style=True)

    def _prepare_cos_sin(
        self,
        custom_pos_emb: tuple[torch.Tensor, torch.Tensor],
        first_step: bool,
        device: torch.device,
    ):
        """Returns cos/sin on the target device based on first_step and caching strategy."""
        if first_step:
            cos_input, sin_input = custom_pos_emb
            cos = cos_input.to(device)
            sin = sin_input.to(device)
            self.custom_pos_emb = None
        else:
            if self.custom_pos_emb is None:
                cos_input, sin_input = custom_pos_emb
                cos = cos_input.to(device)
                sin = sin_input.to(device)
                self.custom_pos_emb = (cos, sin)
            else:
                cos, sin = self.custom_pos_emb
        return cos, sin

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        hidden_states: torch.Tensor,
        custom_pos_emb: tuple[torch.Tensor, torch.Tensor],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states_shape[-1])
        if kwargs.get("mode", "gen_text") != "gen_image":
            return q, k

        first_step = kwargs.get("first_step", False)
        device = q.device
        # 1. Prepare cos/sin
        cos, sin = self._prepare_cos_sin(custom_pos_emb, first_step, device)

        # 2. Shape validation
        query_lens: list[int] = kwargs.get("query_lens")
        bs = len(query_lens)
        q_len = query_lens[0]
        assert hidden_states.shape[0] == bs * q_len, f"{hidden_states.shape[0]} != {bs * q_len}"

        # 3. Reshape + transpose for apply_rotary_pos_emb
        #    Assume q shape [B*L, H*D] -> [2, L, H, D] -> [2, H, L, D]
        q = q.reshape(bs, q_len, self.num_heads, self.head_dim)
        k = k.reshape(bs, q_len, self.num_kv_heads, self.head_dim)

        q = self.rope(q.to(torch.float32), cos, sin)
        k = self.rope(k.to(torch.float32), cos, sin)

        # 5. Restore original shape + convert to bfloat16
        q = q.reshape(hidden_states.shape[0], self.num_heads * self.head_dim).to(torch.bfloat16)
        k = k.reshape(hidden_states.shape[0], self.num_kv_heads * self.head_dim).to(torch.bfloat16)
        hidden_states = hidden_states.reshape(hidden_states_shape)
        return q, k


class ImageKVCacheManager:
    """
    Manages specialized caching and updating of KV-Cache for image tokens in multimodal models.
    """

    def __init__(self, image_token_len: int = 4097):
        """
        Args:
            image_token_len: Number of tokens per image (including special placeholders),
            default 4097 (timestamp + 4096 image tokens).
        """
        self.image_token_len: int = image_token_len
        self.image_kv_cache: tuple[torch.Tensor, torch.Tensor] = None

    def _save_image_kv_caches(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        seq_len: int,
    ) -> None:
        bs, q_len, num_kv_heads, head_dim = key.shape
        assert q_len == seq_len, f"for first-step, {q_len} != {seq_len}"

        key = key.reshape(-1, num_kv_heads, head_dim)
        value = value.reshape(-1, num_kv_heads, head_dim)

        cached_prompt_len = seq_len - self.image_token_len - 1
        cached_key = [key[:cached_prompt_len], key[seq_len - 1 : seq_len]]
        cached_value = [value[:cached_prompt_len], value[seq_len - 1 : seq_len]]

        if bs > 1:
            assert bs == 2, "for cfg case, bs must be 2"
            cached_key.append(key[seq_len : seq_len + cached_prompt_len])
            cached_key.append(key[-1:])

            cached_value.append(value[seq_len : seq_len + cached_prompt_len])
            cached_value.append(value[-1:])

        cached_key = torch.cat(cached_key, dim=0)
        cached_value = torch.cat(cached_value, dim=0)
        self.image_kv_cache_map = (cached_key, cached_value)

    def _update_image_kv_caches(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cached_key, cached_value = self.image_kv_cache_map
        bs, q_len, num_kv_heads, head_dim = key.shape

        cached_prompt_len = cached_key.shape[0] // bs - 1
        assert (cached_prompt_len + 1) == (seq_len - q_len), f"{cached_prompt_len + 1} != {seq_len - q_len}"

        key = key.reshape(-1, num_kv_heads, head_dim)
        value = value.reshape(-1, num_kv_heads, head_dim)

        new_key = [
            cached_key[:cached_prompt_len],
            key[:q_len],
            cached_key[cached_prompt_len : cached_prompt_len + 1],
        ]
        new_value = [
            cached_value[:cached_prompt_len],
            value[:q_len],
            cached_value[cached_prompt_len : cached_prompt_len + 1],
        ]

        if bs > 1:
            assert bs == 2, "for cfg case, bs must be 2"
            new_key.append(cached_key[cached_prompt_len + 1 : cached_prompt_len + 1 + cached_prompt_len])
            new_key.append(key[q_len:])
            new_key.append(cached_key[-1:])

            new_value.append(cached_value[cached_prompt_len + 1 : cached_prompt_len + 1 + cached_prompt_len])
            new_value.append(value[q_len:])
            new_value.append(cached_value[-1:])

        new_key = torch.cat(new_key, dim=0)
        new_value = torch.cat(new_value, dim=0)
        new_key = new_key.reshape(bs, seq_len, num_kv_heads, head_dim)
        new_value = new_value.reshape(bs, seq_len, num_kv_heads, head_dim)

        return new_key.contiguous(), new_value.contiguous()

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        self.image_token_len = kwargs.get("num_image_tokens")
        first_step = kwargs.get("first_step")

        query_lens = kwargs.get("query_lens")
        seq_lens = kwargs.get("seq_lens")
        bs = len(query_lens)
        q_len = query_lens[0]
        seq_len = seq_lens[0]
        assert query.shape[0] == bs * q_len, f"{query.shape[0]} != {bs * q_len}"

        head_num_per_rank = query.shape[1]
        kv_head_num_per_rank = key.shape[1]
        repeat_num = head_num_per_rank // kv_head_num_per_rank
        head_dim = query.shape[2]

        query = query.reshape(bs, q_len, head_num_per_rank, head_dim)
        key = key.reshape(bs, q_len, kv_head_num_per_rank, head_dim)
        value = value.reshape(bs, q_len, kv_head_num_per_rank, head_dim)

        if first_step:
            self.image_kv_cache_map = None
            self._save_image_kv_caches(key, value, seq_len)
        else:
            key, value = self._update_image_kv_caches(key, value, seq_len)

        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()

        key = repeat_kv(key, repeat_num)
        value = repeat_kv(value, repeat_num)

        attention_mask = attention_mask.contiguous()

        attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0)

        attn_output = attn_output.transpose(1, 2).contiguous()  # [bs, q_len, heads, head_dim]
        attn_output = attn_output.reshape(bs * q_len, head_num_per_rank, head_dim)
        return attn_output


@dataclass
class CausalMMOutputWithPast(CausalLMOutputWithPast):
    diffusion_prediction: torch.Tensor | None = None


class HunyuanImage3Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`HunyuanImage3Model`]. It is used to instantiate
    an Hunyuan model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Hunyuan-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Hunyuan Image 3 model. Defines the number of different tokens that can be
            represented by the `inputs_ids` passed when calling [`HunyuanImage3Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations or shared MLP representations.
        moe_intermediate_size (`int` or `List`, *optional*, defaults to 11008):
            Dimension of the MLP representations in MoE. Use a list if you want a different size per layer.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        use_qk_norm (`bool`, *optional*, defaults to `False`):
            Whether query and key in attention use norm
        use_cla (`bool`, *optional*, defaults to `False`):
            Whether to use CLA in attention
        cla_share_factor (`int`, *optional*, defaults to 1):
            The share factor of CLA
        num_experts (`int` or `List`, *optional*, defaults to 1):
            The number of experts for moe. If it is a list, it will be used as the number of experts for each layer.
        num_shared_expert (`int` or `List`, *optional*, defaults to 1):
            The number of shared experts for moe. If it is a list, it will be used as the number of shared experts
            for each layer.
        moe_topk (`int` or `List`, *optional*, defaults to 1):
            The topk value for moe. If it is a list, it will be used as the topk value for each layer.
        capacity_factor (Not used) (`float` or `List`, *optional*, defaults to 1.0):
            The capacity factor for moe. If it is a list, it will be used as the capacity factor for each layer.
        moe_layer_num_skipped (`int`, *optional*, defaults to 0):
            First moe_layer_num_skipped layers do not use MoE.
    """

    model_type = "Hunyuan"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=290943,
        hidden_size=4096,
        intermediate_size: int = 11008,
        moe_intermediate_size: int | list = None,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        attention_head_dim=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        eod_token_id=3,
        im_start_id=4,
        im_end_id=5,
        text_start_id=6,
        text_end_id=7,
        image_token_id=8,
        video_start_id=9,
        video_end_id=10,
        im_newline_id=11,
        mask_init_id=12,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
        use_qk_norm=False,
        use_rotary_pos_emb=True,
        use_cla=False,
        cla_share_factor=1,
        norm_type="hf_rms",
        num_experts: int | list = 1,
        use_mixed_mlp_moe=False,
        num_shared_expert: int | list = 1,
        moe_topk: int | list = 1,
        capacity_factor: float | list = 1.0,
        moe_drop_tokens=False,
        moe_random_routing_dropped_token=False,
        use_mla=False,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_nope_head_dim=128,
        moe_layer_num_skipped=0,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        group_limited_greedy=False,
        n_group=None,
        topk_group=None,
        add_classification_head=False,
        class_num=0,
        pool_type="last",
        pad_id=-1,
        # Added
        moe_impl="eager",
        vae_downsample_factor=(16, 16),  # (h, w)
        img_proj_type="unet",
        patch_size=1,
        patch_embed_hidden_dim=1024,
        image_base_size=1024,
        vae=None,
        vit=None,
        vit_processor=None,
        vit_aligner=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.moe_impl = moe_impl
        self.num_experts = num_experts
        self.use_mixed_mlp_moe = use_mixed_mlp_moe
        self.num_shared_expert = num_shared_expert
        self.moe_topk = moe_topk
        self.capacity_factor = capacity_factor
        self.moe_drop_tokens = moe_drop_tokens
        self.moe_random_routing_dropped_token = moe_random_routing_dropped_token

        if attention_head_dim is not None:
            self.attention_head_dim = attention_head_dim
        else:
            self.attention_head_dim = self.hidden_size // num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.attention_dropout = attention_dropout
        self.use_qk_norm = use_qk_norm
        self.use_rotary_pos_emb = use_rotary_pos_emb
        self.use_cla = use_cla
        self.cla_share_factor = cla_share_factor
        self.norm_type = norm_type
        # MLA args
        self.use_mla = use_mla
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim

        # DeepSeek related args
        self.moe_layer_num_skipped = moe_layer_num_skipped
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.group_limited_greedy = group_limited_greedy
        self.n_group = n_group
        self.topk_group = topk_group
        self.add_classification_head = add_classification_head
        self.class_num = class_num
        self.pool_type = pool_type
        self.pad_id = pad_id

        if self.class_num is not None:
            self.dense_list = [self.hidden_size, self.class_num]

        # ViT args
        self.vit = vit
        self.vit_processor = vit_processor
        self.vit_aligner = vit_aligner

        # Image Gen args
        self.vae = vae
        self.vae_downsample_factor = vae_downsample_factor
        self.img_proj_type = img_proj_type
        self.patch_size = patch_size
        self.patch_embed_hidden_dim = patch_embed_hidden_dim
        self.image_base_size = image_base_size

        # token id
        self.eod_token_id = eod_token_id
        self.im_start_id = im_start_id
        self.im_end_id = im_end_id
        self.text_start_id = text_start_id
        self.text_end_id = text_end_id
        self.image_token_id = image_token_id
        self.video_start_id = video_start_id
        self.video_end_id = video_end_id
        self.im_newline_id = im_newline_id
        self.mask_init_id = mask_init_id

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class HunyuanImage3ImageProcessor:
    def __init__(self, config):
        self.config = config

        self.reso_group = ResolutionGroup(base_size=config.image_base_size)
        self.vae_processor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),  # transform to [-1, 1]
            ]
        )
        self.vision_encoder_processor = Siglip2ImageProcessorFast.from_dict(config.vit_processor)

    def build_image_info(self, image_size):
        # parse image size (HxW, H:W, or <img_ratio_i>)
        if isinstance(image_size, str):
            if image_size.startswith("<img_ratio_"):
                ratio_index = int(image_size.split("_")[-1].rstrip(">"))
                reso = self.reso_group[ratio_index]
                image_size = reso.height, reso.width
            elif "x" in image_size:
                image_size = [int(s) for s in image_size.split("x")]
            elif ":" in image_size:
                image_size = [int(s) for s in image_size.split(":")]
            else:
                raise ValueError(
                    f"`image_size` should be in the format of 'HxW', 'H:W' or <img_ratio_i>, got {image_size}."
                )
            assert len(image_size) == 2, f"`image_size` should be in the format of 'HxW', got {image_size}."
        elif isinstance(image_size, (list, tuple)):
            assert len(image_size) == 2 and all(isinstance(s, int) for s in image_size), (
                f"`image_size` should be a tuple of two integers or a string in the format of 'HxW', got {image_size}."
            )
        else:
            raise ValueError(
                f"`image_size` should be a tuple of two integers or a string in the format of 'WxH', got {image_size}."
            )
        image_width, image_height = self.reso_group.get_target_size(image_size[1], image_size[0])
        token_height = image_height // (self.config.vae_downsample_factor[0] * self.config.patch_size)
        token_width = image_width // (self.config.vae_downsample_factor[1] * self.config.patch_size)
        base_size, ratio_idx = self.reso_group.get_base_size_and_ratio_index(image_size[1], image_size[0])
        image_info = ImageInfo(
            image_type="gen_image",
            image_width=image_width,
            image_height=image_height,
            token_width=token_width,
            token_height=token_height,
            base_size=base_size,
            ratio_index=ratio_idx,
        )
        return image_info


class HunYuanMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            reduce_results=reduce_results,
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class HunYuanSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        layer_id: int = -1,
        prefix: str = "",
        enable_eplb: bool = False,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.n_routed_experts = config.num_experts

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than the number of experts {config.num_experts}."
            )

        # Get layer_id topk if config.moe_topk is a list
        if isinstance(config.moe_topk, list):
            assert layer_id >= 0
            assert len(config.moe_topk) > layer_id
            top_k = config.moe_topk[layer_id]
        else:
            top_k = config.moe_topk

        # If it is moe, moe_intermediate_size is preferred
        intermediate_size = config.intermediate_size
        if config.moe_intermediate_size is not None:
            intermediate_size = (
                config.moe_intermediate_size
                if isinstance(config.moe_intermediate_size, int)
                else config.moe_intermediate_size[layer_id]
            )

        self.enable_eplb = False
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = 0

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        if config.use_mixed_mlp_moe > 0:
            # Get layer_id num_shared_expert if config.num_shared_expert is
            # a list.
            if isinstance(config.num_shared_expert, list):
                assert layer_id >= 0
                assert len(config.num_shared_expert) > layer_id
                num_shared_expert = config.num_shared_expert[layer_id]
            else:
                num_shared_expert = config.num_shared_expert

            self.shared_mlp = HunYuanMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size * num_shared_expert,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_mlp",
            )
        else:
            self.shared_mlp = None

        self.experts = HunyuanFusedMoE(
            shared_experts=self.shared_mlp,
            num_experts=self.n_routed_experts,
            top_k=top_k,
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            reduce_results=False,
            renormalize=top_k > 1,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            pcp_size=1,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=router_logits)
        if self.shared_mlp is not None:
            final_hidden_states = final_hidden_states[0] + final_hidden_states[1]

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(orig_shape)


class HunYuanAttention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: dict[str, Any] | None = None,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
        layer_id: int = -1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.num_key_value_heads = (
            config.num_key_value_heads // tp_size if config.num_key_value_heads else self.num_heads
        )
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        if hasattr(config, "head_dim") and config.head_dim:
            self.head_dim = config.head_dim
        elif hasattr(config, "attention_head_dim"):
            self.head_dim = config.attention_head_dim
        else:
            self.head_dim = self.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        self.layer_id = layer_id

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if rope_scaling is not None:
            # for t2t, fallback rotary_emb type from 'custom' to default.
            rope_scaling["rope_type"] = "default"
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_scaling,
            is_neox_style=True,
        )
        # which attention, from vllm or omni?
        # try use Atten from diffusion
        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            causal=False,
            softmax_scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            # cache_config=cache_config,
            # quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        # default image_token_len = timestamp + 4096*image_tokes
        self.image_attn = ImageKVCacheManager(image_token_len=4097)
        self.image_rope2d_emb = HunYuanRotary2DEmbedder(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )

        if self.use_qk_norm:
            self.query_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_states: tuple[torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        custom_pos_emb: tuple[torch.FloatTensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        past_key_value: Cache | None = kwargs.get("past_key_value", None)
        if past_key_value is not None:
            position_ids = kwargs.get("position_ids")
            key_states = k.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = v.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            cache_kwargs = {"cache_position": position_ids}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_id, cache_kwargs)
        # for image_generation
        if kwargs.get("mode", "gen_text") == "gen_image":
            # assert positions is None, "positions should be None for image attention"
            q, k = self.image_rope2d_emb(q, k, hidden_states, custom_pos_emb, **kwargs)
        else:
            q, k = self.rotary_emb(positions, q, k)
        if self.use_qk_norm:
            q = self.query_layernorm(q.view(-1, self.num_heads, self.head_dim).contiguous())
            k = self.key_layernorm(k.view(-1, self.num_kv_heads, self.head_dim).contiguous())
        # for image_generation
        if kwargs.get("mode", "gen_text") == "gen_image":
            attn_output = self.image_attn(q, k, v, attention_mask=attention_mask, **kwargs)
        else:
            attn_output = self.attn(q, k, v)
        # For o_proj
        attn_output = attn_output.view(q.shape[0], -1)
        output, _ = self.o_proj(attn_output)
        output = output.reshape(bsz, q_len, -1)
        return output, None, past_key_value


class HunyuanFusedMoE(SharedFusedMoE):
    def __init__(self, *, prefix: str = "", **kwargs):
        super().__init__(prefix=prefix, **kwargs)
        self._prefix = prefix

        self._init_hook_handle = self.register_forward_pre_hook(self._initialize_kernel_hook, with_kwargs=True)

    def _initialize_kernel_hook(self, module, args, kwargs):
        if self.quant_method:
            self.quant_method.process_weights_after_loading(self)
        self._init_hook_handle.remove()

    def forward(self, hidden_states, router_logits):
        return super().forward(hidden_states, router_logits)


class HunyuanImage3DecoderLayer(nn.Module):
    def __init__(self, config: HunyuanImage3Config, layer_idx: int, prefix: str = ""):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.intermediate_size = (
            config.intermediate_size
            if isinstance(config.intermediate_size, int)
            else config.intermediate_size[layer_idx]
        )
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = config.original_max_position_embeddings
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        attention_bias = getattr(config, "attention_bias", False) or getattr(config, "bias", False)

        cla_factor = _get_cla_factor(config)
        attention_type = (
            AttentionType.ENCODER_DECODER if layer_idx >= 0 and layer_idx % cla_factor != 0 else AttentionType.DECODER
        )
        if attention_type == AttentionType.DECODER:
            self.self_attn = HunYuanAttention(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                quant_config=None,
                bias=attention_bias,
                cache_config=None,
                prefix=f"{prefix}.self_attn",
                layer_id=layer_idx,
            )
        else:
            raise RuntimeError(f"Unsupported attention type: {attention_type}")

        if (
            (isinstance(config.num_experts, int) and config.num_experts > 1)
            or (isinstance(config.num_experts, list) and max(config.num_experts) > 1)
        ) and layer_idx >= config.moe_layer_num_skipped:
            self.mlp = HunYuanSparseMoeBlock(config, layer_id=layer_idx, prefix=f"{prefix}.mlp")
        else:
            self.mlp = HunYuanMLP(self.hidden_size, self.intermediate_size, config.hidden_act)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        custom_pos_emb: tuple[torch.FloatTensor] | None = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor | Any]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            position_ids (`torch.LongTensor`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            custom_pos_emb (`Tuple[torch.FloatTensor]`, *optional*): custom position embedding for rotary
                position embedding
        """
        if "padding_mask" in kwargs:
            logger.warning(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use "
                "`attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            custom_pos_emb=custom_pos_emb,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class HunyuanImage3PreTrainedModel(PreTrainedModel):
    config_class = HunyuanImage3Config
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _no_split_modules = ["HunyuanImage3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class HunyuanImage3Model(nn.Module):
    def __init__(self, config: HunyuanImage3Config, prefix: str = ""):
        super().__init__()
        quant_config = None
        lora_config = None
        self.num_redundant_experts = 0
        self.config = config
        self.device = get_local_device()
        self.quant_config = quant_config
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size * (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        if get_pp_group().is_first_rank or (config.tie_word_embeddings and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()
        # maybe should not use make_layers()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: HunyuanImage3DecoderLayer(
                config=config,
                layer_idx=int(prefix.split(".")[-1]),
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def _split_qkv_weight(self, qkv: torch.Tensor):
        num_attention_heads = self.config.num_attention_heads
        num_kv_heads = getattr(self.config, "num_key_value_heads", self.config.num_attention_heads)
        num_key_value_groups = num_attention_heads // num_kv_heads
        hidden_size = self.config.hidden_size

        if hasattr(self.config, "head_dim"):
            attention_head_dim = self.config.head_dim
        elif hasattr(self.config, "attention_head_dim"):
            attention_head_dim = self.config.attention_head_dim
        else:
            attention_head_dim = self.config.hidden_size // num_attention_heads

        qkv = qkv.reshape(num_kv_heads, num_key_value_groups + 2, attention_head_dim, hidden_size)
        q, k, v = torch.split(qkv, (num_key_value_groups, 1, 1), dim=1)
        q = q.reshape(-1, hidden_size)
        k = k.reshape(-1, hidden_size)
        v = v.reshape(-1, hidden_size)
        return torch.concat((q, k, v))

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        if _is_moe(self.config):
            # Params for weights, fp8 weight scales, fp8 activation scales
            # (param_name, weight_name, expert_id, shard_id)
            fused_moe_expert_mapping = HunyuanFusedMoE.make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.num_experts,
                num_redundant_experts=self.num_redundant_experts,
            )
            expert_weights_remapping = {
                "gate_proj": ("gate_and_up_proj", 1, 2),
                "up_proj": ("gate_and_up_proj", 0, 2),
            }
            return fused_moe_expert_mapping, expert_weights_remapping
        else:
            return [], {}

    # rename for delay load
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        cla_factor = _get_cla_factor(self.config)
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        num_attention_heads = self.config.num_attention_heads
        num_kv_heads = getattr(self.config, "num_key_value_heads", self.config.num_attention_heads)
        split_params_mapping = [
            (".gate_up_proj", ".gate_and_up_proj", 2, [(1, 1), (0, 1)], None),
            (
                ".qkv_proj",
                ".qkv_proj",
                num_attention_heads + num_kv_heads * 2,
                [("q", num_attention_heads), ("k", num_kv_heads), ("v", num_kv_heads)],
                self._split_qkv_weight,
            ),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping, expert_weights_remapping = self.get_expert_mapping()

        # List of unexpected keywords in weight names
        unexpected_keywords = [
            "vae",
            "vision_aligner",
            "vision_model",
            "final_layer",
            "patch_embed",
            "timestep_emb",
            "time_embed",
            "time_embed_2",
            "guidance_emb",
            "timestep_r_emb",
        ]

        def contains_unexpected_keyword(name, keywords):
            """Check if the name contains any unexpected keywords"""
            for keyword in keywords:
                if keyword in name:
                    return True
            return False

        for name, loaded_weight in weights:
            # print(f"Loading weight name: {name}, tp_rank: {tp_rank}", flush=True)
            if contains_unexpected_keyword(name, unexpected_keywords):
                print(f"Skipping unexpected weight name: {name}")
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            if "gate_proj_bias" in name:
                name = name.replace("gate_proj_bias", "gate_proj.bias")
            if "up_proj_bias" in name:
                name = name.replace("up_proj_bias", "up_proj.bias")
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            # With tie_word_embeddings, we can skip lm_head.weight
            # The weight might appear unnecessarily in the files if the model is
            # processed with quantization, LoRA, fine-tuning, etc.
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            if self.quant_config is not None and (scale_name := self.quant_config.get_cache_scale(name)):
                # Loading kv cache scales for compressed-tensors quantization
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = loaded_weight[0]
                weight_loader(param, loaded_weight)
                continue

            is_found = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                # cross layer only have q_proj, skip qkv pack
                if weight_name == ".q_proj":
                    match = re.search(r"layers\.\d+", name)
                    if match:
                        layer_id = int(match.group(0).split(".")[-1])
                        if cla_factor > 1 and layer_id % cla_factor != 0:
                            continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                is_found = True
                break
            if is_found:
                continue

            for (
                param_name,
                weight_name,
                den,
                split_param,
                func,
            ) in split_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                assert loaded_weight.shape[0] % den == 0
                units = loaded_weight.shape[0] // den
                param = params_dict[name]
                weight_loader = param.weight_loader
                offset = 0
                for shard_id, num in split_param:
                    new_offset = offset + num * units
                    if func:
                        weight_loader(param, func(loaded_weight)[offset:new_offset], shard_id)
                    else:
                        weight_loader(param, loaded_weight[offset:new_offset], shard_id)
                    offset = new_offset

                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                is_expert_weight = False
                is_found = False
                found_num = 0
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    offset = 0
                    den = 1
                    for (
                        mapped_weight_substr,
                        origin_weight_info,
                    ) in expert_weights_remapping.items():
                        if mapped_weight_substr in weight_name:
                            origin_weight_name, offset, den = origin_weight_info
                            weight_name = weight_name.replace(mapped_weight_substr, origin_weight_name)
                            break
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    found_num += 1
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    param = params_dict[name_mapped]
                    weight_loader = cast(Callable[..., bool], param.weight_loader)
                    assert loaded_weight.shape[0] % den == 0
                    units = loaded_weight.shape[0] // den

                    success = weight_loader(
                        param,
                        loaded_weight[offset * units : offset * units + units],
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        loaded_params.add(name_mapped)
                        is_found = True
                        if found_num == den:
                            break
                if is_found:
                    continue
                if is_expert_weight:
                    # We've checked that this is an expert weight
                    # However it's not mapped locally to this rank
                    # So we simply skip it
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                if "mlp.gate.wg." in name:
                    name = name.replace("wg.", "")
                if name == "ln_f.weight":
                    name = "norm.weight"
                if name == "wte.weight":
                    name = "embed_tokens.weight"
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        custom_pos_emb: tuple[torch.FloatTensor] | None = None,
        mode: str = "gen_text",
        first_step: bool | None = None,
        query_lens: list[int] | None = None,
        seq_lens: list[int] | None = None,
        num_image_tokens: int | None = None,
        gen_timestep_scatter_index: torch.Tensor | None = None,
    ) -> tuple | BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                positions=None,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                custom_pos_emb=custom_pos_emb,
                mode=mode,
                first_step=first_step,
                query_lens=query_lens,
                seq_lens=seq_lens,
                num_image_tokens=num_image_tokens,
                gen_timestep_scatter_index=gen_timestep_scatter_index,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class ClassifierFreeGuidance:
    def __init__(
        self,
        use_original_formulation: bool = False,
        start: float = 0.0,
        stop: float = 1.0,
    ):
        super().__init__()
        self.use_original_formulation = use_original_formulation

    def __call__(
        self,
        pred_cond: torch.Tensor,
        pred_uncond: torch.Tensor | None,
        guidance_scale: float,
        step: int,
    ) -> torch.Tensor:
        shift = pred_cond - pred_uncond
        pred = pred_cond if self.use_original_formulation else pred_uncond
        pred = pred + guidance_scale * shift

        return pred


@dataclass
class HunyuanImage3Text2ImagePipelineOutput(BaseOutput):
    samples: list[Any] | np.ndarray


class HunyuanImage3Text2ImagePipeline(DiffusionPipeline):
    r"""
    Pipeline for condition-to-sample generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        model ([`ModelMixin`]):
            A model to denoise the diffused latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `diffusion_model` to denoise the diffused latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """

    model_cpu_offload_seq = ""
    _optional_components = []
    _exclude_from_cpu_offload = []
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        model,
        scheduler: SchedulerMixin,
        vae,
        progress_bar_config: dict[str, Any] | None = None,
    ):
        super().__init__()

        if progress_bar_config is None:
            progress_bar_config = {}
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(progress_bar_config)
        self.register_modules(
            model=model,
            scheduler=scheduler,
            vae=vae,
        )

        # should be a tuple or a list corresponding to the size of latents (batch_size, channel, *size)
        # if None, will be treated as a tuple of 1
        self.latent_scale_factor = self.model.config.vae_downsample_factor
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.latent_scale_factor)

        # Must start with APG_mode_
        self.cfg_operator = ClassifierFreeGuidance()

    @staticmethod
    def denormalize(images: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """
        Denormalize an image array to [0,1].
        """
        return (images / 2 + 0.5).clamp(0, 1)

    @staticmethod
    def pt_to_numpy(images: torch.Tensor) -> np.ndarray:
        """
        Convert a PyTorch tensor to a NumPy image.
        """
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        return images

    @staticmethod
    def numpy_to_pil(images: np.ndarray):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def prepare_extra_func_kwargs(self, func, kwargs):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        extra_kwargs = {}

        for k, v in kwargs.items():
            accepts = k in set(inspect.signature(func).parameters.keys())
            if accepts:
                extra_kwargs[k] = v
        return extra_kwargs

    def prepare_latents(self, batch_size, latent_channel, image_size, dtype, device, generator, latents=None):
        if self.latent_scale_factor is None:
            latent_scale_factor = (1,) * len(image_size)
        elif isinstance(self.latent_scale_factor, int):
            latent_scale_factor = (self.latent_scale_factor,) * len(image_size)
        elif isinstance(self.latent_scale_factor, tuple) or isinstance(self.latent_scale_factor, list):
            assert len(self.latent_scale_factor) == len(image_size), (
                "len(latent_scale_factor) should be the same as len(image_size)"
            )
            latent_scale_factor = self.latent_scale_factor
        else:
            raise ValueError(
                f"latent_scale_factor should be either None, int, tuple of int, or list of int, "
                f"but got {self.latent_scale_factor}"
            )

        latents_shape = (
            batch_size,
            latent_channel,
            *[int(s) // f for s, f in zip(image_size, latent_scale_factor)],
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(latents_shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # Check existence to make it compatible with FlowMatchEulerDiscreteScheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma

        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def set_scheduler(self, new_scheduler):
        self.register_modules(scheduler=new_scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int,
        image_size: list[int],
        num_inference_steps: int = 50,
        timesteps: list[int] | None = None,
        sigmas: list[float] | None = None,
        guidance_scale: float = 5.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        guidance_rescale: float = 0.0,
        callback_on_step_end: Callable[[int, int, dict], None]
        | PipelineCallback
        | MultiPipelineCallbacks
        | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        model_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The text to guide image generation.
            image_size (`Tuple[int]` or `List[int]`):
                The size (height, width) of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate samples closely linked to the
                `condition` at the expense of lower sample quality. Guidance scale is enabled when `guidance_scale > 1`.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for sample
                generation. Can be used to tweak the same generation with different conditions. If not provided,
                a latents tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~DiffusionPipelineOutput`] instead of a
                plain tuple.
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~DiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~DiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated samples.
        """
        callback_steps = kwargs.pop("callback_steps", None)
        pbar_steps = kwargs.pop("pbar_steps", None)

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale

        cfg_factor = 1 + self.do_classifier_free_guidance

        # Define call parameters
        device = self._execution_device

        # Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
        )

        # Prepare latent variables
        latents = self.prepare_latents(
            batch_size=batch_size,
            latent_channel=self.model.config.vae["latent_channels"],
            image_size=image_size,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
            latents=latents,
        )

        # Prepare extra step kwargs.
        _scheduler_step_extra_kwargs = self.prepare_extra_func_kwargs(self.scheduler.step, {"generator": generator})

        # Prepare model kwargs
        input_ids = model_kwargs.pop("input_ids")
        attention_mask = self.model._prepare_attention_mask_for_generation(  # noqa
            input_ids,
            self.model.generation_config,
            model_kwargs=model_kwargs,
        )
        b, _, q_len1, seq_len = attention_mask.shape
        query_lens = [q_len1] * b
        seq_lens = [seq_len] * b
        model_kwargs["query_lens"] = query_lens
        model_kwargs["seq_lens"] = seq_lens
        model_kwargs["attention_mask"] = attention_mask.to(latents.device)

        # Sampling loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * cfg_factor)
                # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                t_expand = t.repeat(latent_model_input.shape[0])

                model_inputs = self.model.prepare_inputs_for_generation(
                    input_ids,
                    images=latent_model_input,
                    timestep=t_expand,
                    **model_kwargs,
                )

                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=True):
                    model_output = self.model.forward_call(**model_inputs, first_step=(i == 0))
                    pred = model_output["diffusion_prediction"]
                pred = pred.to(dtype=torch.float32)

                # perform guidance
                if self.do_classifier_free_guidance:
                    pred_cond, pred_uncond = pred.chunk(2)
                    pred = self.cfg_operator(pred_cond, pred_uncond, self.guidance_scale, step=i)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(pred, t, latents, **_scheduler_step_extra_kwargs, return_dict=False)[0]

                if i != len(timesteps) - 1:
                    model_kwargs = self.model._update_model_kwargs_for_generation(  # noqa
                        model_output,
                        model_kwargs,
                    )
                    if input_ids.shape[1] != model_kwargs["position_ids"].shape[1]:
                        input_ids = torch.gather(input_ids, 1, index=model_kwargs["position_ids"])
                    attention_mask = model_kwargs.get("attention_mask")
                    b, _, q_len1, seq_len = attention_mask.shape
                    query_lens = [q_len1] * b
                    seq_lens = [seq_len] * b
                    model_kwargs["query_lens"] = query_lens
                    model_kwargs["seq_lens"] = seq_lens

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if hasattr(self.vae.config, "scaling_factor") and self.vae.config.scaling_factor:
            latents = latents / self.vae.config.scaling_factor
        if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
            latents = latents + self.vae.config.shift_factor

        if hasattr(self.vae, "ffactor_temporal"):
            latents = latents.unsqueeze(2)

        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=True):
            image = self.vae.decode(latents, return_dict=False, generator=generator)[0]

        if hasattr(self.vae, "ffactor_temporal"):
            assert image.shape[2] == 1, "image should have shape [B, C, T, H, W] and T should be 1"
            image = image.squeeze(2)

        do_denormalize = [True] * image.shape[0]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        if not return_dict:
            return (image,)

        return HunyuanImage3Text2ImagePipelineOutput(samples=image)


def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    Args:
        t (torch.Tensor): a 1-D Tensor of N indices, one per batch element. These may be fractional.
        dim (int): the dimension of the output.
        max_period (int): controls the minimum frequency of the embeddings.

    Returns:
        embedding (torch.Tensor): An (N, D) Tensor of positional embeddings.

    .. ref_link: https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=t.device
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(
        self,
        hidden_size,
        act_layer=nn.GELU,
        frequency_embedding_size=256,
        max_period=10000,
        out_size=None,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        if out_size is None:
            out_size = hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True, **factory_kwargs),
            act_layer(),
            nn.Linear(hidden_size, out_size, bias=True, **factory_kwargs),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size, self.max_period).type(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    r"""
    A residual block that can optionally change the number of channels.

    Args:
        in_channels (`int`):
            The number of input channels.
        emb_channels (`int`):
            The number of timestep embedding channels.
        dropout (`float`):
            The rate of dropout.
        out_channels (`int`, *optional*):
            If specified, the number of output channels.
        use_conv (`bool`, *optional*):
            If True and out_channels is specified, use a spatial convolution instead of a
            smaller 1x1 convolution to change the channels in the skip connection.
        dims (`int`, *optional*):
            Determines if the signal is 1D, 2D, or 3D.
        up (`bool`, *optional*):
            If True, use this block for upsampling.
        down (`bool`, *optional*):
            If True, use this block for downsampling.

    """

    def __init__(
        self,
        in_channels,
        emb_channels,
        out_channels=None,
        dropout=0.0,
        use_conv=False,
        dims=2,
        up=False,
        down=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.in_channels = in_channels
        self.dropout = dropout
        self.out_channels = out_channels or self.in_channels
        self.use_conv = use_conv

        self.in_layers = nn.Sequential(
            normalization(self.in_channels, **factory_kwargs),
            nn.SiLU(),
            conv_nd(dims, self.in_channels, self.out_channels, 3, padding=1, **factory_kwargs),  # noqa: N802
        )

        self.updown = up or down
        self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(nn.SiLU(), linear(emb_channels, 2 * self.out_channels, **factory_kwargs))

        self.out_layers = nn.Sequential(
            normalization(self.out_channels, **factory_kwargs),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1, **factory_kwargs)),  # noqa: N802
        )

        if self.out_channels == self.in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, self.in_channels, self.out_channels, 3, padding=1, **factory_kwargs)  # noqa: N802
        else:
            self.skip_connection = conv_nd(dims, self.in_channels, self.out_channels, 1, **factory_kwargs)  # noqa: N802

    def forward(self, x, emb) -> torch.Tensor:
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        # Adaptive Group Normalization
        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = out_norm(h) * (1.0 + scale) + shift
        h = out_rest(h)

        return self.skip_connection(x) + h


class UNetDown(nn.Module):
    def __init__(
        self, patch_size, in_channels, emb_channels, hidden_channels, out_channels, dropout=0.0, device=None, dtype=None
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()

        self.patch_size = patch_size
        assert self.patch_size in [1, 2, 4, 8]

        self.model = nn.ModuleList(
            [
                conv_nd(
                    2, in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, padding=1, **factory_kwargs
                )  # noqa: N802
            ]
        )

        if self.patch_size == 1:
            self.model.append(
                ResBlock(
                    in_channels=hidden_channels,
                    emb_channels=emb_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    **factory_kwargs,
                )
            )
        else:
            for i in range(self.patch_size // 2):
                self.model.append(
                    ResBlock(
                        in_channels=hidden_channels,
                        emb_channels=emb_channels,
                        out_channels=hidden_channels if (i + 1) * 2 != self.patch_size else out_channels,
                        dropout=dropout,
                        down=True,
                        **factory_kwargs,
                    )
                )

    def forward(self, x, t):
        assert x.shape[2] % self.patch_size == 0 and x.shape[3] % self.patch_size == 0
        for module in self.model:
            if isinstance(module, ResBlock):
                x = module(x, t)
            else:
                x = module(x)
        _, _, token_h, token_w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        return x, token_h, token_w


class UNetUp(nn.Module):
    def __init__(
        self,
        patch_size,
        in_channels,
        emb_channels,
        hidden_channels,
        out_channels,
        dropout=0.0,
        device=None,
        dtype=None,
        out_norm=False,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()

        self.patch_size = patch_size
        assert self.patch_size in [1, 2, 4, 8]

        self.model = nn.ModuleList()

        if self.patch_size == 1:
            self.model.append(
                ResBlock(
                    in_channels=in_channels,
                    emb_channels=emb_channels,
                    out_channels=hidden_channels,
                    dropout=dropout,
                    **factory_kwargs,
                )
            )
        else:
            for i in range(self.patch_size // 2):
                self.model.append(
                    ResBlock(
                        in_channels=in_channels if i == 0 else hidden_channels,
                        emb_channels=emb_channels,
                        out_channels=hidden_channels,
                        dropout=dropout,
                        up=True,
                        **factory_kwargs,
                    )
                )

        if out_norm:
            self.model.append(
                nn.Sequential(
                    normalization(hidden_channels, **factory_kwargs),
                    nn.SiLU(),
                    conv_nd(
                        2,
                        in_channels=hidden_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1,
                        **factory_kwargs,
                    ),  # noqa: N802
                )
            )
        else:
            self.model.append(
                conv_nd(
                    2,
                    in_channels=hidden_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    **factory_kwargs,
                )  # noqa: N802
            )

    # batch_size, seq_len, model_dim
    def forward(self, x, t, token_h, token_w):
        x = rearrange(x, "b (h w) c -> b c h w", h=token_h, w=token_w)
        for module in self.model:
            if isinstance(module, ResBlock):
                x = module(x, t)
            else:
                x = module(x)
        return x
