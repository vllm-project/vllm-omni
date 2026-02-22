# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import typing
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal, TypeAlias

import numpy as np
import regex as re
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from torch import nn
from torchvision import transforms
from transformers import Siglip2ImageProcessorFast
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.hunyuan_v1 import (
    HunYuanModel,
    _get_cla_factor,
    _is_moe,
)
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
    _require_is_multimodal,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    _merge_multimodal_embeddings,
    is_pp_missing_parameter,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_omni.model_executor.models.hunyuan_image3.autoencoder_kl_3d import AutoencoderKLConv3D
from vllm_omni.model_executor.models.hunyuan_image3.siglip2 import LightProjector, Siglip2VisionTransformer

logger = init_logger(__name__)


@support_torch_compile
class HunyuanModel(HunYuanModel):
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

    def get_expert_mapping(self) -> tuple[list[tuple[str, str, int, str]], dict[str, tuple[str, int, int]]]:
        if _is_moe(self.config):
            # Params for weights, fp8 weight scales, fp8 activation scales
            # (param_name, weight_name, expert_id, shard_id)
            fused_moe_expert_mapping = SharedFusedMoE.make_expert_params_mapping(
                model=self,
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
            for keyword in keywords:
                if keyword in name:
                    return True
            return False

        for name, loaded_weight in weights:
            if contains_unexpected_keyword(name, unexpected_keywords):
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

                    # this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True

                    # Do not modify `name` since the loop may continue here
                    # Instead, create a new variable
                    name_mapped = name.replace(weight_name, param_name)
                    found_num += 1

                    if is_pp_missing_parameter(name_mapped, self):
                        continue

                    param = params_dict[name_mapped]
                    # We should ask the weight loader to return success or not
                    # here since otherwise we may skip experts with other
                    # available replicas.
                    weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
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


def conv_nd(dims, *args, **kwargs):
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


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels, **kwargs):
    """
    Make a standard normalization layer.
    """
    return nn.GroupNorm(32, channels, **kwargs)


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1, **factory_kwargs)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1, **factory_kwargs)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.

    :param in_channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
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
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.in_channels = in_channels
        self.dropout = dropout
        self.out_channels = out_channels or self.in_channels
        self.use_conv = use_conv

        self.in_layers = nn.Sequential(
            normalization(self.in_channels, **factory_kwargs),
            nn.SiLU(),
            conv_nd(dims, self.in_channels, self.out_channels, 3, padding=1, **factory_kwargs),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(self.in_channels, False, dims, **factory_kwargs)
            self.x_upd = Upsample(self.in_channels, False, dims, **factory_kwargs)
        elif down:
            self.h_upd = Downsample(self.in_channels, False, dims, **factory_kwargs)
            self.x_upd = Downsample(self.in_channels, False, dims, **factory_kwargs)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(nn.SiLU(), linear(emb_channels, 2 * self.out_channels, **factory_kwargs))

        self.out_layers = nn.Sequential(
            normalization(self.out_channels, **factory_kwargs),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1, **factory_kwargs)),
        )

        if self.out_channels == self.in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, self.in_channels, self.out_channels, 3, padding=1, **factory_kwargs)
        else:
            self.skip_connection = conv_nd(dims, self.in_channels, self.out_channels, 1, **factory_kwargs)

    def forward(self, x, emb):
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
    """
    patch_size: one of [1, 2 ,4 ,8]
    in_channels: vae latent dim
    hidden_channels: hidden dim for reducing parameters
    out_channels: transformer model dim
    """

    def __init__(
        self, patch_size, in_channels, emb_channels, hidden_channels, out_channels, dropout=0.0, device=None, dtype=None
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()

        self.patch_size = patch_size
        assert self.patch_size in [1, 2, 4, 8]

        self.model = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, padding=1, **factory_kwargs
                )
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
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        if out_size is None:
            out_size = hidden_size

        self.mlp = nn.Sequential(
            ColumnParallelLinear(
                frequency_embedding_size, hidden_size, bias=True, gather_output=False, return_bias=False
            ),
            act_layer(),
            RowParallelLinear(hidden_size, out_size, bias=True, input_is_parallel=True, return_bias=False),
        )

        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    @staticmethod
    def _timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t (torch.Tensor): a 1-D Tensor of N indices, one per batch element. These may be fractional.
            dim (int): the dimension of the output.
            max_period (int): controls the minimum frequency of the embeddings.

        Returns:
            embedding (torch.Tensor): An (N, D) Tensor of positional embeddings.
        """
        # Ensure t is a tensor and get device
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
        device = t.device

        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=device) / half
        )
        # Ensure t is 2D: (N, 1) for broadcasting
        if t.ndim == 0:
            t = t.unsqueeze(0)
        if t.ndim == 1:
            t = t[:, None]
        args = t.float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self._timestep_embedding(t, self.frequency_embedding_size, self.max_period).type(
            self.mlp[0].weight.dtype
        )
        # t_emb = self.linear1(t_freq)
        # t_emb = self.act(t_emb)
        # t_emb = self.linear2(t_emb)
        t_emb = self.mlp(t_freq)
        return t_emb


class HunyuanImage3PixelInputs(TensorSchema):
    type: Literal["pixel_values"]

    pixel_values: dict[str, torch.Tensor]


class HunyuanImage3Processor:
    """Image processor for Hunyuan Image 3.0 model."""

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
        """Group of resolutions for image processing."""

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

        def _calc_by_step(self):
            assert self.align <= self.step, f"align {self.align} must be smaller than step {self.step}"

            min_height = self.base_size // 2
            min_width = self.base_size // 2
            max_height = self.base_size * 2
            max_width = self.base_size * 2

            resolutions = [HunyuanImage3Processor.Resolution(self.base_size, self.base_size)]

            cur_height, cur_width = self.base_size, self.base_size
            while True:
                if cur_height >= max_height and cur_width <= min_width:
                    break

                cur_height = min(cur_height + self.step, max_height)
                cur_width = max(cur_width - self.step, min_width)
                resolutions.append(
                    HunyuanImage3Processor.Resolution(
                        cur_height // self.align * self.align, cur_width // self.align * self.align
                    )
                )

            cur_height, cur_width = self.base_size, self.base_size
            while True:
                if cur_height <= min_height and cur_width >= max_width:
                    break

                cur_height = max(cur_height - self.step, min_height)
                cur_width = min(cur_width + self.step, max_width)
                resolutions.append(
                    HunyuanImage3Processor.Resolution(
                        cur_height // self.align * self.align, cur_width // self.align * self.align
                    )
                )

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

    def __init__(self, tokenizer, hf_config, **kwargs: object):
        self.tokenizer = tokenizer
        self.hf_config = hf_config
        self.reso_group = self.ResolutionGroup(base_size=hf_config.image_base_size)
        self.vision_encoder_processor = Siglip2ImageProcessorFast.from_dict(hf_config.vit_processor)
        self.vae_processor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),  # transform to [-1, 1]
            ]
        )

    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput],
        images: ImageInput,
        **kwargs,
    ):
        """
        Main method to prepare for the model one or several sequences(s) and image(s).
        """
        if images is not None:
            image_infors = self.process_image(images)
        else:
            image_infors = None

        text_inputs = self.tokenizer(text, return_tensors="pt", **kwargs) if text is not None else None

        if image_infors and text_inputs is not None:
            # Combine text and image inputs into BatchFeature
            combined = dict(text_inputs)
            combined.update(image_infors)
            return BatchFeature(combined)
        elif image_infors:
            return BatchFeature(image_infors)
        elif text_inputs is not None:
            return BatchFeature(dict(text_inputs))
        else:
            return BatchFeature({})

    def process_image(self, image_input: ImageInput):
        if isinstance(image_input, Image.Image):
            images = [image_input]
        elif isinstance(image_input, list):
            images = image_input
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}.")

        torch_dtype = getattr(self.hf_config, "torch_dtype", torch.bfloat16)

        batch_data = []
        for image in images:
            current_info = {}

            # VIT processing
            vit_pixel_values = self.vision_encoder_processor(image)
            # shape: (seq_len, num_channels * patch_size * patch_size)
            current_info["vit_pixel_values"] = vit_pixel_values["pixel_values"].squeeze(0)
            # shape: (seq_len, )
            current_info["vit_pixel_attention_mask"] = vit_pixel_values["pixel_attention_mask"].squeeze(0)
            # shape: (2, )
            current_info["vit_spatial_shapes"] = vit_pixel_values["spatial_shapes"].squeeze(0)
            logger.info(
                f"vit input image: {image.width} x {image.height}, token grid: {vit_pixel_values['spatial_shapes']}"
            )

            # VAE processing
            image_width, image_height = self.reso_group.get_target_size(image.width, image.height)
            resized_image = self._resize_and_crop(image, (image_width, image_height))
            vae_pixel_values = self.vae_processor(resized_image)
            token_height = image_height // (self.hf_config.vae_downsample_factor[0] * self.hf_config.patch_size)
            token_width = image_width // (self.hf_config.vae_downsample_factor[1] * self.hf_config.patch_size)
            current_info["vae_pixel_values"] = vae_pixel_values.squeeze(0).to(dtype=torch_dtype)
            current_info["vae_token_grid_hw"] = torch.tensor([token_height, token_width])
            logger.info(
                f"vae input image: {image.width} x {image.height}, target size: {image_width} x {image_height}, "
                + f"token grid: {token_width} x {token_height}."
            )

            # size
            base_size, ratio_index = self.reso_group.get_base_size_and_ratio_index(image_width, image_height)
            current_info["base_size"] = torch.tensor(base_size)
            current_info["ratio_index"] = torch.tensor(ratio_index)
            logger.info(f"base size: {base_size} and ratio index: {ratio_index}")

            batch_data.append(current_info)

        # Stack the tensors in the list into a batch dimension (B, ...)
        final_image_info = {}
        if len(batch_data) > 0:
            for key in batch_data[0].keys():
                final_image_info[key] = torch.stack([d[key] for d in batch_data], dim=0)

        return final_image_info

    def _resize_and_crop(self, image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
        tw, th = target_size
        w, h = image.size

        tr = th / tw
        r = h / w

        # resize
        if r < tr:
            resize_height = th
            resize_width = int(round(th / h * w))
        else:
            resize_width = tw
            resize_height = int(round(tw / w * h))

        image = image.resize((resize_width, resize_height), resample=Image.Resampling.LANCZOS)

        # center crop
        crop_top = int(round((resize_height - th) / 2.0))
        crop_left = int(round((resize_width - tw) / 2.0))

        image = image.crop((crop_left, crop_top, crop_left + tw, crop_top + th))
        return image


class HunyuanImage3ProcessingInfo(BaseProcessingInfo):
    """Processing information for HunyuanImage3 model."""

    def get_image_processor(self, **kwargs: object) -> HunyuanImage3Processor:
        tokenizer = self.get_tokenizer()
        hf_config = self.get_hf_config()

        return HunyuanImage3Processor(
            hf_config=hf_config,
            tokenizer=tokenizer,
            **kwargs,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}


class HunyuanImage3DummyInputsBuilder(BaseDummyInputsBuilder[HunyuanImage3ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        """Generate dummy text with image placeholders."""
        num_images = mm_counts.get("image", 0)
        return "<img>" * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        logger.info(f"dummy image count: {num_images}")

        hf_config = self.info.get_hf_config()
        image_size = hf_config.image_base_size
        image_overrides = mm_options.get("image") if mm_options else None

        return {
            "image": self._get_dummy_images(
                width=image_size,
                height=image_size,
                num_images=num_images,
                overrides=image_overrides,
            ),
        }


class HunyuanImage3MultiModalProcessor(BaseMultiModalProcessor[HunyuanImage3ProcessingInfo]):
    """Multimodal processor for HunyuanImage3 model."""

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        image_processor = self.info.get_image_processor(**mm_kwargs)
        images = mm_data.get("images", [])
        logger.info(f"process image count: {len(images)}")
        batch_feature = image_processor(prompt, images, **tok_kwargs)
        return batch_feature

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        config = {}

        if "vit_pixel_values" in hf_inputs:
            config["vit_pixel_values"] = MultiModalFieldConfig.batched("image")
        if "vit_pixel_attention_mask" in hf_inputs:
            config["vit_pixel_attention_mask"] = MultiModalFieldConfig.batched("image")
        if "vit_spatial_shapes" in hf_inputs:
            config["vit_spatial_shapes"] = MultiModalFieldConfig.batched("image")
        if "vae_pixel_values" in hf_inputs:
            config["vae_pixel_values"] = MultiModalFieldConfig.batched("image")
        if "vae_token_grid_hw" in hf_inputs:
            config["vae_token_grid_hw"] = MultiModalFieldConfig.batched("image")
        if "base_size" in hf_inputs:
            config["base_size"] = MultiModalFieldConfig.batched("image")
        if "ratio_index" in hf_inputs:
            config["ratio_index"] = MultiModalFieldConfig.batched("image")
        return config

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """Replace image placeholders with the correct number of tokens."""

        tokenizer = self.info.get_tokenizer()
        img_token_id = tokenizer.convert_tokens_to_ids("<img>")
        if img_token_id is None:
            raise ValueError("Image token '<img>' not found in tokenizer vocabulary")
        joint_img_sep_token_id = tokenizer.convert_tokens_to_ids("<joint_img_sep>")
        if joint_img_sep_token_id is None:
            raise ValueError("Joint image separator token '<joint_img_sep>' not found in tokenizer vocabulary")
        boi_token_id = tokenizer.convert_tokens_to_ids("<boi>")
        if boi_token_id is None:
            raise ValueError("Beginning of image token '<boi>' not found in tokenizer vocabulary")
        eoi_token_id = tokenizer.convert_tokens_to_ids("<eoi>")
        if eoi_token_id is None:
            raise ValueError("End of image token '<eoi>' not found in tokenizer vocabulary")

        out_mm_data = out_mm_kwargs.get_data()
        vit_spatial_shapes = out_mm_data.get("vit_spatial_shapes")
        vae_token_grid_hw = out_mm_data.get("vae_token_grid_hw")
        base_size = out_mm_data.get("base_size")
        ratio_index = out_mm_data.get("ratio_index")

        def get_replacement_image(item_idx: int) -> PromptUpdateDetails:
            _vit_token_grid_hw = vit_spatial_shapes.tolist()[item_idx]
            _vae_token_grid_hw = vae_token_grid_hw.tolist()[item_idx]
            _base_size = base_size.tolist()[item_idx]
            _ratio_index = ratio_index.tolist()[item_idx]

            timestep_token_num = 1
            vae_token_num = _vae_token_grid_hw[0] * _vae_token_grid_hw[1]
            vit_token_num = _vit_token_grid_hw[0] * _vit_token_grid_hw[1]

            base_size_token_id = tokenizer.convert_tokens_to_ids(f"<img_size_{_base_size}>")
            if base_size_token_id is None:
                raise ValueError(f"Base size token '<img_size_{_base_size}>' not found in tokenizer vocabulary")
            ratio_token_id = tokenizer.convert_tokens_to_ids(f"<img_ratio_{_ratio_index}>")
            if ratio_token_id is None:
                raise ValueError(f"Ratio token '<img_ratio_{_ratio_index}>' not found in tokenizer vocabulary")

            replacement = (
                [boi_token_id]
                + [base_size_token_id]
                + [ratio_token_id]
                + [img_token_id] * timestep_token_num
                + [img_token_id] * vae_token_num
                + [joint_img_sep_token_id]
                + [img_token_id] * vit_token_num
                + [eoi_token_id]
            )
            logger.info(f"actual replacement token count: {timestep_token_num + vae_token_num + vit_token_num}")
            return PromptUpdateDetails.select_token_id(replacement, embed_token_id=img_token_id)

        return [
            PromptReplacement(modality="image", target=[img_token_id], replacement=get_replacement_image),
        ]


@MULTIMODAL_REGISTRY.register_processor(
    HunyuanImage3MultiModalProcessor,
    info=HunyuanImage3ProcessingInfo,
    dummy_inputs=HunyuanImage3DummyInputsBuilder,
)
class HunyuanImage3ForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsLoRA, SupportsPP):
    HunyuanImage3Inputs: TypeAlias = HunyuanImage3PixelInputs

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        # Change the rope_type from 'custom' to 'default' in the AR stage
        if isinstance(config.rope_parameters, dict):
            config.rope_parameters["rope_type"] = "default"

        self.config = config
        self.quant_config = quant_config
        self.vllm_config = vllm_config
        self.model = HunyuanModel(vllm_config=vllm_config, prefix="model")
        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head.weight = self.model.embed_tokens.weight

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(self.unpadded_vocab_size, config.vocab_size, logit_scale)
        else:
            self.lm_head = PPMissingLayer()

        # vae
        self.vae = AutoencoderKLConv3D.from_config(config.vae)
        self.patch_embed = UNetDown(
            patch_size=config.patch_size,
            emb_channels=config.hidden_size,
            in_channels=config.vae["latent_channels"],
            hidden_channels=config.patch_embed_hidden_dim,
            out_channels=config.hidden_size,
        )

        # Used when converting VAE-encoded latent space (latents) to token embeddings.
        self.time_embed = TimestepEmbedder(hidden_size=config.hidden_size)

        # vision
        self.vision_model = Siglip2VisionTransformer(config.vit)
        self.vision_aligner = LightProjector(config.vit_aligner)

        # Used to embed timestep information into the input sequence.
        self.timestep_emb = TimestepEmbedder(hidden_size=config.hidden_size)

    def _parse_and_validate_image_input(
        self,
        **kwargs: dict[str, Any],
    ) -> HunyuanImage3Inputs | None:
        """
        Parse and validate image input from kwargs.
        """
        vit_pixel_values = kwargs.pop("vit_pixel_values", None)
        vit_pixel_attention_mask = kwargs.pop("vit_pixel_attention_mask", None)
        vit_spatial_shapes = kwargs.pop("vit_spatial_shapes", None)
        vae_pixel_values = kwargs.pop("vae_pixel_values", None)
        vae_token_grid_hw = kwargs.pop("vae_token_grid_hw", None)

        if vit_pixel_values is None or vae_pixel_values is None:
            return None

        return HunyuanImage3PixelInputs(
            type="pixel_values",
            pixel_values={
                "vit_pixel_values": vit_pixel_values,
                "vit_pixel_attention_mask": vit_pixel_attention_mask,
                "vit_spatial_shapes": vit_spatial_shapes,
                "vae_pixel_values": vae_pixel_values,
                "vae_token_grid_hw": vae_token_grid_hw,
            },
        )

    def _vae_encode(
        self,
        images: torch.Tensor,
        cfg_factor: int = 1,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """
        Encode images through VAE encoder.
        """
        config = self.vae.config

        vae_encode_result = self.vae.encode(images)
        latents = vae_encode_result.latent_dist.sample()

        # Apply shift and scaling factors if present
        if hasattr(config, "shift_factor") and config.shift_factor:
            latents.sub_(config.shift_factor)
        if hasattr(config, "scaling_factor") and config.scaling_factor:
            latents.mul_(config.scaling_factor)

        # Handle temporal dimension if present
        # from (B, C, T, H, W) to (B, C, H, W)
        if latents.ndim == 5:
            # Check if T dimension is 1, then squeeze it
            # This matches HuggingFace implementation: assert latents.shape[2] == 1, then squeeze(2)
            if latents.shape[2] == 1:
                latents = latents.squeeze(2)
            else:
                # If T > 1, take the first frame (shouldn't happen for conditional images)
                latents = latents[:, :, 0, :, :]
                logger.warning(f"T({latents.shape[2]}) > 1, should not happen for conditional images")

        # Always use t=0 to declare it is a clean conditional image
        t = torch.zeros((latents.shape[0],), device=latents.device, dtype=latents.dtype)

        # Apply cfg_factor if needed
        if cfg_factor > 1:
            t = t.repeat(cfg_factor)
            latents = latents.repeat(cfg_factor, 1, 1, 1)

        return t, latents

    def _vit_encode(
        self,
        pixel_values: torch.Tensor,
        vit_attention_mask: torch.Tensor,
        vit_spatial_shapes: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """
        Encode pixel_values through ViT encoder (vision_model and vision_aligner).
        """
        # Handle empty batch
        if pixel_values.shape[0] == 0:
            return tuple()

        vision_output = self.vision_model(
            pixel_values, attention_mask=vit_attention_mask, spatial_shapes=vit_spatial_shapes
        )
        image_embed = vision_output.last_hidden_state
        image_embed = self.vision_aligner(image_embed)
        return image_embed

    def _timestep_encode(
        self,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Encode timestep into embedding representation.
        Args:
            timestep: Timestep tensor, shape (token_size,).
        Returns:
            Timestep embeddings, shape (token_size, hidden_size).
        """
        if timestep.ndim == 0:
            t_flat = timestep.unsqueeze(0)
        else:
            t_flat = timestep.reshape(-1)

        t_emb = self.timestep_emb(t_flat)
        return t_emb

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """Get multimodal embeddings from input."""
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        vae_cfg_factor = 1
        pixel_values = image_input["pixel_values"]
        vit_pixel_values = pixel_values["vit_pixel_values"]
        vit_pixel_attention_mask = pixel_values["vit_pixel_attention_mask"]
        vit_spatial_shapes = pixel_values["vit_spatial_shapes"]
        vae_pixel_values = pixel_values["vae_pixel_values"]

        # Perform ViT encoding
        vit_embeddings = self._vit_encode(vit_pixel_values, vit_pixel_attention_mask, vit_spatial_shapes)

        # Perform VAE encoding
        t, latents = self._vae_encode(vae_pixel_values, vae_cfg_factor)

        # Process VAE latents through patch_embed to convert to token embeddings
        # VAE latents are in (B, C, H, W) format, need to be converted to (B, seq_len, hidden_size)
        vae_token_embeddings = []
        batch_size = latents.shape[0]
        for i in range(batch_size):
            t_i = t[i]
            latents_i = latents[i : i + 1]  # Shape: (1, C, H, W)

            # Time embedding for VAE processing
            t_emb = self.time_embed(t_i)

            # Process VAE latent through patch_embed
            # Input: (1, C, H, W) -> Output: (1, seq_len, hidden_size)
            vae_tokens, _, _ = self.patch_embed(latents_i, t_emb)
            vae_token_embeddings.append(vae_tokens)

        assert vit_embeddings.shape[0] == len(vae_token_embeddings), (
            f"Number of ViT embeddings ({vit_embeddings.shape[0]}) does not match "
            f"number of VAE token embeddings ({len(vae_token_embeddings)}). "
            "Each image should have both VAE and ViT embeddings."
        )

        # Order per image: timestep -> VAE tokens -> ViT tokens
        combined_embeddings: list[torch.Tensor] = []
        num_images = len(vae_token_embeddings)
        for img_idx in range(num_images):
            # 1. Timestep embedding
            timestep = torch.zeros((1,)).to(vit_embeddings.device).to(vit_embeddings.dtype)
            timestep_emb = self._timestep_encode(timestep)

            # 2. VAE image token embeddings
            vae_token_embed = vae_token_embeddings[img_idx]
            # Remove batch dimension if present: (B, seq_len, hidden_size) -> (seq_len, hidden_size)
            if vae_token_embed.ndim == 3:
                vae_token_embed = vae_token_embed.squeeze(0)

            # 3. ViT image embeddings
            vit_embed = vit_embeddings[img_idx]

            stacked_embed = torch.cat([timestep_emb, vae_token_embed, vit_embed], dim=0)
            combined_embeddings.append(stacked_embed)

        return combined_embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        """Embed input IDs with optional multimodal embeddings."""
        # Get text embeddings
        inputs_embeds = self.model.embed_input_ids(input_ids)

        # If no multimodal embeddings, return text embeddings
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        # Merge multimodal embeddings with text embeddings
        merged_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=_require_is_multimodal(is_multimodal),
        )
        return merged_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        sampling_metadata: SamplingMetadata | None = None,
        logits_index: int | None = None,
        sampler=None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        model_input_ids = None if inputs_embeds is not None else input_ids
        model_output = self.model(model_input_ids, positions, intermediate_tensors, inputs_embeds)
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros((batch_size, self.config.hidden_size), dtype=dtype, device=device),
                "residual": torch.zeros((batch_size, self.config.hidden_size), dtype=dtype, device=device),
            }
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = ["lm_head."] if self.config.tie_word_embeddings else []
        unexpected_keywords = [
            "final_layer",
            "time_embed_2",
            "guidance_emb",
            "timestep_r_emb",
        ]
        skip_prefixes.extend(unexpected_keywords)
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=skip_prefixes,
        )

        loaded_params = loader.load_weights(weights)
        return loaded_params

    def get_language_model(self) -> torch.nn.Module:
        return self.model
