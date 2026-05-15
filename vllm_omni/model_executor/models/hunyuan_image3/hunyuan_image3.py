# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import math
import typing
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal, TypeAlias

import numpy as np
import regex as re
import torch
from einops import rearrange
from PIL import Image
from torch import nn
from torchvision import transforms
from transformers import PretrainedConfig, Siglip2ImageProcessorFast
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping

try:
    from vllm.model_executor.layers.fused_moe.shared_fused_moe import SharedFusedMoE
except ImportError:
    # PyPI vllm 0.20.x neither exports `SharedFusedMoE` from the package top-level
    # nor ships a `shared_fused_moe.py` submodule. The functionality lives on
    # `FusedMoE` directly (which gained a `shared_experts` parameter), so alias
    # the symbol — call sites only use the classmethod `make_expert_params_mapping`
    # and `__init__(shared_experts=..., ...)` which are present on `FusedMoE`.
    from vllm.model_executor.layers.fused_moe import FusedMoE as SharedFusedMoE
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
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
    HunYuanMLP,
    HunYuanModel,
    HunYuanSparseMoeBlock,
    _get_cla_factor,
    _is_moe,
)
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMRoPE,
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
from vllm.multimodal.image import rgba_to_rgb
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
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
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.utils.tensor_schema import TensorSchema
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

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
            fused_moe_expert_mapping = fused_moe_make_expert_params_mapping(
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

        skipped_unexpected: set[str] = set()

        for name, loaded_weight in weights:
            if contains_unexpected_keyword(name, unexpected_keywords):
                skipped_unexpected.add(name)
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

                loaded_params.add(name)
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

        if skipped_unexpected:
            logger.warning_once(
                "Skipped %d weights matching unexpected_keywords "
                "(e.g. vae, vision_model, patch_embed, timestep_emb). "
                "If upstream renamed components, these may be silently "
                "lost. Skipped names: %s",
                len(skipped_unexpected),
                sorted(skipped_unexpected)[:10],
            )

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

        self.down = down

        if down:
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
        if self.down:
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

        def __init__(self, base_size=None, step=None, align=1, extra_resolutions=None):
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

            if extra_resolutions is not None:
                for er in extra_resolutions:
                    if not any(r.ratio == er.ratio for r in self.data):
                        self.data.append(er)

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
        # `HUNYUAN_IMAGE3_EXTRA_RESOLUTIONS` mirrors the official
        # `vae_reso_group` extras (image_processor.py:147-152). Build with
        # this processor's inner Resolution class so `data` stays
        # type-homogeneous.
        from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
            HUNYUAN_IMAGE3_EXTRA_RESOLUTIONS,
        )

        self.reso_group = self.ResolutionGroup(
            base_size=hf_config.image_base_size,
            extra_resolutions=[HunyuanImage3Processor.Resolution(s) for s in HUNYUAN_IMAGE3_EXTRA_RESOLUTIONS],
        )
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

        # Each cond image keeps its own VAE bucket (mirrors official HF's
        # ragged behavior in `_encode_cond_image`). VAE pixel tensors have
        # different (H_i, W_i) per image, so they're flattened to 1-D and
        # concatenated; vLLM `flat_from_sizes("image", vae_pixel_size)` slices
        # them back per-image at consumption time. VIT (Siglip2 naflex) pads
        # to `max_num_patches` so VIT fields keep the existing `batched`
        # stack path.
        batch_data = []
        for image in images:
            current_info = {}

            if self.hf_config.vit["num_channels"] == 3 and image.mode == "RGBA":
                image = rgba_to_rgb(image, (255, 255, 255))

            # VIT processing
            vit_pixel_values = self.vision_encoder_processor(image)
            # transformers>=5.x returns lists; stack to tensor when needed
            _pv = vit_pixel_values["pixel_values"]
            if isinstance(_pv, list):
                _pv = torch.stack(_pv, dim=0)
            current_info["vit_pixel_values"] = _pv.squeeze(0)
            _pam = vit_pixel_values["pixel_attention_mask"]
            if isinstance(_pam, list):
                _pam = torch.stack(_pam, dim=0)
            current_info["vit_pixel_attention_mask"] = _pam.squeeze(0)
            _ss = vit_pixel_values["spatial_shapes"]
            if isinstance(_ss, list):
                _ss = torch.tensor(_ss, dtype=torch.long)
            current_info["vit_spatial_shapes"] = _ss.squeeze(0)

            # VAE: per-image bucket via `reso_group.get_target_size`; mirrors
            # HF's `resize_and_crop` (crop_type="center", the official
            # generate_image default with infer_align_image_size=False).
            # Keep fp32 — the VAE encoder casts to model dtype at its
            # boundary (see `_vae_encode`).
            image_width, image_height = self.reso_group.get_target_size(image.width, image.height)
            resized_image = self._resize_and_crop(image, (image_width, image_height))
            vae_pixel_values = self.vae_processor(resized_image).squeeze(0)
            token_height = image_height // (self.hf_config.vae_downsample_factor[0] * self.hf_config.patch_size)
            token_width = image_width // (self.hf_config.vae_downsample_factor[1] * self.hf_config.patch_size)

            current_info["vae_pixel_values_flat"] = vae_pixel_values.reshape(-1)
            current_info["vae_pixel_size"] = torch.tensor(vae_pixel_values.numel(), dtype=torch.long)
            current_info["vae_token_grid_hw"] = torch.tensor([token_height, token_width])

            base_size, ratio_index = self.reso_group.get_base_size_and_ratio_index(image_width, image_height)
            current_info["base_size"] = torch.tensor(base_size)
            current_info["ratio_index"] = torch.tensor(ratio_index)

            batch_data.append(current_info)

        final_image_info: dict[str, torch.Tensor] = {}
        if not batch_data:
            return final_image_info

        # Same-shape fields: stack along a new image-batch dim as before.
        same_shape_keys = [
            "vit_pixel_values",
            "vit_pixel_attention_mask",
            "vit_spatial_shapes",
            "vae_token_grid_hw",
            "vae_pixel_size",
            "base_size",
            "ratio_index",
        ]
        for key in same_shape_keys:
            final_image_info[key] = torch.stack([d[key] for d in batch_data], dim=0)

        # Variable-shape VAE pixels: 1-D concat across images (paired with
        # `vae_pixel_size` via `flat_from_sizes` in `_get_mm_fields_config`).
        final_image_info["vae_pixel_values"] = torch.cat([d["vae_pixel_values_flat"] for d in batch_data], dim=0)

        shapes_info = {k: tuple(v.shape) for k, v in final_image_info.items()}
        logger.info(f"Successfully processed {len(images)} image(s). Final tensor shapes: {shapes_info}")

        return final_image_info

    def _resize_and_crop(
        self,
        image: Image.Image,
        target_size: tuple[int, int],
        crop_type: str = "resize",
    ) -> Image.Image:
        # Default mode mirrors the official `infer_align_image_size=True`
        # path (image_processor.py:355 → crop_type="resize") used by the
        # IT2I demo: stretch the cond image to the bucket dims so its
        # `<img_ratio_*>` tag and ViT/VAE features stay aligned with the
        # bucket, instead of dropping content via center crop.
        tw, th = target_size
        if crop_type == "resize":
            return image.resize((tw, th), resample=Image.Resampling.LANCZOS)
        w, h = image.size
        tr = th / tw
        r = h / w
        if r < tr:
            resize_height = th
            resize_width = int(round(th / h * w))
        else:
            resize_width = tw
            resize_height = int(round(tw / w * h))
        image = image.resize((resize_width, resize_height), resample=Image.Resampling.LANCZOS)
        crop_top = int(round((resize_height - th) / 2.0))
        crop_left = int(round((resize_width - tw) / 2.0))
        return image.crop((crop_left, crop_top, crop_left + tw, crop_top + th))


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
        logger.debug(f"process image count: {len(images)}")
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
        # `vae_pixel_values` is a 1-D concatenation of variable-shape per-image
        # VAE tensors (see `process_image`). `vae_pixel_size` carries the
        # per-image flat length so vLLM can split the buffer back per image.
        if "vae_pixel_values" in hf_inputs and "vae_pixel_size" in hf_inputs:
            config["vae_pixel_values"] = MultiModalFieldConfig.flat_from_sizes("image", hf_inputs["vae_pixel_size"])
        if "vae_pixel_size" in hf_inputs:
            config["vae_pixel_size"] = MultiModalFieldConfig.batched("image")
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
            hf_config = self.info.get_hf_config()
            vit_token_num = hf_config.vit_processor.get("max_num_patches", 729)

            base_size_token_id = tokenizer.convert_tokens_to_ids(f"<img_size_{_base_size}>")
            if base_size_token_id is None:
                raise ValueError(f"Base size token '<img_size_{_base_size}>' not found in tokenizer vocabulary")
            ratio_token_id = tokenizer.convert_tokens_to_ids(f"<img_ratio_{_ratio_index}>")
            if ratio_token_id is None:
                raise ValueError(f"Ratio token '<img_ratio_{_ratio_index}>' not found in tokenizer vocabulary")
            timestep_token_id = tokenizer.convert_tokens_to_ids("<timestep>")
            if timestep_token_id is None:
                raise ValueError("Timestep token '<timestep>' not found in tokenizer vocabulary")

            # Use the real <timestep> token id (HF parity). The trained wte
            # at this slot is overwritten with timestep_emb(0) at runtime by
            # `embed_input_ids`.
            #
            # Mark <img>*VAE + <joint_img_sep> + <img>*ViT as one contiguous
            # embed run so vLLM's prefix-LM mask treats it as a single
            # bidirectional region, mirroring official `joint_image_slices`
            # full-attention range (image_processor.py:388, with
            # cond_token_attn_type effectively spanning VAE+sep+ViT). With the
            # default `select_token_id(<img>)` mask, sep splits the run into
            # two regions; that asymmetry is what biased multi-image AR
            # ratio prediction to the first image's bucket.
            replacement = (
                [boi_token_id]
                + [base_size_token_id]
                + [ratio_token_id]
                + [timestep_token_id] * timestep_token_num
                + [img_token_id] * vae_token_num
                + [joint_img_sep_token_id]
                + [img_token_id] * vit_token_num
                + [eoi_token_id]
            )
            logger.debug(f"actual replacement token count: {timestep_token_num + vae_token_num + vit_token_num}")
            return PromptUpdateDetails.select_token_ids(
                replacement,
                embed_token_ids=[img_token_id, joint_img_sep_token_id],
            )

        return [
            PromptReplacement(modality="image", target=[img_token_id], replacement=get_replacement_image),
        ]


def _hunyuan_image3_unpack_packed_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack pre-computed ``(topk_weights, topk_indices)`` packed by
    :class:`HunyuanImage3SparseMoeBlock` into ``gating_output``.

    Used as ``custom_routing_function`` for the underlying ``SharedFusedMoE``,
    bypassing its bf16 ``topk_softmax`` CUDA op so the routing decision can
    be made in fp32 (matching the reference implementation).

    Layout of ``gating_output`` (shape ``[num_tokens, top_k * 2]``)::

        [:, :top_k]  -> topk_weights (already softmax'd + renormalized in fp32,
                                      stored as fp32 for transport)
        [:, top_k:]  -> topk_indices (cast to fp32 for transport, restored to int32)
    """
    topk_weights = gating_output[:, :topk].contiguous()
    topk_indices = gating_output[:, topk:]
    return topk_weights.to(torch.float32), topk_indices.to(torch.int32)


class HunyuanImage3SparseMoeBlock(HunYuanSparseMoeBlock):
    """MoE block with FP32 routing for byte-level alignment with HF.

    The reference ``modeling_hunyuan_image_3.py`` runs the router in fp32:

    - ``HunyuanTopKGate.wg`` is constructed as ``nn.Linear(..., dtype=torch.float32)``
      and ``hidden_states`` is cast to fp32 before the matmul (line 1114-1116).
    - ``HunyuanMoE.forward`` wraps the gate call in
      ``with torch.autocast('cuda', enabled=False):`` to defeat any AMP cast
      (line 1204-1205), then calls ``easy_topk`` which does
      ``F.softmax`` → ``torch.topk`` → divide by
      ``torch.clamp(weight_sums, min=1e-8)`` → cast back to bf16, all in fp32
      (line 1132-1139, 1206-1207).

    vLLM's stock ``HunYuanSparseMoeBlock`` instead builds the gate as a
    default-dtype ``ReplicatedLinear`` (bf16) and lets ``SharedFusedMoE``'s
    ``topk_softmax`` CUDA op consume bf16 logits, which can flip top-k
    boundary decisions vs HF on close routing scores. With ``num_experts=64``,
    ``top_k=8`` per layer × 32 MoE layers, even a small per-token flip rate
    cascades into divergent expert outputs and KV-cache state, eventually
    flipping the top-1 decoded token.

    This subclass:

    1. Replaces ``self.gate`` with a fp32 ``ReplicatedLinear``.
    2. Replaces ``self.experts`` with a ``SharedFusedMoE`` whose routing is a
       no-op unpack of our pre-computed (topk_weights, topk_indices) — the
       fp32 softmax/topk/renormalize is done in :meth:`forward` here, exactly
       mirroring HF's ``easy_topk`` math (including ``clamp(min=1e-8)``).
    """

    def __init__(
        self,
        config,
        quant_config=None,
        layer_id: int = -1,
        prefix: str = "",
        enable_eplb: bool = False,
    ) -> None:
        # Bypass ``HunYuanSparseMoeBlock.__init__`` — it would build a wasteful
        # bf16 gate + a stub ``SharedFusedMoE`` we'd then have to del+recreate
        # (which trips ``ValueError: Duplicate layer name`` because the stub
        # already registered itself in ``compilation_config.static_forward_context``).
        # Instead, set up ``nn.Module`` ourselves and construct the fp32 gate
        # + ``custom_routing_function``-driven ``SharedFusedMoE`` directly,
        # mirroring the parent's structure 1:1 except for the routing dtype.
        nn.Module.__init__(self)

        self.tp_size = get_tensor_model_parallel_world_size()
        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than the number of experts {config.num_experts}."
            )

        if isinstance(config.moe_topk, list):
            top_k = config.moe_topk[layer_id]
        else:
            top_k = config.moe_topk
        self.top_k = top_k

        intermediate_size = config.intermediate_size
        if config.moe_intermediate_size is not None:
            intermediate_size = (
                config.moe_intermediate_size
                if isinstance(config.moe_intermediate_size, int)
                else config.moe_intermediate_size[layer_id]
            )

        vllm_config = get_current_vllm_config()
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = enable_eplb
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size
        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = self.physical_expert_start + self.n_local_physical_experts

        # FP32 router gate (HF: ``wg = nn.Linear(..., dtype=torch.float32)``).
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            params_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )

        if config.use_mixed_mlp_moe > 0:
            num_shared_expert = (
                config.num_shared_expert[layer_id]
                if isinstance(config.num_shared_expert, list)
                else config.num_shared_expert
            )
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

        # Experts with our ``_hunyuan_image3_unpack_packed_topk`` custom
        # routing — we feed it (topk_weights, topk_indices) packed into
        # ``router_logits`` in ``forward()`` so the bf16 ``topk_softmax``
        # CUDA op is bypassed entirely. ``renormalize=False`` because we
        # already did clamp+divide in fp32 to match HF's
        # ``topk_weight = topk_weight_1 / clamp(sum, min=1e-8)``.
        self.experts = SharedFusedMoE(
            shared_experts=self.shared_mlp,
            num_experts=self.n_routed_experts,
            top_k=top_k,
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            renormalize=False,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            custom_routing_function=_hunyuan_image3_unpack_packed_topk,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        # FP32 router (HF: `with torch.autocast('cuda', enabled=False): ...`
        # plus `if self.wg.weight.dtype == torch.float32: hidden_states.float()`).
        # ``self.gate.weight`` is fp32 (params_dtype=torch.float32), so the
        # ReplicatedLinear matmul runs in fp32 once we cast the input.
        router_logits, _ = self.gate(hidden_states.float())

        # softmax + topk + clamp-divide renormalization, all in fp32 — matches
        # ``HunyuanTopKGate.easy_topk`` exactly.
        gates = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_indices = torch.topk(gates, self.top_k, dim=-1)
        weight_sums = topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights / weight_sums.clamp(min=1e-8)

        # Cast topk weights to model dtype for the expert MLP combine.
        # HF: ``topk_weights = topk_weights.to(hidden_states.dtype)`` (line 1207).
        topk_weights = topk_weights.to(hidden_states.dtype)

        # Pack (weights, indices) into the ``router_logits`` slot so
        # ``_hunyuan_image3_unpack_packed_topk`` can pull them back out
        # inside ``SharedFusedMoE``. Both halves are stored as fp32 for
        # transport — the indices get cast back to int32 on unpack.
        packed_routing = torch.cat([topk_weights.float(), topk_indices.to(torch.float32)], dim=-1)

        # vllm 0.20+ FusedMoE merges shared-experts internally and runs the
        # TP all-reduce inside its forward (we no longer pass
        # `reduce_results=False`). The tuple `(routed, shared)` return shape
        # from the legacy SharedFusedMoE is gone; the result is the
        # already-combined, already-reduced tensor.
        final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=packed_routing)
        return final_hidden_states.view(orig_shape)


class HunyuanImage3RotaryEmbedding(nn.Module):
    """Custom interleaved 2D Rotary Embedding for HunyuanImage3.

    The original HunyuanImage3 ``build_2d_rope`` interleaves y (height) and
    x (width) positions across consecutive frequencies::

        theta = inv_freq.reshape(n_elem // 4, 2)
        idx_theta[2k]   = y_pos * theta[k, 0]   # even freq -> y
        idx_theta[2k+1] = x_pos * theta[k, 1]   # odd  freq -> x

    vLLM's standard ``MRotaryEmbedding`` instead assigns *contiguous* blocks
    of frequencies to each position dimension, producing different encodings
    for the same positions.  This class re-implements the original interleaved
    pattern so that pre-trained weights work correctly.

    Reference: https://kexue.fm/archives/10352
    """

    def __init__(self, head_dim: int, rope_theta: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        assert head_dim % 4 == 0, f"head_dim must be divisible by 4, got {head_dim}"
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.dim() == 2 and positions.shape[0] == 3:
            y_pos = positions[1].float()
            x_pos = positions[2].float()
        else:
            # 1D fallback: both dims get the same position → standard RoPE
            y_pos = positions.float()
            x_pos = positions.float()

        num_tokens = y_pos.shape[0]
        dtype = query.dtype
        query_shape = query.shape
        key_shape = key.shape

        inv_freq = self.inv_freq.to(device=y_pos.device, dtype=torch.float32)

        inv_freq_y = inv_freq[0::2]  # even indices -> y
        inv_freq_x = inv_freq[1::2]  # odd  indices -> x

        y_freqs = y_pos.unsqueeze(-1) * inv_freq_y.unsqueeze(0)
        x_freqs = x_pos.unsqueeze(-1) * inv_freq_x.unsqueeze(0)

        # Interleave: [y*θ₀, x*θ₁, y*θ₂, x*θ₃, ...]
        freqs = torch.stack([y_freqs, x_freqs], dim=-1).reshape(num_tokens, -1)

        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(dtype).unsqueeze(1)
        sin = emb.sin().to(dtype).unsqueeze(1)

        query = query.view(num_tokens, -1, self.head_dim)
        key = key.view(num_tokens, -1, self.head_dim)

        query = query * cos + self._rotate_half(query) * sin
        key = key * cos + self._rotate_half(key) * sin

        return query.view(query_shape), key.view(key_shape)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


@MULTIMODAL_REGISTRY.register_processor(
    HunyuanImage3MultiModalProcessor,
    info=HunyuanImage3ProcessingInfo,
    dummy_inputs=HunyuanImage3DummyInputsBuilder,
)
class HunyuanImage3ForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsLoRA, SupportsPP, SupportsMRoPE):
    """
    HunyuanImage3.0 model for conditional image generation.

    This is the main entry point for HunyuanImage3.0 in vLLM. It wraps:
    - HunyuanModel
    - VAE Encoder (AutoencoderKLConv3D + TimestepEmbedder + UNetDown)
    - ViT Encoder (Siglip2VisionTransformer + LightProjector)
    - LM Head for token prediction

    Supports:
    - Text-to-Text and Image-to-Text generation
    - Tensor Parallelism
    """

    HunyuanImage3Inputs: TypeAlias = HunyuanImage3PixelInputs

    prefer_model_sampler = True

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

        # Use mRoPE to preserve 2D positional encoding for image tokens.
        if isinstance(config.rope_parameters, dict):
            config.rope_parameters["rope_type"] = "default"
            head_dim = getattr(
                config,
                "head_dim",
                getattr(config, "attention_head_dim", config.hidden_size // config.num_attention_heads),
            )
            config.rope_parameters["mrope_section"] = [0, head_dim // 4, head_dim // 4]

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

        # --- AR-stage components ---
        # These are needed for image encoding in the AR stage.
        # If a future text-only stage is added, gate on vllm_config.model_config.model_stage.

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

        tokenizer = get_tokenizer(vllm_config.model_config.tokenizer)
        self._mrope_img_token_id = tokenizer.convert_tokens_to_ids("<img>")
        self._mrope_boi_token_id = tokenizer.convert_tokens_to_ids("<boi>")
        self._mrope_eoi_token_id = tokenizer.convert_tokens_to_ids("<eoi>")
        self._mrope_joint_img_sep_token_id = tokenizer.convert_tokens_to_ids("<joint_img_sep>")
        self._mrope_max_num_patches = config.vit_processor.get("max_num_patches", 729)

        # Special token IDs for logits processors (stage transitions).
        # These mirror the official tokenization_hunyuan_image_3.py setup.
        self._end_of_think_id = tokenizer.convert_tokens_to_ids("</think>")
        self._recaption_id = tokenizer.convert_tokens_to_ids("<recaption>")
        self._end_of_recaption_id = tokenizer.convert_tokens_to_ids("</recaption>")
        self._answer_id = tokenizer.convert_tokens_to_ids("<answer>")
        self._end_of_answer_id = tokenizer.convert_tokens_to_ids("</answer>")
        image_base_size = getattr(config, "image_base_size", 1024)
        self._size_token_id = tokenizer.convert_tokens_to_ids(f"<img_size_{image_base_size}>")
        self._timestep_token_id = tokenizer.convert_tokens_to_ids("<timestep>")
        self._start_ratio_id = tokenizer.convert_tokens_to_ids("<img_ratio_0>")
        self._end_ratio_id = tokenizer.convert_tokens_to_ids("<img_ratio_32>")
        ratio_33 = tokenizer.convert_tokens_to_ids("<img_ratio_33>")
        ratio_36 = tokenizer.convert_tokens_to_ids("<img_ratio_36>")
        self._ratio_other_slices = [(ratio_33, ratio_36 + 1)]
        # Build the full set of ratio token IDs for use as stop tokens.
        self._all_ratio_ids = set(range(self._start_ratio_id, self._end_ratio_id + 1))
        for s, e in self._ratio_other_slices:
            self._all_ratio_ids.update(range(s, e))

        # Determine mode: comprehension (I2T/T2T) vs generation (IT2I/T2I).
        engine_output_type = getattr(vllm_config.model_config, "engine_output_type", None)
        self._is_comprehension = engine_output_type in (None, "text")

        # For comprehension mode, block image generation tokens but allow
        # text structure tokens (<think>, <answer>, etc.) so the model can
        # follow its natural generation pattern. Runtime sampling params
        # decide stop tokens from the active bot_task, matching the official
        # HunyuanImage3 generation path without hard-coded YAML token ids.
        self._blocked_token_ids: set[int] = set()
        if self._is_comprehension:
            self._blocked_token_ids.update(
                [
                    self._mrope_boi_token_id,  # <boi>
                    self._mrope_eoi_token_id,  # <eoi>
                    self._size_token_id,  # <img_size_*>
                ]
            )
            self._blocked_token_ids.update(self._all_ratio_ids)

        # For generation mode, build stage transition map.
        # Official logic: </think> → [<recaption>],
        #   </recaption> → [<answer>, <boi>, <img_size_*>]
        # After <img_size_*>, restrict vocab to ratio tokens only.
        # Stage-transition forced sequences, keyed by trigger token.
        self._stage_transitions: dict[int, list[int]] = {}
        if not self._is_comprehension:
            self._stage_transitions[self._end_of_think_id] = [
                self._recaption_id,
            ]
            self._stage_transitions[self._end_of_recaption_id] = [
                self._answer_id,
                self._mrope_boi_token_id,
                self._size_token_id,
            ]

        self._sampler: Sampler | None = None
        self._eos_token_id: int = tokenizer.eos_token_id

        self._replace_rotary_embeddings()
        self._patch_moe_blocks()

    def _patch_moe_blocks(self):
        """Replace stock ``HunYuanSparseMoeBlock`` instances with
        :class:`HunyuanImage3SparseMoeBlock`, which routes in fp32 to match
        the HF reference (``modeling_hunyuan_image_3.HunyuanMoE``).

        Stock vLLM builds the router gate as a default-dtype (bf16)
        ``ReplicatedLinear`` and lets ``SharedFusedMoE``'s ``topk_softmax``
        kernel consume bf16 logits, which is the largest deterministic
        precision gap remaining vs HF after the prompt/preprocessing
        alignment fixes. See ``HunyuanImage3SparseMoeBlock`` docstring for
        the full rationale.

        Must run before weight loading (still inside ``__init__``) so the
        replacement gate's fp32 ``params_dtype`` is honored when the
        checkpoint is loaded.
        """
        if not _is_moe(self.config):
            return
        enable_eplb = getattr(self.vllm_config.parallel_config, "enable_eplb", False)
        ccfg = self.vllm_config.compilation_config
        replaced = 0
        for layer_id, layer in enumerate(self.model.layers):
            mlp = getattr(layer, "mlp", None)
            if isinstance(mlp, HunYuanSparseMoeBlock) and not isinstance(mlp, HunyuanImage3SparseMoeBlock):
                # Pop the OLD experts' registration from
                # ``static_forward_context`` first — otherwise the new
                # ``SharedFusedMoE`` built inside
                # :class:`HunyuanImage3SparseMoeBlock` will trip
                # ``ValueError: Duplicate layer name`` (see
                # vllm/model_executor/layers/fused_moe/layer.py:327).
                old_prefix = f"model.layers.{layer_id}.mlp.experts"
                ccfg.static_forward_context.pop(old_prefix, None)
                if old_prefix in ccfg.static_all_moe_layers:
                    ccfg.static_all_moe_layers.remove(old_prefix)

                # Free the OLD MoE block's GPU buffers BEFORE allocating
                # the replacement. The parent ``SharedFusedMoE`` pre-
                # allocates the full ``[num_experts, ...]`` expert weight
                # tensors at ``__init__`` (~750 MiB per layer per worker
                # on this 80B model with TP=2), so without this drop we
                # transiently double the MoE footprint and OOM near the
                # gpu_memory_utilization cap.
                layer.mlp = None
                del mlp
                gc.collect()
                torch.accelerator.empty_cache()

                layer.mlp = HunyuanImage3SparseMoeBlock(
                    config=self.config,
                    quant_config=self.quant_config,
                    layer_id=layer_id,
                    prefix=f"model.layers.{layer_id}.mlp",
                    enable_eplb=enable_eplb,
                )
                replaced += 1
        logger.info(
            "Replaced %d HunYuanSparseMoeBlock layers with "
            "HunyuanImage3SparseMoeBlock (fp32 router matching HF reference)",
            replaced,
        )
        if replaced == 0:
            logger.warning(
                "HunyuanImage3: _patch_moe_blocks replaced 0 layers. "
                "Routing will run in bf16 instead of fp32 — output will "
                "diverge from the HF reference more than necessary. "
                "Check that model.layers[*].mlp is HunYuanSparseMoeBlock."
            )

    def _replace_rotary_embeddings(self):
        """Replace vLLM's standard MRotaryEmbedding with the custom
        interleaved 2D RoPE that matches the original HunyuanImage3 model."""
        rope_theta = getattr(self.config, "rope_theta", 10000.0)
        head_dim = getattr(
            self.config,
            "head_dim",
            getattr(
                self.config,
                "attention_head_dim",
                self.config.hidden_size // self.config.num_attention_heads,
            ),
        )
        custom_rope = HunyuanImage3RotaryEmbedding(
            head_dim=head_dim,
            rope_theta=rope_theta,
        )
        replaced = 0
        for layer in self.model.layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "rotary_emb"):
                layer.self_attn.rotary_emb = custom_rope
                replaced += 1
        logger.info(
            "Replaced %d rotary embeddings with HunyuanImage3RotaryEmbedding "
            "(interleaved 2D RoPE, head_dim=%d, rope_theta=%.1f)",
            replaced,
            head_dim,
            rope_theta,
        )
        if replaced == 0:
            raise RuntimeError(
                "HunyuanImage3: _replace_rotary_embeddings replaced 0 layers. "
                "The custom interleaved 2D mRoPE is not active — model outputs "
                "will be incorrect. Check that model.layers[*].self_attn.rotary_emb exists."
            )

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
        # vae_pixel_size is only metadata for vLLM's flat_from_sizes split;
        # we reconstruct per-image shapes from vae_token_grid_hw below.
        kwargs.pop("vae_pixel_size", None)
        vae_token_grid_hw = kwargs.pop("vae_token_grid_hw", None)

        if vit_pixel_values is None or vae_pixel_values is None:
            return None

        # Handle empty batch (e.g., during profiling with 0 images / T2T mode)
        if vit_pixel_values.numel() == 0 or vae_pixel_values.numel() == 0:
            return None

        # `vae_pixel_values` arrives as a 1-D concatenation of per-image flat
        # buffers (see `process_image` + `flat_from_sizes`). Reconstruct a
        # list of per-image (3, H_i, W_i) tensors using the per-image grid
        # dims so the downstream VAE encoder can run image-by-image.
        vae_factor_h = self.config.vae_downsample_factor[0] * self.config.patch_size
        vae_factor_w = self.config.vae_downsample_factor[1] * self.config.patch_size
        num_images = vae_token_grid_hw.shape[0]
        vae_image_list: list[torch.Tensor] = []
        offset = 0
        flat = vae_pixel_values.reshape(-1)
        for i in range(num_images):
            token_h, token_w = vae_token_grid_hw[i].tolist()
            h_i = int(token_h) * vae_factor_h
            w_i = int(token_w) * vae_factor_w
            n_i = 3 * h_i * w_i
            vae_image_list.append(flat[offset : offset + n_i].reshape(3, h_i, w_i))
            offset += n_i
        if offset != flat.numel():
            raise ValueError(
                f"vae_pixel_values size mismatch: consumed {offset} of {flat.numel()} elements "
                f"across {num_images} images (token_grid_hw={vae_token_grid_hw.tolist()})"
            )

        return HunyuanImage3PixelInputs(
            type="pixel_values",
            pixel_values={
                "vit_pixel_values": vit_pixel_values,
                "vit_pixel_attention_mask": vit_pixel_attention_mask,
                "vit_spatial_shapes": vit_spatial_shapes,
                "vae_pixel_values": vae_image_list,
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

        # Cast pixel input to model dtype here (at the encoder boundary)
        # rather than inside HunyuanImage3Processor.process_image. This
        # matches HF's path which keeps fp32 pixels in build_cond_images and
        # only casts inside the VAE forward — preserving fp32 precision in
        # the multimodal_data dict and minimizing precision drift vs HF.
        # Verified by pixel-tensor diff: removing the early bf16 cast brings
        # omni's vae_pixel_values byte-identical to HF's (within fp32 noise),
        # whereas an early cast leaves a ~7e-4 mean-abs-diff bf16 quantization
        # error on every element.
        if images.dtype != self.vae.dtype:
            images = images.to(dtype=self.vae.dtype)

        vae_encode_result = self.vae.encode(images)
        # Match HunyuanImage-3's cond encode path: sample the posterior, but
        # use a fixed generator so online requests do not consume the global
        # RNG and drift across a long-running server.
        _cond_vae_gen = torch.Generator(device=images.device).manual_seed(0)
        latents = vae_encode_result.latent_dist.sample(_cond_vae_gen)

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
                logger.warning(f"T({latents.shape[2]}) > 1, should not happen for conditional images")
                latents = latents[:, :, 0, :, :]

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
    ) -> torch.Tensor | None:
        """
        Encode pixel_values through ViT encoder (vision_model and vision_aligner).
        """
        # Handle empty batch
        if pixel_values.shape[0] == 0:
            return None

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

        # VAE encode + patch_embed per image — each cond image is at its own
        # `reso_group` bucket so shapes are ragged across the image-batch dim.
        vae_token_embeddings = []
        for vae_image_i in vae_pixel_values:
            t_i, latents_i = self._vae_encode(vae_image_i.unsqueeze(0), vae_cfg_factor)
            t_emb = self.time_embed(t_i[0])
            vae_tokens, _, _ = self.patch_embed(latents_i, t_emb)
            vae_token_embeddings.append(vae_tokens)

        assert vit_embeddings is not None and vit_embeddings.shape[0] == len(vae_token_embeddings), (
            f"Number of ViT embeddings ({vit_embeddings.shape[0]}) does not match "
            f"number of VAE token embeddings ({len(vae_token_embeddings)}). "
            "Each image should have both VAE and ViT embeddings."
        )

        # Order per image: VAE tokens -> <joint_img_sep> wte -> ViT tokens.
        # The <joint_img_sep> wte is included so it joins the bidirectional
        # MM region (matching the official `joint_image_slices` full-attn
        # range that spans VAE+sep+ViT). The merger replaces the sep slot
        # with this wte tensor, which is numerically identical to what
        # `model.embed_input_ids` would produce — no semantic change for
        # single-image, but with multi-image the sep position now sits
        # inside the bidirectional region (matching how the model was
        # trained).
        sep_token_id = self._mrope_joint_img_sep_token_id
        sep_input_ids = torch.tensor([sep_token_id], device=vit_embeddings.device, dtype=torch.long)
        sep_embed = self.model.embed_input_ids(sep_input_ids).to(vit_embeddings.dtype)

        # The <timestep> slot at the head of each per-image scaffold is NOT
        # included here — its embedding is patched in by `embed_input_ids`
        # via a token-id mask, mirroring HF's `instantiate_continuous_tokens`
        # scatter-replace.
        combined_embeddings: list[torch.Tensor] = []
        num_images = len(vae_token_embeddings)
        for img_idx in range(num_images):
            vae_token_embed = vae_token_embeddings[img_idx]
            if vae_token_embed.ndim == 3:
                vae_token_embed = vae_token_embed.squeeze(0)
            vit_embed = vit_embeddings[img_idx]
            stacked_embed = torch.cat([vae_token_embed, sep_embed, vit_embed], dim=0)
            combined_embeddings.append(stacked_embed)

        return combined_embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed input IDs with optional multimodal embeddings."""
        inputs_embeds = self.model.embed_input_ids(input_ids)

        # Patch <timestep> slots with timestep_emb(0). HF parity: the trained
        # wte at this slot is irrelevant; runtime uses
        # `instantiate_continuous_tokens(timestep_emb(0))`. With multi-image,
        # keeping these slots as <img> ids merged the timestep position into
        # the bidirectional MM region and biased AR ratio prediction toward
        # the first image's bucket.
        timestep_mask = input_ids == self._timestep_token_id
        n_timestep = int(timestep_mask.sum().item())
        if n_timestep > 0:
            timestep_input = torch.zeros((n_timestep,), device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            inputs_embeds[timestep_mask] = self._timestep_encode(timestep_input)

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

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

    # ------------------------------------------------------------------
    # Custom sampler — applies HunyuanImage3-specific logits processors
    # before the standard sampling step.
    #
    # Comprehension (I2T / T2T):
    #   Block generation-specific special tokens so sampling can't
    #   accidentally produce <answer>, <boi>, ratio tokens, etc.
    #
    # Generation (IT2I / T2I think):
    #   1. _StageTransitionLogitsProcessor — force token sequences at
    #      transition boundaries (</think> → <recaption>, etc.)
    #   2. _ConditionalSliceVocabLogitsProcessor — after <img_size_*>,
    #      restrict vocab to ratio tokens only (greedy).
    # ------------------------------------------------------------------

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        if logits is None or logits.numel() == 0:
            return None

        if self._sampler is None:
            self._sampler = Sampler()

        min_score = torch.finfo(logits.dtype).min

        for req_idx in range(logits.shape[0]):
            decoded_tokens: list[int] = (
                sampling_metadata.output_token_ids[req_idx] if req_idx < len(sampling_metadata.output_token_ids) else []
            )
            last_token = decoded_tokens[-1] if decoded_tokens else -1

            if self._is_comprehension:
                for tid in self._blocked_token_ids:
                    logits[req_idx, tid] = min_score
            else:
                forced = self._get_forced_token(decoded_tokens)
                if forced is not None:
                    logits[req_idx].fill_(min_score)
                    logits[req_idx, forced] = 0
                elif last_token == self._size_token_id:
                    self._apply_ratio_restriction(logits, req_idx, min_score)
                elif last_token in self._all_ratio_ids:
                    logits[req_idx].fill_(min_score)
                    logits[req_idx, self._eos_token_id] = 0

        return self._sampler(logits=logits, sampling_metadata=sampling_metadata)

    def _get_forced_token(self, decoded_tokens: list[int]) -> int | None:
        """Derive the next forced token from output history (stateless).

        Scans decoded_tokens backwards for the most recent trigger token,
        then prefix-matches the forced sequence against what followed.
        Returns the next token to force, or None if the sequence is complete
        or history has diverged from the expected forced sequence.
        """
        for i in range(len(decoded_tokens) - 1, -1, -1):
            trigger = decoded_tokens[i]
            if trigger not in self._stage_transitions:
                continue

            forced_seq = self._stage_transitions[trigger]
            emitted = decoded_tokens[i + 1 :]

            matched = 0
            for expected, actual in zip(forced_seq, emitted):
                if actual != expected:
                    # History diverged from the expected forced sequence.
                    # Stop applying transition forcing for safety.
                    return None
                matched += 1

            if matched < len(forced_seq):
                return forced_seq[matched]
            return None

        return None

    def _apply_ratio_restriction(
        self,
        logits: torch.Tensor,
        req_idx: int,
        min_score: float,
    ) -> None:
        """Port of official _ConditionalSliceVocabLogitsProcessor.__call__.

        After the size token, only allow ratio tokens and pick greedily.
        """
        original = logits[req_idx].clone()
        logits[req_idx].fill_(min_score)
        # Allow primary ratio range.
        logits[req_idx, self._start_ratio_id : self._end_ratio_id + 1] = original[
            self._start_ratio_id : self._end_ratio_id + 1
        ]
        # Allow extra ratio slices.
        for s, e in self._ratio_other_slices:
            logits[req_idx, s:e] = original[s:e]
        # Force greedy: keep only the argmax.
        max_id = logits[req_idx].argmax().item()
        logits[req_idx].fill_(min_score)
        logits[req_idx, max_id] = 0

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

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec] | None = None,
        *,
        hf_config: PretrainedConfig | None = None,
        image_grid_thw: list[list[int]] | torch.Tensor | None = None,
        video_grid_thw: list[list[int]] | torch.Tensor | None = None,
        second_per_grid_ts: list[float] | None = None,
        context_len: int = 0,
        seq_len: int | None = None,
        audio_feature_lengths: torch.Tensor | None = None,
        use_audio_in_video: bool = False,
    ) -> tuple[torch.Tensor, int]:
        """Compute mRoPE positions for HunyuanImage-3.

        Maps the original model's build_2d_rope logic into vLLM's 3-dim
        mRoPE position tensor ``[3, seq_len]`` where dim-1 is height and
        dim-2 is width.  dim-0 is unused (temporal) and kept equal to 1D.

        For text tokens and auxiliary image tokens (timestep, separators):
            All three dims get the same flat 1D position id.
        For VAE / ViT image tokens:
            dim-0 (T): flat 1D position id at the region start
            dim-1 (H): 2D y-position using build_2d_rope centering
            dim-2 (W): 2D x-position using build_2d_rope centering
        """

        # Extract per-image VAE and ViT grid dims from mm_features
        vae_grids: list[tuple[int, int]] = []
        vit_grids: list[tuple[int, int]] = []
        if mm_features is not None:
            for mm_feature in mm_features:
                mm_item = mm_feature.data
                if mm_item is None:
                    continue
                mm_input = mm_item.get_data()
                vae_hw = mm_input.get("vae_token_grid_hw")
                if vae_hw is not None:
                    grid = vae_hw.tolist()
                    vae_grids.append((int(grid[0]), int(grid[1])))
                vit_hw = mm_input.get("vit_spatial_shapes")
                if vit_hw is not None:
                    grid = vit_hw.tolist()
                    vit_grids.append((int(grid[0]), int(grid[1])))

        img_token_id = self._mrope_img_token_id
        boi_token_id = self._mrope_boi_token_id
        eoi_token_id = self._mrope_eoi_token_id
        joint_img_sep_token_id = self._mrope_joint_img_sep_token_id
        timestep_token_id = self._timestep_token_id

        # Build position arrays
        t_pos: list[int] = []  # temporal (same as 1D for this model)
        h_pos: list[int] = []  # height
        w_pos: list[int] = []  # width

        pos = 0
        image_idx = 0
        i = 0
        n = len(input_tokens)

        while i < n:
            tok = input_tokens[i]

            if tok == boi_token_id:
                # Found start of image block.
                # Structure: <boi> <size> <ratio> <timestep> <img>*vae
                #            <joint_img_sep> <img>*vit <eoi>
                # <boi> token
                t_pos.append(pos)
                h_pos.append(pos)
                w_pos.append(pos)
                pos += 1
                i += 1

                # <size> token
                if i < n:
                    t_pos.append(pos)
                    h_pos.append(pos)
                    w_pos.append(pos)
                    pos += 1
                    i += 1

                # <ratio> token
                if i < n:
                    t_pos.append(pos)
                    h_pos.append(pos)
                    w_pos.append(pos)
                    pos += 1
                    i += 1

                # <timestep> token (1 token)
                if i < n and input_tokens[i] == timestep_token_id:
                    t_pos.append(pos)
                    h_pos.append(pos)
                    w_pos.append(pos)
                    pos += 1
                    i += 1

                # VAE tokens: 2D grid positions
                if image_idx < len(vae_grids):
                    vae_h, vae_w = vae_grids[image_idx]
                else:
                    vae_h, vae_w = 0, 0
                L = pos
                wh = vae_w * vae_h
                beta_y = L + (wh - vae_h) / 2
                beta_x = L + (wh - vae_w) / 2

                for row in range(vae_h):
                    for col in range(vae_w):
                        if i < n and input_tokens[i] == img_token_id:
                            t_pos.append(L)
                            h_pos.append(int(beta_y + row))
                            w_pos.append(int(beta_x + col))
                            i += 1

                pos = L + wh

                # <joint_img_sep> token
                if i < n and input_tokens[i] == joint_img_sep_token_id:
                    t_pos.append(pos)
                    h_pos.append(pos)
                    w_pos.append(pos)
                    pos += 1
                    i += 1

                # ViT tokens: 2D grid positions
                if image_idx < len(vit_grids):
                    vit_h, vit_w = vit_grids[image_idx]
                else:
                    vit_h, vit_w = 0, 0
                L = pos
                wh = vit_w * vit_h
                beta_y = L + (wh - vit_h) / 2
                beta_x = L + (wh - vit_w) / 2

                for row in range(vit_h):
                    for col in range(vit_w):
                        if i < n and input_tokens[i] == img_token_id:
                            t_pos.append(L)
                            h_pos.append(int(beta_y + row))
                            w_pos.append(int(beta_x + col))
                            i += 1

                pos = L + wh

                # Remaining ViT padding tokens (when max_num_patches >
                # vit_h * vit_w) get flat 1D positions.
                while i < n and input_tokens[i] == img_token_id:
                    t_pos.append(pos)
                    h_pos.append(pos)
                    w_pos.append(pos)
                    pos += 1
                    i += 1

                # <eoi> token
                if i < n and input_tokens[i] == eoi_token_id:
                    t_pos.append(pos)
                    h_pos.append(pos)
                    w_pos.append(pos)
                    pos += 1
                    i += 1

                image_idx += 1
            else:
                # Regular text token — flat 1D position
                t_pos.append(pos)
                h_pos.append(pos)
                w_pos.append(pos)
                pos += 1
                i += 1

        llm_positions = torch.tensor([t_pos, h_pos, w_pos], dtype=torch.long)
        mrope_position_delta = llm_positions.max() + 1 - len(input_tokens)

        if seq_len is not None:
            llm_positions = llm_positions[:, context_len:seq_len]

        return llm_positions, mrope_position_delta
