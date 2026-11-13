# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Online MXFP8 for the native MiniMax H3 video decoder on Ascend A5."""

from __future__ import annotations

from collections.abc import Mapping
from types import MethodType
from typing import TYPE_CHECKING

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.platforms.npu import is_a5
from vllm_omni.quantization.component_config import ComponentQuantizationConfig

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

    from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

logger = init_logger(__name__)
_MXFP8_FA_METADATA = AttentionMetadata(extra={"kv_cache_dtype": "mxfp8"})


def resolve_h3_vae_mxfp8(
    config: QuantizationConfig | None, device: torch.device
) -> DiffusionMXFP8Config | None:
    if not isinstance(config, ComponentQuantizationConfig):
        return None

    # Global or inherited MXFP8 still only affects the original components.
    decoder_config = config.component_configs.get("video_vae")
    if decoder_config is None or decoder_config.get_name() != "mxfp8":
        return None
    if device.type != "npu" or not is_a5(device):
        raise ValueError("video_vae MXFP8 requires an Ascend A5 device")

    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    if not isinstance(decoder_config, DiffusionMXFP8Config):
        return None
    if decoder_config.is_checkpoint_mxfp8_serialized or decoder_config.ignored_layers:
        raise ValueError("video_vae MXFP8 requires a floating checkpoint and all decoder block projections")
    return decoder_config


class _NPUMxfp8VideoVAELinear(nn.Module):
    """Eager Linear adapter using the same MXFP8 layout and matmul as the DiT."""

    def __init__(
        self,
        linear: nn.Linear,
        config: DiffusionMXFP8Config,
        device: torch.device,
        load_device: torch.device,
    ) -> None:
        super().__init__()
        from vllm_omni.quantization.mxfp8_config import NPUMxfp8OnlineLinearMethod

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = nn.Parameter(linear.weight.detach().to(device=device, dtype=torch.bfloat16), requires_grad=False)
        self.bias = linear.bias
        self._method = NPUMxfp8OnlineLinearMethod(config)
        self._method.process_weights_after_loading(self)

        # Host staging reconstructs tensors from strided storage. Keep its master
        # in ND, since CPU transfer drops the device's private NZ format anyway.
        if load_device.type == "npu":
            from vllm_ascend.utils import maybe_trans_nz

            weight = maybe_trans_nz(self.weight.detach(), customize_dtype=self.weight.dtype)
            self.weight = nn.Parameter(weight, requires_grad=False)
        self.to(device=load_device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The native VAE can run with FP32 residuals; MX quant accepts FP16/BF16.
        dtype = hidden_states.dtype
        autocast = torch.is_autocast_enabled("npu")
        if dtype == torch.float32:
            compute_dtype = torch.get_autocast_dtype("npu") if autocast else torch.bfloat16
            hidden_states = hidden_states.to(compute_dtype)
        result = self._method.apply(self, hidden_states, self.bias)
        return result if autocast else result.to(dtype)


@torch.compiler.disable
def _perform_mxfp8_attention_npu(
    self: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    pack_info: Mapping[str, object] | None = None,
) -> torch.Tensor:
    pack_info = {} if pack_info is None else pack_info
    if (
        self.training
        or query.device.type != "npu"
        or query.ndim != 4
        or query.numel() == 0
        or query.dtype not in (torch.float16, torch.bfloat16)
        or any(t.shape != query.shape or t.dtype != query.dtype or t.device != query.device for t in (key, value))
        or any(pack_info.get(name) is not None for name in ("cu_seqlens", "mask_mod", "block_sparse"))
    ):
        return type(self).perform_attention(self, query, key, value, pack_info)

    logger.info_once("MiniMax H3 Video VAE uses MindIE-SD MXFP8 quantized attention.")
    return _run_mxfp8_fa(self._omni_mxfp8_fa, query, key, value)


def _run_mxfp8_fa(
    backend: FlashAttentionImpl,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    quant_fa = backend.forward_fa_quant_npu
    if query.shape[0] == 1 or query.shape[1] % 64 == 0:
        return quant_fa(query, key, value, _MXFP8_FA_METADATA)

    # MindIE flattens BSND into TND and groups V scales across 64 sequence
    # rows. An unaligned tile must keep its own scale groups and sequence end.
    # Write into the result as each tile completes to keep scratch bounded.
    output = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    for index in range(query.shape[0]):
        tile = slice(index, index + 1)
        output[tile].copy_(quant_fa(query[tile], key[tile], value[tile], _MXFP8_FA_METADATA))
    return output


def install_h3_vae_mxfp8(
    decoder: nn.Module | None,
    config: DiffusionMXFP8Config,
    *,
    device: torch.device,
    load_device: torch.device,
) -> None:
    blocks = getattr(decoder, "transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        raise ValueError("video_vae MXFP8 requires a native MiniMax H3 decoder with transformer blocks")

    from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl

    targets: list[tuple[nn.Module, str, nn.Linear]] = []
    attention_layers: list[tuple[nn.Module, FlashAttentionImpl]] = []
    for block in blocks:
        attention = getattr(block, "attn", None)
        if not isinstance(attention, nn.Module) or not callable(getattr(type(attention), "perform_attention", None)):
            raise ValueError("video_vae MXFP8 requires the native decoder attention entry point")
        heads = getattr(attention, "heads", None)
        head_dim = getattr(attention, "dim_head", None)
        if (
            "perform_attention" in vars(attention)
            or not isinstance(heads, int)
            or heads <= 0
            or not isinstance(head_dim, int)
            or head_dim < 32
            or head_dim & (head_dim - 1)
        ):
            raise ValueError("Unsupported native decoder attention for video_vae MXFP8")
        backend = FlashAttentionImpl(
            num_heads=heads, head_size=head_dim, softmax_scale=head_dim**-0.5, qkv_layout="BSND"
        )
        attention_layers.append((attention, backend))
        projections = (
            (attention, ("to_qkv", "to_out")),
            (getattr(block, "ff", None), ("w1", "w2")),
        )
        for owner, names in projections:
            if not isinstance(owner, nn.Module):
                raise ValueError(f"Missing video_vae MXFP8 projections: {names}")
            for name in names:
                linear = getattr(owner, name, None)
                if (
                    not isinstance(linear, nn.Linear)
                    or linear.weight.dtype not in (torch.float32, torch.float16, torch.bfloat16)
                    or linear.in_features % 32
                    or linear.out_features % 32
                ):
                    raise ValueError(f"Unsupported video_vae MXFP8 projection: {name}")
                targets.append((owner, name, linear))

    for owner, name, linear in targets:
        setattr(owner, name, _NPUMxfp8VideoVAELinear(linear, config, device, load_device))
    for attention, backend in attention_layers:
        attention._omni_mxfp8_fa = backend
        attention.perform_attention = MethodType(_perform_mxfp8_attention_npu, attention)
    logger.info(
        "Applied online A5 MXFP8 to %d MiniMax H3 video decoder projections and %d attention layers",
        len(targets),
        len(attention_layers),
    )
