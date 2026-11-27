# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from __future__ import annotations

import json
import os
from functools import cache, lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index

from vllm_omni.diffusion.attention.backends.abstract import AttentionBackend, AttentionImpl, AttentionMetadata
from vllm_omni.diffusion.config import get_current_diffusion_config_or_none
from vllm_omni.diffusion.forward_context import get_forward_context

if TYPE_CHECKING:
    from vllm_omni.diffusion.attention.layer import Attention

logger = init_logger(__name__)

SOL_MORTON_ORDER = "sol_morton_order"


def _has_cann_kernel(root: Path, arch: str, binary: str) -> bool:
    """Follow the CANN manifest to the compiled object for each kernel variant."""
    manifest = root / "config" / arch / f"{binary}.json"
    try:
        variants = json.loads(manifest.read_bytes())["binList"]
        if not isinstance(variants, list) or not variants:
            return False
        for variant in variants:
            metadata_path = root / variant["binInfo"]["jsonFilePath"]
            metadata = json.loads(metadata_path.read_bytes())
            obj = metadata_path.with_name(metadata["binFileName"] + metadata["binFileSuffix"])
            if not obj.is_file():
                return False
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError, KeyError, TypeError, ValueError):
        return False
    return True


def _sol_unavailable_reason() -> str | None:
    try:
        import mindiesd
    except ModuleNotFoundError as exc:
        if exc.name != "mindiesd":
            raise
        return "MindIE-SD is not installed"
    if not callable(getattr(mindiesd, "sol_attn", None)):
        return "MindIE-SD does not export a callable sol_attn"

    import torch_npu

    # MindIE-SD #929 uses three common torch operators. On A2 its short
    # sequence and fused routing paths also need four A2-specific operators.
    soc = torch_npu.npu.get_soc_version()
    ops: tuple[str, ...] = ("prepare_qkv", "prepare_route_pack", "fwd_m128")
    if 220 <= soc <= 225:
        arch = "ascend910b"
        ops += ("prepare_diag_thresh", "prepare_short", "prepare_route_stats", "short_fwd_m128")
        binaries = tuple(f"sol_attn_{op}_kernel" for op in ops)
    elif soc == 260:
        arch = "ascend950"
        binaries = ("sol_attn_a5_summary", "sol_attn_a5_route", "sol_attn_a5_forward")
    else:
        return f"MindIE-SD Sol-Attn does not support NPU SoC {soc}"

    for op in ops:
        name = f"sol_attn_{op}"
        if not hasattr(torch.ops.mindiesd, name) or not torch._C._dispatch_has_kernel_for_dispatch_key(
            f"mindiesd::{name}", "PrivateUse1"
        ):
            return f"missing NPU operator mindiesd::{name}"

    # The torch plugin registers schemas independently of the SoC binaries
    # shipped in the CANN custom OPP package.
    roots = [
        Path(path) / "op_impl/ai_core/tbe/kernel"
        for path in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
        if path
    ]
    for binary in binaries:
        if not any(_has_cann_kernel(root, arch, binary) for root in roots):
            return f"missing or incomplete CANN kernel {binary} for {arch}"
    return None


@cache
def _sol_available() -> bool:
    """Resolve once in the single-NPU worker, before choosing a token layout."""
    reason = _sol_unavailable_reason()
    if reason is not None:
        logger.warning_once(
            "SOL_ATTN unavailable: %s. This worker will continue Wan self-attention with "
            "FLASH_ATTN (full attention), without Sol sparse acceleration.",
            reason,
        )
    return reason is None


@lru_cache(maxsize=8)
def get_morton_indices(grid: tuple[int, int, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Map raster video tokens to 3D Z-order (W, H, T bits) and back."""
    frames, height, width = grid
    raster = torch.arange(frames * height * width, device="cpu")
    w, h, t = raster % width, raster // width % height, raster // (height * width)
    code = torch.zeros_like(raster)
    for bit in range((max(grid) - 1).bit_length()):
        for axis, coord in enumerate((w, h, t)):
            code |= ((coord >> bit) & 1) << (3 * bit + axis)
    order = code.argsort()
    return order.to(device), order.argsort().to(device)


def wan_sol_layout(last_attention: Attention) -> dict[str, bool]:
    """All Wan blocks share a profile. With a dense prefix, the last layer is sparse if any layer is."""
    impl, _, _ = last_attention.effective_attention()
    if not isinstance(impl, SolAttnImpl):
        return {}
    step = get_forward_context().denoise_step_idx
    return {
        SOL_MORTON_ORDER: not impl.dense_layer and step is not None and step >= impl.dense_steps and _sol_available()
    }


class SolAttnBackend(AttentionBackend):
    supported_platforms = ("npu",)

    @staticmethod
    def get_name() -> str:
        return "SOL_ATTN"

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [128]

    @staticmethod
    def get_impl_cls() -> type[SolAttnImpl]:
        return SolAttnImpl


class SolAttnImpl(AttentionImpl[AttentionMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        qkv_layout: str | None = None,
        role: str = "self",
        prefix: str = "",
        backend_kwargs: dict[str, float | int] | None = None,
        **kwargs: object,
    ) -> None:
        if (
            role != "self"
            or causal
            or qkv_layout != "BSND"
            or head_size != 128
            or num_kv_heads not in (None, num_heads)
        ):
            raise ValueError("SOL_ATTN requires noncausal Wan self-attention with BSND layout and head size 128.")
        config = get_current_diffusion_config_or_none()
        if config is not None:
            if config.dtype != torch.bfloat16:
                raise ValueError("SOL_ATTN requires BF16. Set the diffusion dtype to bfloat16.")
            if config.parallel_config.world_size != 1:
                raise ValueError("SOL_ATTN currently supports single-device Wan inference.")
            if config.diffusion_use_graph or config.cache_backend not in (None, "none"):
                raise ValueError("SOL_ATTN step-dependent token ordering requires uncached, non-graphed DiT execution.")
        options = backend_kwargs or {}
        self.tau = options.get("tau", 1.0)
        self.scale = softmax_scale
        self.dense_steps = options.get("dense_steps", 4)
        self.dense_layer = extract_layer_index(prefix) < options.get("dense_layers", 1)
        from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionBackend

        self.dense = FlashAttentionBackend.get_impl_cls()(
            num_heads=num_heads, head_size=head_size, softmax_scale=softmax_scale, qkv_layout=qkv_layout
        )

    def forward_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None or SOL_MORTON_ORDER not in attn_metadata.extra or attn_metadata.attn_mask is not None:
            raise ValueError("SOL_ATTN needs an unmasked Wan video token sequence with a declared token order.")
        if self.dense_layer or not attn_metadata.extra[SOL_MORTON_ORDER]:
            return self.dense.forward_npu(query, key, value, attn_metadata)
        from mindiesd import sol_attn

        return sol_attn(query.contiguous(), key.contiguous(), value.contiguous(), scale=self.scale, tau=self.tau)
