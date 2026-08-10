# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionBackend
from vllm_omni.diffusion.config import get_current_diffusion_config_or_none
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available

logger = init_logger(__name__)

# The rf_v2 kernel only implements a 128-token block.
_BLOCK_SIZE = 128

# Below this many video blocks, the pooling and gather that block selection adds
# cost more than the QK work it removes, so stay dense.
_MIN_VIDEO_BLOCKS = 32

# rf_v2's ``input_layout`` describes the caller's tensors, and everything below
# slices the sequence on dim 1. vLLM-Omni diffusion attention hands the impl
# [B, S, N, D], so that is the only layout this backend accepts.
_INPUT_LAYOUT = "BSND"

# rf_v2 precision mode: 0 = high precision, 1 = high performance. Kept at the
# precise setting because the sparsity knob is the intended perf lever, and on
# A5 devices the kernel is routed to rf_v3, which overrides this anyway.
_INNER_PRECISE = 0

_WRONG_PLATFORM = (
    "RAINFUSION_ATTN runs the MindIE-SD rf_v2 kernel and is available on Ascend NPU only. "
    "Select FLASH_ATTN or TORCH_SDPA on this platform."
)

_MISSING_MINDIESD = (
    "RAINFUSION_ATTN requires MindIE-SD. Please install MindIE-SD to enable RainFusion sparse "
    "attention on Ascend NPU. For installation details, see https://gitcode.com/Ascend/MindIE-SD "
    "Otherwise, use FlashAttention by setting DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN"
)


def _try_extract_layer_index(prefix: str) -> int | None:
    if not prefix:
        return None
    try:
        return extract_layer_index(prefix)
    except (AssertionError, ValueError):
        return None


@dataclass(frozen=True)
class RainFusionConfig:
    """Resolved RainFusion controls for one attention layer.

    ``sparsity`` is the nominal fraction of key blocks dropped per query block.
    The realized sparsity is lower because rf_v2 always keeps the prefix rows and
    the first-frame blocks. ``start_step`` and ``skip_layers`` are the accuracy
    knobs: early denoise steps and specific DiT blocks stay dense.
    """

    sparsity: float = 0.0
    start_step: int = 0
    skip_layers: frozenset[int] = frozenset()

    @classmethod
    def from_backend_kwargs(cls, backend_kwargs: dict | None) -> RainFusionConfig:
        bk = backend_kwargs or {}
        return cls(
            sparsity=float(bk.get("sparsity", 0.0)),
            start_step=int(bk.get("start_step", 0)),
            skip_layers=frozenset(bk.get("skip_layers") or ()),
        )

    @property
    def enabled(self) -> bool:
        return self.sparsity > 0.0


@dataclass(frozen=True)
class RainFusionPlan:
    """Per-forward geometry handed to the rf_v2 kernel."""

    prefix_len: int
    used_len: int
    latent_shape: list[int]


class RainFusionAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_platforms: tuple[str, ...] = ("npu",)

    @classmethod
    def validate_available(cls) -> None:
        from importlib.util import find_spec

        if find_spec("mindiesd") is None:
            raise ValueError(_MISSING_MINDIESD)

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 96, 128, 192, 256]

    @staticmethod
    def get_name() -> str:
        return "RAINFUSION_ATTN"

    @staticmethod
    def get_impl_cls() -> type[RainFusionAttentionImpl]:
        return RainFusionAttentionImpl


class RainFusionAttentionImpl(AttentionImpl):
    """Block-sparse video attention via MindIE-SD RainFusion (rf_v2) on Ascend NPU.

    Sparsity applies only to the video segment of a packed multimodal sequence,
    whose extent the model publishes as ``AttentionMetadata.video_layout``. Every
    other case — warmup denoise steps, exempt layers, a layer that does not declare
    ``qkv_layout="BSND"``, sequences without a published video segment, video
    segments too short to pay for block selection — delegates to FlashAttention,
    so a model can select this backend unconditionally. MindIE-SD handles an
    irregular video tail internally, retaining it outside the sparse blocks.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        qkv_layout: str | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        **extra_impl_args,
    ) -> None:
        self.num_heads = num_heads
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.qkv_layout = qkv_layout

        self.rainfusion = RainFusionConfig.from_backend_kwargs(backend_kwargs)
        self.layer_idx = _try_extract_layer_index(prefix)

        if self.rainfusion.enabled:
            self._validate_parallel_config()
            if causal:
                raise ValueError(
                    "RAINFUSION_ATTN does not support causal attention: rf_v2 selects key "
                    "blocks by pooled relevance and cannot express a causal mask. Select "
                    "FLASH_ATTN for causal roles."
                )
            if qkv_layout is not None and qkv_layout.upper() != _INPUT_LAYOUT:
                raise ValueError(
                    f"RAINFUSION_ATTN needs {_INPUT_LAYOUT} tensors to locate the video segment along "
                    f"the sequence axis, but this layer declares qkv_layout={qkv_layout!r}. Select "
                    "FLASH_ATTN for this role."
                )

        self.dense_fallback = FlashAttentionBackend.get_impl_cls()(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
            qkv_layout=qkv_layout,
        )

    def _validate_parallel_config(self) -> None:
        config = get_current_diffusion_config_or_none()
        parallel_config = getattr(config, "parallel_config", None)
        ring_degree = getattr(parallel_config, "ring_degree", 1)
        if ring_degree > 1:
            # Ring gives each rank a slice of the sequence, so block selection
            # would score only local keys and the layer bypasses the backend
            # entirely (see Attention._run_ring_attention).
            raise ValueError(
                "RAINFUSION_ATTN is not compatible with ring sequence parallelism "
                f"(ring_degree={ring_degree}): rf_v2 needs the whole key sequence to rank "
                "blocks. Use Ulysses SP (ring_degree=1) instead."
            )

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        # ROCm and MUSA route through forward_cuda by default, so this covers them too.
        raise NotImplementedError(_WRONG_PLATFORM)

    def forward_xpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError(_WRONG_PLATFORM)

    def forward_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        plan = self._resolve_plan(attn_metadata)
        if plan is None:
            return self.dense_fallback.forward_npu(query, key, value, attn_metadata)
        return self._forward_sparse_npu(query, key, value, plan)

    def _resolve_plan(self, attn_metadata: AttentionMetadata | None) -> RainFusionPlan | None:
        """Return the rf_v2 geometry, or None when this forward must stay dense."""
        rf = self.rainfusion
        if not rf.enabled:
            return None
        if self.layer_idx is not None and self.layer_idx in rf.skip_layers:
            return None
        if is_forward_context_available():
            step_idx = get_forward_context().denoise_step_idx
            if step_idx is not None and step_idx < rf.start_step:
                return None
        if self.qkv_layout is None:
            # The sparse path reads the sequence off dim 1, which the tensors alone
            # do not establish, and the dense fallback resolves an absent layout its
            # own way. Sparsifying on an assumption would put the two paths on
            # different axes, so an undeclared layout stays dense.
            logger.warning_once(
                "RAINFUSION_ATTN staying dense: this layer does not declare qkv_layout, and rf_v2 "
                "needs %s to locate the video segment along the sequence axis. Set qkv_layout=%r on "
                "the Attention layer to enable sparsity.",
                _INPUT_LAYOUT,
                _INPUT_LAYOUT,
            )
            return None

        if attn_metadata is None:
            return None

        layout = attn_metadata.video_layout
        if layout is None:
            logger.warning_once(
                "RAINFUSION_ATTN staying dense: this attention role carries no video segment. The "
                "model must publish AttentionMetadata.video_layout for the sequence to be sparsified."
            )
            return None
        max_seqlen_q = attn_metadata.extra.get("max_seqlen_q")
        if max_seqlen_q is None:
            logger.warning_once(
                "RAINFUSION_ATTN staying dense: attention metadata is missing max_seqlen_q, so the "
                "video segment cannot be confirmed to be the tail of packed document 0."
            )
            return None

        prefix_len = int(layout.prefix_len)
        latent_shape = [int(dim) for dim in layout.latent_grid]
        # rf_v2 splits the sequence as [prefix | t*h*w video rows]. Document 0 of
        # the packed sequence holds those rows; anything past it is alignment
        # padding that rf_v2 must not see.
        video_len = math.prod(latent_shape)
        used_len = prefix_len + video_len

        if used_len != int(max_seqlen_q):
            logger.warning_once(
                "RAINFUSION_ATTN staying dense: prefix (%d) plus latent grid %s does not fill "
                "packed document 0 (%d rows). rf_v2 requires the video segment to be its tail.",
                prefix_len,
                tuple(latent_shape),
                int(max_seqlen_q),
            )
            return None
        if video_len < _MIN_VIDEO_BLOCKS * _BLOCK_SIZE:
            logger.warning_once(
                "RAINFUSION_ATTN staying dense: %d video rows is under the %d-row "
                "(%d block) threshold where sparse selection pays off.",
                video_len,
                _MIN_VIDEO_BLOCKS * _BLOCK_SIZE,
                _MIN_VIDEO_BLOCKS,
            )
            return None
        logger.info_once(
            "RAINFUSION_ATTN active: sparsity=%.2f, start_step=%d, exempt_layers=%d, "
            "latent_grid=%s, prefix_rows=%d, video_rows=%d. Realized sparsity is lower than nominal "
            "because prefix and first-frame blocks are always kept.",
            rf.sparsity,
            rf.start_step,
            len(rf.skip_layers),
            tuple(latent_shape),
            prefix_len,
            video_len,
        )
        return RainFusionPlan(
            prefix_len=prefix_len,
            used_len=used_len,
            latent_shape=latent_shape,
        )

    def _forward_sparse_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        plan: RainFusionPlan,
    ) -> torch.Tensor:
        try:
            from mindiesd import sparse_attention
        except ImportError:
            raise ImportError(_MISSING_MINDIESD)

        used = plan.used_len
        q, k, v = (tensor[:, :used] for tensor in (query, key, value))
        # Ulysses has already gathered the full sequence onto this rank and split
        # the heads, so read the head count off the tensor rather than num_heads.
        out = sparse_attention(
            q,
            k,
            v,
            scale=self.softmax_scale,
            head_num=query.shape[-2],
            input_layout=_INPUT_LAYOUT,
            inner_precise=_INNER_PRECISE,
            sparse_type="rf_v2",
            txt_len=plan.prefix_len,
            block_size=_BLOCK_SIZE,
            latent_shape_q=plan.latent_shape,
            latent_shape_k=plan.latent_shape,
            sparsity=self.rainfusion.sparsity,
        )
        if used == query.shape[1]:
            return out
        padded = torch.zeros_like(query)
        padded[:, :used] = out
        return padded
