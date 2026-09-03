# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import functools
import inspect
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

_INCOMPATIBLE_MINDIESD = (
    "RAINFUSION_ATTN requires a MindIE-SD build whose sparse_attention supports the video_spans "
    "argument. Please upgrade MindIE-SD or select FLASH_ATTN."
)


# Whether the installed mindiesd ``sparse_attention`` accepts ``precision=``.
# Releases without it accept the kwarg through ``**kwargs`` but silently ignore
# it, so a requested mix/fp8 mode would silently run the BF16 path. Cached
# because forwards run per layer per denoise step.
@functools.cache
def _mindiesd_supports_precision() -> bool:
    try:
        from inspect import signature

        from mindiesd import sparse_attention

        return "precision" in signature(sparse_attention).parameters
    except Exception:
        return False


def _try_extract_layer_index(prefix: str) -> int | None:
    if not prefix:
        return None
    try:
        return extract_layer_index(prefix)
    except (AssertionError, ValueError):
        return None


def _supports_video_spans(sparse_attention: Any) -> bool:
    try:
        return "video_spans" in inspect.signature(sparse_attention).parameters
    except (TypeError, ValueError):
        return False


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
    end_step: int = 0
    precision: str = "bf16"
    skip_layers: frozenset[int] = frozenset()

    @classmethod
    def from_backend_kwargs(cls, backend_kwargs: dict | None) -> RainFusionConfig:
        bk = backend_kwargs or {}
        return cls(
            sparsity=float(bk.get("sparsity", 0.0)),
            start_step=int(bk.get("start_step", 0)),
            end_step=int(bk.get("end_step", 0)),
            precision=str(bk.get("precision", "bf16")),
            skip_layers=frozenset(bk.get("skip_layers") or ()),
        )

    @property
    def enabled(self) -> bool:
        return self.sparsity > 0.0


@dataclass(frozen=True)
class RainFusionPlan:
    """Per-forward geometry handed to the rf_v2 kernel."""

    used_len: int
    prefix_len: int | None = None
    latent_shape: list[int] | None = None
    video_spans: list[dict[str, object]] | None = None


class RainFusionAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_platforms: tuple[str, ...] = ("npu",)

    @classmethod
    def validate_available(cls) -> None:
        from importlib.util import find_spec

        if find_spec("mindiesd") is None:
            raise ValueError(_MISSING_MINDIESD)
        try:
            from mindiesd import sparse_attention
        except ImportError as exc:
            raise ValueError(_MISSING_MINDIESD) from exc
        if not _supports_video_spans(sparse_attention):
            raise ValueError(_INCOMPATIBLE_MINDIESD)

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
            total_steps = get_forward_context().total_denoise_steps
            if step_idx is not None and step_idx < rf.start_step:
                return None
            # Tail fallback: keep the last ``end_step`` denoise steps dense.
            if (
                rf.end_step > 0
                and step_idx is not None
                and total_steps is not None
                and step_idx >= total_steps - rf.end_step
            ):
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

        if layout.video_spans:
            return self._resolve_multi_span_plan(layout, max_seqlen_q)

        if layout.prefix_len is None or layout.latent_grid is None:
            logger.warning_once(
                "RAINFUSION_ATTN staying dense: video layout has neither a legacy video tail nor multi-video spans."
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
            used_len=used_len,
            prefix_len=prefix_len,
            latent_shape=latent_shape,
        )

    def _resolve_multi_span_plan(self, layout, max_seqlen_q: int) -> RainFusionPlan | None:
        """Validate Ref2VA's non-contiguous video grids before sparse dispatch."""
        used_len = layout.used_len
        if used_len is None or int(used_len) != int(max_seqlen_q):
            logger.warning_once(
                "RAINFUSION_ATTN staying dense: multi-video layout used_len=%r does not match packed document 0 (%d).",
                used_len,
                int(max_seqlen_q),
            )
            return None

        spans: list[dict[str, object]] = []
        span_summaries: list[str] = []
        previous_end = 0
        target_count = 0
        video_seqlen = 0
        boundary_dense_seqlen = 0
        previous_video_length: int | None = None
        for span in sorted(layout.video_spans, key=lambda item: item.start):
            grid = tuple(int(dim) for dim in span.latent_grid)
            length = math.prod(grid)
            start = int(span.start)
            if (
                len(grid) != 3
                or any(dim <= 0 for dim in grid)
                or start < previous_end
                or start + length > int(used_len)
            ):
                logger.warning_once(
                    "RAINFUSION_ATTN staying dense: invalid multi-video span start=%d grid=%s used_len=%d.",
                    start,
                    grid,
                    int(used_len),
                )
                return None
            role = span.role
            if role not in ("reference", "target"):
                logger.warning_once("RAINFUSION_ATTN staying dense: unsupported multi-video span role %r.", role)
                return None
            if role == "target":
                target_count += 1
            if previous_video_length is not None:
                # rf_v2 works on fixed 128-token blocks. The preceding clip
                # needs these dense rows to complete its tail block before
                # this clip begins, otherwise one sparse block would cross a
                # clip boundary.
                boundary_dense_seqlen += (-previous_video_length) % _BLOCK_SIZE
            spans.append({"start": start, "latent_shape": list(grid)})
            span_summaries.append(f"role={role}, start={start}, seqlen={length}, latent_shape={grid}")
            previous_end = start + length
            video_seqlen += length
            previous_video_length = length

        if target_count != 1:
            logger.warning_once(
                "RAINFUSION_ATTN staying dense: Ref2VA layout must contain exactly one target video span, got %d.",
                target_count,
            )
            return None
        if video_seqlen < _MIN_VIDEO_BLOCKS * _BLOCK_SIZE:
            logger.warning_once(
                "RAINFUSION_ATTN staying dense: multi-video seqlen=%d is under the sparse threshold seqlen=%d.",
                video_seqlen,
                _MIN_VIDEO_BLOCKS * _BLOCK_SIZE,
            )
            return None
        dense_context_seqlen = int(used_len) - video_seqlen
        if boundary_dense_seqlen > dense_context_seqlen:
            logger.warning_once(
                "RAINFUSION_ATTN staying dense: multi-video spans need dense_context_seqlen=%d "
                "to isolate clip block boundaries, but this layout has only %d.",
                boundary_dense_seqlen,
                dense_context_seqlen,
            )
            return None
        logger.info_once(
            "RAINFUSION_ATTN multi-video active: sparsity=%.2f, spans=[%s], "
            "video_seqlen=%d, dense_context_seqlen=%d, valid_packed_seqlen=%d.",
            self.rainfusion.sparsity,
            "; ".join(span_summaries),
            video_seqlen,
            dense_context_seqlen,
            int(used_len),
        )
        return RainFusionPlan(used_len=int(used_len), video_spans=spans)

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
        if self.rainfusion.precision != "bf16" and not _mindiesd_supports_precision():
            raise RuntimeError(
                f"block_sparse.precision={self.rainfusion.precision!r} requires MindIE-SD "
                "with sparse_attention(precision=...) support; the installed mindiesd "
                "silently ignores it and would run the BF16 path. Install a compatible "
                "MindIE-SD release or use precision='bf16'."
            )

        used = plan.used_len
        q, k, v = (tensor[:, :used] for tensor in (query, key, value))
        # Ulysses has already gathered the full sequence onto this rank and split
        # the heads, so read the head count off the tensor rather than num_heads.
        common_kwargs: dict[str, object] = {
            "scale": self.softmax_scale,
            "head_num": query.shape[-2],
            "input_layout": _INPUT_LAYOUT,
            "inner_precise": _INNER_PRECISE,
            "block_size": _BLOCK_SIZE,
            "sparsity": self.rainfusion.sparsity,
            "precision": self.rainfusion.precision,
        }
        if plan.video_spans is not None:
            out = sparse_attention(
                q,
                k,
                v,
                sparse_type="rf_v2",
                video_spans=plan.video_spans,
                **common_kwargs,
            )
        else:
            assert plan.prefix_len is not None and plan.latent_shape is not None
            common_kwargs.update(
                sparse_type="rf_v2",
                txt_len=plan.prefix_len,
                latent_shape_q=plan.latent_shape,
                latent_shape_k=plan.latent_shape,
            )
            out = sparse_attention(q, k, v, **common_kwargs)
        if used == query.shape[1]:
            return out
        padded = torch.zeros_like(query)
        padded[:, :used] = out
        return padded
