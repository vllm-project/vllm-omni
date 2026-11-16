# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in visual-reference isolation, without a quadratic attention mask.

Reference queries read visual-reference keys only. All other valid queries
read all valid keys. Text, target/reference audio and padding are deliberately
excluded from the static reference group. This changes H3's learned attention
semantics and needs quality evaluation; it is not a lossless original-model
optimization. Dynamic queries may use the existing rectangular RainFusion
path, while reference queries always execute on their reference-only K/V.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from vllm_omni.diffusion.kv_offload_config import REFERENCE_ISOLATION_METRICS

if TYPE_CHECKING:
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, VideoTokenLayout

_LOG = logging.getLogger(__name__)
POLICY_VERSION = "visual_reference_only_rf_v2"


def reference_isolation_enabled(config: object | None = None) -> bool:
    """Read the serialized serving option; environment variables are ignored."""
    value = getattr(config, "kv_isolate", False)
    if value is None:
        return False
    if type(value) is not bool:
        raise ValueError("kv_isolate must be a boolean")
    return value


class MiniMaxH3ReferenceIsolation:
    """One request/CFG-distilled branch's immutable global row partition.

    Rows are selected after Ulysses, where each rank has the complete sequence
    and a balanced head shard. Only full-loop, single-document, noncausal
    local/pure-Ulysses execution is supported. Unsupported paths fail closed;
    removing this restriction mid-request would invalidate populated K/V.
    """

    policy_version = POLICY_VERSION

    def __init__(
        self,
        reference_positions: torch.Tensor,
        *,
        sequence_rows: int,
        used_rows: int,
        condition_timestep: float,
        device: torch.device,
    ) -> None:
        if not 0 < used_rows <= sequence_rows:
            raise ValueError("isolation requires 0 < used_rows <= sequence_rows")
        if not math.isfinite(condition_timestep) or not 0 <= condition_timestep <= 1:
            raise ValueError("isolated reference condition timestep must be finite and within [0, 1]")
        if reference_positions.ndim != 1 or reference_positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("reference_positions must be a 1-D integer tensor")
        # The denoise branch already owns CPU row positions. Validate once,
        # outside the layer stack, and never read NPU scalars per layer.
        positions = reference_positions.detach().to(device="cpu", dtype=torch.long)
        if positions.numel() and (int(positions.min()) < 0 or int(positions.max()) >= used_rows):
            raise ValueError("reference positions must lie inside the valid document, never padding")
        if positions.numel() > 1 and not bool((positions[1:] > positions[:-1]).all()):
            raise ValueError("reference positions must be unique and sorted")
        selected = torch.zeros(used_rows, dtype=torch.bool)
        selected[positions] = True
        self.reference_rows = int(positions.numel())
        self.sequence_rows = int(sequence_rows)
        self.used_rows = int(used_rows)
        self.condition_timestep = float(condition_timestep)
        self.reference_positions = positions.to(device=device)
        self.dynamic_positions = torch.nonzero(~selected, as_tuple=False).flatten().to(device=device)
        self.key_order = torch.cat((self.reference_positions, self.dynamic_positions))
        self.geometry_ready = False
        self.dynamic_video_layout: VideoTokenLayout | None = None
        self.kv_video_layout: VideoTokenLayout | None = None
        self.full_layer_calls = 0
        self.compact_layer_calls = 0
        self.dynamic_backend_calls = 0
        self.sparse_layer_calls = 0
        self._packed_lengths: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self.closed = False
        _LOG.warning(
            "MiniMax-H3 reference isolation enabled: policy=%s, reference_rows=%d, "
            "valid_rows=%d, reference_timestep=%g, reference_reads=visual_reference_only, "
            "dynamic_reads=all_valid, reference_attention=dense_rr, dynamic_attention=configured_backend; "
            "experimental quality-changing mode",
            self.policy_version,
            self.reference_rows,
            self.used_rows,
            self.condition_timestep,
        )

    def validate_presentation(self, *, sequence_rows: int, used_rows: int, num_requests: int) -> None:
        if self.closed:
            raise RuntimeError("reference isolation state is closed")
        if num_requests != 1:
            raise ValueError("reference isolation currently requires one request per forward")
        if (sequence_rows, used_rows) != (self.sequence_rows, self.used_rows):
            raise ValueError("reference isolation layout changed within a request")

    def _dense_metadata(self, metadata: AttentionMetadata, q: torch.Tensor, kv_rows: int) -> AttentionMetadata:
        q_rows = q.shape[1]
        # Real document plus empty padding document, matching H3 Flash's packed
        # contract on CUDA and Ascend. K/V are already sliced to valid rows.
        lengths = self._packed_lengths.get((q_rows, kv_rows))
        if lengths is None:
            lengths = (
                torch.tensor([0, q_rows, q_rows], dtype=torch.int32, device=q.device),
                torch.tensor([0, kv_rows, kv_rows], dtype=torch.int32, device=q.device),
            )
            self._packed_lengths[(q_rows, kv_rows)] = lengths
        extra = {
            "cu_seqlens_q": lengths[0],
            "cu_seqlens_k": lengths[1],
            "max_seqlen_q": q_rows,
            "max_seqlen_k": kv_rows,
            "valid_kv_length": kv_rows,
            "npu_attn_varlen": True,
        }
        for name in ("laser_input_scale", "kv_cache_dtype"):
            if name in metadata.extra:
                extra[name] = metadata.extra[name]
        return replace(metadata, attn_mask=None, packed_padding=None, video_layout=None, extra=extra)

    def set_video_geometry(self, dynamic_layout: VideoTokenLayout | None, kv_layout: VideoTokenLayout | None) -> None:
        """Reuse H3's Q compaction/KV-prefix geometry builders once per request."""
        if self.geometry_ready:
            raise RuntimeError("isolation video geometry is immutable within a request")
        if dynamic_layout is not None:
            if dynamic_layout.used_len != self.used_rows - self.reference_rows:
                raise ValueError("isolated dynamic video layout has an incorrect valid Q length")
            if kv_layout is None or kv_layout.used_len != self.used_rows:
                raise ValueError("isolated KV geometry must cover the full valid KV document")
        self.dynamic_video_layout = dynamic_layout
        self.kv_video_layout = kv_layout
        self.geometry_ready = True

    def _dynamic_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        metadata: AttentionMetadata,
        dense_forward: Callable[..., torch.Tensor],
        dynamic_forward: Callable[..., torch.Tensor] | None,
    ) -> torch.Tensor:
        prepared = self._dense_metadata(metadata, q, self.used_rows)
        if dynamic_forward is None or self.dynamic_video_layout is None:
            return dense_forward(q, k, v, prepared)
        # Only dynamic Q reaches the sparse backend, using the same reference
        # KV prefix as Tier2 reuse. Protected keys cannot reopen R->dynamic.
        extra = {
            **prepared.extra,
            "minimax_h3_compact_reference_rows": self.reference_rows,
            "minimax_h3_reference_kv_video_layout": self.kv_video_layout,
            REFERENCE_ISOLATION_METRICS: self,
        }
        result = dynamic_forward(q, k, v, replace(prepared, video_layout=self.dynamic_video_layout, extra=extra))
        self.dynamic_backend_calls += 1  # Includes RF's intended dense warmup/fallbacks.
        return result

    def record_sparse_layer(self) -> None:
        """Called only after the existing RF hybrid sparse execution succeeds."""
        self.sparse_layer_calls += 1

    def attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: AttentionMetadata,
        *,
        compact: bool,
        dense_forward: Callable[..., torch.Tensor],
        dynamic_forward: Callable[..., torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Apply the four-region policy with separate reference and dynamic calls."""
        if self.closed:
            raise RuntimeError("reference isolation state is closed")
        if query.ndim != 4 or query.shape[0] != 1 or key.ndim != 4 or key.shape[0] != 1 or key.shape != value.shape:
            raise ValueError("reference isolation requires BSND Q and equal BSND K/V, batch=1")
        if query.shape[2:] != key.shape[2:] or query.device != key.device or key.device != value.device:
            raise ValueError("isolated Q/K/V head layout or device mismatch")
        if query.dtype != key.dtype or key.dtype != value.dtype:
            raise ValueError("isolated Q/K/V must have matching dtypes")
        if self.reference_positions.device != query.device:
            raise ValueError("reference isolation row indices must be on the Q/K/V device")
        if (
            metadata.attn_mask is not None
            or getattr(metadata, "full_attn_spans", None) is not None
            or getattr(metadata, "query_ranges", None) is not None
        ):
            raise ValueError("reference isolation cannot discard another explicit attention mask")
        if any(
            getattr(metadata, name, None) is not None
            for name in ("joint_query", "joint_key", "joint_value", "joint_attn_mask")
        ):
            raise ValueError("reference isolation does not support joint attention")
        q_rows = self.used_rows - self.reference_rows if compact else self.used_rows
        if q_rows <= 0 or query.shape[1] < q_rows or key.shape[1] < self.used_rows:
            raise ValueError("isolated Q/K/V do not cover the valid document")
        if compact and metadata.extra.get("minimax_h3_compact_reference_rows") != self.reference_rows:
            raise ValueError("compact isolation requires the existing reference-prefix K/V contract")
        key, value = key[:, : self.used_rows], value[:, : self.used_rows]
        output = torch.zeros_like(query)
        if compact:
            q = query[:, :q_rows]
            output[:, :q_rows] = self._dynamic_attention(q, key, value, metadata, dense_forward, dynamic_forward)
            self.compact_layer_calls += 1
            return output
        if self.reference_rows:
            q = query.index_select(1, self.reference_positions)
            k = key.index_select(1, self.reference_positions)
            v = value.index_select(1, self.reference_positions)
            reference = dense_forward(q, k, v, self._dense_metadata(metadata, q, self.reference_rows))
            output.index_copy_(1, self.reference_positions, reference)
        if self.dynamic_positions.numel():
            q = query.index_select(1, self.dynamic_positions)
            if dynamic_forward is not None and self.dynamic_video_layout is not None:
                key = key.index_select(1, self.key_order)
                value = value.index_select(1, self.key_order)
            dynamic = self._dynamic_attention(q, key, value, metadata, dense_forward, dynamic_forward)
            output.index_copy_(1, self.dynamic_positions, dynamic)
        self.full_layer_calls += 1
        return output

    def close(self) -> None:
        _LOG.info(
            "MiniMax-H3 reference isolation summary: policy=%s, full_layer_calls=%d, compact_layer_calls=%d, "
            "dynamic_backend_calls=%d, sparse_layer_calls=%d",
            self.policy_version,
            self.full_layer_calls,
            self.compact_layer_calls,
            self.dynamic_backend_calls,
            self.sparse_layer_calls,
        )
        self.closed = True
        self.reference_positions = self.reference_positions.new_empty((0,))
        self.dynamic_positions = self.dynamic_positions.new_empty((0,))
        self.key_order = self.key_order.new_empty((0,))
        self.dynamic_video_layout = self.kv_video_layout = None
        self._packed_lengths.clear()
