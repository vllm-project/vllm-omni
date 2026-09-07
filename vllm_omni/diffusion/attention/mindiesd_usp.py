# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Optional MindIE-SD unified sequence-parallel attention adapter."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Protocol

import torch
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

logger = init_logger(__name__)


class SequenceParallelGroups(Protocol):
    """Process groups owned by vLLM-Omni's SP coordinator."""

    ulysses_group: object
    ring_group: object


@dataclass(frozen=True, slots=True)
class MindIESDUSPOptions:
    """vLLM-Omni-owned, normalized options for the MindIE-SD USP call."""

    enabled: bool = False
    ulysses_degree: int = 1
    ring_degree: int = 1
    allgather_degree: int = 1
    ulysses_mode: str = "strict"

    @classmethod
    def from_parallel_config(cls, config: object) -> MindIESDUSPOptions:
        return cls(
            enabled=bool(getattr(config, "enable_mindiesd_usp", False)),
            ulysses_degree=int(getattr(config, "ulysses_degree", 1)),
            ring_degree=int(getattr(config, "ring_degree", 1)),
            allgather_degree=int(getattr(config, "allgather_degree", 1)),
            ulysses_mode=str(getattr(config, "ulysses_mode", "strict")),
        )


class MindIESDUSPAdapter:
    """Translate vLLM-Omni attention state to MindIE-SD's explicit USP ABI.

    The adapter deliberately returns ``None`` when the optional fast path does
    not cover the current semantics. The caller then executes its existing
    native Ulysses/Ring path unchanged.
    """

    def __init__(
        self,
        options: MindIESDUSPOptions,
        sp_group: SequenceParallelGroups,
    ) -> None:
        self.options = options
        self.sp_group = sp_group
        self._usp_module: ModuleType | None = None
        self._load_attempted = False

    def _load_usp_module(self) -> ModuleType | None:
        if self._load_attempted:
            return self._usp_module
        self._load_attempted = True
        try:
            module = importlib.import_module("mindiesd.layers.usp")
        except ImportError as exc:
            logger.warning_once(
                "MindIE-SD USP is enabled but mindiesd.layers.usp is unavailable; "
                "using vLLM-Omni native sequence-parallel attention: %s",
                exc,
            )
            return None

        if not callable(getattr(module, "usp_attention", None)) or not isinstance(
            getattr(module, "USPError", None), type
        ):
            logger.warning_once(
                "MindIE-SD USP is enabled but its installed API is incompatible; "
                "using vLLM-Omni native sequence-parallel attention."
            )
            return None
        self._usp_module = module
        logger.info_once("Using MindIE-SD unified sequence-parallel attention.")
        return module

    def _groups(self):
        ulysses_group = self.sp_group.ulysses_group if self.options.ulysses_degree > 1 else None
        if self.options.ring_degree > 1:
            # MindIE-SD currently materializes K/V with AllGather over this
            # group. It is numerically equivalent to Omni's Ring path but is
            # intentionally not described as a true P2P Ring implementation.
            kv_gather_group = self.sp_group.ring_group
        else:
            kv_gather_group = None
        return ulysses_group, kv_gather_group

    def _supports_call(
        self,
        query: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
        *,
        backend_name: str,
        causal: bool,
        softmax_scale: float,
        scatter_dim: int,
        gather_dim: int,
    ) -> bool:
        options = self.options
        if not options.enabled or backend_name != "FLASH_ATTN":
            return False
        if options.allgather_degree > 1 or options.ulysses_degree * options.ring_degree == 1:
            return False
        if options.ulysses_mode != "strict" or causal:
            return False
        if scatter_dim != 2 or gather_dim != 1 or query.ndim != 4:
            return False
        if not math.isclose(float(softmax_scale), query.shape[-1] ** -0.5, rel_tol=1e-6, abs_tol=1e-8):
            return False
        if attn_metadata is None:
            return True
        if any(
            tensor is not None
            for tensor in (
                attn_metadata.joint_query,
                attn_metadata.joint_key,
                attn_metadata.joint_value,
                attn_metadata.joint_attn_mask,
            )
        ):
            return False
        # Omni may still hold a rank-local mask here; its native SP strategy
        # normalizes that mask only after the interception point.
        if attn_metadata.attn_mask is not None:
            return False
        if attn_metadata.full_attn_spans is not None or attn_metadata.query_ranges is not None:
            return False
        if attn_metadata.packed_padding is not None:
            return False
        unsupported_extra = {
            "cu_seqlens_q",
            "cu_seqlens_k",
            "gate_compress",
            "kv_cache_dtype",
            "laser_input_scale",
            "npu_attn_varlen",
            "seq_lens",
            "valid_kv_length",
        }
        return not unsupported_extra.intersection(attn_metadata.extra)

    def try_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attn_metadata: AttentionMetadata | None,
        backend_name: str,
        causal: bool,
        softmax_scale: float,
        scatter_dim: int,
        gather_dim: int,
    ) -> torch.Tensor | None:
        """Run MindIE-SD once when compatible, otherwise request native fallback."""
        if not self._supports_call(
            query,
            attn_metadata,
            backend_name=backend_name,
            causal=causal,
            softmax_scale=softmax_scale,
            scatter_dim=scatter_dim,
            gather_dim=gather_dim,
        ):
            return None

        module = self._load_usp_module()
        if module is None:
            return None

        ulysses_group, kv_gather_group = self._groups()
        try:
            return module.usp_attention(
                query,
                key,
                value,
                ulysses_group=ulysses_group,
                kv_gather_group=kv_gather_group,
            )
        except module.USPError as exc:
            logger.warning_once(
                "MindIE-SD USP rejected the current attention contract; "
                "using vLLM-Omni native sequence-parallel attention: %s",
                exc,
            )
            return None


__all__ = ["MindIESDUSPAdapter", "MindIESDUSPOptions"]
