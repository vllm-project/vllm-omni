# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Ascend implementation of unified sequence-parallel attention."""

from __future__ import annotations

import importlib
import math
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


class AscendUSPExecutor:
    """Execute supported Ascend SP attention through MindIE-SD.

    Unsupported calls return ``None`` so the portable attention layer can use
    its existing Ulysses/Ring implementation without duplicating collectives.
    """

    def __init__(
        self,
        *,
        sp_group: SequenceParallelGroups,
        ulysses_degree: int,
        ring_degree: int,
        allgather_degree: int,
        ulysses_mode: str,
    ) -> None:
        self.sp_group = sp_group
        self.ulysses_degree = ulysses_degree
        self.ring_degree = ring_degree
        self.allgather_degree = allgather_degree
        self.ulysses_mode = ulysses_mode
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
                "Ascend USP is enabled but mindiesd.layers.usp is unavailable; "
                "using vLLM-Omni native sequence-parallel attention: %s",
                exc,
            )
            return None

        if not callable(getattr(module, "usp_attention", None)) or not isinstance(
            getattr(module, "USPError", None), type
        ):
            logger.warning_once(
                "Ascend USP is enabled but the installed MindIE-SD API is incompatible; "
                "using vLLM-Omni native sequence-parallel attention."
            )
            return None
        self._usp_module = module
        logger.info_once("Using the Ascend unified sequence-parallel attention executor.")
        return module

    def _groups(self) -> tuple[object | None, object | None]:
        ulysses_group = self.sp_group.ulysses_group if self.ulysses_degree > 1 else None
        kv_gather_group = self.sp_group.ring_group if self.ring_degree > 1 else None
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
        if backend_name != "FLASH_ATTN":
            return False
        if self.allgather_degree > 1 or self.ulysses_degree * self.ring_degree == 1:
            return False
        if self.ulysses_mode != "strict" or causal:
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
        """Run the Ascend USP executor when compatible, else request fallback."""
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
                "Ascend USP rejected the current attention contract; "
                "using vLLM-Omni native sequence-parallel attention: %s",
                exc,
            )
            return None


__all__ = ["AscendUSPExecutor"]
