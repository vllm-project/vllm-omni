# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger

# import torch.distributed as dist # Not used directly here, but good practice if needed
from vllm_omni.diffusion.attention.backends.ring.ring_globals import (
    FA3_SUPPORTED_CUDA_MAJORS,
    HAS_AITER,
    HAS_FA3,
    HAS_FA4,
    HAS_FLASH_ATTN,
)
from vllm_omni.diffusion.attention.backends.ring.ring_selector import AttnType
from vllm_omni.diffusion.attention.parallel.base import (
    ParallelAttentionContext,
    # ParallelAttentionStrategy, # Not used in type hint below currently
)
from vllm_omni.diffusion.distributed.group_coordinator import SequenceParallelGroupCoordinator

if TYPE_CHECKING:
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

logger = init_logger(__name__)


def _can_use_fa3(device: torch.device) -> bool:
    """Return whether the installed FA3 kernels support ``device``.

    Importing an extension only proves that its Python module is installed.
    When the source publishes a supported-major contract, also check it before
    entering a CUDA launcher that may abort instead of raising an exception.
    """
    if not HAS_FA3 or device.type != "cuda":
        return False
    if FA3_SUPPORTED_CUDA_MAJORS is None:
        return True
    major, _minor = torch.cuda.get_device_capability(device)
    return major in FA3_SUPPORTED_CUDA_MAJORS


def _can_use_fa4(device: torch.device) -> bool:
    """Return whether the installed FA4 kernels support ``device``."""
    if not HAS_FA4 or device.type != "cuda":
        return False
    major, _minor = torch.cuda.get_device_capability(device)
    return major >= 10


def _can_use_fa2(device: torch.device) -> bool:
    """Return whether the FA2 backend supports ``device``.

    FA2's CUDA backend supports Ampere, Ada, and Hopper.  Blackwell support is
    provided by the separate FA4 implementation.
    """
    if not HAS_FLASH_ATTN or device.type != "cuda":
        return False
    major, _minor = torch.cuda.get_device_capability(device)
    return major in (8, 9)


@dataclass(frozen=True, slots=True)
class _RingCtx(ParallelAttentionContext):
    """Per-forward context for Ring sequence-parallel attention."""

    # Ring attention typically doesn't need complex context for post-processing
    # as the output is already correctly sharded along sequence dimension.
    pass


class RingParallelAttention:
    """Ring sequence-parallel strategy.

    This strategy prepares inputs for Ring Attention.
    Key responsibilities:
    - Concatenate joint_query (Text) to query (Image) if present.
    - Keep joint_key/value separate in metadata for the Ring kernel to handle as static prefix.
    """

    def __init__(
        self,
        sp_group: SequenceParallelGroupCoordinator,
        attn_backend_pref: str | None = None,
        attn_backend_explicit: bool = False,
    ) -> None:
        self._sp_group = sp_group
        self.attn_backend_pref = attn_backend_pref
        self.attn_backend_explicit = attn_backend_explicit

    @property
    def enabled(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "ring"

    def pre_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ):
        joint_tensor_query = None
        joint_strategy = "front"

        if attn_metadata is not None:
            joint_tensor_query = attn_metadata.joint_query
            joint_strategy = attn_metadata.joint_strategy

        if joint_tensor_query is not None:
            supported_joint_strategy = ["front", "rear"]
            if joint_strategy not in supported_joint_strategy:
                raise ValueError(f"joint_strategy: {joint_strategy} not supported.")

            if joint_strategy == "front":
                query = torch.cat([joint_tensor_query, query], dim=1)
            else:
                query = torch.cat([query, joint_tensor_query], dim=1)

            # Note: We do NOT concatenate joint_key/value here.
            # They are preserved in attn_metadata and will be passed
            # explicitly to ring_flash_attn_func.

        ctx = _RingCtx(name=self.name)
        return query, key, value, attn_metadata, ctx

    def post_attention(self, attn_output: torch.Tensor, ctx: ParallelAttentionContext | None) -> torch.Tensor:
        # Ring attention output is already sharded correctly along sequence dimension.
        return attn_output

    def run_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
        softmax_scale: float | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """Run the actual Ring Attention kernel."""
        if softmax_scale is None:
            softmax_scale = query.shape[-1] ** -0.5

        backend_pref = self.attn_backend_pref
        if backend_pref is not None:
            backend_pref = backend_pref.lower()

        # Extract joint tensors
        joint_key, joint_value = None, None
        joint_strategy = "front"
        if attn_metadata is not None:
            joint_key = attn_metadata.joint_key
            joint_value = attn_metadata.joint_value
            if attn_metadata.joint_strategy is not None:
                joint_strategy = attn_metadata.joint_strategy

        def _run_sdpa() -> torch.Tensor:
            from vllm_omni.diffusion.attention.backends.ring_pytorch_attn import ring_pytorch_attn_func

            return ring_pytorch_attn_func(
                query,
                key,
                value,
                softmax_scale=softmax_scale,
                causal=causal,
                group=self._sp_group.ring_group,
                op_type="efficient",
                joint_tensor_key=joint_key,
                joint_tensor_value=joint_value,
                joint_strategy=joint_strategy,
            )

        # Ring only implements local Flash/AITER and SDPA kernels. HuggingFace
        # Hub FA and other non-local backends can use the SDPA ring path only
        # when they came from automatic platform selection. An explicit
        # request must never silently substitute local FA4/FA3/FA2.
        _sdpa_prefs = {
            "sdpa",
            "torch",
            "torch_sdpa",
        }
        _non_ring_prefs = {
            "cudnn_attn",
            "flashinfer_attn",
            "trtllm_attn",
            "sage_attn",
            "sage_attn_3",
            "flash_attn_hub",
            "flash_attn_3_hub",
        }
        if backend_pref in _sdpa_prefs:
            return _run_sdpa()
        if backend_pref in _non_ring_prefs:
            if self.attn_backend_explicit:
                raise ValueError(
                    f"{self.attn_backend_pref} was explicitly selected, but ring sequence parallelism "
                    "has no implementation for that backend. Select TORCH_SDPA/FLASH_ATTN or use Ulysses SP."
                )
            return _run_sdpa()

        if query.dtype == torch.float32:
            if self.attn_backend_explicit:
                raise ValueError(
                    f"{self.attn_backend_pref} was explicitly selected for ring attention, "
                    "but its ring kernel does not support float32. Select TORCH_SDPA or use a supported dtype."
                )
            return _run_sdpa()

        can_use_fa4 = _can_use_fa4(query.device)
        can_use_fa3 = _can_use_fa3(query.device)
        can_use_fa2 = _can_use_fa2(query.device)
        if not can_use_fa4 and not can_use_fa3 and not can_use_fa2 and not HAS_AITER:
            if self.attn_backend_explicit:
                raise RuntimeError(
                    f"{self.attn_backend_pref} was explicitly selected, but no compatible ring kernel "
                    f"is available for device {query.device}."
                )
            logger.warning_once(
                "Automatic ring backend selection chose TORCH_SDPA because no compatible "
                "FA2/FA3/FA4/AITER ring kernel is available for this device."
            )
            return _run_sdpa()

        from vllm_omni.diffusion.attention.backends.ring_flash_attn import ring_flash_attn_func

        # Prefer FA4 on Blackwell. An importable Hopper-only FA3 wheel can
        # otherwise be selected and fail at launch with "no kernel image".
        if can_use_fa4:
            attn_type = AttnType.FA4
        # Prefer FA3 over FA2 on Ampere/Ada/Hopper. On ROCm, use AITER.
        elif can_use_fa3:
            attn_type = AttnType.FA3
        elif HAS_AITER:
            attn_type = AttnType.AITER
        elif can_use_fa2:
            attn_type = AttnType.FA
        else:
            raise RuntimeError("No compatible Flash Attention backend is available for ring attention.")

        return ring_flash_attn_func(
            query,
            key,
            value,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            deterministic=False,
            group=self._sp_group.ring_group,
            attn_type=attn_type,
            joint_tensor_key=joint_key,
            joint_tensor_value=joint_value,
            joint_strategy=joint_strategy,
        )
