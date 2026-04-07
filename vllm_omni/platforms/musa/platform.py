# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from vllm.logger import init_logger
from vllm_musa.platform import MUSAPlatformBase

from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

logger = init_logger(__name__)


class MUSAOmniPlatform(OmniPlatform, MUSAPlatformBase):
    """MUSA/Moore Threads GPU implementation of OmniPlatform.

    Inherits all MUSA-specific implementations from vllm-musa's MUSAPlatformBase,
    and adds Omni-specific interfaces from OmniPlatform.
    """

    _omni_enum = OmniPlatformEnum.MUSA

    @classmethod
    def get_omni_ar_worker_cls(cls) -> str:
        return "vllm_omni.platforms.musa.worker.musa_ar_worker.MUSAARWorker"

    @classmethod
    def get_omni_generation_worker_cls(cls) -> str:
        return "vllm_omni.platforms.musa.worker.musa_generation_worker.MUSAGenerationWorker"

    @classmethod
    def get_default_stage_config_path(cls) -> str:
        return "vllm_omni/model_executor/stage_configs"

    @classmethod
    def get_diffusion_model_impl_qualname(cls, op_name: str) -> str:
        # MUSA uses default implementations for diffusion ops
        if op_name == "hunyuan_fused_moe":
            return "vllm_omni.diffusion.models.hunyuan_image_3.hunyuan_fused_moe.HunyuanFusedMoEDefault"
        return super().get_diffusion_model_impl_qualname(op_name)

    @classmethod
    def prepare_diffusion_op_runtime(cls, op_name: str, **kwargs: Any) -> None:
        # MUSA uses default runtime preparation
        return None

    @classmethod
    def get_diffusion_attn_backend_cls(
        cls,
        selected_backend: str | None,
        head_size: int,
    ) -> str:
        """Get the diffusion attention backend class path for MUSA platform.

        MUSA currently supports SDPA (Scaled Dot Product Attention) as the
        primary backend. Flash Attention support may be added in future
        when MUSA-specific implementations are available.

        Args:
            selected_backend: User-selected backend name (e.g., "FLASH_ATTN",
                "TORCH_SDPA"). If None, uses platform default.
            head_size: Attention head size.

        Returns:
            Fully qualified class path of the selected backend.
        """
        if selected_backend is not None:
            backend_upper = selected_backend.upper()
            backend = DiffusionAttentionBackendEnum[backend_upper]
            logger.info("Using diffusion attention backend '%s'", backend_upper)
            return backend.get_path()

        # Default to SDPA for MUSA as it's the most compatible backend
        logger.info("Defaulting to diffusion attention backend SDPA")
        return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()

    @classmethod
    def supports_torch_inductor(cls) -> bool:
        """MUSA supports torch.compile with inductor backend."""
        return True

    @classmethod
    def get_torch_device(cls, local_rank: int | None = None) -> torch.device:
        """Get the torch device for MUSA platform.

        Args:
            local_rank: Optional local rank for multi-GPU setups.

        Returns:
            torch.device for MUSA GPU.
        """
        if local_rank is None:
            return torch.device("musa")
        return torch.device("musa", local_rank)

    @classmethod
    def get_device_count(cls) -> int:
        """Get the number of available MUSA devices."""
        return torch.musa.device_count()

    @classmethod
    def synchronize(cls) -> None:
        """Synchronize all MUSA operations."""
        torch.musa.synchronize()

    @classmethod
    def get_free_memory(cls, device: torch.device | None = None) -> int:
        """Get the free memory on the MUSA device.

        Args:
            device: Optional device to query. If None, uses current device.

        Returns:
            Free memory in bytes.
        """
        free, _ = torch.musa.mem_get_info(device)
        return free

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.musa.get_device_name(device_id)
