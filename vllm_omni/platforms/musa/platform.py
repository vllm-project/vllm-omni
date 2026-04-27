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
        return "vllm_omni.worker.gpu_ar_worker.GPUARWorker"

    @classmethod
    def get_omni_generation_worker_cls(cls) -> str:
        return "vllm_omni.worker.gpu_generation_worker.GPUGenerationWorker"

    @classmethod
    def get_default_stage_config_path(cls) -> str:
        return "vllm_omni/model_executor/stage_configs"

    @classmethod
    def get_diffusion_model_impl_qualname(cls, op_name: str) -> str:
        # MUSA uses default implementations for diffusion ops
        if op_name == "hunyuan_fused_moe":
            return "vllm_omni.diffusion.models.hunyuan_image3.hunyuan_fused_moe.HunyuanFusedMoEDefault"
        return super().get_diffusion_model_impl_qualname(op_name)

    @classmethod
    def prepare_diffusion_op_runtime(cls, op_name: str, **kwargs: Any) -> None:
        # MUSA uses default runtime preparation
        return None

    @classmethod
    def has_flash_attn_package(cls) -> bool:
        from vllm_omni.diffusion.attention.backends.utils.fa import is_flash_attn_installed

        return is_flash_attn_installed()

    @classmethod
    def get_diffusion_attn_backend_cls(
        cls,
        selected_backend: str | None,
        head_size: int,
    ) -> str:
        """Get the diffusion attention backend class path for MUSA platform.

        MUSA supports FLASH_ATTN via the mate package, and SDPA as fallback.

        Args:
            selected_backend: User-selected backend name (e.g., "FLASH_ATTN",
                "TORCH_SDPA"). If None, uses platform default.
            head_size: Attention head size.

        Returns:
            Fully qualified class path of the selected backend.
        """
        from vllm_omni.diffusion.envs import PACKAGES_CHECKER

        # Check compute capability for Flash Attention support
        # Flash Attention requires compute capability >= 3.1
        compute_capability = cls.get_device_capability()
        compute_supported = False
        if compute_capability is not None:
            major, minor = compute_capability
            capability = major * 10 + minor
            compute_supported = capability >= 31

        # Check if FA packages are available
        packages_info = PACKAGES_CHECKER.get_packages_info()
        packages_available = packages_info.get("has_flash_attn", False)

        # Both compute capability and packages must be available for FA
        flash_attn_supported = compute_supported and packages_available

        if selected_backend is not None:
            backend_upper = selected_backend.upper()
            if backend_upper == "FLASH_ATTN" and not flash_attn_supported:
                if not compute_supported:
                    logger.warning(
                        "Flash Attention requires MUSA GPU with compute capability >= 3.1. "
                        "Falling back to TORCH_SDPA backend."
                    )
                elif not packages_available:
                    logger.warning("Flash Attention (mate package) not available. Falling back to TORCH_SDPA backend.")
                logger.info("Defaulting to diffusion attention backend SDPA")
                return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()
            backend = DiffusionAttentionBackendEnum[backend_upper]
            logger.info("Using diffusion attention backend '%s'", backend_upper)
            return backend.get_path()

        if flash_attn_supported:
            logger.info("Defaulting to diffusion attention backend FLASH_ATTN")
            return DiffusionAttentionBackendEnum.FLASH_ATTN.get_path()

        logger.info("Defaulting to diffusion attention backend SDPA")
        return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()

    @classmethod
    def supports_torch_inductor(cls) -> bool:
        """MUSA supports torch.compile with inductor backend."""
        return True

    @classmethod
    def supports_float64(cls) -> bool:
        """MUSA does not support float64 yet."""
        return False

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
