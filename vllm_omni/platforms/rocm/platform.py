# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.logger import init_logger
from vllm.platforms.rocm import RocmPlatform

from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

logger = init_logger(__name__)


class RocmOmniPlatform(OmniPlatform, RocmPlatform):
    """ROCm/AMD GPU implementation of OmniPlatform.

    Inherits all ROCm-specific implementations from vLLM's RocmPlatform,
    and adds Omni-specific interfaces from OmniPlatform.
    """

    _omni_enum = OmniPlatformEnum.ROCM

    @classmethod
    def get_omni_ar_worker_cls(cls) -> str:
        return "vllm_omni.worker.gpu_ar_worker.GPUARWorker"

    @classmethod
    def get_omni_generation_worker_cls(cls) -> str:
        return "vllm_omni.worker.gpu_generation_worker.GPUGenerationWorker"

    @classmethod
    def get_diffusion_attn_backend_cls(
        cls,
        selected_backend: str | None,
        head_size: int,
    ) -> str:
        from vllm._aiter_ops import is_aiter_found_and_supported

        # Check if aiter is available for Flash Attention support
        # aiter currently only is supported on gfx942 and gfx950
        # https://github.com/vllm-project/vllm/blob/main/vllm/_aiter_ops.py
        compute_capability = torch.cuda.get_device_capability()
        major, minor = compute_capability
        capability = major * 10 + minor
        aiter_supported = is_aiter_found_and_supported() and 90 < capability < 100

        if selected_backend is not None:
            backend_upper = selected_backend.upper()
            if backend_upper == "FLASH_ATTN" and not aiter_supported:
                logger.warning(
                    "Flash Attention requires `aiter` library which is only supported "
                    "on gfx942 and gfx950. Falling back to TORCH_SDPA backend."
                )
                logger.info("Defaulting to diffusion attention backend SDPA")
                return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()
            backend = DiffusionAttentionBackendEnum[backend_upper]
            logger.info("Using diffusion attention backend '%s'", backend_upper)
            return backend.get_path()

        # Choose to enable Flash Attention by default on ROCm
        # whenever possible as it is the fastest backend
        if aiter_supported:
            logger.info("Defaulting to diffusion attention backend FLASH_ATTN")
            return DiffusionAttentionBackendEnum.FLASH_ATTN.get_path()

        logger.info("Defaulting to diffusion attention backend SDPA")
        return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()

    @classmethod
    def supports_torch_inductor(cls) -> bool:
        return True

    @classmethod
    def get_default_stage_config_path(cls) -> str:
        return "vllm_omni/platforms/rocm/stage_configs"

    @classmethod
    def get_torch_device(cls, local_rank: int | None = None) -> torch.device:
        if local_rank is None:
            return torch.device("cuda")
        return torch.device("cuda", local_rank)

    @classmethod
    def get_device_count(cls) -> int:
        return torch.cuda.device_count()

    @classmethod
    def get_device_version(cls) -> str | None:
        if torch.version.hip is not None:
            hip_version = torch.version.hip
            return hip_version.split("-")[0]
        return None

    @classmethod
    def synchronize(cls) -> None:
        torch.cuda.synchronize()

    @classmethod
    def get_free_memory(cls, device: torch.device | None = None) -> int:
        free, _ = torch.cuda.mem_get_info(device)
        return free
