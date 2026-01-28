# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.logger import init_logger
from vllm.platforms.xpu import XPUPlatform

from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

logger = init_logger(__name__)


class XPUOmniPlatform(OmniPlatform, XPUPlatform):
    """XPU/Intel GPU implementation of OmniPlatform.

    Inherits all XPU-specific implementations from vLLM's XPUPlatform,
    and adds Omni-specific interfaces from OmniPlatform.
    """

    _omni_enum = OmniPlatformEnum.XPU

    @classmethod
    def get_omni_ar_worker_cls(cls) -> str:
        return "vllm_omni.platforms.xpu.worker.xpu_ar_worker.XPUARWorker"

    @classmethod
    def get_omni_generation_worker_cls(cls) -> str:
        return "vllm_omni.platforms.xpu.worker.xpu_generation_worker.XPUGenerationWorker"

    @classmethod
    def get_diffusion_attn_backend_cls(
        cls,
        selected_backend: str | None,
        head_size: int,
    ) -> str:
        if selected_backend is not None:
            backend_upper = selected_backend.upper()
            backend = DiffusionAttentionBackendEnum[backend_upper]
            logger.info("Using diffusion attention backend '%s'", backend_upper)
            return backend.get_path()

        logger.info("Defaulting to diffusion attention backend SDPA")
        return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()

    @classmethod
    def get_default_stage_config_path(cls) -> str:
        return "vllm_omni/platforms/xpu/stage_configs"

    @classmethod
    def get_torch_device(cls, local_rank: int | None = None) -> torch.device:
        if local_rank is None:
            return torch.device("xpu")
        return torch.device("xpu", local_rank)

    @classmethod
    def get_device_count(cls) -> int:
        return torch.xpu.device_count()

    @classmethod
    def get_device_version(cls) -> str | None:
        # XPU does not have a version string like CUDA
        return None

    @classmethod
    def synchronize(cls) -> None:
        torch.xpu.synchronize()

    @classmethod
    def get_free_memory(cls, device: torch.device | None = None) -> int:
        if device is None:
            device_id = 0
        else:
            device_id = device.index if device.index is not None else 0
        props = torch.xpu.get_device_properties(device_id)
        return props.total_memory
