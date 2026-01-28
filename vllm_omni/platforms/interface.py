# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum

import torch
from vllm.platforms import Platform


class OmniPlatformEnum(Enum):
    """Enum for supported Omni platforms."""

    CUDA = "cuda"
    ROCM = "rocm"
    NPU = "npu"
    XPU = "xpu"
    UNSPECIFIED = "unspecified"


class OmniPlatform(Platform):
    """
    Abstract base class for vllm-omni Platform.

    Inherits from vLLM's Platform and adds Omni-specific interfaces.
    This gives OmniPlatform all vLLM Platform capabilities plus
    Omni-specific methods.
    """

    _omni_enum: OmniPlatformEnum

    def is_npu(self) -> bool:
        return self._omni_enum == OmniPlatformEnum.NPU

    @classmethod
    def get_omni_ar_worker_cls(cls) -> str:
        raise NotImplementedError

    @classmethod
    def get_omni_generation_worker_cls(cls) -> str:
        raise NotImplementedError

    @classmethod
    def get_default_stage_config_path(cls) -> str:
        raise NotImplementedError

    @classmethod
    def get_diffusion_attn_backend_cls(
        cls,
        selected_backend: str | None,
        head_size: int,
    ) -> str:
        """Get the diffusion attention backend class path for this platform.

        This method selects the appropriate attention backend for diffusion
        models based on platform capabilities and user preferences.

        Args:
            selected_backend: User-selected backend name (e.g., "FLASH_ATTN",
                "TORCH_SDPA", "SAGE_ATTN"). If None, uses platform default.
            head_size: Attention head size.

        Returns:
            Fully qualified class path of the selected backend.
        """
        raise NotImplementedError

    @classmethod
    def get_torch_device(cls, local_rank: int | None = None) -> torch.device:
        raise NotImplementedError

    @classmethod
    def get_device_count(cls) -> int:
        raise NotImplementedError

    @classmethod
    def get_device_version(cls) -> str | None:
        raise NotImplementedError

    @classmethod
    def synchronize(cls) -> None:
        raise NotImplementedError

    @classmethod
    def get_free_memory(cls, device: torch.device | None = None) -> int:
        raise NotImplementedError


class UnspecifiedOmniPlatform(OmniPlatform):
    _omni_enum = OmniPlatformEnum.UNSPECIFIED
    device_type = ""
