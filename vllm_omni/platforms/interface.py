# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from enum import Enum
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.platforms import Platform

logger = init_logger(__name__)


class OmniPlatformEnum(Enum):
    """Enum for supported Omni platforms."""

    CUDA = "cuda"
    ROCM = "rocm"
    NPU = "npu"
    XPU = "xpu"
    MUSA = "musa"
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

    def is_xpu(self) -> bool:
        return self._omni_enum == OmniPlatformEnum.XPU

    def is_cuda(self) -> bool:
        return self._omni_enum == OmniPlatformEnum.CUDA

    def is_rocm(self) -> bool:
        return self._omni_enum == OmniPlatformEnum.ROCM

    def is_musa(self) -> bool:
        return self._omni_enum == OmniPlatformEnum.MUSA

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
    def get_diffusion_model_impl_qualname(cls, op_name: str) -> str:
        if op_name == "hunyuan_fused_moe":
            return "vllm_omni.diffusion.models.hunyuan_image3.hunyuan_fused_moe.HunyuanFusedMoEDefault"
        raise NotImplementedError(f"Unsupported diffusion model op: {op_name}")

    @classmethod
    def prepare_diffusion_op_runtime(cls, op_name: str, **kwargs: Any) -> None:
        return None

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
    def supports_torch_inductor(cls) -> bool:
        """Check if the platform supports torch.compile with inductor backend."""
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

    @classmethod
    def create_autocast_context(
        cls,
        *,
        device_type: str,
        dtype: torch.dtype,
        enabled: bool = True,
    ):
        if not enabled:
            return nullcontext()

        try:
            return torch.autocast(device_type=device_type, dtype=dtype, enabled=True)
        except (RuntimeError, TypeError, ValueError) as exc:
            logger.warning("autocast unavailable for device_type=%s dtype=%s: %s", device_type, dtype, exc)
            return nullcontext()

    @classmethod
    def supports_cpu_offload(cls) -> bool:
        return True

    @classmethod
    def supports_float64(cls) -> bool:
        return True

    @classmethod
    def set_device_control_env_var(cls, devices: str | int | None) -> None:
        import os

        os.environ[cls.device_control_env_var] = devices

    @classmethod
    def unset_device_control_env_var(cls) -> None:
        import os

        os.environ.pop(cls.device_control_env_var, None)

    @classmethod
    def get_profiler_cls(cls) -> str:
        """Get the profiler class for this platform.

        Returns:
            Fully qualified class path of the profiler.
            Default returns the base OmniTorchProfilerWrapper.
        """
        return "vllm_omni.profiler.omni_torch_profiler.OmniTorchProfilerWrapper"


class UnspecifiedOmniPlatform(OmniPlatform):
    _omni_enum = OmniPlatformEnum.UNSPECIFIED
    device_type = ""
