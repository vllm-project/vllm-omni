# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm_ascend.platform import NPUPlatform

from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

logger = init_logger(__name__)

_DIFFUSION_PACKED_MODULES_MAPPING = {
    "HunyuanImage3Pipeline": {
        "experts": ["experts.0.gate_up_proj", "experts.0.down_proj"],
    },
}


class NPUOmniPlatform(OmniPlatform, NPUPlatform):
    """NPU/Ascend implementation of OmniPlatform.

    Inherits all NPU-specific implementations from vllm-ascend's NPUPlatform,
    and adds Omni-specific interfaces from OmniPlatform.
    """

    _omni_enum = OmniPlatformEnum.NPU
    dist_backend: str = "hccl"

    @classmethod
    def get_omni_ar_worker_cls(cls) -> str:
        return "vllm_omni.platforms.npu.worker.npu_ar_worker.NPUARWorker"

    @classmethod
    def get_omni_generation_worker_cls(cls) -> str:
        return "vllm_omni.platforms.npu.worker.npu_generation_worker.NPUGenerationWorker"

    @classmethod
    def get_default_stage_config_path(cls) -> str:
        return "vllm_omni/platforms/npu/stage_configs"

    @classmethod
    def get_diffusion_model_impl_qualname(cls, op_name: str) -> str:
        if op_name == "hunyuan_fused_moe":
            return "vllm_omni.platforms.npu.models.hunyuan_fused_moe.AscendHunyuanFusedMoE"
        return super().get_diffusion_model_impl_qualname(op_name)

    @classmethod
    def prepare_diffusion_op_runtime(cls, op_name: str, **kwargs: Any) -> None:
        if op_name != "hunyuan_fused_moe":
            return

        from vllm_omni.platforms.npu.models.hunyuan_fused_moe import (
            prepare_hunyuan_fused_moe_runtime,
        )

        prepare_hunyuan_fused_moe_runtime()

    @classmethod
    def get_diffusion_packed_modules_mapping(
        cls,
        model_class: type[nn.Module],
    ) -> dict[str, list[str]] | None:
        return _DIFFUSION_PACKED_MODULES_MAPPING.get(model_class.__name__, None)

    @classmethod
    def get_diffusion_attn_backend_cls(
        cls,
        selected_backend: str | None,
        head_size: int,
    ) -> str:
        from importlib.util import find_spec

        if selected_backend is not None:
            backend_upper = selected_backend.upper()
            backend = DiffusionAttentionBackendEnum[backend_upper]
            logger.debug("Using diffusion attention backend '%s'", backend_upper)
            return backend.get_path()

        # Try FLASH_ATTN if mindiesd is available, otherwise fall back to SDPA
        if find_spec("mindiesd"):
            # Configure ASCEND_CUSTOM_OPP_PATH for mindiesd custom ops upon import
            import mindiesd  # noqa: F401

            logger.debug("Defaulting to diffusion attention backend FLASH_ATTN")
            return DiffusionAttentionBackendEnum.FLASH_ATTN.get_path()

        logger.debug("Falling back to diffusion attention backend SDPA")
        return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()

    @classmethod
    def supports_torch_inductor(cls) -> bool:
        return False

    @classmethod
    def get_torch_device(cls, local_rank: int | None = None) -> torch.device:
        if local_rank is None:
            return torch.device("npu")
        return torch.device("npu", local_rank)

    @classmethod
    def get_device_count(cls) -> int:
        return torch.npu.device_count()

    @classmethod
    def get_device_version(cls) -> str | None:
        return None

    @classmethod
    def synchronize(cls) -> None:
        torch.npu.synchronize()

    @classmethod
    def get_free_memory(cls, device: torch.device | None = None) -> int:
        free, _ = torch.npu.mem_get_info(device)
        return free

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.npu.get_device_properties(device_id)
        return device_props.total_memory

    @classmethod
    def create_autocast_context(cls, *, device_type, dtype, enabled=True):
        if device_type != "npu":
            return super().create_autocast_context(
                device_type=device_type,
                dtype=dtype,
                enabled=enabled,
            )
        if not enabled:
            return nullcontext()

        # NPU-specific fallback
        try:
            return torch.npu.amp.autocast(dtype=dtype)
        except (RuntimeError, TypeError, ValueError) as exc:
            logger.warning("autocast unavailable for device_type=%s dtype=%s: %s", device_type, dtype, exc)
        return nullcontext()

    @classmethod
    def get_profiler_cls(cls) -> str:
        return "vllm_omni.platforms.npu.profiler.NPUTorchProfilerWrapper"
