# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm import envs
from vllm.config import VllmConfig
from vllm.config.kernel import IrOpPriorityConfig
from vllm.logger import init_logger
from vllm.platforms.rocm import RocmPlatform

from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

logger = init_logger(__name__)


class RocmOmniPlatform(OmniPlatform, RocmPlatform):
    """ROCm/AMD GPU implementation of OmniPlatform.

    Inherits all ROCm-specific implementations from vLLM's RocmPlatform,
    and adds Omni-specific interfaces from OmniPlatform.


    NOTE: AR Attention Backend Overriding Logic:
    ------------------------------------------
    Since vLLM v0.19.0, the default attention backend is ROCM_ATTN for ROCm.
    However, the compatibility of ROCM_ATTN with Omni is not guaranteed.
    Therefore, we still use TRITON_ATTN as the default attention backend,
    when the selected_backend is not specified.

    So the behaviour of the attention backend overriding logic currently lives in
    extract_stage_metadata in `vllm_omni/engine/stage_init_utils.py`

    ```
    if current_omni_platform.is_rocm():
        print(f"engine_args: {str(engine_args)}")
        if engine_args.get("attention_backend") is None:
            from vllm._aiter_ops import rocm_aiter_ops

            if rocm_aiter_ops.is_enabled():
                engine_args["attention_backend"] = "ROCM_AITER_FA"
            # Before vLLM v0.19.0, the default attention backend is TRITON_ATTN for ROCm.
            # Since vLLM v0.19.0, the default attention backend is ROCM_ATTN for ROCm.
            # However, the compatibility of ROCM_ATTN with Omni is not guaranteed.
            # Therefore, we still use TRITON_ATTN as the default attention backend,
            # when the selected_backend is not specified.
            engine_args["attention_backend"] = "TRITON_ATTN"
    ```

    """

    _omni_enum = OmniPlatformEnum.ROCM

    @classmethod
    def get_omni_ar_worker_cls(cls) -> str:
        return "vllm_omni.worker.gpu_ar_worker.GPUARWorker"

    @classmethod
    def get_omni_generation_worker_cls(cls) -> str:
        return "vllm_omni.worker.gpu_generation_worker.GPUGenerationWorker"

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
                logger.debug("Defaulting to diffusion attention backend SDPA")
                return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()
            backend = DiffusionAttentionBackendEnum[backend_upper]
            logger.debug("Using diffusion attention backend '%s'", backend_upper)
            return backend.get_path()

        # Choose to enable Flash Attention by default on ROCm
        # whenever possible as it is the fastest backend
        if aiter_supported:
            logger.debug("Defaulting to diffusion attention backend FLASH_ATTN")
            return DiffusionAttentionBackendEnum.FLASH_ATTN.get_path()

        logger.debug("Defaulting to diffusion attention backend SDPA")
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
        return torch.accelerator.device_count()

    @classmethod
    def get_device_version(cls) -> str | None:
        if torch.version.hip is not None:
            hip_version = torch.version.hip
            return hip_version.split("-")[0]
        return None

    @classmethod
    def synchronize(cls) -> None:
        torch.accelerator.synchronize()

    @classmethod
    def get_free_memory(cls, device: torch.device | None = None) -> int:
        free, _ = torch.cuda.mem_get_info(device)
        return free

    @classmethod
    def set_device_control_env_var(cls, devices: str | int | None) -> None:
        import os

        os.environ["HIP_VISIBLE_DEVICES"] = devices
        os.environ["CUDA_VISIBLE_DEVICES"] = devices

    @classmethod
    def unset_device_control_env_var(cls) -> None:
        import os

        os.environ.pop("HIP_VISIBLE_DEVICES", None)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    @classmethod
    def get_default_ir_op_priority(cls, vllm_config: VllmConfig) -> IrOpPriorityConfig:
        """Copied from vllm/platforms/rocm/platform.py v0.20.0 with force using vllm_c kernels"""
        # TODO(luka/TJ) use aiter, vllm_c, native by default on ROCm
        cc = vllm_config.compilation_config
        default = ["vllm_c", "native"]  # Originally using "native" here when compiling

        # This (mostly) preserves previous CustomOp behavior
        # Necessary on ROCm because it's common that users
        # enable rms_norm to use the aiter kernel.
        # TODO(luka/TJ) remove env vars completely
        if cc.is_custom_op_enabled("rms_norm") and envs.VLLM_ROCM_USE_AITER and envs.VLLM_ROCM_USE_AITER_RMSNORM:
            rms_norm = ["aiter"] + default
        else:
            rms_norm = default

        return IrOpPriorityConfig.with_default(default, rms_norm=rms_norm)
