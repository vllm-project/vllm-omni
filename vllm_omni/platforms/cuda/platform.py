# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

import torch
import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.config.kernel import IrOpPriorityConfig
from vllm.logger import init_logger
from vllm.platforms.cuda import CudaPlatformBase
from vllm.platforms.interface import DeviceCapability

from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

logger = init_logger(__name__)


class CudaOmniPlatform(OmniPlatform, CudaPlatformBase):
    """CUDA/GPU implementation of OmniPlatform (default).

    Inherits all CUDA-specific implementations from vLLM's CudaPlatform,
    and adds Omni-specific interfaces from OmniPlatform.
    """

    _omni_enum = OmniPlatformEnum.CUDA

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
    def has_flash_attn_package(cls) -> bool:
        from vllm_omni.diffusion.attention.backends.utils.fa import is_flash_attn_installed

        # Turing/Tesla/T4 GPUs don't support flash attention well
        gpu_name = cls.get_device_name()
        if "Turing" in gpu_name or "Tesla" in gpu_name or "T4" in gpu_name:
            return False

        if not is_flash_attn_installed():
            return False

        return True

    @classmethod
    def get_diffusion_attn_backend_cls(
        cls,
        selected_backend: str | None,
        head_size: int,
    ) -> str:
        from vllm_omni.diffusion.envs import PACKAGES_CHECKER

        # Check compute capability for Flash Attention support.
        # FA requires sm_80+. Blackwell (sm_10x/sm_12x) only works with FA builds
        # that include the Blackwell CUTE kernel — plain FA2 will crash there.
        #
        # Known Blackwell SKUs:
        #   sm_100 = B200 / GB200 (datacenter)
        #   sm_103 = B300 / GB300 (Blackwell Ultra)
        #   sm_120 = RTX Pro 6000, RTX 50-series (consumer)
        #   sm_121 = consumer Blackwell refresh
        _known_blackwell_sms = {(10, 0), (10, 3), (12, 0), (12, 1)}
        compute_capability = cls.get_device_capability()
        compute_supported = False
        is_blackwell = False
        sm_str = ""
        if compute_capability is not None:
            major, minor = compute_capability
            capability = major * 10 + minor
            compute_supported = capability >= 80
            sm_str = f"sm_{major}{minor}"
            # Accept major in {10, 11, 12} to cover future Blackwell refreshes.
            is_blackwell = major in (10, 11, 12)
            if is_blackwell and (major, minor) not in _known_blackwell_sms:
                logger.info(
                    "Detected Blackwell-class GPU %s (untested variant); routing to CUDNN_ATTN with SDPA fallback.",
                    sm_str,
                )

        # Check if FA packages are available
        packages_info = PACKAGES_CHECKER.get_packages_info()
        packages_available = packages_info.get("has_flash_attn", False)

        # Both compute capability and packages must be available for FA
        flash_attn_supported = compute_supported and packages_available

        # cuDNN 9.5+ ships Blackwell FMHA kernels. If the runtime is older,
        # the CUDNN_ATTN default would still work via internal fallback but
        # without the tuned Blackwell path, so we skip routing there.
        cudnn_version = torch.backends.cudnn.version() or 0
        cudnn_blackwell_ready = cudnn_version >= 90500

        # FlashInfer edges cuDNN by ~4% at the kernel level on sm_120 but
        # regresses ~2x at e2e on HV-1.5 because its dense-prefill path can't
        # take 2D attn_masks and the SDPA fallback dispatches to
        # EFFICIENT_ATTENTION (~25 ms) instead of the cuDNN mask path (~11 ms).
        # CUDNN_ATTN pins sdpa_kernel([CUDNN_ATTENTION]) directly so masked
        # calls keep the cuDNN path. Blackwell default prefers CUDNN_ATTN;
        # users can opt into FLASHINFER_ATTN explicitly for no-mask workloads.
        flashinfer_available = False
        try:
            import flashinfer  # noqa: F401

            flashinfer_available = True
        except Exception as e:
            # A partially installed / ABI-mismatched wheel can raise OSError or
            # RuntimeError from extension loading, not just ImportError. This
            # runs during default backend selection, so a probe failure must
            # not abort startup — just treat FlashInfer as unavailable.
            logger.debug("FlashInfer probe failed (%s); treating as unavailable", e)

        if selected_backend is not None:
            backend_upper = selected_backend.upper()
            if backend_upper == "FLASH_ATTN" and not flash_attn_supported:
                if not compute_supported:
                    logger.warning(
                        "Flash Attention requires GPU with compute capability >= 8.0. "
                        "Falling back to TORCH_SDPA backend."
                    )
                elif not packages_available:
                    logger.warning("Flash Attention packages not available. Falling back to TORCH_SDPA backend.")
                logger.debug("Defaulting to diffusion attention backend SDPA")
                return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()
            if backend_upper == "SAGE_ATTN_3":
                sage_attn3_supported = compute_capability is not None and compute_capability.major >= 10
                if not sage_attn3_supported:
                    logger.warning(
                        "SageAttention3 requires a Blackwell-class GPU with compute capability >= 10.0. "
                        "Falling back to TORCH_SDPA backend."
                    )
                    return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()
                try:
                    importlib.import_module("sageattn3")
                except ImportError:
                    logger.warning(
                        "SageAttention3 package not available. Install it from "
                        "SageAttention/sageattention3_blackwell. Falling back to TORCH_SDPA backend."
                    )
                    return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()
            backend = DiffusionAttentionBackendEnum[backend_upper]
            logger.debug("Using diffusion attention backend '%s'", backend_upper)
            return backend.get_path()

        if is_blackwell and cudnn_blackwell_ready:
            logger.info(
                "Defaulting to diffusion attention backend CUDNN_ATTN (Blackwell %s, cuDNN %d)",
                sm_str,
                cudnn_version,
            )
            return DiffusionAttentionBackendEnum.CUDNN_ATTN.get_path()

        if is_blackwell and flashinfer_available:
            logger.info(
                "Defaulting to diffusion attention backend FLASHINFER_ATTN (Blackwell %s, cuDNN unavailable)",
                sm_str,
            )
            return DiffusionAttentionBackendEnum.FLASHINFER_ATTN.get_path()

        if is_blackwell and not cudnn_blackwell_ready:
            logger.warning(
                "Detected Blackwell %s but cuDNN %d < 9.5 — no tuned Blackwell FMHA. "
                "Falling through to FLASH_ATTN / SDPA.",
                sm_str,
                cudnn_version,
            )

        if flash_attn_supported:
            logger.debug("Defaulting to diffusion attention backend FLASH_ATTN")
            return DiffusionAttentionBackendEnum.FLASH_ATTN.get_path()

        logger.debug("Defaulting to diffusion attention backend SDPA")
        return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()

    @classmethod
    def supports_torch_inductor(cls) -> bool:
        return True

    @classmethod
    def get_torch_device(cls, local_rank: int | None = None) -> torch.device:
        if local_rank is None:
            return torch.device("cuda")
        return torch.device("cuda", local_rank)

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    def get_device_count(cls) -> int:
        return torch.accelerator.device_count()

    @classmethod
    def get_device_version(cls) -> str | None:
        return torch.version.cuda

    @classmethod
    def synchronize(cls) -> None:
        torch.accelerator.synchronize()

    @classmethod
    def get_free_memory(cls, device: torch.device | None = None) -> int:
        free, _ = torch.cuda.mem_get_info(device)
        return free

    @classmethod
    def get_device_memory(cls, device: torch.device | None = None) -> tuple[int, int]:
        free, total = torch.cuda.mem_get_info(device)
        return free, total

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(device_id)

    @classmethod
    def get_default_ir_op_priority(cls, vllm_config: VllmConfig) -> IrOpPriorityConfig:
        """Copied from vllm/platforms/cuda/platform.py v0.20.0 with force using vllm_c kernels"""
        default = ["vllm_c", "native"]  # Originally using "native" here when compiling

        # Use oink if enabled for rms_norm
        # TODO(Laurawly/luka): remove this env var,
        #  users can just use IR op priority directly
        rms_norm = default
        if envs.VLLM_USE_OINK_OPS:
            rms_norm = ["oink"] + default

        return IrOpPriorityConfig.with_default(default, rms_norm=rms_norm, fused_add_rms_norm=rms_norm)
