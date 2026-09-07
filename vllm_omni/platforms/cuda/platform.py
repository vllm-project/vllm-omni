# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

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
        return "vllm_omni/deploy"

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
    def has_flash_attn_4(cls) -> bool:
        """Return whether CuTe FA4 is importable (Blackwell-capable FLASH_ATTN)."""
        from vllm_omni.diffusion.attention.backends.utils.fa import is_flash_attn_4_available

        return is_flash_attn_4_available()

    @classmethod
    def supports_diffusion_dense_flash_attention(cls) -> bool:
        """Dense FLASH_ATTN is FA4 on Blackwell; FA2/FA3 on older CUDA GPUs."""
        capability = cls.get_device_capability()
        if capability is not None and capability.major in (10, 11, 12):
            return cls.has_flash_attn_4()
        return cls.has_flash_attn_package()

    @classmethod
    def get_diffusion_attn_backend_cls(
        cls,
        selected_backend: str | None,
        head_size: int,
        allow_trtllm_default: bool = True,
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
                    "Detected Blackwell-class GPU %s (untested variant); routing to CUDNN_ATTN.",
                    sm_str,
                )

        # Check if FA packages are available
        packages_info = PACKAGES_CHECKER.get_packages_info()
        packages_available = packages_info.get("has_flash_attn", False)

        # FA2/FA3 wheels are often importable on Blackwell but only ship
        # Ampere/Ada/Hopper kernels; the first forward then dies with
        # "no kernel image". FA4 (flash_attn.cute / vllm-omni[fa4]) is the
        # FLASH_ATTN implementation that actually runs on Blackwell.
        if is_blackwell:
            flash_attn_supported = compute_supported and cls.has_flash_attn_4()
        else:
            flash_attn_supported = compute_supported and packages_available

        # cuDNN 9.5+ ships Blackwell FMHA kernels. If the runtime is older,
        # skip the CUDNN_ATTN default rather than selecting a backend whose
        # kernel selector may reject the runtime shape.
        cudnn_version = torch.backends.cudnn.version() or 0
        cudnn_blackwell_ready = cudnn_version >= 90500

        # FlashInfer edges cuDNN by ~4% at the kernel level on sm_120 but
        # regresses ~2x at e2e on HV-1.5 because its dense-prefill path cannot
        # represent every diffusion attention mask without changing semantics.
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

        # FLASHINFER_ATTN needs BatchPrefillWithRaggedKVCacheWrapper, not just a
        # top-level flashinfer package. Probe the actual symbol so a stub /
        # partial wheel is not auto-routed then crash at layer init.
        flashinfer_prefill_available = False
        if flashinfer_available:
            try:
                from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper  # noqa: F401

                flashinfer_prefill_available = True
            except Exception as e:
                logger.debug("FlashInfer prefill wrapper probe failed (%s); treating as unavailable", e)

        # TRTLLM_ATTN needs the trtllm-gen kernel specifically, not just any FlashInfer
        # wheel. Probe the actual symbol so a released wheel lacking it does not get
        # auto-routed to TRTLLM_ATTN and then crash on the first forward.
        trtllm_gen_available = False
        if flashinfer_available:
            try:
                from flashinfer.prefill import trtllm_ragged_attention_deepseek  # noqa: F401

                trtllm_gen_available = True
            except Exception as e:
                logger.debug("trtllm-gen kernel probe failed (%s); treating as unavailable", e)

        if selected_backend is not None:
            backend_upper = selected_backend.upper()
            # Architecture / package probes for optional kernels must run before
            # validate_diffusion_attn_backend, which imports the backend module
            # and would otherwise load sageattention / sageattn3 on unsupported GPUs.
            if backend_upper == "SAGE_ATTN":
                sage_supported_sms = {(8, 0), (8, 6), (8, 9), (9, 0), (12, 0), (12, 1)}
                if compute_capability is None or tuple(compute_capability) not in sage_supported_sms:
                    raise ValueError(
                        f"SAGE_ATTN was explicitly selected but does not provide a kernel for {sm_str or 'this GPU'}. "
                        "Select a compatible backend."
                    )
                try:
                    importlib.import_module("sageattention")
                except ImportError as e:
                    raise ImportError(
                        "SAGE_ATTN was explicitly selected, but the sageattention package is not available."
                    ) from e
            if backend_upper == "SAGE_ATTN_3":
                sage_attn3_supported = compute_capability is not None and compute_capability.major >= 10
                if not sage_attn3_supported:
                    raise ValueError(
                        "SAGE_ATTN_3 was explicitly selected but requires a Blackwell-class GPU "
                        "with compute capability >= 10.0. Select a compatible backend."
                    )
                try:
                    importlib.import_module("sageattn3")
                except ImportError as e:
                    raise ImportError(
                        "SAGE_ATTN_3 was explicitly selected, but the sageattn3 package is not available. "
                        "Install SageAttention/sageattention3_blackwell or select a different backend."
                    ) from e
            cls.validate_diffusion_attn_backend(backend_upper)
            if backend_upper == "SAGE_ATTN_3" and head_size > 0:
                sage3_cls = DiffusionAttentionBackendEnum.SAGE_ATTN_3.get_class()
                if not sage3_cls.supports_head_size(head_size):
                    raise ValueError(
                        f"SAGE_ATTN_3 was explicitly selected but head_size={head_size} is unsupported. "
                        f"Supported head sizes: {sage3_cls.get_supported_head_sizes()}. "
                        "Select TORCH_SDPA or another backend."
                    )
            if backend_upper in ("FLASH_ATTN_HUB", "FLASH_ATTN_3_HUB"):
                try:
                    importlib.import_module("kernels")
                    logger.info("Using HuggingFace kernels-backed attention backend '%s'", backend_upper)
                except ImportError as e:
                    raise ImportError(
                        f"{backend_upper} was explicitly selected, but the HuggingFace `kernels` "
                        "library is not available. Install `kernels` or select a different backend."
                    ) from e

            if backend_upper == "FLASH_ATTN_HUB":
                fa2_hub_supported = compute_capability is not None and compute_capability.major in (8, 9)
                if not fa2_hub_supported:
                    raise ValueError(
                        "FLASH_ATTN_HUB was explicitly selected but its current FA2 kernels require "
                        "an Ampere, Ada, or Hopper GPU (compute capability 8.x/9.x)."
                    )

            if backend_upper == "FLASH_ATTN_3_HUB":
                fa3_hub_supported = compute_capability is not None and compute_capability.major == 9
                if not fa3_hub_supported:
                    raise ValueError(
                        "FLASH_ATTN_3_HUB was explicitly selected but its current kernels require a Hopper GPU "
                        "with compute capability 9.x. Select a compatible backend."
                    )

            if backend_upper == "FLASH_ATTN" and not flash_attn_supported:
                if is_blackwell:
                    reason = (
                        "Blackwell requires CuTe FlashAttention-4 "
                        "(flash_attn.cute / vllm-omni[fa4]); FA2/FA3 kernels are Hopper-only"
                    )
                elif not compute_supported:
                    reason = "compute capability < 8.0"
                else:
                    reason = "Flash Attention package unavailable"
                raise ValueError(
                    f"FLASH_ATTN was explicitly selected but is unsupported ({reason}). Select a compatible backend."
                )
            if backend_upper == "FLASHINFER_ATTN" and not flashinfer_available:
                raise ValueError(
                    "FLASHINFER_ATTN was explicitly selected, but FlashInfer is unavailable. "
                    "Install a compatible FlashInfer build or select a different backend."
                )
            if backend_upper == "FLASHINFER_ATTN" and not flashinfer_prefill_available:
                raise ValueError(
                    "FLASHINFER_ATTN was explicitly selected, but the installed FlashInfer build does not "
                    "provide BatchPrefillWithRaggedKVCacheWrapper. Install a compatible build."
                )
            if backend_upper == "FLASHINFER_ATTN":
                from vllm_omni.diffusion.attention.backends.flashinfer_attn import FlashInferAttentionBackend

                # head_size <= 0 is the capability-probe sentinel. Skip geometry
                # validation; Attention construction still checks the real size.
                if head_size > 0 and not FlashInferAttentionBackend.supports_head_size(head_size):
                    raise ValueError(
                        f"FLASHINFER_ATTN was explicitly selected but head_size={head_size} is unsupported. "
                        f"Supported head sizes: {FlashInferAttentionBackend.get_supported_head_sizes()}. "
                        "Select TORCH_SDPA or another backend."
                    )
            if backend_upper == "CUDNN_ATTN":
                from vllm_omni.diffusion.attention.backends.cudnn_attn import CuDNNAttentionBackend

                # head_size <= 0 is the capability-probe sentinel (auto-pad does
                # not know head_dim). Skip geometry validation; the real
                # Attention construction still checks the configured size.
                if head_size > 0 and not CuDNNAttentionBackend.supports_head_size(head_size):
                    raise ValueError(
                        f"CUDNN_ATTN was explicitly selected but head_size={head_size} is unsupported. "
                        "Blackwell cuDNN FMHA requires head_dim divisible by 8 and no larger than 256. "
                        "Select FLASHINFER_ATTN or TORCH_SDPA."
                    )
            if backend_upper == "TRTLLM_ATTN":
                trtllm_attn_supported = compute_capability is not None and compute_capability.major == 10
                if not trtllm_attn_supported:
                    raise ValueError(
                        "TRTLLM_ATTN diffusion attention backend requires a datacenter "
                        "Blackwell GPU (SM100 / SM103, compute capability 10.x). Select a "
                        "different --diffusion-attention-backend."
                    )
                if not trtllm_gen_available:
                    raise ValueError(
                        "TRTLLM_ATTN was explicitly selected, but the installed FlashInfer build does not "
                        "provide trtllm_ragged_attention_deepseek. Install a compatible build."
                    )
            backend = DiffusionAttentionBackendEnum[backend_upper]
            logger.debug("Using diffusion attention backend '%s'", backend_upper)
            return backend.get_path()

        trtllm_attn_default_ok = (
            allow_trtllm_default
            and compute_capability is not None
            and compute_capability.major == 10
            and head_size == 128
            and trtllm_gen_available
        )
        if trtllm_attn_default_ok:
            logger.info(
                "Defaulting to diffusion attention backend TRTLLM_ATTN (datacenter Blackwell %s, head_dim %d)",
                sm_str,
                head_size,
            )
            return DiffusionAttentionBackendEnum.TRTLLM_ATTN.get_path()

        if is_blackwell and cudnn_blackwell_ready:
            from vllm_omni.diffusion.attention.backends.cudnn_attn import CuDNNAttentionBackend

            # Unknown head_size is a capability probe (SP auto-pad), not a
            # real DiT geometry. Keep CUDNN so mask support is reported.
            if head_size <= 0 or CuDNNAttentionBackend.supports_head_size(head_size):
                logger.info(
                    "Defaulting to diffusion attention backend CUDNN_ATTN (Blackwell %s, cuDNN %d, head_dim %d)",
                    sm_str,
                    cudnn_version,
                    head_size,
                )
                return DiffusionAttentionBackendEnum.CUDNN_ATTN.get_path()
            logger.info(
                "Skipping CUDNN_ATTN on Blackwell %s: head_dim %d is outside cuDNN FMHA "
                "(multiples of 8, 8-256). Automatic selection will use FlashInfer when available, otherwise SDPA.",
                sm_str,
                head_size,
            )

        if is_blackwell and flashinfer_prefill_available:
            from vllm_omni.diffusion.attention.backends.flashinfer_attn import FlashInferAttentionBackend

            if FlashInferAttentionBackend.supports_head_size(head_size):
                logger.info(
                    "Defaulting to diffusion attention backend FLASHINFER_ATTN (Blackwell %s, head_dim %d)",
                    sm_str,
                    head_size,
                )
                return DiffusionAttentionBackendEnum.FLASHINFER_ATTN.get_path()

        if is_blackwell and not cudnn_blackwell_ready:
            logger.warning(
                "Detected Blackwell %s but cuDNN %d < 9.5 — no tuned Blackwell FMHA. "
                "Automatic backend selection will use FlashInfer when available, otherwise SDPA.",
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
    def record_device_event(cls) -> torch.Event | None:
        """Record a device event on the default stream to mark tensor readiness."""
        try:
            event = torch.Event()
            event.record()
            return event
        except Exception:
            logger.warning("Failed to record device event for cross-stream sync")
            return None

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
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.cuda.get_device_properties(device_id)
        return device_props.total_memory

    @classmethod
    def is_fully_connected(cls, device_ids: list[int]) -> bool:
        logger.debug("NVLink detection not available on CudaOmniPlatform; assuming no NVLink.")
        return False

    @classmethod
    def get_default_ir_op_priority(cls, vllm_config: VllmConfig) -> IrOpPriorityConfig:
        """Prefer ``vllm_c`` CUDA kernels over ``native`` for diffusion IR ops."""
        default = ["vllm_c", "native"]

        # Use oink if enabled for rms_norm
        # TODO(Laurawly/luka): remove this env var,
        #  users can just use IR op priority directly
        rms_norm = default
        if envs.VLLM_USE_OINK_OPS:
            rms_norm = ["oink"] + default

        return IrOpPriorityConfig.with_default(default, rms_norm=rms_norm, fused_add_rms_norm=rms_norm)
