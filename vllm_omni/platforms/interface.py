# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger
from vllm.platforms import Platform
from vllm.platforms.interface import PlatformEnum

logger = init_logger(__name__)


class OmniPlatformEnum(Enum):
    """Enum for supported Omni platforms."""

    CUDA = "cuda"
    ROCM = "rocm"
    NPU = "npu"
    XPU = "xpu"
    MUSA = "musa"
    OOT = "oot"
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

    def is_out_of_tree(self) -> bool:
        return self._omni_enum == OmniPlatformEnum.OOT

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
    def prepare_diffusion_op_runtime(cls, op_name: str, **kwargs: Any) -> None:
        return None

    @classmethod
    def register_additional_diffusion_fused_moe_hooks(cls, moe_runner: Any) -> None:
        # One-shot lazy kernel initialisation on the first forward (no-op unless
        # the runner exposes an uninitialised quant_method). Mirrors the prior
        # wrapper behaviour exactly, just bound to the runner module.
        init_handle: Any = None

        def _kernel_init_pre_hook(module: Any, args: Any, kwargs: Any) -> None:
            nonlocal init_handle
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None and getattr(quant_method, "moe_kernel", None) is None:
                quant_method.process_weights_after_loading(module)
            if init_handle is not None:
                init_handle.remove()

        init_handle = moe_runner.register_forward_pre_hook(
            _kernel_init_pre_hook,
            with_kwargs=True,
        )

    @classmethod
    def reset_diffusion_fused_moe_forward_context(cls) -> None:
        return None

    @classmethod
    def get_diffusion_packed_modules_mapping(
        cls,
        model_class: type[nn.Module],
    ) -> dict[str, list[str]] | None:
        return None

    @classmethod
    def get_diffusion_attn_backend_cls(
        cls,
        selected_backend: str | None,
        head_size: int,
        allow_trtllm_default: bool = False,
    ) -> str:
        """Get the diffusion attention backend class path for this platform.

        This method selects the appropriate attention backend for diffusion
        models based on platform capabilities and user preferences.

        Args:
            selected_backend: User-selected backend name (e.g., "FLASH_ATTN",
                "TORCH_SDPA", "SAGE_ATTN"). If None, uses platform default.
            head_size: Attention head size.
            allow_trtllm_default: Whether TRTLLM may be chosen as the default.

        Returns:
            Fully qualified class path of the selected backend.
        """
        raise NotImplementedError

    @classmethod
    def supports_torch_inductor(cls) -> bool:
        """Check if the platform supports torch.compile with inductor backend."""
        raise NotImplementedError

    @classmethod
    def has_flash_attn_package(cls) -> bool:
        """Check if a Flash Attention package is available and usable on this platform."""
        return False

    @classmethod
    def get_diffusion_worker_cls(cls) -> str:
        """Get the diffusion worker class path for this platform.

        Returns a fully qualified class path string that will be resolved
        and instantiated by WorkerWrapperBase. The class must be compatible
        with the DiffusionWorker interface.
        """
        return "vllm_omni.diffusion.worker.diffusion_worker.DiffusionWorker"

    @classmethod
    def get_diffusion_model_runner_cls(cls) -> str:
        """Get the diffusion model runner class path for this platform.

        Returns a fully qualified class path string. The class must be
        compatible with the DiffusionModelRunner interface.
        """
        return "vllm_omni.diffusion.worker.diffusion_model_runner.DiffusionModelRunner"

    @classmethod
    def init_diffusion_worker_vllm_config(
        cls,
        vllm_config: Any,
    ) -> None:
        """Initialize platform-specific state for diffusion worker VllmConfig."""
        return None

    @classmethod
    def init_diffusion_model_runner_runtime(
        cls,
        vllm_config: Any,
        od_config: Any,
        device: torch.device,
    ) -> None:
        """Initialize platform-specific runtime state for diffusion model runners."""
        return None

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

    # ── Async diffusion output: cross-stream sync ──

    @classmethod
    def record_device_event(cls):
        """Record a device event on the default stream to mark tensor readiness.

        On platforms where distributed communication (e.g. HCCL) may use
        internal streams not visible to the default stream, this method
        should synchronize the default stream before recording the event
        to ensure the event captures all completed work including
        cross-device communication results.

        Returns ``None`` by default so that platforms without a native
        implementation (ROCm, XPU, MUSA) fall through to a safe no-op.
        Override in platform subclasses to provide real event support.
        """
        return None

    @classmethod
    def get_free_memory(cls, device: torch.device | None = None) -> int:
        raise NotImplementedError

    @classmethod
    def get_device_memory(cls, device: torch.device | None = None) -> tuple[int, int]:
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

    @classmethod
    def get_graph_wrapper_cls(cls) -> type:
        """Return the platform's full-graph wrapper class.

        Defaults to vLLM's CUDAGraphWrapper; NPU overrides with ACLGraphWrapper.
        """
        from vllm.compilation.cuda_graph import CUDAGraphWrapper

        return CUDAGraphWrapper

    @classmethod
    def set_forward_context(
        cls,
        attn_metadata: Any,
        vllm_config: VllmConfig,
        *,
        cudagraph_runtime_mode: CUDAGraphMode,
        batch_descriptor: BatchDescriptor,
    ):
        """Platform-neutral wrapper around the device's set_forward_context.

        Defaults to vLLM's ``set_forward_context``; NPU overrides to dispatch
        to ``set_ascend_forward_context`` (renaming ``cudagraph_runtime_mode``
        to ``aclgraph_runtime_mode``).
        """
        from vllm.forward_context import set_forward_context

        return set_forward_context(
            attn_metadata,
            vllm_config,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=batch_descriptor,
        )


class UnspecifiedOmniPlatform(OmniPlatform):
    _omni_enum = OmniPlatformEnum.UNSPECIFIED
    _enum = PlatformEnum.UNSPECIFIED
    device_type = "cpu"

    @classmethod
    def get_torch_device(cls, local_rank: int | None = None) -> torch.device:
        return torch.device("cpu")

    @classmethod
    def get_device_count(cls) -> int:
        return 0
