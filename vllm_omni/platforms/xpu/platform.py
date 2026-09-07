# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os
from typing import ClassVar

import torch
from vllm.config import VllmConfig
from vllm.config.kernel import IrOpPriorityConfig
from vllm.logger import init_logger
from vllm.platforms.xpu import XPUPlatform

from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

logger = init_logger(__name__)


# Use the native query; vLLM's XPU custom op reports free=0 in spawned workers.
torch.accelerator.get_memory_info = lambda device=None: torch.xpu.mem_get_info(device)

# The XPU sampler kernel is broken for omni on the current vLLM; use the native path.
os.environ.setdefault("VLLM_XPU_USE_SAMPLER_KERNEL", "0")


class XPUOmniPlatform(OmniPlatform, XPUPlatform):
    """XPU/Intel GPU implementation of OmniPlatform.

    Inherits all XPU-specific implementations from vLLM's XPUPlatform,
    and adds Omni-specific interfaces from OmniPlatform.
    """

    _omni_enum = OmniPlatformEnum.XPU

    # One flat all-gather landing buffer per (dtype, device), held on the class
    # so every coordinator in the process lands at the same address; see
    # `all_gather_into_tensor`.
    _all_gather_buffers: ClassVar[dict[tuple[torch.dtype, torch.device], torch.Tensor]] = {}

    @classmethod
    def all_gather_into_tensor(cls, input_: torch.Tensor, world_size: int, group) -> torch.Tensor:
        """All-gather into a reused landing buffer, returning a private copy.

        A fresh output per call means the collective sees a new receive address
        after any request-path `empty_cache()`, and the driver-side registration
        the communication library keeps for each such address is likely never
        reclaimed -- an inference from intervention experiments (B70, 8 cards,
        torch 2.13.0+xpu, xccl), not something read out of a driver registry. A
        model-free 2x2 sandbox separated the factors: changing the collective
        API does not help, reusing the receive buffer does. In an H3 server A/B
        it moved per-request growth outside PyTorch's accounting
        (`card_used - memory_reserved`) from 1726 MiB to 1626 MiB: all-gather is
        one share of that total, not all of it.

        One buffer per (dtype, device), grown monotonically, so residency is
        bounded by the largest gather seen and the address count by the number
        of record highs. `clone()` keeps the caller owning its result, and the
        clone's memory is never handed to the library. The slot is shared, so
        this assumes one execution stream per process, which is what the
        diffusion worker does today.
        """
        output_size = list(input_.size())
        output_size[0] *= world_size
        numel = input_.numel() * world_size
        key = (input_.dtype, input_.device)
        buffer = cls._all_gather_buffers.get(key)
        if buffer is None or buffer.numel() < numel:
            buffer = torch.empty(numel, dtype=input_.dtype, device=input_.device)
            cls._all_gather_buffers[key] = buffer
        landing = buffer[:numel].view(output_size)
        torch.distributed.all_gather_into_tensor(landing, input_.contiguous(), group=group)
        return landing.clone()

    @classmethod
    def reset_all_gather_buffers(cls) -> None:
        cls._all_gather_buffers.clear()

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
        allow_trtllm_default: bool = False,
    ) -> str:
        # XPU has no TRTLLM backend; arg accepted for signature parity but unused.
        compute_capability = torch.xpu.get_device_capability()
        # Intel Max 1100 and 1550 will not support flash_attn currently
        flash_attn_supported = compute_capability["architecture"] not in [13136561920]

        if selected_backend is not None:
            backend_upper = selected_backend.upper()
            cls.validate_diffusion_attn_backend(backend_upper)
            if backend_upper in ("FLASH_ATTN_HUB", "FLASH_ATTN_3_HUB"):
                logger.warning(
                    "HuggingFace kernels-backed FlashAttention is "
                    "not supported on XPU. Falling back to local "
                    "FLASH_ATTN."
                )
                backend_upper = "FLASH_ATTN"

            backend = DiffusionAttentionBackendEnum[backend_upper]
            logger.debug("Using diffusion attention backend '%s'", backend_upper)
            return backend.get_path()

        if flash_attn_supported:
            logger.debug("Defaulting to diffusion attention backend FLASH_ATTN")
            return DiffusionAttentionBackendEnum.FLASH_ATTN.get_path()

        logger.debug("Defaulting to diffusion attention backend SDPA")
        return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()

    @classmethod
    def supports_torch_inductor(cls) -> bool:
        return True

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
    def record_device_event(cls) -> torch.Event | None:
        """Record an XPU event on the current stream to mark tensor readiness.

        Deliberately a device-agnostic ``torch.Event`` rather than a
        ``torch.xpu.Event``. The consumer (the async diffusion output thread)
        waits with ``torch.Stream.wait_event`` on a generic ``torch.Stream``,
        and that C-level binding silently no-ops for a ``torch.xpu.Event``
        instead of enqueuing the dependency — the side stream then starts its
        D2H copy while the compute stream is still writing the tensor, so the
        host reads a partially-written image (garbage rows at the bottom of the
        output). ``torch.Event`` dispatches through the accelerator hooks and
        the wait is honored, which is the actual fix.
        """
        try:
            event = torch.Event()
            event.record()
            return event
        except Exception:
            logger.warning("Failed to record XPU device event for cross-stream sync")
            return None

    @classmethod
    def get_free_memory(cls, device: torch.device | None = None) -> int:
        free, _ = torch.xpu.mem_get_info(device)
        return free

    @classmethod
    def get_device_memory(cls, device: torch.device | None = None) -> tuple[int, int]:
        free, total = torch.xpu.mem_get_info(device)
        return free, total

    @classmethod
    def get_profiler_cls(cls) -> str:
        """Return XPU-specific profiler that handles XPU events."""
        return "vllm_omni.platforms.xpu.profiler.XPUTorchProfilerWrapper"

    @classmethod
    def get_default_ir_op_priority(cls, vllm_config: VllmConfig) -> IrOpPriorityConfig:
        """Copied from upstream XPUPlatform with inductor-aware logic.

        When inductor is active (compiling) use native as the default;
        otherwise prefer vllm_c where available.
        """
        from vllm.config.compilation import CompilationMode

        cc = vllm_config.compilation_config
        using_inductor = cc.backend == "inductor" and cc.mode != CompilationMode.NONE
        default = ["native"] if using_inductor else ["vllm_c", "native"]

        return IrOpPriorityConfig.with_default(default)
