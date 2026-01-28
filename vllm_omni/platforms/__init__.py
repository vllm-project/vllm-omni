# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import traceback
from itertools import chain
from typing import TYPE_CHECKING

from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.torch_utils import supports_xccl

from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum
from vllm_omni.plugins import (
    OMNI_PLATFORM_PLUGINS_GROUP,
    load_omni_plugins_by_group,
)

logger = logging.getLogger(__name__)


def cuda_omni_platform_plugin() -> str | None:
    """Check if CUDA OmniPlatform should be activated."""
    is_cuda = False
    logger.debug("Checking if CUDA OmniPlatform is available.")
    try:
        from vllm.utils.import_utils import import_pynvml

        pynvml = import_pynvml()
        pynvml.nvmlInit()
        try:
            if pynvml.nvmlDeviceGetCount() > 0:
                is_cuda = True
                logger.debug("Confirmed CUDA OmniPlatform is available.")
            else:
                logger.debug("CUDA OmniPlatform is not available because no GPU is found.")
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        logger.debug("CUDA OmniPlatform is not available because: %s", str(e))

    return "vllm_omni.platforms.cuda.platform.CudaOmniPlatform" if is_cuda else None


def rocm_omni_platform_plugin() -> str | None:
    """Check if ROCm OmniPlatform should be activated."""
    is_rocm = False
    logger.debug("Checking if ROCm OmniPlatform is available.")
    try:
        import amdsmi

        amdsmi.amdsmi_init()
        try:
            if len(amdsmi.amdsmi_get_processor_handles()) > 0:
                is_rocm = True
                logger.debug("Confirmed ROCm OmniPlatform is available.")
            else:
                logger.debug("ROCm OmniPlatform is not available because no GPU is found.")
        finally:
            amdsmi.amdsmi_shut_down()
    except Exception as e:
        logger.debug("ROCm OmniPlatform is not available because: %s", str(e))

    return "vllm_omni.platforms.rocm.platform.RocmOmniPlatform" if is_rocm else None


def npu_omni_platform_plugin() -> str | None:
    """Check if NPU OmniPlatform should be activated."""
    is_npu = False
    logger.debug("Checking if NPU OmniPlatform is available.")
    try:
        import torch

        if hasattr(torch, "npu") and torch.npu.is_available():
            is_npu = True
            logger.debug("Confirmed NPU OmniPlatform is available.")
    except Exception as e:
        logger.debug("NPU OmniPlatform is not available because: %s", str(e))

    return "vllm_omni.platforms.npu.platform.NPUOmniPlatform" if is_npu else None


def xpu_omni_platform_plugin() -> str | None:
    """Check if XPU OmniPlatform should be activated."""
    is_xpu = False
    logger.debug("Checking if XPU OmniPlatform is available.")
    try:
        # installed IPEX if the machine has XPUs.
        import intel_extension_for_pytorch  # noqa: F401
        import torch

        if supports_xccl():
            dist_backend = "xccl"
        else:
            dist_backend = "ccl"
            import oneccl_bindings_for_pytorch  # noqa: F401

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            is_xpu = True
            from vllm_omni.platforms.xpu import XPUOmniPlatform

            XPUOmniPlatform.dist_backend = dist_backend
            logger.debug("Confirmed %s backend is available.", XPUOmniPlatform.dist_backend)
            logger.debug("Confirmed XPU platform is available.")
    except Exception as e:
        logger.debug("XPU omni platform is not available because: %s", str(e))

    return "vllm_omni.platforms.xpu.platform.XPUOmniPlatform" if is_xpu else None


builtin_omni_platform_plugins = {
    "cuda": cuda_omni_platform_plugin,
    "rocm": rocm_omni_platform_plugin,
    "npu": npu_omni_platform_plugin,
    "xpu": xpu_omni_platform_plugin,
}


def resolve_current_omni_platform_cls_qualname() -> str:
    """Resolve the current OmniPlatform class qualified name."""
    platform_plugins = load_omni_plugins_by_group(OMNI_PLATFORM_PLUGINS_GROUP)

    activated_plugins = []

    for name, func in chain(builtin_omni_platform_plugins.items(), platform_plugins.items()):
        try:
            assert callable(func)
            platform_cls_qualname = func()
            if platform_cls_qualname is not None:
                activated_plugins.append(name)
        except Exception:
            pass

    activated_builtin_plugins = list(set(activated_plugins) & set(builtin_omni_platform_plugins.keys()))
    activated_oot_plugins = list(set(activated_plugins) & set(platform_plugins.keys()))

    if len(activated_oot_plugins) >= 2:
        raise RuntimeError(f"Only one OmniPlatform plugin can be activated, but got: {activated_oot_plugins}")
    elif len(activated_oot_plugins) == 1:
        platform_cls_qualname = platform_plugins[activated_oot_plugins[0]]()
        logger.info("OmniPlatform plugin %s is activated", activated_oot_plugins[0])
    elif len(activated_builtin_plugins) >= 2:
        raise RuntimeError(f"Only one OmniPlatform plugin can be activated, but got: {activated_builtin_plugins}")
    elif len(activated_builtin_plugins) == 1:
        platform_cls_qualname = builtin_omni_platform_plugins[activated_builtin_plugins[0]]()
        logger.debug("Automatically detected OmniPlatform %s.", activated_builtin_plugins[0])
    else:
        platform_cls_qualname = "vllm_omni.platforms.interface.UnspecifiedOmniPlatform"
        logger.debug("No platform detected, vLLM-Omni is running on UnspecifiedOmniPlatform")

    return platform_cls_qualname


_current_omni_platform = None
_init_trace: str = ""

if TYPE_CHECKING:
    current_omni_platform: OmniPlatform


def __getattr__(name: str):
    if name == "current_omni_platform":
        # Lazy init current_omni_platform
        global _current_omni_platform
        if _current_omni_platform is None:
            platform_cls_qualname = resolve_current_omni_platform_cls_qualname()
            _current_omni_platform = resolve_obj_by_qualname(platform_cls_qualname)()
            global _init_trace
            _init_trace = "".join(traceback.format_stack())
        return _current_omni_platform
    elif name in globals():
        return globals()[name]
    else:
        raise AttributeError(f"No attribute named '{name}' exists in {__name__}.")


def __setattr__(name: str, value):  # noqa: N807
    if name == "current_omni_platform":
        global _current_omni_platform
        _current_omni_platform = value
    elif name in globals():
        globals()[name] = value
    else:
        raise AttributeError(f"No attribute named '{name}' exists in {__name__}.")


__all__ = [
    "OmniPlatform",
    "OmniPlatformEnum",
    "current_omni_platform",
    "_init_trace",
]
