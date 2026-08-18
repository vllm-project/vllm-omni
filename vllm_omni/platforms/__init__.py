# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import traceback
from itertools import chain
from typing import TYPE_CHECKING

from vllm.utils.import_utils import resolve_obj_by_qualname

from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum
from vllm_omni.plugins import (
    OMNI_PLATFORM_PLUGINS_GROUP,
    load_omni_plugins_by_group,
)

logger = logging.getLogger(__name__)

# Stores diagnostic information about why each platform check failed.
# Populated by platform plugin functions and consumed by
# resolve_current_omni_platform_cls_qualname() when no platform is detected.
_platform_diagnostics: str = ""

# Side-channel for built-in plugins to report why they couldn't activate.
# Built-in plugins internally catch exceptions (to keep normal startup quiet
# when hardware is absent), so the outer try/except in the resolver can't see
# their errors.  Each plugin writes a short error string here before returning
# None, and the resolver consumes the dict only when falling back to
# UnspecifiedOmniPlatform.
_plugin_error_details: dict[str, str] = {}


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
    except ImportError:
        logger.debug("CUDA OmniPlatform is not available because pynvml is not installed")
    except Exception as e:
        # pynvml imported successfully but init/query failed — the hardware
        # (or at least its driver stack) is present; record for diagnostics.
        _plugin_error_details["cuda"] = str(e)
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
    except ImportError:
        logger.debug("ROCm OmniPlatform is not available because amdsmi is not installed")
    except Exception as e:
        # amdsmi imported successfully but init/query failed.
        _plugin_error_details["rocm"] = str(e)
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
    except ImportError:
        logger.debug("NPU OmniPlatform is not available because torch is not installed")
    except Exception as e:
        _plugin_error_details["npu"] = str(e)
        logger.debug("NPU OmniPlatform is not available because: %s", str(e))

    return "vllm_omni.platforms.npu.platform.NPUOmniPlatform" if is_npu else None


def xpu_omni_platform_plugin() -> str | None:
    """Check if XPU OmniPlatform should be activated."""
    is_xpu = False
    logger.debug("Checking if XPU OmniPlatform is available.")
    try:
        import torch

        if torch.distributed.is_xccl_available():
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
    except ImportError:
        logger.debug("XPU omni platform is not available because required packages are not installed")
    except Exception as e:
        # Required packages are installed but init/query failed.
        _plugin_error_details["xpu"] = str(e)
        logger.debug("XPU omni platform is not available because: %s", str(e))

    return "vllm_omni.platforms.xpu.platform.XPUOmniPlatform" if is_xpu else None


def musa_omni_platform_plugin() -> str | None:
    """Check if MUSA OmniPlatform should be activated."""
    is_musa = False
    logger.debug("Checking if MUSA OmniPlatform is available.")
    try:
        import torchada

        if torchada.is_musa_platform():
            is_musa = True
            logger.debug("Confirmed MUSA OmniPlatform is available.")
    except ImportError:
        logger.debug("MUSA OmniPlatform is not available because torchada is not installed")
    except Exception as e:
        # torchada imported successfully but is_musa_platform() failed.
        _plugin_error_details["musa"] = str(e)
        logger.debug("MUSA OmniPlatform is not available because: %s", str(e))

    return "vllm_omni.platforms.musa.platform.MUSAOmniPlatform" if is_musa else None


builtin_omni_platform_plugins = {
    "cuda": cuda_omni_platform_plugin,
    "rocm": rocm_omni_platform_plugin,
    "npu": npu_omni_platform_plugin,
    "xpu": xpu_omni_platform_plugin,
    "musa": musa_omni_platform_plugin,
}


def resolve_current_omni_platform_cls_qualname() -> str:
    """Resolve the current OmniPlatform class qualified name."""
    global _platform_diagnostics, _plugin_error_details
    # Clear the side-channel before probing so stale errors from a previous
    # resolution (shouldn't happen in practice) can't leak in.
    _plugin_error_details = {}
    platform_plugins = load_omni_plugins_by_group(OMNI_PLATFORM_PLUGINS_GROUP)

    activated_plugins = []
    plugin_errors: dict[str, str] = {}

    for name, func in chain(builtin_omni_platform_plugins.items(), platform_plugins.items()):
        try:
            assert callable(func)
            platform_cls_qualname = func()
            if platform_cls_qualname is not None:
                activated_plugins.append(name)
        except Exception as e:
            plugin_errors[name] = str(e)

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
        # Merge side-channel errors from built-in plugins with exceptions
        # from OOT plugins (built-in plugs catch internally and don't raise).
        all_errors = {**_plugin_error_details, **plugin_errors}
        _platform_diagnostics = _build_unspecified_diagnostics(all_errors)
        logger.debug("No platform detected, vLLM-Omni is running on UnspecifiedOmniPlatform")

    return platform_cls_qualname


def _build_unspecified_diagnostics(plugin_errors: dict[str, str]) -> str:
    """Build a diagnostic message explaining why no platform was detected."""
    parts = [
        "No hardware accelerator detected, vLLM-Omni is running on UnspecifiedOmniPlatform.",
    ]
    if plugin_errors:
        parts.append("The following error(s) occurred while checking platform (see earlier debug logs for details):")
        for name, error in plugin_errors.items():
            parts.append(f"  - {name}: {error}")
    parts.append(
        "Platform-specific methods such as set_device() will raise NotImplementedError.\n"
        "Please check your device drivers and the output of "
        "'nvidia-smi' (for CUDA), 'npu-smi info' (for Ascend NPU), 'rocm-smi' (for ROCm), "
        "or your vendor's diagnostic tool."
    )
    return "\n".join(parts)


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
    "_platform_diagnostics",
]
