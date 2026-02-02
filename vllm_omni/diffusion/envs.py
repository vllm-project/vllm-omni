# Copyright 2024 xDiT team.
# Adapted from
# https://github.com/xdit-project/xDiT/blob/main/xfuser/envs.py
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    MASTER_ADDR: str = ""
    MASTER_PORT: int | None = None
    CUDA_HOME: str | None = None
    LOCAL_RANK: int = 0

environment_variables: dict[str, Callable[[], Any]] = {
    # ================== Runtime Env Vars ==================
    # used in distributed environment to determine the master address
    "MASTER_ADDR": lambda: os.getenv("MASTER_ADDR", ""),
    # used in distributed environment to manually set the communication port
    "MASTER_PORT": lambda: (int(os.getenv("MASTER_PORT", "0")) if "MASTER_PORT" in os.environ else None),
    # path to cudatoolkit home directory, under which should be bin, include,
    # and lib directories.
    "CUDA_HOME": lambda: os.environ.get("CUDA_HOME", None),
    # local rank of the process in the distributed setting, used to determine
    # the GPU device id
    "LOCAL_RANK": lambda: int(os.environ.get("LOCAL_RANK", "0")),
}

logger = init_logger(__name__)


class PackagesEnvChecker:
    """Singleton class for checking package availability."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        packages_info = {}
        packages_info["has_flash_attn"] = self._check_flash_attn(packages_info)
        self.packages_info = packages_info

    def _check_flash_attn(self, packages_info) -> bool:
        """Check if flash attention is available and compatible."""
        platform = current_omni_platform

        # Flash attention requires CUDA-like platforms (CUDA or ROCm)
        if not platform.is_cuda_alike():
            return False

        # Check if devices are available
        if platform.get_device_count() == 0:
            return False

        try:
            gpu_name = platform.get_device_name()
            # Turing/Tesla/T4 GPUs don't support flash attention well
            if "Turing" in gpu_name or "Tesla" in gpu_name or "T4" in gpu_name:
                return False

            # Check for any FA backend: FA3 (fa3_fwd_interface, flash_attn_interface) or FA2 (flash_attn)
            # Try FA3 from fa3-fwd PyPI package
            try:
                import fa3_fwd_interface  # noqa: F401

                return True
            except (ImportError, ModuleNotFoundError):
                pass

            # Try FA3 from flash-attention source build
            try:
                import flash_attn_interface  # noqa: F401

                return True
            except (ImportError, ModuleNotFoundError):
                pass

            # Try FA2 from flash-attn package
            from flash_attn import __version__

            if __version__ < "2.6.0":
                raise ImportError("install flash_attn >= 2.6.0")
            return True
        except (ImportError, ModuleNotFoundError):
            if not packages_info.get("has_aiter", False):
                logger.warning("No Flash Attention backend found, using pytorch SDPA implementation")
            return False

    def get_packages_info(self) -> dict:
        """Get the packages info dictionary."""
        return self.packages_info


PACKAGES_CHECKER = PackagesEnvChecker()


def __getattr__(name):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
