# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2024 xDiT team.
# Adapted from
# https://github.com/xdit-project/xDiT/blob/main/xfuser/envs.py
import os
import tempfile
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    MASTER_ADDR: str = ""
    MASTER_PORT: int | None = None
    CUDA_HOME: str | None = None
    LOCAL_RANK: int = 0
    VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS: str | None = None
    VLLM_OMNI_SEEDVR2_MAX_FRAMES: int = 0
    VLLM_OMNI_SEEDVR2_SHARDED_FRAME_PIXELS: int = 0
    VLLM_OMNI_SEEDVR2_SHARDED_CLIP_PIXELS: int | None = None
    VLLM_OMNI_SEEDVR2_LONG_MAX_UPLOAD_BYTES: int = 0
    VLLM_OMNI_SEEDVR2_LONG_MAX_WINDOW: int = 0
    VLLM_OMNI_SEEDVR2_LONG_JOB_TTL_SECONDS: int = 0
    VLLM_OMNI_SEEDVR2_LONG_OUTPUT_DIR: str = ""


def _positive_int(name: str, default: int | None) -> int | None:
    """Read an operator-tuned limit, rejecting values that would disable it."""
    value = os.environ.get(name)
    if value is None:
        return default
    if not value.strip().isdigit() or not int(value):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


environment_variables: dict[str, Callable[[], Any]] = {
    # ================== Runtime Env Vars ==================
    # used in distributed environment to determine the master address
    "MASTER_ADDR": lambda: os.getenv("MASTER_ADDR", ""),
    # used in distributed environment to manually set the communication port
    "MASTER_PORT": lambda: int(os.getenv("MASTER_PORT", "0")) if "MASTER_PORT" in os.environ else None,
    # path to cudatoolkit home directory, under which should be bin, include,
    # and lib directories.
    "CUDA_HOME": lambda: os.environ.get("CUDA_HOME", None),
    # local rank of the process in the distributed setting, used to determine
    # the GPU device id
    "LOCAL_RANK": lambda: int(os.environ.get("LOCAL_RANK", "0")),
    # Minimum rotary-table span (B*S positions) at which consumers of the shared
    # fused_qk_norm_rope op take the fused path instead of their eager chain;
    # "0" = always fuse; unset = each consumer's own measured default. Raw
    # string or None; validated by fused_qk_norm_rope_min_tokens().
    "VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS": lambda: os.environ.get("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", None),
    # ================== SeedVR2 Restoration Env Vars ==================
    # Admission budgets for the sharded serving profile. The defaults are
    # calibrated for the smallest qualified device, so larger accelerators raise
    # them here; setting them beyond what the device holds surfaces as a CUDA
    # OOM during the forward pass. See docs/models/seedvr2.md.
    "VLLM_OMNI_SEEDVR2_MAX_FRAMES": lambda: _positive_int("VLLM_OMNI_SEEDVR2_MAX_FRAMES", 257),
    "VLLM_OMNI_SEEDVR2_SHARDED_FRAME_PIXELS": lambda: _positive_int(
        "VLLM_OMNI_SEEDVR2_SHARDED_FRAME_PIXELS", 2560 * 1472
    ),
    # Unset, the clip budget scales with device memory from the calibrated
    # 32 GB value; see seedvr2.video.sharded_budget.
    "VLLM_OMNI_SEEDVR2_SHARDED_CLIP_PIXELS": lambda: _positive_int("VLLM_OMNI_SEEDVR2_SHARDED_CLIP_PIXELS", None),
    # Bounds for the long-video restoration route.
    "VLLM_OMNI_SEEDVR2_LONG_MAX_UPLOAD_BYTES": lambda: _positive_int(
        "VLLM_OMNI_SEEDVR2_LONG_MAX_UPLOAD_BYTES", 8 * 1024**3
    ),
    # Longest model window. The clip budget usually binds first; this caps it
    # at 30 latent frames, the DiT's largest temporal attention window.
    "VLLM_OMNI_SEEDVR2_LONG_MAX_WINDOW": lambda: _positive_int("VLLM_OMNI_SEEDVR2_LONG_MAX_WINDOW", 121),
    "VLLM_OMNI_SEEDVR2_LONG_JOB_TTL_SECONDS": lambda: _positive_int("VLLM_OMNI_SEEDVR2_LONG_JOB_TTL_SECONDS", 3600),
    "VLLM_OMNI_SEEDVR2_LONG_OUTPUT_DIR": lambda: os.environ.get(
        "VLLM_OMNI_SEEDVR2_LONG_OUTPUT_DIR", tempfile.gettempdir()
    ),
}


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
        packages_info["has_flash_attn"] = self._check_flash_attn()
        self.packages_info = packages_info

    def _check_flash_attn(self) -> bool:
        """Check if flash attention is available and compatible."""
        platform = current_omni_platform

        if platform.get_device_count() == 0:
            return False

        return platform.has_flash_attn_package()

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
