# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""torch_npu-native NPU platform for diffusion (DiT) stages.

Used when vllm-ascend is disabled (``VLLM_OMNI_DISABLE_VLLM_ASCEND=true``).
Pure diffusion stages (e.g. Qwen-Image) run with the torch_npu backend +
mindiesd attention and never need vllm-ascend; this module stays free of any
``vllm_ascend`` import so it can be imported and unit-tested in environments
without vllm-ascend.
"""

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.platforms.npu.platform import NPUOmniPlatform

logger = init_logger(__name__)


class DiTNPUOmniPlatform(NPUOmniPlatform):
    """torch_npu implementation of the NPU platform for diffusion stages.

    Implements the vLLM ``Platform`` entries a diffusion stage needs
    (mindiesd attention, hccl distributed) with torch_npu-native calls. The
    torch_npu method bodies mirror the corresponding implementations in
    vllm-ascend's ``NPUPlatform`` (vllm_ascend/platform.py); keep them in
    sync when either side changes.
    """

    def __init__(self) -> None:
        # 310P worker patches are torch_npu-level and apply on both backends.
        from vllm_omni.platforms.npu._310p import apply_patches as apply_310p_patches

        apply_310p_patches()

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        torch.npu.set_device(device)

        # Ascend quantized weights are converted from ND to FRACTAL_NZ
        # after loading. Enable internal format so the NZ storage layout
        # is preserved for fused NPU kernels.
        torch.npu.config.allow_internal_format = True

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.npu.get_device_name(device_id)

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        device_props = torch.npu.get_device_properties(device_id)
        if not hasattr(device_props, "uuid") or device_props.uuid is None:
            raise RuntimeError(f"Device {device_id} does not have a valid UUID.")
        return device_props.uuid

    @classmethod
    def num_compute_units(cls, device_id: int = 0) -> int:
        props = torch.npu.get_device_properties(device_id)
        cube_core_num = getattr(props, "cube_core_num", None)
        if cube_core_num is not None and cube_core_num > 0:
            return int(cube_core_num)
        vector_core_num = getattr(props, "vector_core_num", None)
        if vector_core_num is not None and vector_core_num > 0:
            return int(vector_core_num)
        return 24  # safe default (24 Cube Cores)

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        torch.npu.manual_seed_all(seed)

    @classmethod
    def get_current_memory_usage(cls, device=None) -> float:
        torch.npu.reset_peak_memory_stats(device)
        return torch.npu.max_memory_allocated(device)

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        # NOTE: vllm-ascend deliberately leaves this as NotImplementedError to
        # avoid initializing torch_npu too early, but vLLM's engine startup
        # (vllm/v1/worker/startup_plan.py) calls it unconditionally. Keep this
        # torch_npu implementation so the standalone path satisfies the call.
        device_props = torch.npu.get_device_properties(device_id)
        return device_props.total_memory

    @classmethod
    def get_diffusion_kv_block_tables_cls(cls) -> type:
        raise NotImplementedError(
            "DiTNPUOmniPlatform (standalone torch_npu) does not implement the "
            "native diffusion paged-KV path. Use ARNPUOmniPlatform "
            "(vllm-ascend) for models that enable diffusion_kv, or set "
            "VLLM_OMNI_DISABLE_VLLM_ASCEND=false."
        )

    @classmethod
    def build_diffusion_kv_attn_metadata(cls, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError(
            "DiTNPUOmniPlatform (standalone torch_npu) does not implement the "
            "native diffusion paged-KV path. Use ARNPUOmniPlatform "
            "(vllm-ascend) for models that enable diffusion_kv, or set "
            "VLLM_OMNI_DISABLE_VLLM_ASCEND=false."
        )
