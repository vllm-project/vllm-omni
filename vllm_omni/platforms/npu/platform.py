# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm_ascend.platform import NPUPlatform

from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

logger = init_logger(__name__)

_DIFFUSION_PACKED_MODULES_MAPPING = {
    "HunyuanImage3Pipeline": {
        "experts": ["experts.0.gate_up_proj", "experts.0.down_proj"],
    },
}


class NPUOmniPlatform(OmniPlatform, NPUPlatform):
    """NPU/Ascend implementation of OmniPlatform.

    Inherits all NPU-specific implementations from vllm-ascend's NPUPlatform,
    and adds Omni-specific interfaces from OmniPlatform.
    """

    _omni_enum = OmniPlatformEnum.NPU
    dist_backend: str = "hccl"

    # conv2d convolution operator in the code2wav module of Qwen3-TTS not being able to run on Aclnn
    def __init__(self) -> None:
        from vllm_ascend.utils import adapt_patch

        from vllm_omni.platforms.npu._310p import apply_patches as apply_310p_patches
        from vllm_omni.platforms.npu.models.minimax_h3 import apply_minimax_h3_qwen3vl_patch
        from vllm_omni.platforms.npu.models.qwen3_tts_code2wav import (
            apply_qwen3_tts_code2wav_patch,
        )
        from vllm_omni.platforms.npu.models.qwen3_tts_tokenizer_v2 import (
            apply_qwen3_tts_tokenizer_v2_patch,
        )

        adapt_patch(is_global_patch=True)
        apply_minimax_h3_qwen3vl_patch()
        apply_qwen3_tts_code2wav_patch()
        apply_qwen3_tts_tokenizer_v2_patch()
        apply_310p_patches()

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        super().set_device(device)

        # Register vllm_ascend custom ops (torch.ops._C_ascend.*).
        from vllm_ascend.utils import enable_custom_op

        enable_custom_op()

        # Ascend quantized weights are converted from ND to FRACTAL_NZ
        # after loading. Enable internal format so the NZ storage layout
        # is preserved for fused NPU kernels.
        torch.npu.config.allow_internal_format = True

    @classmethod
    def get_omni_ar_worker_cls(cls) -> str:
        return "vllm_omni.platforms.npu.worker.npu_ar_worker.NPUARWorker"

    @classmethod
    def get_omni_generation_worker_cls(cls) -> str:
        return "vllm_omni.platforms.npu.worker.npu_generation_worker.NPUGenerationWorker"

    @classmethod
    def init_diffusion_worker_vllm_config(cls, vllm_config: Any) -> None:
        from vllm_ascend.ascend_config import init_ascend_config

        init_ascend_config(vllm_config)

    @classmethod
    def init_diffusion_model_runner_runtime(cls, vllm_config: Any, od_config: Any, device: torch.device) -> None:
        from vllm_ascend.ascend_forward_context import set_mc2_mask, set_mc2_tokens_capacity

        from vllm_omni.platforms.npu.models.minimax_h3 import (
            apply_minimax_h3_qwen3vl_swiglu_patch,
        )

        # The patch imports the MiniMax encoder, which depends on
        # current_omni_platform. Run it only after platform construction has
        # completed, but before the diffusion pipeline is loaded.
        apply_minimax_h3_qwen3vl_swiglu_patch()
        set_mc2_tokens_capacity(vllm_config, od_config.max_num_seqs, 1)
        set_mc2_mask(vllm_config, device)

    @classmethod
    def get_default_stage_config_path(cls) -> str:
        return "vllm_omni/deploy"

    @classmethod
    def prepare_diffusion_op_runtime(cls, op_name: str, **kwargs: Any) -> None:
        if op_name != "fused_moe":
            return

        from vllm_omni.platforms.npu.layers.fused_moe import prepare_fused_moe_runtime

        prepare_fused_moe_runtime()

    @classmethod
    def register_additional_diffusion_fused_moe_hooks(cls, moe_runner: Any) -> None:
        from vllm_omni.platforms.npu.layers.fused_moe import fused_moe_forward_context_pre_hook

        moe_runner.register_forward_pre_hook(
            fused_moe_forward_context_pre_hook,
            with_kwargs=True,
        )

    @classmethod
    def reset_diffusion_fused_moe_forward_context(cls) -> None:
        from vllm_omni.platforms.npu.layers.fused_moe import reset_fused_moe_forward_context

        reset_fused_moe_forward_context()

    @classmethod
    def get_diffusion_packed_modules_mapping(
        cls,
        model_class: type[nn.Module],
    ) -> dict[str, list[str]] | None:
        return _DIFFUSION_PACKED_MODULES_MAPPING.get(model_class.__name__, None)

    @classmethod
    def get_diffusion_attn_backend_cls(
        cls,
        selected_backend: str | None,
        head_size: int,
        allow_trtllm_default: bool = True,
    ) -> str:
        # NPU has no TRTLLM backend; accepted for signature parity, ignored.
        from importlib.util import find_spec

        if selected_backend is not None:
            backend_upper = selected_backend.upper()
            cls.validate_diffusion_attn_backend(backend_upper)
            if backend_upper in ("FLASH_ATTN_HUB", "FLASH_ATTN_3_HUB"):
                logger.warning(
                    "HuggingFace kernels-backed FlashAttention is "
                    "not supported on NPU. Falling back to local "
                    "FLASH_ATTN."
                )
                backend_upper = "FLASH_ATTN"

            if backend_upper == "FLASH_ATTN" and find_spec("mindiesd"):
                # The NPU FLASH_ATTN backend imports mindiesd lazily at first
                # forward, but CANN snapshots the custom-op registry at the
                # first custom-op regInfo lookup in the process (e.g. a
                # vllm-ascend custom op during model load/warmup). Import
                # mindiesd here so its env.py prepends the mindiesd vendor
                # dirs (aie_ascendc etc.) to ASCEND_CUSTOM_OPP_PATH before
                # that snapshot; otherwise aclnnLaserAttention /
                # FusedAttentionScore fail with EZ1001 "does not support
                # opType" for the rest of the process.
                import mindiesd  # noqa: F401

            backend = DiffusionAttentionBackendEnum[backend_upper]
            logger.debug("Using diffusion attention backend '%s'", backend_upper)
            return backend.get_path()

        # Try FLASH_ATTN if mindiesd is available, otherwise fall back to SDPA
        if find_spec("mindiesd"):
            # Configure ASCEND_CUSTOM_OPP_PATH for mindiesd custom ops upon import
            import mindiesd  # noqa: F401

            logger.debug("Defaulting to diffusion attention backend FLASH_ATTN")
            return DiffusionAttentionBackendEnum.FLASH_ATTN.get_path()

        logger.debug("Falling back to diffusion attention backend SDPA")
        return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()

    @classmethod
    def supports_torch_inductor(cls) -> bool:
        return False

    @classmethod
    def get_torch_device(cls, local_rank: int | None = None) -> torch.device:
        if local_rank is None:
            return torch.device("npu")
        return torch.device("npu", local_rank)

    @classmethod
    def get_device_count(cls) -> int:
        return torch.npu.device_count()

    @classmethod
    def get_device_version(cls) -> str | None:
        return None

    @classmethod
    def synchronize(cls) -> None:
        torch.npu.synchronize()

    @classmethod
    def record_device_event(cls) -> torch.Event | None:
        """Record a NPU event on the default stream to mark tensor readiness.

        On NPU/Ascend with HCCL, distributed communication may use internal
        streams not visible to the default stream. Synchronize the default
        stream first so that HCCL results are written back before we record
        the event, ensuring d2h_stream.wait_event() captures the complete
        output data.
        """
        try:
            torch.npu.current_stream().synchronize()
            event = torch.npu.Event()
            event.record()
            return event
        except Exception:
            logger.warning("Failed to record NPU event for cross-stream sync")
            return None

    @classmethod
    def get_free_memory(cls, device: torch.device | None = None) -> int:
        free, _ = torch.npu.mem_get_info(device)
        return free

    @classmethod
    def get_device_memory(cls, device: torch.device | None = None) -> tuple[int, int]:
        free, total = torch.npu.mem_get_info(device)
        return free, total

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.npu.get_device_properties(device_id)
        return device_props.total_memory

    @classmethod
    def create_autocast_context(cls, *, device_type, dtype, enabled=True):
        if device_type != "npu":
            return super().create_autocast_context(
                device_type=device_type,
                dtype=dtype,
                enabled=enabled,
            )
        if not enabled:
            return nullcontext()

        # NPU-specific fallback
        try:
            return torch.npu.amp.autocast(dtype=dtype)
        except (RuntimeError, TypeError, ValueError) as exc:
            logger.warning("autocast unavailable for device_type=%s dtype=%s: %s", device_type, dtype, exc)
        return nullcontext()

    @classmethod
    def get_profiler_cls(cls) -> str:
        return "vllm_omni.platforms.npu.profiler.NPUTorchProfilerWrapper"

    @classmethod
    def get_graph_wrapper_cls(cls) -> type:
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

        return ACLGraphWrapper

    @classmethod
    def set_forward_context(
        cls,
        attn_metadata,
        vllm_config,
        *,
        cudagraph_runtime_mode,
        batch_descriptor,
    ):
        from vllm_ascend.ascend_forward_context import set_ascend_forward_context

        return set_ascend_forward_context(
            attn_metadata,
            vllm_config,
            aclgraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=batch_descriptor,
        )
