# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""vllm-ascend-enhanced NPU platform for AR/generation stages.

AR/generation stages (Qwen3-Omni, TTS, etc.) run on the vllm-ascend backend.
This module is only imported when the AR backend is selected (the platform
plugin resolves it from ``VLLM_OMNI_DISABLE_VLLM_ASCEND``, which defaults to
false), so the module-level ``vllm_ascend`` import is safe: it is only
reached when the vllm-ascend backend is actually required.
"""

from functools import cache
from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger
from vllm_ascend.platform import NPUPlatform

from vllm_omni.platforms.npu.platform import NPUOmniPlatform, _vllm_ascend_available

logger = init_logger(__name__)


@cache
def _get_strict_ulysses_paged_backend() -> type:
    """Return an Ascend backend that bypasses vLLM PCP dispatch."""

    from vllm_ascend.attention.attention_v1 import (
        AscendAttentionBackend,
        AscendAttentionBackendImpl,
        AscendAttentionMetadataBuilder,
    )

    class AscendStrictUlyssesPagedBackend(AscendAttentionBackend):
        @staticmethod
        def get_impl_cls() -> type:
            return AscendAttentionBackendImpl

        @staticmethod
        def get_builder_cls() -> type:
            return AscendAttentionMetadataBuilder

    return AscendStrictUlyssesPagedBackend


class ARNPUOmniPlatform(NPUOmniPlatform, NPUPlatform):
    """vllm-ascend implementation of the NPU platform.

    Inherits the vLLM ``Platform`` entries from vllm-ascend's ``NPUPlatform``
    and the shared interface from :class:`NPUOmniPlatform` (which only holds
    methods vllm-ascend does not define, so the MRO is conflict-free). The
    class body keeps the Ascend enhancements vllm-ascend does not provide:
    custom-op registration, ACL graph wrapper, ascend forward context, ascend
    config/logging, model patches, and the torch_npu total-memory override
    (vllm-ascend deliberately raises there, but ``startup_plan.py`` calls it
    unconditionally).
    """

    def __init__(self) -> None:
        if not _vllm_ascend_available():
            raise RuntimeError(
                "ARNPUOmniPlatform requires the vllm-ascend backend, but "
                "vllm-ascend is not installed. Pure diffusion stages do NOT "
                "need it; install vllm-ascend, or set "
                "VLLM_OMNI_DISABLE_VLLM_ASCEND=true in the stage env "
                "to use the standalone torch_npu backend (DiTNPUOmniPlatform)."
            )
        # Preserve the original application order: vllm-ascend global/model
        # patches first, then the 310P worker patches.
        from vllm_ascend.utils import adapt_patch

        from vllm_omni.platforms.npu._310p import apply_patches as apply_310p_patches
        from vllm_omni.platforms.npu.models.minicpmo_4_5_code2wav import (
            apply_minicpmo_4_5_code2wav_patch,
        )
        from vllm_omni.platforms.npu.models.qwen3_tts_code2wav import (
            apply_qwen3_tts_code2wav_patch,
        )
        from vllm_omni.platforms.npu.models.qwen3_tts_tokenizer_v2 import (
            apply_qwen3_tts_tokenizer_v2_patch,
        )

        adapt_patch(is_global_patch=True)
        apply_minicpmo_4_5_code2wav_patch()
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
    def init_diffusion_worker_vllm_config(cls, vllm_config: Any) -> None:
        from vllm_ascend.ascend_config import init_ascend_config
        from vllm_ascend.utils import adapt_patch

        # Omni's custom DiffusionWorker does not pass through vLLM-Ascend's
        # NPUWorker constructor, where worker-local patches are normally
        # installed.  In particular, AscendBlockTables needs the patched
        # non-UVA buffer implementation on NPU.
        adapt_patch()
        init_ascend_config(vllm_config)

    @classmethod
    def configure_diffusion_vllm_config(cls, vllm_config: Any, od_config: Any) -> None:
        """Use the block geometry required by Ascend's native paged kernel."""
        if getattr(od_config, "diffusion_kv_mode", None) is None:
            return
        from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode

        if od_config.diffusion_kv_mode is not DiffusionKVCacheMode.PAGED_SCHEDULER:
            return
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackend

        supported_sizes = [
            size for size in AscendAttentionBackend.get_supported_kernel_block_sizes() if type(size) is int and size > 0
        ]
        if not supported_sizes:
            raise RuntimeError("Ascend paged attention did not expose an integer kernel block size")
        # vLLM's generic default is 16, while the Ascend FIA backend stores
        # cache pages as 128-token blocks. Set the Manager geometry
        # before KV specs are collected so Scheduler and Worker agree.
        vllm_config.cache_config.block_size = supported_sizes[0]

    @classmethod
    def requires_diffusion_paged_kv_prewrite(cls) -> bool:
        """Write the full K/V span once before piecewise FIA segments."""

        return True

    @classmethod
    def get_diffusion_paged_kv_attn_backend(cls, attn_backend: type, *, ulysses_degree: int) -> type:
        """Keep strict Ulysses paged FIA out of vLLM's PCP implementation."""

        del cls
        if ulysses_degree <= 1:
            return attn_backend
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackend

        if not isinstance(attn_backend, type) or not issubclass(attn_backend, AscendAttentionBackend):
            return attn_backend
        return _get_strict_ulysses_paged_backend()

    @classmethod
    def get_diffusion_kv_block_tables_cls(cls) -> type:
        from vllm_ascend.worker.v2.block_table import AscendBlockTables

        return AscendBlockTables

    @classmethod
    def build_diffusion_kv_attn_metadata(cls, **kwargs: Any) -> dict[str, Any]:
        """Build the Ascend metadata required by the native NPU backend."""
        from vllm_ascend.attention.attention_v1 import AscendAttentionState
        from vllm_ascend.worker.v2.attn_utils import build_attn_metadata

        kwargs = dict(kwargs)
        seq_lens_cpu = kwargs.pop("seq_lens_cpu")
        kwargs["seq_lens_np"] = seq_lens_cpu.detach().cpu().numpy()
        # The diffusion adapter always supplies a paged cache and the current
        # K/V write span. ChunkedPrefill is Ascend's cache-backed FIA state for
        # both multi-token updates and single-token updates in this path.
        kwargs["attn_state"] = AscendAttentionState.ChunkedPrefill
        return build_attn_metadata(**kwargs)

    @classmethod
    def init_diffusion_model_runner_runtime(cls, vllm_config: Any, od_config: Any, device: torch.device) -> None:
        super().init_diffusion_model_runner_runtime(vllm_config, od_config, device)
        from vllm_ascend.ascend_forward_context import set_mc2_mask, set_mc2_tokens_capacity

        set_mc2_tokens_capacity(vllm_config, od_config.max_num_seqs, 1)
        set_mc2_mask(vllm_config, device)

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
        cudagraph_runtime_mode: CUDAGraphMode,
        batch_descriptor: BatchDescriptor,
    ):
        from vllm_ascend.ascend_forward_context import set_ascend_forward_context

        return set_ascend_forward_context(
            attn_metadata,
            vllm_config,
            aclgraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=batch_descriptor,
        )

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        # Keep vllm-ascend's own config checks (parallel-config validation,
        # draft/decode context checks, incompatible-config fixes), then apply
        # the ascend config and logging setup.
        super().check_and_update_config(vllm_config)
        from vllm_ascend.ascend_config import init_ascend_config
        from vllm_ascend.logger import configure_ascend_file_logging, configure_ascend_logging

        init_ascend_config(vllm_config)
        configure_ascend_file_logging()
        configure_ascend_logging()

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        # NOTE: vllm-ascend deliberately leaves this as NotImplementedError to
        # avoid initializing torch_npu too early, but vLLM's engine startup
        # (vllm/v1/worker/startup_plan.py) calls it unconditionally. Keep this
        # torch_npu implementation so the AR path satisfies the call.
        device_props = torch.npu.get_device_properties(device_id)
        return device_props.total_memory
