# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen2.5-Omni image encoder contract for the upstream CUDA graph manager."""

import dataclasses
from typing import Any

from vllm.config import CUDAGraphMode
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsEncoderCudaGraph
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphConfig

logger = init_logger(__name__)

_GRAPH_ATTN_BACKENDS = {
    AttentionBackendEnum.FLASH_ATTN,
    AttentionBackendEnum.ROCM_AITER_FA,
    AttentionBackendEnum.TRITON_ATTN,
}

# The thinker's vision tower is Qwen2.5-VL's, so buffer layout, window
# metadata, capture and replay are that model's protocol methods.
_VL = Qwen2_5_VLForConditionalGeneration


class Qwen2_5OmniVisionEncoderCudaGraphMixin:
    """Image-only protocol methods; video and audio retain embed_multimodal.

    The capability attribute is set on eligible model instances after tower
    construction. It is deliberately absent from this mixin's class: the
    runtime protocol checks attribute presence, not the value of a false flag.
    """

    # Read by Qwen2.5-VL's protocol methods. The capability is only set when
    # video pruning is off (see _enable_image_encoder_cudagraph), matching
    # Qwen2.5-VL, which drops every graph modality under EVS pruning.
    is_multimodal_pruning_enabled = False

    # The eager tower is launch-bound, so a group that needs several replays
    # or per-image eager calls runs as one eager call instead (A800: three
    # 169-token images take +59% as three replays, +0% as one eager call); see
    # SingleReplayEncoderCudaGraphManager.
    encoder_cudagraph_single_replay = True

    def _enable_image_encoder_cudagraph(self) -> None:
        # Only the first pipeline rank runs the encoder. Any FP8 encoder
        # attention keeps the eager tower: dynamic scaling advances a host-side
        # amax slot per call, which a graph would freeze, and static scales
        # (mm_encoder_fp8_scale_path) share that attention path. The runner
        # captures no graph when cudagraph_mode is NONE (or still unresolved,
        # which CompilationConfig also treats as NONE), as an eager thinker
        # implies; the eager case gets its own message because it is the
        # deploy default. The pipeline-group query comes last so a thinker
        # built without a vision tower or distributed state never reaches it.
        mm_config = self.multimodal_config
        backend = getattr(self.visual, "attn_backend", None)
        if mm_config.enable_mm_embeds:
            reason = "enable_mm_embeds is set"
        elif mm_config.mm_encoder_attn_dtype is not None:
            reason = f"FP8 encoder attention is set (mm_encoder_attn_dtype={mm_config.mm_encoder_attn_dtype})"
        elif mm_config.is_multimodal_pruning_enabled():
            reason = "video pruning is enabled"
        elif self.vllm_config.model_config.enforce_eager:
            reason = "the thinker runs with enforce_eager; set enforce_eager: false on stage 0"
        elif self.vllm_config.compilation_config.cudagraph_mode in (None, CUDAGraphMode.NONE):
            reason = "compilation_config.cudagraph_mode is NONE"
        elif mm_config.get_limit_per_prompt("image") <= 0:
            reason = "images are disabled"
        elif backend not in _GRAPH_ATTN_BACKENDS:
            reason = f"the vision attention backend is {backend}"
        elif not get_pp_group().is_first_rank:
            return
        else:
            self.supports_encoder_cudagraph = True
            return
        if self.vllm_config.compilation_config.cudagraph_mm_encoder:
            logger.warning("Qwen2.5-Omni image encoder CUDA graphs are off: %s.", reason)

    def get_encoder_cudagraph_config(self) -> EncoderCudaGraphConfig:
        return dataclasses.replace(_VL.get_encoder_cudagraph_config(self), modalities=["image"])

    def get_input_modality(self, mm_kwargs: dict[str, Any]) -> str:
        if "image_grid_thw" not in mm_kwargs:
            raise ValueError("Qwen2.5-Omni encoder graphs only take images")
        return "image"

    def get_max_frames_per_video(self) -> int:
        return 1

    def get_encoder_cudagraph_budget_range(self, vllm_config) -> tuple[int, int]:
        # The eager tower costs about the same up to ~500 tokens (launch-bound),
        # so replay saves most at small sizes: on A800, -46% to -72% up to 256
        # tokens, but only 5-8% for 300-506 tokens padded to 512, and +21% for
        # 784 tokens padded to 1024. Each budget also adds graph memory on the
        # thinker's GPU. Auto-inferred budgets therefore stop at 256 tokens;
        # encoder_cudagraph_token_budgets can ask for more.
        maximum = min(vllm_config.scheduler_config.max_num_batched_tokens, vllm_config.model_config.max_model_len, 256)
        return min(64, maximum), maximum

    _get_pixel_values_by_modality = _VL._get_pixel_values_by_modality
    _get_grid_thw_by_modality = _VL._get_grid_thw_by_modality
    get_encoder_cudagraph_item_specs = _VL.get_encoder_cudagraph_item_specs
    select_encoder_cudagraph_items = _VL.select_encoder_cudagraph_items
    prepare_encoder_cudagraph_capture_inputs = _VL.prepare_encoder_cudagraph_capture_inputs
    prepare_encoder_cudagraph_replay_buffers = _VL.prepare_encoder_cudagraph_replay_buffers
    encoder_cudagraph_forward = _VL.encoder_cudagraph_forward
    encoder_eager_forward = _VL.encoder_eager_forward
    postprocess_encoder_output = SupportsEncoderCudaGraph.postprocess_encoder_output
