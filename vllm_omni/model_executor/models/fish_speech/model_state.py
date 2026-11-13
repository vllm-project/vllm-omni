# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fish Speech Slow AR request state on Model Runner V2.

The Fast AR runs through the generic current-step MTP of ``OmniModelState``.
This state only adds what the V1 runner gives the Fish kvcache decode path:
capture workspaces prewarmed at load and CPU sequence-length upper bounds on
the attention metadata.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.utils import AttentionGroup

from vllm_omni.attention.fish_kvcache_backend import (
    attach_fish_kvcache_seq_lens_upper_bound_mrv2,
    prewarm_fish_kvcache_attn_capture_workspaces,
)
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState


class FishSpeechModelState(OmniModelState):
    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        capture_sizes = vllm_config.compilation_config.cudagraph_capture_sizes
        if capture_sizes:
            prewarm_fish_kvcache_attn_capture_workspaces(
                model_config=vllm_config.model_config,
                device=device,
                dtype=self.dtype,
                capture_sizes=capture_sizes,
            )

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    ) -> dict[str, Any]:
        attn_metadata = super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
            ubatch_idx=ubatch_idx,
            model_specific_attn_metadata=model_specific_attn_metadata,
        )
        attach_fish_kvcache_seq_lens_upper_bound_mrv2(
            model_config=self.model_config,
            attn_metadata=attn_metadata,
            seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
            for_cudagraph_capture=for_capture,
        )
        return attn_metadata
