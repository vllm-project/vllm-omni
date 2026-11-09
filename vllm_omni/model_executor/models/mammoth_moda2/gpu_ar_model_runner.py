# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MammothModa2 AR runner: MoE specialty capture/replay lives here (#9865).

Shared ``OmniGPUModelRunner`` / ``GPUARModelRunner`` only expose thin hooks so
non-Mammoth models keep an unchanged capture path. All classify / select /
warmup detail stays under ``mammoth_moda2``.
"""

from __future__ import annotations

from vllm.config import CUDAGraphMode

from vllm_omni.model_executor.models.mammoth_moda2.moe_cudagraph import (
    annotate_moe_routing_batch_descriptor,
    select_moe_replay_batch_descriptor,
    warmup_and_capture_with_moe_routing,
)
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner


class MammothGPUARModelRunner(GPUARModelRunner):
    """AR runner used for MammothModa2 stages that may enable MoE specialty graphs."""

    def _annotate_moe_routing_batch_descriptor(
        self,
        batch_desc,
        *,
        num_tokens: int,
        force_moe_routing_key: int | None = None,
    ):
        return annotate_moe_routing_batch_descriptor(
            self,
            batch_desc,
            num_tokens=num_tokens,
            force_moe_routing_key=force_moe_routing_key,
        )

    def _select_moe_replay_batch_descriptor(self, mode, batch_desc, num_tokens: int):
        return select_moe_replay_batch_descriptor(self, mode, batch_desc, num_tokens)

    def _warmup_and_capture(
        self,
        desc,
        cudagraph_runtime_mode: CUDAGraphMode,
        profile_seq_lens: int | None = None,
        allow_microbatching: bool = False,
        num_warmups: int | None = None,
        profiler=None,
    ):
        warmup_and_capture_with_moe_routing(
            self,
            desc,
            cudagraph_runtime_mode,
            profile_seq_lens=profile_seq_lens,
            allow_microbatching=allow_microbatching,
            num_warmups=num_warmups,
            profiler=profiler,
        )
