# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.v1.request import RequestStatus

import vllm_omni.platforms as omni_platform
from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler

from .runtime_config import _VoxCPM2RuntimeConfig


class VoxCPM2OmniARAsyncScheduler(OmniARAsyncScheduler):
    """Preserve MRV1 decode-only graph admission; MRV2 accepts mixed batches.

    MRV2 full and chunked prefills share the mixed varlen graph path, so running
    decode requests do not require deferring new waiting requests.
    """

    def _unified_decode_graph_enabled(self) -> bool:
        runtime_config = _VoxCPM2RuntimeConfig.from_vllm_config(self.vllm_config)
        return runtime_config.unified_decode_graph_available(
            use_cuda_graph=omni_platform.current_omni_platform.is_cuda()
        )

    def _should_defer_waiting_for_unified_decode_graph(self) -> bool:
        if not self._unified_decode_graph_enabled():
            return False
        if getattr(self.vllm_config.model_config, "use_v2_model_runner", False):
            # Mixed batches admit both full and chunked prefills.
            return False
        if not (self.waiting or self.kv_holding_waiting) or not self.running:
            return False

        for request in self.running:
            if getattr(request, "status", None) != RequestStatus.RUNNING or request.is_finished():
                continue
            if self._get_confirmed_num_computed_tokens(request) >= request.num_prompt_tokens:
                return True
        return False

    def _should_defer_waiting_admission(self) -> bool:
        return self._should_defer_waiting_for_unified_decode_graph()
