# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.v1.request import RequestStatus

from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler


class VoxCPM2OmniARAsyncScheduler(OmniARAsyncScheduler):
    """VoxCPM2 scheduler variant for full unified decode graph serving.

    VoxCPM2's full unified decode graph only applies to pure decode batches.
    When a decode-ready request is already running, this scheduler defers new
    waiting admissions for the current tick so the decode batch can stay on the
    VoxCPM2 graph path. This is a model-local serving policy, not a generic AR
    scheduler rule.
    """

    def _unified_decode_graph_enabled(self) -> bool:
        hf_config = getattr(self.vllm_config.model_config, "hf_config", None)
        runtime_config = getattr(hf_config, "voxcpm2_runtime_config", None)
        if isinstance(runtime_config, dict):
            return bool(runtime_config.get("enable_unified_decode_graph", False))
        return bool(getattr(runtime_config, "enable_unified_decode_graph", False))

    def _should_defer_waiting_for_unified_decode_graph(self) -> bool:
        if not self._unified_decode_graph_enabled():
            return False
        if not self.waiting or not self.running:
            return False

        for request in self.running:
            if getattr(request, "status", None) != RequestStatus.RUNNING or request.is_finished():
                continue
            if self._get_confirmed_num_computed_tokens(request) >= request.num_prompt_tokens:
                return True
        return False

    def _should_defer_waiting_admission(self) -> bool:
        return self._should_defer_waiting_for_unified_decode_graph()
