# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Auxiliary-output delivery for Ascend's legacy model runner."""

from types import SimpleNamespace
from typing import Any

import numpy as np


def _install_ascend_capture_sources(model: Any) -> None:
    # AscendRoutedExperts invokes the router's private selection method, before
    # logical-to-physical remapping and shared-expert expansion. It bypasses
    # BaseRouter.select_experts(), where upstream normally invokes capture_fn.
    for module in model.modules():
        if not type(module).__module__.startswith("vllm_ascend.") or not hasattr(module, "layer_id"):
            continue
        router = getattr(module, "router", None)
        select_experts = getattr(router, "_select_experts", None)
        if not callable(select_experts) or getattr(module, "_omni_npu_capture_source", False):
            continue
        module.capture_fn = None

        def select_and_capture(*args, _select=select_experts, _source=module, **kwargs):
            weights, logical_ids = _select(*args, **kwargs)
            if _source.capture_fn is not None:
                _source.capture_fn(logical_ids)
            return weights, logical_ids

        router._select_experts = select_and_capture
        # layer_id + capture_fn implement upstream RoutedExpertsCaptureSource.
        module._omni_npu_capture_source = True


class NPUAuxOutputMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_output_connector = None

    def _init_omni_aux_output(self, kv_cache_config) -> None:
        if not self.vllm_config.aux_output_config.enabled:
            return
        from vllm.distributed.aux_output_connector.worker import get_aux_output_connector

        self._close_omni_aux_output()
        _install_ascend_capture_sources(self.model)
        self.aux_output_connector = get_aux_output_connector(self.model, self.vllm_config, kv_cache_config)

    def _begin_omni_aux_output_step(self, scheduler_output) -> None:
        if self.aux_output_connector is not None:
            # Apply terminal/hash-only metadata even on no-forward steps.
            self.aux_output_connector.begin_step(scheduler_output.aux_output_connector_metadata)

    def _prepare_omni_aux_output(self):
        if self.aux_output_connector is None:
            return None
        num_reqs = len(self.input_batch.req_ids)
        # Bookkeeping and the next async batch mutate these legacy buffers.
        batch = SimpleNamespace(
            req_ids=list(self.input_batch.req_ids),
            query_start_loc_np=self.query_start_loc.np[: num_reqs + 1].copy(),
            num_computed_tokens_np=self.input_batch.num_computed_tokens_cpu[:num_reqs].copy(),
        )
        return self.aux_output_connector.prepare_output(batch)

    def _finish_omni_aux_output(self, pending, scheduler_output, sampled_token_ids=None, invalid_req_indices=()):
        if pending is None:
            return None
        num_reqs = len(pending.request_ids)
        num_sampled = np.zeros(num_reqs, dtype=np.int32)
        num_rejected = np.zeros(num_reqs, dtype=np.int32)
        if sampled_token_ids is not None:
            # Instrumentation opts into a synchronized copy. The normal NPU
            # async sampling path remains unchanged when capture is disabled.
            sampled = sampled_token_ids.detach().to("cpu").numpy()
            num_sampled[:] = (sampled >= 0).sum(axis=-1)
            for i, request_id in enumerate(pending.request_ids):
                draft_count = len(scheduler_output.scheduled_spec_decode_tokens.get(request_id, ()))
                if draft_count:
                    num_rejected[i] = max(0, draft_count + 1 - int(num_sampled[i]))
            # Partial prefills execute valid routing rows but emit no token.
            num_sampled[list(invalid_req_indices)] = 0
        pending.enqueue_cpu_copy(num_sampled=num_sampled, num_rejected=num_rejected)
        self._sync_device()
        return pending.process_output()

    def _close_omni_aux_output(self) -> None:
        connector = self.aux_output_connector
        self.aux_output_connector = None
        if connector is not None:
            try:
                connector.close()
            finally:
                for module in self.model.modules():
                    if getattr(module, "_omni_npu_capture_source", False):
                        module.capture_fn = None
