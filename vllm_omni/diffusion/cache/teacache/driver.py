# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache pool driver for TeaCache."""

from __future__ import annotations

from typing import Any

import torch

from vllm_omni.diffusion.cache.dit_cache_manager import DiTCacheStateDriverBase
from vllm_omni.diffusion.cache.teacache.hook import TeaCacheHook
from vllm_omni.diffusion.worker.utils import CacheBackendSlot


class TeaCacheStateDriver(DiTCacheStateDriverBase):
    """Manage per-request TeaCache hook state for stepwise serving."""

    def __init__(self, pipeline: Any):
        self._hook = self._get_hook(pipeline)
        if self._hook is None:
            raise ValueError("TeaCache is enabled but no TeaCacheHook was found on pipeline.transformer.")

    @property
    def backend_name(self) -> str:
        return "tea_cache"

    def create_empty_slot(self) -> CacheBackendSlot:
        return CacheBackendSlot(
            backend_name=self.backend_name,
            payload={
                "states": {},
                "forward_cnt": 0,
            },
        )

    def install_slot(self, slot: CacheBackendSlot) -> None:
        payload = self._get_payload(slot)
        self._hook.state_manager._states = payload["states"]
        self._hook._forward_cnt = payload["forward_cnt"]

    def initialize_fresh_slot(self, slot: CacheBackendSlot, num_inference_steps: int) -> None:
        payload = self._get_payload(slot)
        payload["states"] = {}
        payload["forward_cnt"] = 0
        slot.metadata["num_inference_steps"] = num_inference_steps
        self.install_slot(slot)

    def is_slot_compatible(self, slot: CacheBackendSlot, num_inference_steps: int) -> bool:
        del slot, num_inference_steps
        return True

    def deactivate_slot(self, slot: CacheBackendSlot | None) -> None:
        if slot is None:
            return

        payload = self._get_payload(slot)
        payload["states"] = self._hook.state_manager._states
        payload["forward_cnt"] = self._hook._forward_cnt
        self._hook.state_manager._states = {}
        self._hook.state_manager.set_context("teacache")
        self._hook._forward_cnt = 0

    def clear_slot(self, slot: CacheBackendSlot) -> None:
        payload = self._get_payload(slot)
        for state in payload["states"].values():
            for name, value in vars(state).items():
                if isinstance(value, torch.Tensor):
                    setattr(state, name, None)
        payload["states"].clear()
        payload["forward_cnt"] = 0
        slot.metadata.clear()
        slot.resident_bytes = 0

    def estimate_slot_bytes(self, slot: CacheBackendSlot) -> int:
        total_bytes = 0
        seen_tensor_ids: set[int] = set()
        payload = self._get_payload(slot)
        for state in payload["states"].values():
            for value in vars(state).values():
                if not isinstance(value, torch.Tensor):
                    continue
                value_id = id(value)
                if value_id in seen_tensor_ids:
                    continue
                seen_tensor_ids.add(value_id)
                total_bytes += value.nelement() * value.element_size()
        return total_bytes

    @staticmethod
    def _get_hook(pipeline: Any) -> TeaCacheHook | None:
        transformer = getattr(pipeline, "transformer", None)
        registry = getattr(transformer, "_hook_registry", None)
        if registry is None:
            return None
        hook = registry.get_hook(TeaCacheHook._HOOK_NAME)
        if hook is None or not isinstance(hook, TeaCacheHook):
            return None
        return hook

    @staticmethod
    def _get_payload(slot: CacheBackendSlot) -> dict[str, Any]:
        payload = slot.payload
        if not isinstance(payload, dict):
            raise TypeError(f"Invalid tea_cache slot payload: {type(payload)}")
        return payload
