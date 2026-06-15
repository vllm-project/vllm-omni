# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache pool driver for cache-dit."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from cache_dit.caching.cache_contexts.cache_manager import CachedContextManager

from vllm_omni.diffusion.cache.cache_dit_batch import (
    clear_batch_contexts,
    set_batch_contexts,
)
from vllm_omni.diffusion.cache.dit_cache_manager import DiTCacheStateDriverBase
from vllm_omni.diffusion.worker.utils import CacheBackendSlot, DiffusionRequestState


@dataclass(frozen=True)
class _CacheDiTHandle:
    context_manager: CachedContextManager
    context_names: tuple[str, ...]
    templates: dict[str, Any]


class CacheDiTStateDriver(DiTCacheStateDriverBase):
    """Manage per-request cache-dit context dictionaries for stepwise serving."""

    def __init__(self, backend: Any, pipeline: Any):
        self._backend = backend
        self._pipeline = pipeline
        self._handles = self._discover_handles(pipeline)
        if not self._handles:
            raise ValueError("cache-dit is enabled but no persistent cache context handles were found.")

    @property
    def backend_name(self) -> str:
        return "cache_dit"

    def create_empty_slot(self) -> CacheBackendSlot:
        payload = tuple(self._build_fresh_contexts(handle) for handle in self._handles)
        return CacheBackendSlot(backend_name=self.backend_name, payload=payload)

    def install_slot(self, slot: CacheBackendSlot) -> None:
        payload = self._get_payload(slot)
        for handle, contexts in zip(self._handles, payload):
            handle.context_manager._cached_context_manager = contexts
            handle.context_manager._current_context = None

    def initialize_fresh_slot(self, slot: CacheBackendSlot, num_inference_steps: int) -> None:
        self.install_slot(slot)
        self._backend.force_refresh(self._pipeline, num_inference_steps, verbose=False)
        slot.metadata["num_inference_steps"] = num_inference_steps

    def is_slot_compatible(self, slot: CacheBackendSlot, num_inference_steps: int) -> bool:
        return slot.metadata.get("num_inference_steps") == num_inference_steps

    def deactivate_slot(self, slot: CacheBackendSlot | None) -> None:
        payload = self._get_payload(slot) if slot is not None else None
        for handle_idx, handle in enumerate(self._handles):
            handle.context_manager._current_context = None
            if payload is not None and handle.context_manager._cached_context_manager is payload[handle_idx]:
                handle.context_manager._cached_context_manager = self._build_fresh_contexts(handle)

    def clear_slot(self, slot: CacheBackendSlot) -> None:
        for handle, contexts in zip(self._handles, self._get_payload(slot)):
            if handle.context_manager._cached_context_manager is contexts:
                handle.context_manager._current_context = None
                handle.context_manager._cached_context_manager = self._build_fresh_contexts(handle)
            for context in contexts.values():
                context.clear_buffers()
            contexts.clear()
        slot.metadata.clear()
        slot.resident_bytes = 0

    def estimate_slot_bytes(self, slot: CacheBackendSlot) -> int:
        total_bytes = 0
        seen_tensor_ids: set[int] = set()
        for contexts in self._get_payload(slot):
            for context in contexts.values():
                for value in context.buffers.values():
                    if not isinstance(value, torch.Tensor):
                        continue
                    value_id = id(value)
                    if value_id in seen_tensor_ids:
                        continue
                    seen_tensor_ids.add(value_id)
                    total_bytes += value.nelement() * value.element_size()
        return total_bytes

    # ── Batch-mode support ──

    def install_batch_slots(self, states: list[DiffusionRequestState]) -> None:
        """Install multiple request slots in batch mode.

        For each context-manager handle, collects the per-request context dicts
        and calls ``set_batch_contexts`` with the corresponding row counts.
        """
        row_counts = [int(state.latents.shape[0]) for state in states]
        for handle_idx, handle in enumerate(self._handles):
            batch_context_list: list[dict[str, Any]] = []
            for state in states:
                payload = self._get_payload(state.cache_slot)
                contexts = payload[handle_idx]
                batch_context_list.append(contexts)
            set_batch_contexts(handle.context_manager, batch_context_list, row_counts)

    def deactivate_batch_slots(self) -> None:
        """Exit batch mode on all context-manager handles."""
        for handle in self._handles:
            clear_batch_contexts(handle.context_manager)

    # ── Handle discovery ──

    def _discover_handles(self, pipeline: Any) -> tuple[_CacheDiTHandle, ...]:
        grouped: dict[int, dict[str, Any]] = {}
        for module in self._candidate_modules(pipeline):
            context_manager = getattr(module, "_context_manager", None)
            context_names = tuple(getattr(module, "_context_names", ()) or ())
            if context_manager is None or not context_names:
                continue

            manager_id = id(context_manager)
            handle_data = grouped.setdefault(
                manager_id,
                {
                    "context_manager": context_manager,
                    "context_names": [],
                    "templates": {},
                },
            )
            for name in context_names:
                if name in handle_data["templates"]:
                    continue
                handle_data["context_names"].append(name)
                handle_data["templates"][name] = context_manager.get_context(name)

        handles = []
        for handle_data in grouped.values():
            handles.append(
                _CacheDiTHandle(
                    context_manager=handle_data["context_manager"],
                    context_names=tuple(handle_data["context_names"]),
                    templates=handle_data["templates"],
                )
            )
        return tuple(handles)

    @staticmethod
    def _candidate_modules(pipeline: Any) -> list[Any]:
        modules = []
        seen_ids: set[int] = set()

        def add(module: Any) -> None:
            if module is None:
                return
            module_id = id(module)
            if module_id in seen_ids:
                return
            seen_ids.add(module_id)
            modules.append(module)

        add(pipeline)
        add(getattr(pipeline, "transformer", None))
        add(getattr(pipeline, "transformer_2", None))
        add(getattr(pipeline, "bagel", None))

        language_model = getattr(pipeline, "language_model", None)
        add(language_model)
        add(getattr(language_model, "model", None))
        return modules

    @staticmethod
    def _clone_fresh_context(source: Any) -> Any:
        init_args = copy.deepcopy(getattr(source, "_init_args", ()))
        init_kwargs = copy.deepcopy(getattr(source, "_init_kwargs", {}))
        fresh_context = type(source)(*init_args, **init_kwargs)
        fresh_context._init_args = init_args
        fresh_context._init_kwargs = init_kwargs
        return fresh_context

    def _build_fresh_contexts(self, handle: _CacheDiTHandle) -> dict[str, Any]:
        return {name: self._clone_fresh_context(handle.templates[name]) for name in handle.context_names}

    @staticmethod
    def _get_payload(slot: CacheBackendSlot) -> tuple[dict[str, Any], ...]:
        payload = slot.payload
        if not isinstance(payload, tuple):
            raise TypeError(f"Invalid cache-dit slot payload: {type(payload)}")
        return payload
