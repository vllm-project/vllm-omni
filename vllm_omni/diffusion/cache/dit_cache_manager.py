# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-local cache pool lifecycle management for stepwise diffusion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.worker.utils import CacheBackendSlot, DiffusionRequestState

logger = init_logger(__name__)


class DiTCacheStateDriverBase(ABC):
    """Backend-specific adapter used by ``DiTCacheManager``."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the cache backend name associated with this driver."""

    @abstractmethod
    def create_empty_slot(self) -> CacheBackendSlot:
        """Create an empty request slot owned by this backend."""

    @abstractmethod
    def install_slot(self, slot: CacheBackendSlot) -> None:
        """Install slot state onto the live pipeline/backend objects."""

    @abstractmethod
    def initialize_fresh_slot(self, slot: CacheBackendSlot, num_inference_steps: int) -> None:
        """Initialize a just-created slot for a new request."""

    @abstractmethod
    def is_slot_compatible(self, slot: CacheBackendSlot, num_inference_steps: int) -> bool:
        """Return whether an existing slot can resume under the requested shape."""

    @abstractmethod
    def deactivate_slot(self, slot: CacheBackendSlot | None) -> None:
        """Capture any live mutable state back into the slot and clear active pointers."""

    @abstractmethod
    def clear_slot(self, slot: CacheBackendSlot) -> None:
        """Release tensors/resources owned by the slot."""

    @abstractmethod
    def estimate_slot_bytes(self, slot: CacheBackendSlot) -> int:
        """Estimate resident bytes currently owned by the slot."""


class DiTCacheManager:
    """Runner-facing lifecycle manager for request-local cache slots.

    ``activate`` / ``deactivate`` accept either a single
    ``DiffusionRequestState`` or a list.  When a list with more than one
    element is passed, the driver's batch-mode API
    (``install_batch_slots`` / ``deactivate_batch_slots``) is used so the
    backend can keep per-request cache state during one batched forward pass.
    """

    def __init__(self, driver: DiTCacheStateDriverBase):
        self.driver = driver
        # Single-request tracking (used when activate receives one state).
        self._active_req_id: str | None = None
        self._active_slot: CacheBackendSlot | None = None
        # Batch tracking (used when activate receives >1 states).
        self._batch_active: bool = False

    # ── Public API ──

    @property
    def supports_batch_activation(self) -> bool:
        """Return whether this backend can install multiple live slots at once."""
        return callable(getattr(self.driver, "install_batch_slots", None)) and callable(
            getattr(self.driver, "deactivate_batch_slots", None)
        )

    def activate(
        self,
        state: DiffusionRequestState | list[DiffusionRequestState],
    ) -> bool | list[bool]:
        """Install or restore cache slot(s).

        Args:
            state: A single request state **or** a list of states.

        Returns:
            ``True``/``False`` per request — ``True`` when an existing
            compatible slot was resumed.
        """
        if isinstance(state, list):
            if len(state) == 1:
                return [self._activate_single(state[0])]
            if not self.supports_batch_activation:
                raise ValueError(
                    f"Cache backend '{self.driver.backend_name}' does not support batched slot activation."
                )
            return self._activate_batch(state)
        return self._activate_single(state)

    def deactivate(
        self,
        state: DiffusionRequestState | list[DiffusionRequestState] | None = None,
    ) -> None:
        """Deactivate slot(s) after a denoise step finishes."""
        if self._batch_active:
            self._deactivate_batch(state if isinstance(state, list) else None)
        else:
            self._deactivate_single(state if not isinstance(state, list) else None)

    def free(self, state: DiffusionRequestState) -> None:
        """Release any resident cache state owned by ``state``."""
        if state.cache_slot is None:
            return

        if self._active_req_id == state.request_id:
            self._deactivate_single(state)

        self.driver.clear_slot(state.cache_slot)
        state.cache_slot = None

    # ── Single-request internals ──

    def _activate_single(self, state: DiffusionRequestState) -> bool:
        num_inference_steps = self._get_num_inference_steps(state)
        if self._active_req_id is not None and self._active_req_id != state.request_id:
            self._deactivate_single()

        slot = state.cache_slot
        restored_existing = True
        if slot is None or slot.backend_name != self.driver.backend_name:
            if slot is not None:
                self.driver.clear_slot(slot)
            slot = self.driver.create_empty_slot()
            state.cache_slot = slot
            restored_existing = False
        elif not self.driver.is_slot_compatible(slot, num_inference_steps):
            self.driver.clear_slot(slot)
            slot = self.driver.create_empty_slot()
            state.cache_slot = slot
            restored_existing = False

        try:
            self.driver.install_slot(slot)
            if not restored_existing:
                self.driver.initialize_fresh_slot(slot, num_inference_steps)
                slot.metadata["num_inference_steps"] = num_inference_steps
            slot.resident_bytes = self.driver.estimate_slot_bytes(slot)
        except Exception:
            self.driver.deactivate_slot(slot)
            raise

        self._active_req_id = state.request_id
        self._active_slot = slot
        return restored_existing

    def _deactivate_single(self, state: DiffusionRequestState | None = None) -> None:
        if self._active_slot is None:
            return
        if state is not None and self._active_req_id != state.request_id:
            return

        self.driver.deactivate_slot(self._active_slot)
        self._active_slot.resident_bytes = self.driver.estimate_slot_bytes(self._active_slot)
        self._active_req_id = None
        self._active_slot = None

    # ── Batch internals ──

    def _activate_batch(self, states: list[DiffusionRequestState]) -> list[bool]:
        # Clean up any leftover single-request activation.
        if self._active_req_id is not None:
            self._deactivate_single()

        restored_flags: list[bool] = []
        installed_slots: list[CacheBackendSlot] = []
        batch_install_started = False
        try:
            for state in states:
                num_inference_steps = self._get_num_inference_steps(state)
                slot = state.cache_slot
                restored = True

                if slot is None or slot.backend_name != self.driver.backend_name:
                    if slot is not None:
                        self.driver.clear_slot(slot)
                    slot = self.driver.create_empty_slot()
                    state.cache_slot = slot
                    restored = False
                elif not self.driver.is_slot_compatible(slot, num_inference_steps):
                    self.driver.clear_slot(slot)
                    slot = self.driver.create_empty_slot()
                    state.cache_slot = slot
                    restored = False

                if not restored:
                    # Fresh slot: must install + init *before* entering batch mode
                    # because initialize_fresh_slot calls install_slot + force_refresh
                    # which operate in single-request mode.
                    installed_slots.append(slot)
                    self.driver.install_slot(slot)
                    self.driver.initialize_fresh_slot(slot, num_inference_steps)
                    slot.metadata["num_inference_steps"] = num_inference_steps
                    slot.resident_bytes = self.driver.estimate_slot_bytes(slot)
                else:
                    slot.resident_bytes = self.driver.estimate_slot_bytes(slot)
                restored_flags.append(restored)

            # Switch into batch mode (sets _batch_contexts on context managers).
            batch_install_started = True
            self.driver.install_batch_slots(states)
        except Exception:
            if batch_install_started:
                try:
                    self.driver.deactivate_batch_slots()
                except Exception:
                    logger.exception(
                        "Failed to deactivate batch cache slots for backend %s "
                        "after activation failure; cache context may still be "
                        "installed on the live pipeline.",
                        self.driver.backend_name,
                    )
            for slot in reversed(installed_slots):
                try:
                    self.driver.deactivate_slot(slot)
                except Exception:
                    logger.exception(
                        "Failed to deactivate cache slot for backend %s after activation failure.",
                        self.driver.backend_name,
                    )
            self._batch_active = False
            raise
        self._batch_active = True
        return restored_flags

    def _deactivate_batch(self, states: list[DiffusionRequestState] | None = None) -> None:
        self.driver.deactivate_batch_slots()
        if states is not None:
            for state in states:
                if state.cache_slot is not None:
                    state.cache_slot.resident_bytes = self.driver.estimate_slot_bytes(state.cache_slot)
        self._batch_active = False

    # ── Helpers ──

    @staticmethod
    def _get_num_inference_steps(state: DiffusionRequestState) -> int:
        sampling = state.sampling
        timesteps = getattr(sampling, "timesteps", None)
        if timesteps is not None:
            return DiTCacheManager._sequence_length(timesteps)

        sigmas = getattr(sampling, "sigmas", None)
        if sigmas is not None:
            return len(sigmas)

        num_inference_steps = int(getattr(sampling, "num_inference_steps", 0) or 0)
        if num_inference_steps <= 0:
            raise ValueError(f"Request {state.request_id} has invalid num_inference_steps={num_inference_steps}")
        return num_inference_steps

    @staticmethod
    def _sequence_length(values: Any) -> int:
        ndim = getattr(values, "ndim", None)
        if ndim == 0:
            return 1

        shape = getattr(values, "shape", None)
        if shape is not None:
            return int(shape[0])

        return len(values)
