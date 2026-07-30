# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapter routing Cosmos3's per-CFG-branch UND text K/V through session state.

Cosmos3's cross-step cache is the UND (reasoner) text K/V, currently kept as
instance fields on ``Cosmos3VFMTransformer`` (``cached_kv``).
``Cosmos3StateAdapter`` moves its durable ownership into the shared
``SessionStateManager`` (RFC #4480), behind the
``enable_session_state_manager`` opt-in flag. The load-to-transformer handoff
still uses the transformer's fields, so this is session-state groundwork rather
than synchronization for overlapping forwards.

Each (layer, CFG branch) pair is a model-specific ``Cosmos3UNDKV`` state object.
The shared package deliberately does not define a dense attention-KV object:
DreamZero's attention KV belongs to the paged AR-Diffusion engine, whereas this
is Cosmos3's fixed, encode-once conditioning KV. Keeping the concrete object in
the model adapter preserves that ownership boundary while retaining the shared
lifecycle and byte accounting.

``freqs_gen`` (M-RoPE cos/sin for the GEN pathway) is also session-scoped. It
is constant for one request, so retaining it in ``SessionState.attrs`` avoids
recomputing both UND and GEN RoPE frequencies on every denoise step.
"""

from __future__ import annotations

from typing import Protocol

import torch

from vllm_omni.experimental.world_models.session_state.accounting import (
    SeenStorages,
    device_bytes,
)
from vllm_omni.experimental.world_models.session_state.base import StateObject
from vllm_omni.experimental.world_models.session_state.manager import (
    SessionStateManager,
)

KVPair = tuple[torch.Tensor, torch.Tensor]


class _Cosmos3KVTransformer(Protocol):
    cached_kv: list[KVPair] | None
    cached_freqs_gen: tuple[torch.Tensor, torch.Tensor] | None


class Cosmos3UNDKV(StateObject[KVPair, KVPair]):
    """One layer of fixed, encode-once Cosmos3 UND conditioning K/V."""

    def __init__(self) -> None:
        self._kv: KVPair | None = None

    def allocate(self) -> None:
        self._kv = None

    def commit(self, payload: KVPair | None = None) -> None:
        value = self._staged if payload is None else payload
        if value is None:
            raise RuntimeError("Cosmos3UNDKV.commit requires K/V payload")
        self._kv = value
        self.discard()

    def view(self, *, include_staged: bool = True) -> KVPair:
        if include_staged and self._staged is not None:
            return self._staged
        if self._kv is None:
            raise RuntimeError("Cosmos3UNDKV is not resident")
        return self._kv

    def reset(self) -> None:
        self._kv = None
        self.discard()

    def nbytes_by_device(self, seen: SeenStorages) -> dict[str, int]:
        return {} if self._kv is None else device_bytes(self._kv, seen)

    @property
    def resident(self) -> bool:
        return self._kv is not None


def _layer_key(layer_index: int, is_negative: bool) -> str:
    """Session key for one (layer, CFG branch) UND K/V object (cf. DreamZero ``_xattn_key``)."""
    return f"und_kv_neg/{layer_index}" if is_negative else f"und_kv/{layer_index}"


def _freqs_key(is_negative: bool) -> str:
    return "und_freqs_gen_neg" if is_negative else "und_freqs_gen"


class Cosmos3StateAdapter:
    """Session-keyed view over Cosmos3's per-branch UND text K/V.

    A fresh adapter is built per forward; the durable state lives in the manager
    (mirroring ``DreamZeroStateAdapter``).
    """

    def __init__(self, session_id: str | None, manager: SessionStateManager) -> None:
        self._session_id = session_id
        # Pin the session: the manager may evict it from its lookup table under
        # LRU pressure, but an adapter holding a reference keeps its state alive.
        self._session = manager.get_or_create_session(session_id)

    @property
    def _num_layers(self) -> int | None:
        value = self._session.attrs.get("_num_layers")
        return value if isinstance(value, int) else None

    def is_branch_initialized(self, is_negative: bool) -> bool:
        """Whether this CFG branch's UND K/V has been stored (encode-once)."""
        if self._num_layers is None:
            return False
        obj = self._session.get(_layer_key(0, is_negative))
        return obj is not None and obj.resident

    def get_branch_kv(self, is_negative: bool) -> list[KVPair] | None:
        """Return the branch's ``cached_kv`` as ``list[(K, V)]`` by layer, or ``None``."""
        if not self.is_branch_initialized(is_negative):
            return None
        num_layers = self._num_layers
        assert num_layers is not None
        out: list[KVPair] = []
        for i in range(num_layers):
            obj = self._session.get(_layer_key(i, is_negative))
            if not isinstance(obj, Cosmos3UNDKV) or not obj.resident:
                raise RuntimeError("Cosmos3 UND K/V is partially initialized.")
            out.append(obj.view())
        return out

    def set_branch_kv(self, is_negative: bool, cached_kv: list[KVPair]) -> None:
        """Store this branch's freshly computed ``cached_kv`` once."""
        self._session.attrs["_num_layers"] = len(cached_kv)
        for i, (k, v) in enumerate(cached_kv):
            obj = self._session.get(_layer_key(i, is_negative))
            if not isinstance(obj, Cosmos3UNDKV):
                obj = Cosmos3UNDKV()
                obj.allocate()
                self._session.put(_layer_key(i, is_negative), obj)
            obj.commit((k, v))

    def get_branch_freqs_gen(self, is_negative: bool) -> tuple[torch.Tensor, torch.Tensor] | None:
        value = self._session.attrs.get(_freqs_key(is_negative))
        if isinstance(value, tuple) and len(value) == 2 and all(isinstance(freq, torch.Tensor) for freq in value):
            return value
        return None

    def load_into_transformer(self, transformer: _Cosmos3KVTransformer, is_negative: bool) -> None:
        """Assign this branch's UND K/V and GEN RoPE frequencies onto the transformer."""
        transformer.cached_kv = self.get_branch_kv(is_negative)
        transformer.cached_freqs_gen = self.get_branch_freqs_gen(is_negative)

    def capture_from_transformer(self, transformer: _Cosmos3KVTransformer, is_negative: bool) -> None:
        """Capture the freshly computed ``cached_kv`` into the session if not already stored."""
        if not self.is_branch_initialized(is_negative):
            cached_kv = transformer.cached_kv
            if cached_kv is None:
                raise RuntimeError("Cosmos3 transformer did not produce UND K/V")
            freqs_gen = transformer.cached_freqs_gen
            if freqs_gen is None:
                raise RuntimeError("Cosmos3 transformer did not produce GEN RoPE frequencies")
            self.set_branch_kv(is_negative, cached_kv)
            self._session.attrs[_freqs_key(is_negative)] = freqs_gen

    def reset(self) -> None:
        """Clear all UND K/V objects + metadata for this session (replaces ``reset_cache()``)."""
        self._session.reset()
