# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Entry-scoped BDE KV ownership for DreamZero async lookahead.

The original BDE state is a linear session window. W8 needs a narrower primitive:
name the real/simulated observation entries a forward may read, lease those
entries while the forward is running, and release simulated entries after the
matching real observation arrives. Blocks are not copied between entries.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm_omni.experimental.bde.kv_cache.adapter import BDERequestAdapter
from vllm_omni.experimental.bde.kv_cache.gather import (
    build_window_slots,
    pool_gather_window,
    pool_write_chunk,
)
from vllm_omni.experimental.bde.kv_cache.slot_mapping import resident_block_ids


@dataclass(frozen=True, order=True)
class BDECacheEntryKey:
    session_id: str
    session_epoch: int
    observation_index: int
    sim_depth: int


def bde_cache_entry_key(value: BDECacheEntryKey | Mapping[str, Any]) -> BDECacheEntryKey:
    if isinstance(value, BDECacheEntryKey):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"BDE cache entry key must be a mapping, got {type(value).__name__}")
    return BDECacheEntryKey(
        session_id=_required_str(value, "session_id"),
        session_epoch=_required_int(value, "session_epoch"),
        observation_index=_required_int(value, "observation_index"),
        sim_depth=_required_int(value, "sim_depth"),
    )


def bde_cache_entry_key_dict(key: BDECacheEntryKey) -> dict[str, Any]:
    return {
        "session_id": key.session_id,
        "session_epoch": key.session_epoch,
        "observation_index": key.observation_index,
        "sim_depth": key.sim_depth,
    }


@dataclass
class BDECacheEntry:
    key: BDECacheEntryKey
    pos: BDERequestAdapter
    neg: BDERequestAdapter
    lease_count: int = 0
    owner_released: bool = False
    pending_slots: dict[bool, list[torch.Tensor]] = field(default_factory=lambda: {False: [], True: []})

    def adapter(self, is_negative: bool) -> BDERequestAdapter:
        return self.neg if is_negative else self.pos


class BDECacheEntryLease:
    def __init__(self, store: BDECacheEntryStore, keys: tuple[BDECacheEntryKey, ...]) -> None:
        self._store = store
        self.keys = keys
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._store.release_lease(self.keys)

    def __enter__(self) -> BDECacheEntryLease:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def get_kv_caches(self, *, is_negative: bool) -> list[torch.Tensor]:
        if self._closed:
            raise RuntimeError("BDE cache entry lease is closed")
        return self._store.get_leased_prefix_kv_caches(self, is_negative=is_negative)


class BDECacheEntryStore:
    """Owns W8 cache entries and prefix views for one BDE KV cache.

    Each entry owns two adapters, one positive and one negative CFG branch. A
    prefix view is a tuple of entry keys. The read path gathers the blocks owned
    by exactly those entries, in key order, and trims to the BDE attention window.
    """

    def __init__(self, kv_cache: Any) -> None:
        self.kv_cache = kv_cache
        self._entries: dict[BDECacheEntryKey, BDECacheEntry] = {}

    def create_entry(self, key: BDECacheEntryKey) -> BDECacheEntry:
        existing = self._entries.get(key)
        if existing is not None:
            if existing.owner_released:
                raise RuntimeError(f"BDE cache entry was released: {key}")
            return existing

        prefix = self._request_prefix(key)
        entry = BDECacheEntry(
            key=key,
            pos=self.kv_cache.begin_request(prefix),
            neg=self.kv_cache.begin_request(f"{prefix}__neg"),
        )
        self._entries[key] = entry
        return entry

    def has_entry(self, key: BDECacheEntryKey) -> bool:
        return key in self._entries and not self._entries[key].owner_released

    def lease_entries(self, keys: Iterable[BDECacheEntryKey]) -> BDECacheEntryLease:
        key_tuple = tuple(keys)
        if not key_tuple:
            raise ValueError("BDE prefix lease requires at least one entry")
        for key in key_tuple:
            entry = self._require_entry(key)
            entry.lease_count += 1
        return BDECacheEntryLease(self, key_tuple)

    def release_lease(self, keys: Iterable[BDECacheEntryKey]) -> None:
        for key in keys:
            entry = self._entries.get(key)
            if entry is None:
                continue
            entry.lease_count = max(0, entry.lease_count - 1)
            self._free_if_unowned(entry)

    def release_owner(self, key: BDECacheEntryKey) -> None:
        entry = self._entries.get(key)
        if entry is None:
            return
        entry.owner_released = True
        self._free_if_unowned(entry)

    def close(self) -> None:
        for entry in list(self._entries.values()):
            entry.owner_released = True
            entry.lease_count = 0
            self._free_if_unowned(entry)

    def update_entry_kv(
        self,
        key: BDECacheEntryKey,
        *,
        layer_idx: int,
        updated_kv: torch.Tensor,
        is_negative: bool,
        seq_len: int,
    ) -> None:
        entry = self.create_entry(key)
        adapter = entry.adapter(is_negative)
        chunk_size = self.kv_cache.spec.chunk_size
        if layer_idx == 0:
            if seq_len % chunk_size != 0:
                raise ValueError(f"seq_len must be divisible by chunk_size: {seq_len} % {chunk_size}")
            slots = []
            for _ in range(seq_len // chunk_size):
                self.kv_cache.allocate_chunk(adapter)
                slots.append(self.kv_cache.chunk_write_slots(adapter))
                adapter.on_chunk_committed()
            entry.pending_slots[is_negative] = slots

        slots = entry.pending_slots[is_negative]
        if not slots:
            return
        n_tokens = len(slots) * chunk_size
        k_all = _drop_batch(updated_kv[0])[-n_tokens:]
        v_all = _drop_batch(updated_kv[1])[-n_tokens:]
        k_pool = self.kv_cache._k_pools[layer_idx]
        v_pool = self.kv_cache._v_pools[layer_idx]
        for chunk_idx, slot_mapping in enumerate(slots):
            start = chunk_idx * chunk_size
            end = start + chunk_size
            pool_write_chunk(
                k_pool,
                v_pool,
                k_all[start:end].unsqueeze(0).to(k_pool.dtype),
                v_all[start:end].unsqueeze(0).to(v_pool.dtype),
                slot_mapping,
            )

    def get_prefix_kv_caches(
        self,
        keys: Iterable[BDECacheEntryKey],
        *,
        is_negative: bool,
    ) -> list[torch.Tensor]:
        block_ids = self.prefix_block_ids(keys, is_negative=is_negative)
        return self._gather_block_ids(block_ids)

    def get_leased_prefix_kv_caches(
        self,
        lease: BDECacheEntryLease,
        *,
        is_negative: bool,
    ) -> list[torch.Tensor]:
        block_ids = self._prefix_block_ids(
            lease.keys,
            is_negative=is_negative,
            allow_owner_released=True,
        )
        return self._gather_block_ids(block_ids)

    def get_leased_prefix_with_owned_suffix_kv_caches(
        self,
        lease: BDECacheEntryLease,
        suffix_keys: Iterable[BDECacheEntryKey],
        *,
        is_negative: bool,
    ) -> list[torch.Tensor]:
        block_ids = self._prefix_block_ids(
            lease.keys,
            is_negative=is_negative,
            allow_owner_released=True,
        )
        block_ids.extend(
            self._prefix_block_ids(
                suffix_keys,
                is_negative=is_negative,
                allow_owner_released=False,
            )
        )
        return self._gather_block_ids(block_ids)

    def _gather_block_ids(self, block_ids: list[int]) -> list[torch.Tensor]:
        if not block_ids:
            raise ValueError("BDE prefix view has no resident blocks")
        slots = build_window_slots(block_ids, self.kv_cache.block_size, self.kv_cache._k_pools[0].device)
        return [
            pool_gather_window(
                self.kv_cache._k_pools[layer_idx],
                self.kv_cache._v_pools[layer_idx],
                block_ids,
                self.kv_cache.block_size,
                self.kv_cache.spec.sliding_window,
                slots=slots,
            )
            for layer_idx in range(self.kv_cache.num_layers)
        ]

    def prefix_block_ids(
        self,
        keys: Iterable[BDECacheEntryKey],
        *,
        is_negative: bool,
    ) -> list[int]:
        return self._prefix_block_ids(keys, is_negative=is_negative, allow_owner_released=False)

    def _prefix_block_ids(
        self,
        keys: Iterable[BDECacheEntryKey],
        *,
        is_negative: bool,
        allow_owner_released: bool,
    ) -> list[int]:
        block_ids: list[int] = []
        for key in keys:
            entry = self._require_entry(key, allow_owner_released=allow_owner_released)
            adapter = entry.adapter(is_negative)
            block_ids.extend(self.kv_cache.window_block_ids(adapter))
        return resident_block_ids(block_ids, self.kv_cache.null_block_id)

    def _require_entry(
        self,
        key: BDECacheEntryKey,
        *,
        allow_owner_released: bool = False,
    ) -> BDECacheEntry:
        entry = self._entries.get(key)
        if entry is None or (entry.owner_released and not allow_owner_released):
            raise KeyError(f"BDE cache entry is not available: {key}")
        return entry

    def _free_if_unowned(self, entry: BDECacheEntry) -> None:
        if not entry.owner_released or entry.lease_count > 0:
            return
        self.kv_cache.end_request(entry.pos)
        self.kv_cache.end_request(entry.neg)
        self._entries.pop(entry.key, None)

    @staticmethod
    def _request_prefix(key: BDECacheEntryKey) -> str:
        return (
            f"bde_entry__{key.session_id}__e{key.session_epoch}"
            f"__o{key.observation_index}__sim{key.sim_depth}"
        )


def _drop_batch(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 4 and t.shape[0] == 1:
        return t[0]
    return t


def _required_str(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"BDE cache entry key field {key!r} must be a non-empty string")
    return item


def _required_int(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int):
        raise ValueError(f"BDE cache entry key field {key!r} must be an integer")
    return item
