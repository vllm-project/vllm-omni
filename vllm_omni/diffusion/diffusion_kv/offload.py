# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Host store, Scheduler index and transport for diffusion KV page offload.

RFC #9192 phase 1 scope: only *stable prefix pages* left behind by completed
requests are offloaded.  Active-request pages and mutable image pages stay on
the device.

Ownership split
---------------
* The **Worker** owns the KV bytes.  It keeps a host-resident
  :class:`KVPageOffloadStore` with the only copy of an evicted page.
* The **Scheduler** owns identity.  It keeps a :class:`DiffusionKVOffloadIndex`
  that maps a page's group-scoped block hash to its refcount, so it can decide
  whether a prefix is available off-device and how many pages may be dropped.

The two sides never exchange payloads.  They exchange *names* only, through the
:class:`KVPageOffloadTransport` boundary:

* ``save_pages``   - Worker copies the named device pages into host storage.
* ``restore_pages`` - Worker copies host-resident pages back into device blocks.
* ``discard_pages`` - Worker drops host-resident payloads.

Because a page's identity is the group-scoped block hash, an offloaded page and
a GPU-resident cached page are interchangeable from the Scheduler's point of
view: the same hash chain decides prefix reuse either way.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id

logger = init_logger(__name__)


def make_page_key(block_hash: BlockHash, group_index: int) -> bytes:
    """Return the group-scoped key of one KV page.

    The encoding matches the native ``BlockPool`` prefix-cache key so an
    offloaded page is indistinguishable from a cached device page.
    """

    return bytes(make_block_hash_with_group_id(block_hash, group_index))


@dataclass(frozen=True)
class OffloadedKVPage:
    """Identity of a single offloaded KV page.

    Attributes:
        key: Group-scoped block hash. This is the page's identity.
        group_index: KV cache group the page belongs to.
        block_hash: Per-group block hash of the page content.
        source_block_id: Physical device block the payload was captured from.
            The block may be reused after the offload, so a restore must not
            assume the page still lives there.
        num_tokens: Number of tokens covered by the page.
    """

    key: bytes
    group_index: int
    block_hash: bytes
    source_block_id: int
    num_tokens: int

    @classmethod
    def create(
        cls,
        *,
        group_index: int,
        block_hash: BlockHash,
        source_block_id: int,
        num_tokens: int,
    ) -> OffloadedKVPage:
        if group_index < 0:
            raise ValueError(f"group_index must be non-negative, got {group_index}")
        if source_block_id < 0:
            raise ValueError(f"source_block_id must be non-negative, got {source_block_id}")
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}")
        return cls(
            key=make_page_key(block_hash, group_index),
            group_index=group_index,
            block_hash=bytes(block_hash),
            source_block_id=source_block_id,
            num_tokens=num_tokens,
        )


@dataclass(frozen=True)
class KVPageRestoreRequest:
    """One page restore: copy a host-resident page into a device block.

    The target block is chosen by the Scheduler at restore time; it is never
    the page's ``source_block_id`` because that device block may already have
    been reused for unrelated content.
    """

    key: bytes
    target_block_id: int

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("key must be non-empty")
        if self.target_block_id < 0:
            raise ValueError(f"target_block_id must be non-negative, got {self.target_block_id}")


class KVPageOffloadStore:
    """Host-resident store for offloaded diffusion KV pages.

    The store owns the only copy of a page payload once the page has left the
    device.  Payloads are keyed by the group-scoped block hash, so two rows
    that share a prefix page share a single host copy.

    The store never evicts on its own: the Scheduler knows which pages are
    still referenced and drives eviction explicitly through
    ``discard``.  A Worker-side store therefore never disagrees with the
    Scheduler index.
    """

    def __init__(self, *, pin_memory: bool = False) -> None:
        self._pin_memory = pin_memory
        self._payloads: dict[bytes, torch.Tensor] = {}
        self._pages: dict[bytes, OffloadedKVPage] = {}
        self._bytes_used = 0

    @property
    def pin_memory(self) -> bool:
        return self._pin_memory

    @property
    def bytes_used(self) -> int:
        """Bytes currently held by the store."""

        return self._bytes_used

    def __len__(self) -> int:
        return len(self._pages)

    def __bool__(self) -> bool:
        # An empty store is still a usable store; without this the object
        # would be falsy and idioms such as ``store or KVPageOffloadStore()``
        # would silently drop it.
        return True

    def __contains__(self, key: bytes) -> bool:
        return key in self._payloads

    def pages(self) -> tuple[OffloadedKVPage, ...]:
        """Return the tracked pages in insertion order."""

        return tuple(self._pages.values())

    def page(self, key: bytes) -> OffloadedKVPage | None:
        """Return the page tracked under ``key``, if any."""

        return self._pages.get(key)

    def save(self, page: OffloadedKVPage, payload: torch.Tensor) -> int:
        """Store one page payload and return its size in bytes.

        The payload is copied into host memory, so the caller may release the
        device tensor immediately.  Re-saving a known page replaces its
        payload and keeps a single copy.
        """

        if not isinstance(payload, torch.Tensor):
            raise TypeError(f"payload must be a torch.Tensor, got {type(payload)!r}")
        host = payload.detach().to(device="cpu", copy=True)
        if not host.is_contiguous():
            host = host.contiguous()
        if self._pin_memory and host.device.type == "cpu":
            # Pinned memory requires a CUDA context; keep the store usable on
            # CPU-only hosts instead of failing the offload.
            try:
                host = host.pin_memory()
            except RuntimeError as exc:
                logger.warning("Could not pin offloaded KV page memory: %s", exc)
        size = host.numel() * host.element_size()
        previous = self._payloads.get(page.key)
        if previous is not None:
            self._bytes_used -= previous.numel() * previous.element_size()
        self._payloads[page.key] = host
        self._pages[page.key] = page
        self._bytes_used += size
        return size

    def restore(self, key: bytes) -> torch.Tensor:
        """Return the host payload of one page.

        The returned tensor is the store's own buffer and must be treated as
        read-only.
        """

        payload = self._payloads.get(key)
        if payload is None:
            raise KeyError(f"No offloaded KV page for key {key!r}")
        return payload

    def discard(self, key: bytes) -> bool:
        """Drop one page payload. Returns whether the page was present."""

        payload = self._payloads.pop(key, None)
        self._pages.pop(key, None)
        if payload is None:
            return False
        self._bytes_used -= payload.numel() * payload.element_size()
        return True


class DiffusionKVOffloadIndex:
    """Scheduler-side index of offloaded KV pages.

    The index tracks identity and refcounts only; payloads live in the
    Worker-side store.  A page is *tracked* as soon as it is offloaded and is
    owned by the store until it is dropped.  References are taken by requests
    that use the page (as a published owner or as a restore source) and
    released when they are done.
    """

    def __init__(self) -> None:
        self._pages: dict[bytes, OffloadedKVPage] = {}
        self._refs: dict[bytes, int] = {}
        self._bytes: dict[bytes, int] = {}

    def __len__(self) -> int:
        return len(self._pages)

    def __contains__(self, key: bytes) -> bool:
        return key in self._pages

    @property
    def bytes_used(self) -> int:
        """Bytes reported by the Worker for the tracked pages."""

        return sum(self._bytes.values())

    def page(self, key: bytes) -> OffloadedKVPage | None:
        return self._pages.get(key)

    def pages(self) -> tuple[OffloadedKVPage, ...]:
        return tuple(self._pages.values())

    def refs(self, key: bytes) -> int:
        """Return the live reference count of one page."""

        if key not in self._refs:
            raise KeyError(f"No offloaded KV page for key {key!r}")
        return self._refs[key]

    def track(self, page: OffloadedKVPage, num_bytes: int = 0) -> None:
        """Register an offloaded page.

        Tracking is idempotent: re-tracking a known page keeps its identity
        and reference count, and only refreshes its reported size.
        """

        if num_bytes < 0:
            raise ValueError(f"num_bytes must be non-negative, got {num_bytes}")
        known = self._pages.get(page.key)
        if known is not None:
            if known.block_hash != page.block_hash or known.group_index != page.group_index:
                raise ValueError(f"Conflicting offloaded KV page identity for key {page.key!r}")
            self._bytes[page.key] = num_bytes
            return
        self._pages[page.key] = page
        self._refs[page.key] = 0
        self._bytes[page.key] = num_bytes

    def acquire(self, key: bytes) -> None:
        """Take one reference on an offloaded page."""

        if key not in self._refs:
            raise KeyError(f"No offloaded KV page for key {key!r}")
        self._refs[key] += 1

    def release(self, key: bytes) -> None:
        """Drop one reference on an offloaded page.

        The page stays tracked when its last reference goes: it is still
        owned by the host store and can be restored again later.
        """

        if key not in self._refs:
            raise KeyError(f"No offloaded KV page for key {key!r}")
        refs = self._refs[key]
        if refs <= 0:
            raise RuntimeError(f"Offloaded KV page {key!r} has no live reference to release")
        self._refs[key] = refs - 1

    def remove(self, key: bytes) -> OffloadedKVPage:
        """Drop a page from the index entirely.

        Only pages without live references may be removed.
        """

        page = self._pages.get(key)
        if page is None:
            raise KeyError(f"No offloaded KV page for key {key!r}")
        if self._refs[key] > 0:
            raise RuntimeError(f"Offloaded KV page {key!r} still has live references")
        del self._pages[key]
        del self._refs[key]
        self._bytes.pop(key, None)
        return page

    def find_prefix(self, block_hashes: Sequence[BlockHash], group_index: int) -> tuple[OffloadedKVPage, ...]:
        """Return the longest available off-device prefix of one hash chain.

        The chain is contiguous from its root: the first missing hash ends the
        prefix, because a later hash cannot be reused without its ancestors.
        """

        found: list[OffloadedKVPage] = []
        for block_hash in block_hashes:
            key = make_page_key(block_hash, group_index)
            page = self._pages.get(key)
            if page is None:
                break
            found.append(page)
        return tuple(found)


class KVPageOffloadTransport(Protocol):
    """Byte movement between the Scheduler and the Worker-side host store."""

    def save_pages(self, pages: Sequence[OffloadedKVPage]) -> Mapping[bytes, int]:
        """Copy the named device pages into host storage.

        Returns the size in bytes of every saved page, keyed by page key.
        Implementations either save every page or raise; a failed save must
        not leave the Scheduler index claiming an available page.
        """
        ...

    def restore_pages(self, requests: Sequence[KVPageRestoreRequest]) -> None:
        """Copy host-resident pages back into the named device blocks."""

        ...

    def discard_pages(self, keys: Sequence[bytes]) -> None:
        """Drop host-resident payloads."""

        ...


class LocalKVPageOffloadTransport:
    """In-process transport used by tests and single-process deployments.

    The device accessors are injected so the same transport logic serves a
    CPU-only test and the Worker's CUDA tensors.
    """

    def __init__(
        self,
        store: KVPageOffloadStore | None = None,
        *,
        device_reader,
        device_writer,
    ) -> None:
        # Do not select the store with ``store or ...``: an empty store has
        # ``__len__`` and would be treated as falsy.
        self._store = store if store is not None else KVPageOffloadStore()
        self._device_reader = device_reader
        self._device_writer = device_writer

    @property
    def store(self) -> KVPageOffloadStore:
        return self._store

    def save_pages(self, pages: Sequence[OffloadedKVPage]) -> Mapping[bytes, int]:
        sizes: dict[bytes, int] = {}
        for page in pages:
            payload = self._device_reader(page)
            sizes[page.key] = self._store.save(page, payload)
        return sizes

    def restore_pages(self, requests: Sequence[KVPageRestoreRequest]) -> None:
        for request in requests:
            payload = self._store.restore(request.key)
            self._device_writer(request, payload)

    def discard_pages(self, keys: Sequence[bytes]) -> None:
        for key in keys:
            self._store.discard(key)


class DiffusionKVOffloadCoordinator:
    """Scheduler-side entry point for diffusion KV page offload.

    The coordinator owns the index, drives the transport and enforces the host
    byte budget.  Pages are evicted from the host store only when nothing
    references them, so a restore can never race an eviction.
    """

    def __init__(
        self,
        *,
        transport: KVPageOffloadTransport | None = None,
        max_bytes: int | None = None,
    ) -> None:
        if max_bytes is not None and (type(max_bytes) is not int or max_bytes <= 0):
            raise ValueError(f"max_bytes must be a positive integer when set, got {max_bytes!r}")
        if transport is None and max_bytes is not None:
            # A budget needs a discard path.  Without a transport the
            # coordinator can only track pages, so a budget would be a
            # promise the coordinator cannot keep.
            raise ValueError("max_bytes requires a transport that can discard host-resident pages")
        self._transport = transport
        self._max_bytes = max_bytes
        self._index = DiffusionKVOffloadIndex()

    @property
    def transport(self) -> KVPageOffloadTransport | None:
        """The local transport, if this coordinator moves bytes itself."""

        return self._transport

    def _require_transport(self, operation: str) -> KVPageOffloadTransport:
        if self._transport is None:
            raise RuntimeError(
                f"Diffusion KV page offload {operation} requires a local transport; "
                "in a multi-process deployment the Engine relays pages to the Workers by RPC"
            )
        return self._transport

    @property
    def index(self) -> DiffusionKVOffloadIndex:
        return self._index

    @property
    def max_bytes(self) -> int | None:
        return self._max_bytes

    @property
    def bytes_used(self) -> int:
        return self._index.bytes_used

    def offload_pages(self, pages: Sequence[OffloadedKVPage]) -> tuple[OffloadedKVPage, ...]:
        """Move pages to host storage and register them in the index.

        Returns the pages that are now available off-device.  A failed save is
        propagated: the caller keeps the pages on the device, which is safe
        because the index was not touched.
        """

        if not pages:
            return ()
        unique: dict[bytes, OffloadedKVPage] = {}
        for page in pages:
            known = unique.get(page.key)
            if known is not None and known != page:
                raise ValueError(f"Conflicting offloaded KV page identity for key {page.key!r}")
            unique[page.key] = page
        ordered = list(unique.values())
        # Make room by evicting old unreferenced pages first.  Evicting after
        # the save would drop freshly saved pages instead, which wastes the
        # copy work and leaves the caller with pages it can no longer restore.
        self.enforce_budget()
        sizes = self._require_transport("save").save_pages(ordered)
        for page in ordered:
            self._index.track(page, int(sizes.get(page.key, 0)))
        return tuple(ordered)

    def commit_saved_pages(
        self,
        pages: Sequence[OffloadedKVPage],
        sizes: Mapping[bytes, int],
    ) -> tuple[OffloadedKVPage, ...]:
        """Register pages whose bytes a Worker already copied to the host.

        A multi-process deployment never moves bytes inside the Scheduler:
        the Engine relays page *names* to the Workers by RPC and reports the
        saved sizes back, so the Scheduler-side coordinator only commits the
        resulting index state.
        """

        for page in pages:
            self._index.track(page, int(sizes.get(page.key, 0)))
        return tuple(pages)

    def find_restore_candidates(
        self,
        block_hashes_by_group: Sequence[Sequence[BlockHash]],
    ) -> tuple[tuple[OffloadedKVPage, ...], ...]:
        """Return the per-group off-device prefix that every group can serve.

        Only a prefix shared by all groups may be restored: a row that cannot
        be restored would execute with missing KV.
        """

        if not block_hashes_by_group:
            return ()
        per_group = [
            self._index.find_prefix(block_hashes, group_index)
            for group_index, block_hashes in enumerate(block_hashes_by_group)
        ]
        common = min(len(prefix) for prefix in per_group)
        return tuple(prefix[:common] for prefix in per_group)

    def restore_pages(self, requests: Sequence[KVPageRestoreRequest]) -> None:
        """Restore host-resident pages into device blocks.

        The transport must finish the copies before this returns: a cache hit
        may only execute once its pages are back on the device.
        """

        if not requests:
            return
        for request in requests:
            if request.key not in self._index:
                raise KeyError(f"No offloaded KV page for key {request.key!r}")
        self._require_transport("restore").restore_pages(requests)
        for request in requests:
            self._index.acquire(request.key)

    def release_pages(self, keys: Iterable[bytes]) -> None:
        """Release the references taken by a restore."""

        for key in keys:
            self._index.release(key)

    def enforce_budget(self) -> tuple[bytes, ...]:
        """Drop unreferenced pages until the host store fits its budget.

        Returns the keys that were dropped.  Referenced pages are never
        dropped, so the budget can temporarily be exceeded while requests are
        in flight.
        """

        if self._max_bytes is None:
            return ()
        dropped: list[bytes] = []
        while self.bytes_used > self._max_bytes:
            candidate = self._find_evictable()
            if candidate is None:
                break
            self._require_transport("discard").discard_pages([candidate])
            self._index.remove(candidate)
            dropped.append(candidate)
        return tuple(dropped)

    def _find_evictable(self) -> bytes | None:
        """Return the oldest unreferenced page key, if any."""

        for page in self._index.pages():
            if self._index.refs(page.key) == 0:
                return page.key
        return None
