# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Same-node GPU transport: SHM control plane, CUDA IPC data plane.

Everything this connector adds lives in this file. ``SharedMemoryConnector``
is used untouched: a payload without a tensor above ``inline_tensor_bytes``
takes its exact msgpack path (``super().put()`` / the legacy read), so
edges that do not name ``CudaIpcConnector`` never execute this code.
"""

from __future__ import annotations

import fcntl
import functools
import os
import threading
import time
from collections.abc import Callable, Sequence
from multiprocessing import shared_memory as shm_pkg
from typing import Any

import msgspec
import numpy as np
import torch
from vllm.utils.math_utils import round_up

from ..utils.logging import get_connector_logger
from .shm_connector import SharedMemoryConnector

logger = get_connector_logger(__name__)

# Positions in the args tuple of torch.multiprocessing.reductions.reduce_tensor.
_ARG_TENSOR_CLS = 0
_ARG_STORAGE_CLS = 4
_ARG_DTYPE = 5
_ARG_DEVICE = 6
_ARG_REF_COUNTER_HANDLE = 11
_ARG_REF_COUNTER_OFFSET = 12
_COLLECT_INTERVAL_S = 1.0

# ── Framed segment format ────────────────────────────────────────────
# A framed payload stores large tensors as raw bytes after a msgpack header:
#   magic | u32 header length | header | pad | tensor bytes (64 B aligned)...
# 0xC1 is never valid msgpack, so legacy payloads cannot be mistaken for it.
_FRAME_MAGIC = b"\xc1OMF"
_FRAME_PREFIX = 8
_FRAME_ALIGN = 64
_DEFAULT_INLINE_TENSOR_BYTES = 64 * 1024
_UNCLAIMED = object()

# (header, [(tensor, offset)], segment size, producer handles or None)
_Frame = tuple[bytes, list[tuple[torch.Tensor, int]], int, list[Any] | None]


def _data_start(header_len: int) -> int:
    return round_up(_FRAME_PREFIX + header_len, _FRAME_ALIGN)


def tensor_desc(tensor: torch.Tensor) -> dict[str, Any]:
    """dtype/shape descriptor of a framed tensor."""
    return {"dtype": str(tensor.dtype).removeprefix("torch."), "shape": list(tensor.shape)}


def desc_dtype(desc: dict[str, Any]) -> torch.dtype:
    dtype = getattr(torch, str(desc.get("dtype")), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Invalid framed tensor dtype: {desc.get('dtype')!r}")
    return dtype


# ── Tensor-tree codec ────────────────────────────────────────────────
# Splits tensors out of a nested payload so they can travel out of band
# (into the frame / IPC descriptors) while only the small tensor-free
# skeleton is msgpack-encoded.

_TENSOR = "__omni_tensor__"
_NDARRAY = "__omni_ndarray__"
_TUPLE = "__omni_tuple__"
# Leaves that can never hold a tensor; skipping them keeps long id lists cheap.
_SCALARS = (int, float, str, bool, bytes, type(None))


def extract_tensors(
    obj: Any,
    select: Callable[[torch.Tensor], bool] | None = None,
    *,
    ndarrays: bool = True,
) -> tuple[Any, list[torch.Tensor]]:
    """Replace selected tensor leaves with index markers.

    ``select`` decides per tensor (numeric ndarrays are offered as CPU
    tensors when ``ndarrays``); rejected leaves stay inline. Subtrees without
    selected tensors are returned as-is, so a payload with none comes back
    unchanged. A ``msgspec.Struct`` holding one becomes a dict, the shape
    ``OmniSerializer`` decodes to; tuples keep their type.
    """
    tensors: list[torch.Tensor] = []

    def take(tensor: torch.Tensor, marker: str) -> dict[str, int] | None:
        if select is not None and not select(tensor):
            return None
        tensors.append(tensor)
        return {marker: len(tensors) - 1}

    def visit(value: Any) -> Any:
        if type(value) in _SCALARS:
            return value
        if isinstance(value, torch.Tensor):
            return take(value, _TENSOR) or value
        if ndarrays and isinstance(value, np.ndarray) and value.dtype.kind not in "OVUS":
            return take(torch.from_numpy(np.ascontiguousarray(value)), _NDARRAY) or value
        if isinstance(value, msgspec.Struct):
            fields = msgspec.structs.asdict(value)
            visited = visit(fields)
            return value if visited is fields else visited
        if isinstance(value, dict):
            copied = None
            for key, item in value.items():
                if type(item) not in _SCALARS and (new := visit(item)) is not item:
                    copied = copied or dict(value)
                    copied[key] = new
            return value if copied is None else copied
        if isinstance(value, (list, tuple)):
            items: list[Any] | None = None
            for index, item in enumerate(value):
                if type(item) not in _SCALARS and (new := visit(item)) is not item:
                    items = items or list(value)
                    items[index] = new
            if items is None:
                return value
            return {_TUPLE: items} if isinstance(value, tuple) else items
        return value

    return visit(obj), tensors


def restore_tensors(skeleton: Any, tensors: Sequence[torch.Tensor]) -> Any:
    """Inverse of :func:`extract_tensors`; fills containers in place."""

    def leaf(value: Any) -> Any:
        if isinstance(value, dict) and len(value) == 1:
            if _TENSOR in value:
                return tensors[value[_TENSOR]]
            if _NDARRAY in value:
                return tensors[value[_NDARRAY]].cpu().numpy()
            if _TUPLE in value:
                return tuple(leaf(item) for item in value[_TUPLE])
        return visit(value)

    def visit(value: Any) -> Any:
        if isinstance(value, dict):
            for key, item in value.items():
                if type(item) not in _SCALARS:
                    value[key] = leaf(item)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if type(item) not in _SCALARS:
                    value[index] = leaf(item)
        return value

    return leaf(skeleton) if tensors else skeleton


# ── Transport snapshot ────────────────────────────────────────────────


def _clone_cuda_tensors(value: Any, cloned: list[torch.Tensor]) -> Any:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cuda":
            return value
        out = value.detach().clone()
        cloned.append(out)
        return out
    if isinstance(value, dict):
        return {k: _clone_cuda_tensors(v, cloned) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_cuda_tensors(v, cloned) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_cuda_tensors(v, cloned) for v in value)
    return value


def _create_segment(size: int, name: str | None = None) -> shm_pkg.SharedMemory:
    """Create a SharedMemory segment, replacing a stale one with the same name."""
    try:
        return shm_pkg.SharedMemory(create=True, size=size, name=name)
    except FileExistsError:
        if not name:
            raise
        try:
            existing = shm_pkg.SharedMemory(name=name)
            existing.close()
            existing.unlink()
        except Exception:
            pass
        return shm_pkg.SharedMemory(create=True, size=size, name=name)


@functools.cache
def _gpu_uuid(device_index: int) -> str:
    return str(torch.cuda.get_device_properties(device_index).uuid)


@functools.cache
def _visible_gpus() -> dict[str, int]:
    return {_gpu_uuid(index): index for index in range(torch.accelerator.device_count())}


@functools.cache
def _copy_stream(device_index: int) -> torch.cuda.Stream:
    return torch.cuda.Stream(device=device_index)


def _wire_args(args: tuple) -> list[Any]:
    """``reduce_tensor`` args as plain msgpack values.

    The class and dtype slots are fixed for the exported contiguous clone, so
    they are dropped here and restored by :func:`_rebuild_args`.
    """
    types = (_ARG_TENSOR_CLS, _ARG_STORAGE_CLS, _ARG_DTYPE)
    return [None if i in types else list(a) if isinstance(a, tuple) else a for i, a in enumerate(args)]


def _rebuild_args(desc: dict[str, Any]) -> list[Any]:
    """``rebuild_cuda_tensor`` args for a descriptor written by the producer."""
    args = list(desc["ipc"])
    args[_ARG_TENSOR_CLS] = torch.Tensor
    args[_ARG_STORAGE_CLS] = torch.storage.TypedStorage
    args[_ARG_DTYPE] = desc_dtype(desc)
    return args


def _release_counter(args: Sequence[Any]) -> None:
    """Drop the reference torch counts for a consumer that will never rebuild ``args``."""
    try:
        torch.UntypedStorage._release_ipc_counter_cuda(args[_ARG_REF_COUNTER_HANDLE], args[_ARG_REF_COUNTER_OFFSET])
    except Exception:
        logger.debug("Failed to release a CUDA IPC counter", exc_info=True)


class CudaIpcConnector(SharedMemoryConnector):
    """Move large CUDA tensors between stage processes on one node by CUDA IPC.

    The SHM control plane of ``SharedMemoryConnector`` is inherited as is:
    ``put()`` creates a ``/dev/shm`` segment per payload, ownership is
    claim-by-unlink, and payloads without a large tensor are written and read
    by the base class byte-for-byte. Large tensors travel as a frame: the
    segment carries the tensor-free skeleton plus descriptors, and each CUDA
    tensor is cloned on the producer and exported as a torch CUDA IPC handle
    instead of being staged through host memory. The consumer maps each
    clone, copies it onto its own device and drops the mapping; torch's IPC
    refcount keeps the clone alive until then.

    The consumer must see the producer's GPU: the same GPU, or a peer made
    visible to its process. ``use_ipc: false`` turns the connector into
    framed SHM (host staging), which works for any placement.
    """

    supports_raw_data = True

    def __init__(self, config: dict[str, Any]):
        config = {"inline_tensor_bytes": _DEFAULT_INLINE_TENSOR_BYTES, **(config or {})}
        super().__init__(config)
        self._inline_tensor_bytes = max(1, int(config["inline_tensor_bytes"]))
        self._use_ipc = bool(config.get("use_ipc", True)) and torch.cuda.is_available()
        # put_key -> exports awaiting a consumer (empty list for plain frames).
        self._exports: dict[str, list[Any]] = {}
        self._exports_lock = threading.Lock()
        # Imports land on the device this connector was built on. A process
        # that never initialized CUDA (e.g. a TP>1 scheduler) gets host copies.
        self._receive_device = torch.device("cpu")
        if torch.cuda.is_initialized():
            self._receive_device = torch.device("cuda", torch.accelerator.current_device_index())
        self._last_collect = 0.0
        self._export_failure_logged = False

    # ------------------------------------------------------------------ #
    #  Producer
    # ------------------------------------------------------------------ #

    def put(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        data: Any,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        frame = None
        try:
            self.reap_consumed()
            # Re-put replaces the payload: the previous exports have no
            # reader left, whether that payload was framed or legacy.
            self._drop_exports(put_key, consumed=False)
            frame = self._build_frame(data)
            if frame is None:
                return super().put(from_stage, to_stage, put_key, data)

            lock_file = f"/dev/shm/shm_{put_key}_lockfile.lock"
            with open(lock_file, "wb+") as lockf:
                fcntl.flock(lockf, fcntl.LOCK_EX)
                meta = self._write_frame(put_key, frame)
                fcntl.flock(lockf, fcntl.LOCK_UN)
            size = int(meta["size"])

            with self._exports_lock:
                self._exports[put_key] = frame[3] or []
            # Track the segment like the base put() does, so close(),
            # cleanup() and reap_consumed() keep working through the
            # base-class paths (the exports themselves live in _exports).
            with self._pending_keys_lock:
                self._pending_keys[put_key] = None
            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += size
            return True, size, {"shm": meta, "size": size}

        except Exception as e:
            logger.error(f"CudaIpcConnector put failed for req {put_key}: {e}")
            if frame and frame[3]:
                self._retire(put_key, frame[3], consumed=False)
                with self._exports_lock:
                    self._exports.pop(put_key, None)
            return False, 0, None

    def snapshot_for_transport(self, payload: Any) -> tuple[Any, torch.cuda.Event | None]:
        """Isolate ``payload``'s CUDA tensors on the caller's current stream.

        The save-loop thread reads tensors only when ``put()`` runs, by which
        time a CUDA-graph static buffer or reused runner input may hold the
        next step's data; clone them on the model thread now. Host tensors
        and other leaves pass through untouched; the event is ``None`` when
        nothing was cloned.
        """
        cloned: list[torch.Tensor] = []
        snapshot = _clone_cuda_tensors(payload, cloned)
        if not cloned:
            return snapshot, None
        ready_event = torch.cuda.Event()
        ready_event.record(torch.cuda.current_stream())
        return snapshot, ready_event

    def _export_tensor(self, tensor: torch.Tensor) -> tuple[dict[str, Any], Any] | None:
        """Send ``tensor`` out of band: return its header descriptor and a
        producer handle, or ``None`` to copy its bytes into the segment."""
        if not (self._use_ipc and tensor.is_cuda):
            return None
        from torch.multiprocessing.reductions import reduce_tensor

        # reduce_tensor shares the whole allocation and the producer may reuse
        # the source (CUDA-graph outputs, KV blocks) once put() returns, so
        # export a private clone. torch records a CUDA event after the clone
        # that the consumer waits on before reading.
        owned = tensor.detach().clone(memory_format=torch.contiguous_format)
        try:
            _, args = reduce_tensor(owned)
        except Exception as exc:  # e.g. expandable segments, pluggable allocators
            if not self._export_failure_logged:
                logger.warning("CUDA IPC export failed (%s); staging through host memory instead", exc)
                self._export_failure_logged = True
            return None
        desc = {**tensor_desc(owned), "ipc": _wire_args(args), "gpu": _gpu_uuid(owned.device.index)}
        return desc, (args, owned.nbytes)

    def _retire(self, key: str, handles: list[Any], *, consumed: bool) -> None:
        if not consumed:
            # The handle left without an owner to rebuild it (cleanup, TTL,
            # re-put): drop the reference torch would otherwise count forever.
            # A claimed payload's consumer releases what it fails to import.
            for args, _ in handles:
                _release_counter(args)
        self._maybe_collect()

    def _maybe_collect(self, force: bool = False) -> None:
        now = time.monotonic()
        if force or now - self._last_collect >= _COLLECT_INTERVAL_S:
            self._last_collect = now
            # Free clones whose consumers have released their mappings.
            torch.cuda.ipc_collect()

    def _drop_exports(self, key: str, *, consumed: bool) -> None:
        """Retire this instance's exports for ``key``, if it produced any."""
        with self._exports_lock:
            handles = self._exports.pop(key, None)
        if handles:
            self._retire(key, handles, consumed=consumed)

    # ------------------------------------------------------------------ #
    #  Framed payloads
    # ------------------------------------------------------------------ #

    def _frames_tensor(self, tensor: torch.Tensor) -> bool:
        return tensor.nbytes >= self._inline_tensor_bytes

    def _build_frame(self, data: Any) -> _Frame | None:
        """Split large tensors out of ``data``; ``None`` keeps the legacy format."""
        skeleton, tensors = extract_tensors(data, self._frames_tensor)
        if not tensors:
            return None
        descs: list[dict[str, Any]] = []
        raw: list[tuple[torch.Tensor, int]] = []
        handles: list[Any] = []
        offset = 0
        try:
            for tensor in tensors:
                exported = self._export_tensor(tensor)
                if exported is None:
                    exported = ({**tensor_desc(tensor), "offset": offset}, None)
                    raw.append((tensor, offset))
                    offset = round_up(offset + tensor.nbytes, _FRAME_ALIGN)
                else:
                    handles.append(exported[1])
                descs.append(exported[0])
            header = self.serialize_obj({"skeleton": skeleton, "tensors": descs})
        except Exception:
            if handles:
                self._retire("", handles, consumed=False)
            raise
        return header, raw, _data_start(len(header)) + offset, handles or None

    @staticmethod
    def _write_frame(name: str, frame: _Frame) -> dict[str, Any]:
        header, raw, size, _ = frame
        shm = _create_segment(size, name)
        try:
            buf = shm.buf
            buf[:4] = _FRAME_MAGIC
            buf[4:_FRAME_PREFIX] = len(header).to_bytes(4, "little")
            buf[_FRAME_PREFIX : _FRAME_PREFIX + len(header)] = header
            data_start = _data_start(len(header))
            for tensor, offset in raw:
                dst = torch.frombuffer(buf, dtype=torch.uint8, count=tensor.nbytes, offset=data_start + offset)
                dst.view(tensor.dtype).view(tensor.shape).copy_(tensor)
                del dst  # release the buffer export before close()
            del buf
        finally:
            shm.close()
        return {"name": name, "size": size}

    # ------------------------------------------------------------------ #
    #  Consumer
    # ------------------------------------------------------------------ #

    def _get_data_with_lock(self, lock_file: str, shm_handle: dict[str, Any]) -> tuple[Any, int] | None:
        try:
            with open(lock_file, "rb+") as lockf:
                fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                obj = self._consume_segment(shm_handle)
                fcntl.flock(lockf, fcntl.LOCK_UN)
            if obj is _UNCLAIMED:
                return None
            # This instance also produced the key: its exports found a reader.
            self._drop_exports(shm_handle["name"], consumed=True)
            return obj, int(shm_handle.get("size", 0))
        except BlockingIOError:
            return None
        except Exception as e:
            logger.warning(f"CudaIpcConnector get failed for req {shm_handle['name']}: {e}")
            try:
                shm_pkg.SharedMemory(name=shm_handle["name"]).unlink()
            except FileNotFoundError:
                pass
            return None

    def _consume_segment(self, shm_handle: dict[str, Any]) -> Any:
        """Read a segment and unlink it.

        Legacy (msgpack) segments are read then unlinked, as the base
        connector does. A frame is claimed by unlinking it first: the claim
        makes this reader the owner of any exported handles in it, so a
        concurrent ``cleanup()`` cannot release them underneath the import.
        Losing that unlink means the payload was discarded; returns
        ``_UNCLAIMED`` then.
        """
        shm = shm_pkg.SharedMemory(name=shm_handle["name"])
        try:
            buf = shm.buf[: shm_handle["size"]]
            try:
                if buf[:4] != _FRAME_MAGIC:
                    data = bytes(buf)
                    buf.release()
                    shm.unlink()
                    return self.deserialize_obj(data)
                try:
                    shm.unlink()
                except FileNotFoundError:
                    return _UNCLAIMED
                return self._decode_frame(buf)
            finally:
                buf.release()
        finally:
            shm.close()

    def _decode_frame(self, buf: memoryview) -> Any:
        header_len = int.from_bytes(buf[4:_FRAME_PREFIX], "little")
        header = self.deserialize_obj(buf[_FRAME_PREFIX : _FRAME_PREFIX + header_len])
        tensors = self._import_tensors(header["tensors"], buf, _data_start(header_len))
        return restore_tensors(header["skeleton"], tensors)

    def _import_tensors(self, descs: list[dict[str, Any]], buf: memoryview, data_start: int) -> list[torch.Tensor]:
        ipc = [i for i, desc in enumerate(descs) if "ipc" in desc]
        if not ipc:
            return self._copy_from_segment(descs, buf, data_start)
        from torch.multiprocessing.reductions import rebuild_cuda_tensor

        tensors: list[Any] = [None] * len(descs)
        host = [i for i in range(len(descs)) if i not in ipc]
        for i, tensor in zip(host, self._copy_from_segment([descs[i] for i in host], buf, data_start)):
            tensors[i] = tensor

        # Claiming the segment made this reader the owner of every handle: a
        # mapping returns its clone when dropped, the rest are released here.
        handles = [_rebuild_args(descs[i]) for i in ipc]
        mappings: list[torch.Tensor] = []
        streams: dict[int, torch.cuda.Stream] = {}
        try:
            sources = [self._local_device_index(descs[i]["gpu"]) for i in ipc]
            for i, args, source in zip(ipc, handles, sources):
                desc = descs[i]
                args[_ARG_DEVICE] = source
                out = torch.empty(desc["shape"], dtype=desc_dtype(desc), device=self._receive_device)
                stream = streams.setdefault(source, _copy_stream(source))
                if out.is_cuda:
                    stream.wait_stream(torch.cuda.current_stream(out.device))
                with torch.cuda.stream(stream):
                    # Rebuild waits on the producer's event on this stream; a
                    # copy to another GPU also runs on the source device.
                    mappings.append(rebuild_cuda_tensor(*args))
                    out.copy_(mappings[-1], non_blocking=True)
                tensors[i] = out
        except BaseException:
            for args in handles[len(mappings) :]:
                _release_counter(args)
            raise
        finally:
            for stream in streams.values():
                stream.synchronize()
            mappings.clear()  # unmap only after the copies finished
        return tensors

    @staticmethod
    def _copy_from_segment(descs: list[dict[str, Any]], buf: memoryview, data_start: int) -> list[torch.Tensor]:
        tensors = []
        for desc in descs:
            if "offset" not in desc:
                raise RuntimeError(f"CudaIpcConnector cannot import an out-of-band tensor ({sorted(desc)})")
            out = torch.empty(desc["shape"], dtype=desc_dtype(desc))
            if out.nbytes:
                src = torch.frombuffer(buf, dtype=torch.uint8, count=out.nbytes, offset=data_start + desc["offset"])
                out.view(-1).view(torch.uint8).copy_(src)
                del src  # release the buffer export before the segment closes
            tensors.append(out)
        return tensors

    @staticmethod
    def _local_device_index(uuid: str) -> int:
        index = _visible_gpus().get(uuid)
        if index is None:
            raise RuntimeError(
                f"CUDA IPC source GPU {uuid} is not visible to this process "
                f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}); "
                "make the producer GPU peer-visible or set use_ipc: false on this edge"
            )
        return index

    # ------------------------------------------------------------------ #
    #  Producer-side reclamation
    # ------------------------------------------------------------------ #

    def cleanup(self, request_id: str) -> None:
        # An unread payload's exports have no reader left; a consumed one had
        # them released by its reader (the segment is already unlinked).
        self._drop_exports(request_id, consumed=not os.path.exists(f"/dev/shm/{request_id}"))
        super().cleanup(request_id)

    def reap_consumed(self) -> None:
        # Exports whose segment the receiver already unlinked found a reader.
        with self._exports_lock:
            claimed = [key for key in self._exports if not os.path.exists(f"/dev/shm/{key}")]
        for key in claimed:
            self._drop_exports(key, consumed=True)
        super().reap_consumed()

    # ------------------------------------------------------------------ #

    def close(self) -> None:
        super().close()
        if self._use_ipc and torch.cuda.is_initialized():
            self._maybe_collect(force=True)

    def health(self) -> dict[str, Any]:
        with self._exports_lock:
            handles = [handle for items in self._exports.values() for handle in items]
        return {**super().health(), "ipc_exports": len(handles), "ipc_export_bytes": sum(n for _, n in handles)}
