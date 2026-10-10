# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803 - Triton constexpr parameters use kernel-style capitals.
"""Cheaper staged writes for the MRv2 runner's persistent state tensors.

vLLM's ``StagedWriteTensor`` keeps staged contents as one flat Python list
(prompt token ids and M-RoPE rows are unboxed element by element) and, for a
device tensor, rebuilds a pinned tensor from that list and launches a copy on
every ``apply_write``. With requests joining most steps at high concurrency
this costs ~0.6 ms of engine-loop time per step for a TTS Talker.

Here the contents stay typed NumPy chunks until ``apply_write``, which
concatenates them once into a pinned UVA buffer that the write kernel reads in
place, exactly like the indices, starts and lengths already are. The buffers
come from the same round-robin ``UvaBufferPool`` vLLM uses for those, so the
contents live exactly as long as the other write arguments. The same applies
to ``FusedStagedWriter`` (block tables). M-RoPE prefill positions are staged as
NumPy rows instead of Python lists.

Kept in step with vllm-project/vllm#51334 (typed staging chunks), which does
not remove the per-write pinned allocation and copy launch.

Inside :func:`deferred_writes` (the runner wraps ``add_requests`` in it), each
``apply_write`` only queues its tensor; the queued writes of every tensor then
land with one byte-copy launch when the window closes, or earlier when a
reader needs them (the penalty ``bincount`` reads the new prompt tokens).
Joining requests otherwise cost one write launch per state tensor per step.
A window can flush more than once per step, so the flush stages its arguments
through fenced ``DeviceStager`` rings rather than a two-slot ``UvaBufferPool``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Sequence
from typing import Any, NamedTuple

import numpy as np
import torch
from vllm.logger import init_logger

from vllm_omni.utils.device_copy import DeviceStager

logger = init_logger(__name__)

_installed = False


class _PendingWrite(NamedTuple):
    """The staged writes one ``apply_write`` queued for its tensor."""

    tensor: Any  # vLLM StagedWriteTensor
    indices: list[int]
    starts: list[int]
    contents: list
    cu_lens: list[int]


class _Deferred:
    depth = 0
    pending: list[_PendingWrite] = []
    flush: Callable[[], None] | None = None  # set by install(): applies and clears ``pending``


@contextlib.contextmanager
def deferred_writes() -> Iterator[None]:
    """Queue ``StagedWriteTensor.apply_write`` calls and apply them together on exit."""
    flush = _Deferred.flush
    if flush is None:
        yield
        return
    _Deferred.depth += 1
    try:
        yield
    finally:
        _Deferred.depth -= 1
        if _Deferred.depth == 0:
            flush()


def _flush_deferred_writes() -> None:
    """Apply the queued writes now (before a kernel that reads them)."""
    flush = _Deferred.flush
    if flush is not None:
        flush()


def _stage_values(chunks: list, x: Any) -> int:
    """Append ``x`` to the staged contents; returns its length.

    Python values go into a trailing list, converted once at apply time like
    the upstream flat list; arrays and CPU tensors are kept as private copies
    (no per-element boxing).
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        chunk = np.array(x, copy=True).reshape(-1)
        if chunk.size:
            chunks.append(chunk)
        return int(chunk.size)
    if not isinstance(x, Sequence):
        x = list(x)
    if not x:
        return 0
    if chunks and type(chunks[-1]) is list:
        chunks[-1].extend(x)
    else:
        chunks.append(list(x))
    return len(x)


def _concat(chunks: list, np_dtype: np.dtype) -> np.ndarray:
    arrays = [np.asarray(chunk, dtype=np_dtype) for chunk in chunks]
    return arrays[0] if len(arrays) == 1 else np.concatenate(arrays)


def install() -> None:
    """Patch ``StagedWriteTensor``/``FusedStagedWriter``/``RopeState`` in this process (idempotent)."""
    global _installed
    if _installed:
        return
    from vllm.utils.platform_utils import is_uva_available
    from vllm.v1.worker.gpu import buffer_utils
    from vllm.v1.worker.gpu.mm import rope

    if not is_uva_available():
        return

    swt = buffer_utils.StagedWriteTensor
    fused = buffer_utils.FusedStagedWriter
    apply_kernel = buffer_utils._apply_write_kernel
    pool_cls = buffer_utils.UvaBufferPool
    orig_init = swt.__init__
    orig_fused_init = fused.__init__

    def swt_init(self, size, dtype, device, max_concurrency=None, uva_instead_of_gpu=False):
        orig_init(self, size, dtype, device, max_concurrency, uva_instead_of_gpu)
        self._np_dtype = torch.empty((), dtype=dtype).numpy().dtype
        self._staged_len = 0
        if self.write_contents is None:
            self.write_contents = pool_cls(1, dtype=dtype, max_concurrency=self.max_concurrency)

    def stage_write(self, index: int, start: int, x) -> None:
        assert index >= 0
        assert start >= 0
        size = _stage_values(self._staged_write_contents, x)
        if size == 0:
            return
        self._staged_write_indices.append(index)
        self._staged_write_starts.append(start)
        self._staged_len += size
        self._staged_write_cu_lens.append(self._staged_len)

    def stage_write_elem(self, index: int, x) -> None:
        assert index >= 0
        self._staged_write_indices.append(index)
        self._staged_write_starts.append(0)
        contents = self._staged_write_contents
        if contents and type(contents[-1]) is list:
            contents[-1].append(x)
        else:
            contents.append([x])
        self._staged_len += 1
        self._staged_write_cu_lens.append(self._staged_len)

    def apply_write(self) -> None:
        n = len(self._staged_write_indices)
        if n == 0:
            return
        if _Deferred.depth:
            if any(write.tensor is self for write in _Deferred.pending):
                # Writes of one launch run in parallel: keep a later batch for the same tensor ordered.
                flush()
            # Take this batch as an immediate apply would; later stages start fresh lists.
            _Deferred.pending.append(
                _PendingWrite(
                    self,
                    self._staged_write_indices,
                    self._staged_write_starts,
                    self._staged_write_contents,
                    self._staged_write_cu_lens,
                )
            )
            self._staged_write_indices = []
            self._staged_write_starts = []
            self._staged_write_contents = []
            self._staged_write_cu_lens = []
            self._staged_len = 0
            return
        indices_uva = self.write_indices.copy_to_uva(np.asarray(self._staged_write_indices, dtype=np.int32))
        starts_uva = self.write_starts.copy_to_uva(np.asarray(self._staged_write_starts, dtype=np.int32))
        cu_lens_uva = self.write_cu_lens.copy_to_uva(np.asarray(self._staged_write_cu_lens, dtype=np.int32))
        contents_uva = self.write_contents.copy_to_uva(_concat(self._staged_write_contents, self._np_dtype))
        apply_kernel[(n,)](
            self.gpu,
            self.gpu.stride(0),
            indices_uva,
            starts_uva,
            contents_uva,
            cu_lens_uva,
            None,
            BLOCK_SIZE=1024,
            MULTI_GROUP=False,
        )
        self.clear_staged_writes()

    def clear_staged_writes(self) -> None:
        self._staged_write_indices.clear()
        self._staged_write_starts.clear()
        self._staged_write_contents.clear()
        self._staged_write_cu_lens.clear()
        self._staged_len = 0

    from vllm.triton_utils import tl, triton

    @triton.jit
    def _byte_scatter_kernel(meta_ptr, contents_ptr, BLOCK: tl.constexpr):
        # meta row w: destination address, source byte offset, byte count
        w = tl.program_id(0).to(tl.int64)
        dst = tl.cast(tl.load(meta_ptr + 3 * w), tl.pointer_type(tl.uint8))
        src = tl.load(meta_ptr + 3 * w + 1)
        nbytes = tl.load(meta_ptr + 3 * w + 2)
        for off in range(0, nbytes, BLOCK):
            offs = off + tl.arange(0, BLOCK)
            m = offs < nbytes
            tl.store(dst + offs, tl.load(contents_ptr + src + offs, mask=m), mask=m)

    meta_stager = DeviceStager(dtype=torch.int64)
    contents_stager = DeviceStager(dtype=torch.uint8)

    def flush() -> None:
        pending = _Deferred.pending
        if not pending:
            return
        _Deferred.pending = []
        metas: list[np.ndarray] = []
        blobs: list[np.ndarray] = []
        offset = 0
        for t, indices, starts, contents, cu_lens in pending:
            elem = t.gpu.element_size()
            cu = np.asarray(cu_lens, dtype=np.int64)
            prev = np.concatenate(([0], cu[:-1]))
            meta = np.empty((cu.size, 3), dtype=np.int64)
            meta[:, 0] = (
                t.gpu.data_ptr()
                + np.asarray(indices, dtype=np.int64) * (t.gpu.stride(0) * elem)
                + np.asarray(starts, dtype=np.int64) * elem
            )
            meta[:, 1] = offset + prev * elem
            meta[:, 2] = (cu - prev) * elem
            blob = np.ascontiguousarray(_concat(contents, t._np_dtype)).view(np.uint8)
            metas.append(meta)
            blobs.append(blob)
            offset += blob.size
        device = pending[0][0].gpu.device
        meta_uva = meta_stager(np.concatenate(metas).reshape(-1), device)
        contents_uva = contents_stager(blobs[0] if len(blobs) == 1 else np.concatenate(blobs), device)
        _byte_scatter_kernel[(meta_uva.shape[0] // 3,)](meta_uva, contents_uva, BLOCK=1024)

    def fused_init(self, device, max_writes, max_concurrency=None):
        orig_fused_init(self, device, max_writes, max_concurrency)
        self.contents = pool_cls(1, dtype=torch.int32, max_concurrency=max_concurrency)

    def fused_apply(self, tensors, output_ptrs, output_strides) -> None:
        group_ids: list[int] = []
        indices: list[int] = []
        starts: list[int] = []
        chunks: list[np.ndarray] = []
        cu_lens: list[np.ndarray] = []
        base = 0
        for group_id, t in enumerate(tensors):
            n = len(t._staged_write_indices)
            if n == 0:
                continue
            group_ids.extend([group_id] * n)
            indices.extend(t._staged_write_indices)
            starts.extend(t._staged_write_starts)
            chunks.extend(t._staged_write_contents)
            cu_lens.append(np.asarray(t._staged_write_cu_lens, dtype=np.int32) + base)
            base += t._staged_len
        if not group_ids:
            return
        group_ids_uva = self.group_ids.copy_to_uva(np.asarray(group_ids, dtype=np.int32))
        indices_uva = self.indices.copy_to_uva(np.asarray(indices, dtype=np.int32))
        starts_uva = self.starts.copy_to_uva(np.asarray(starts, dtype=np.int32))
        cu_lens_uva = self.cu_lens.copy_to_uva(_concat(cu_lens, np.dtype(np.int32)))
        contents_uva = self.contents.copy_to_uva(_concat(chunks, np.dtype(np.int32)))
        apply_kernel[(len(group_ids),)](
            output_ptrs,
            output_strides,
            indices_uva,
            starts_uva,
            contents_uva,
            cu_lens_uva,
            group_ids_uva,
            BLOCK_SIZE=1024,
            MULTI_GROUP=True,
        )
        for t in tensors:
            t.clear_staged_writes()

    def init_prefill_positions(self, req_idx, model, prefill_token_ids, mm_features) -> None:
        prefill_positions, delta = model.get_mrope_input_positions(prefill_token_ids, mm_features)
        self.prefill_delta.np[req_idx] = delta
        if isinstance(prefill_positions, torch.Tensor):
            prefill_positions = prefill_positions.detach().cpu().numpy()
        for i in range(self.num_dims):
            self.prefill_positions.stage_write(self.num_dims * req_idx + i, 0, prefill_positions[i])

    _install_staged_h2d()
    _install_fused_bincount()
    swt.__init__ = swt_init
    swt.stage_write = stage_write
    swt.stage_write_elem = stage_write_elem
    swt.apply_write = apply_write
    swt.clear_staged_writes = clear_staged_writes
    _Deferred.flush = flush
    fused.__init__ = fused_init
    fused.apply = fused_apply
    rope.RopeState.init_prefill_positions = init_prefill_positions
    _installed = True
    logger.debug("Installed NumPy/UVA staged writes")


_STAGED_H2D_MAX_ELEMENTS = 1 << 16


def _install_fused_bincount() -> None:
    """Penalty statistics of new requests in one launch.

    Upstream ``bincount`` casts the slot indices, zeroes both statistic rows
    with two ``index_fill_`` launches and then counts; one program per request
    here zeroes its rows and counts its prompt / output tokens.
    """
    from vllm.triton_utils import tl, triton
    from vllm.v1.worker.gpu.sample import penalties

    @triton.jit
    def _fused_bincount_kernel(
        idx_ptr,
        token_ids_ptr,
        token_ids_stride,
        prompt_len_ptr,
        prefill_len_ptr,
        mask_ptr,
        mask_stride,
        num_words,
        counts_ptr,
        counts_stride,
        vocab_size,
        BLOCK: tl.constexpr,
    ):
        req = tl.load(idx_ptr + tl.program_id(0)).to(tl.int64)
        for start in range(0, num_words, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            tl.store(mask_ptr + req * mask_stride + offs, tl.zeros([BLOCK], tl.int32), mask=offs < num_words)
        for start in range(0, vocab_size, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            tl.store(counts_ptr + req * counts_stride + offs, tl.zeros([BLOCK], tl.int32), mask=offs < vocab_size)
        # The zeroed rows must be visible to the atomics below.
        tl.debug_barrier()
        prompt_len = tl.load(prompt_len_ptr + req)
        prefill_len = tl.load(prefill_len_ptr + req)
        for start in range(0, prefill_len, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            valid = offs < prefill_len
            tokens = tl.load(token_ids_ptr + req * token_ids_stride + offs, mask=valid, other=0)
            in_prompt = valid & (offs < prompt_len)
            bit = tl.full([BLOCK], 1, tl.int32) << (tokens % 32)
            tl.atomic_or(mask_ptr + req * mask_stride + tokens // 32, bit, mask=in_prompt)
            tl.atomic_add(counts_ptr + req * counts_stride + tokens, 1, mask=valid & (offs >= prompt_len))

    def fused_bincount(
        expanded_idx_mapping,
        all_token_ids,
        prompt_len,
        prefill_len,
        prompt_bin_mask,
        output_bin_counts,
        max_prefill_len,
    ) -> None:
        n = expanded_idx_mapping.shape[0]
        if n == 0:
            return
        # Counts the prompt tokens a deferred write may still be holding.
        _flush_deferred_writes()
        _fused_bincount_kernel[(n,)](
            expanded_idx_mapping,
            all_token_ids,
            all_token_ids.stride(0),
            prompt_len,
            prefill_len,
            prompt_bin_mask,
            prompt_bin_mask.stride(0),
            prompt_bin_mask.shape[1],
            output_bin_counts,
            output_bin_counts.stride(0),
            output_bin_counts.shape[1],
            BLOCK=1024,
        )

    penalties.bincount = fused_bincount


def _install_staged_h2d() -> None:
    """Small per-step host arrays go up through a resident UVA ring instead of a fresh pinned buffer.

    ``async_tensor_h2d`` allocates a pinned host tensor and launches a copy for
    every call (``idx_mapping`` and ``query_start_loc`` each step, the new
    penalty rows whenever requests join). The ring's slots are only reused
    after the copies that read them have run (see ``DeviceStager``).
    """
    from vllm.utils import torch_utils
    from vllm.v1.worker.gpu import model_runner
    from vllm.v1.worker.gpu.sample import penalties

    original = torch_utils.async_tensor_h2d
    stagers: dict[torch.dtype, DeviceStager] = {}

    def staged_h2d(data, device=None, dtype=None, out=None):
        if dtype is None and out is not None:
            dtype = out.dtype
        if isinstance(data, torch.Tensor) or (dtype is None and not isinstance(data, np.ndarray)):
            return original(data, device=device, dtype=dtype, out=out)
        arr = np.asarray(data) if dtype is None else np.asarray(data, dtype=torch.empty((), dtype=dtype).numpy().dtype)
        target = out.device if out is not None else torch.device(device)
        if target.type != "cuda" or arr.size > _STAGED_H2D_MAX_ELEMENTS or arr.dtype.kind not in "biuf":
            return original(data, device=device, dtype=dtype, out=out)
        torch_dtype = torch.from_numpy(arr[:0]).dtype
        stager = stagers.get(torch_dtype)
        if stager is None:
            stager = stagers[torch_dtype] = DeviceStager(dtype=torch_dtype, slots=32, group=8)
        view = stager(arr, target)
        if out is not None:
            return out.copy_(view, non_blocking=True)
        return view.clone()

    model_runner.async_tensor_h2d = staged_h2d
    penalties.async_tensor_h2d = staged_h2d


__all__ = ["deferred_writes", "install"]
