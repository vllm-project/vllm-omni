# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""IPC utilities for transferring large tensors via POSIX shared memory.

Tensors are streamed chunk-by-chunk with simple request-response protocol:
Producer writes a chunk, waits for consumer ACK, then writes next chunk.
This ensures only one chunk exists in /dev/shm at a time.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from multiprocessing import shared_memory

import numpy as np
import torch
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.diffusion import envs
from vllm_omni.diffusion.data import DiffusionOutput

logger = init_logger(__name__)

# Minimum tensor size to use SHM instead of inline transfer
_SHM_TENSOR_THRESHOLD = 1_000_000  # 1 MB

# Default maximum size for a single SHM segment (64MB)
_DEFAULT_SHM_MAX_SEGMENT_SIZE = 64 * 1024 * 1024
_SHM_ACK_TIMEOUT = 30  # seconds to wait for ACK between SHM chunks

# Key used in placeholder dicts that replace tensors extracted for SHM transfer.
# These placeholder dicts survive pickling through MessageQueue and are replaced
# with the actual tensors on the consumer side.
_SHM_PLACEHOLDER_KEY = "__shm_placeholder__"


def _get_max_shm_segment_size() -> int:
    """Get max chunk size for the reusable SHM buffer.

    Since only one chunk-sized segment is resident in /dev/shm at a time
    (write → wait ACK → overwrite), the buffer can use up to 90 % of
    available space.  The ``VLLM_SHM_MAX_SEGMENT_SIZE`` env var overrides
    this heuristic.
    """
    if envs.VLLM_SHM_MAX_SEGMENT_SIZE is not None:
        return envs.VLLM_SHM_MAX_SEGMENT_SIZE

    try:
        stat = os.statvfs("/dev/shm")
        available = stat.f_frsize * stat.f_bavail
        # 90 % leaves 10 % headroom for other processes.
        return int(available * 0.9)
    except Exception as e:
        logger.warning("Failed to get /dev/shm space: %s, using default 64MB", e)

    return _DEFAULT_SHM_MAX_SEGMENT_SIZE


def _tensor_to_bytes(tensor: torch.Tensor) -> tuple[np.ndarray, int]:
    flat = tensor.view(torch.uint8).reshape(-1).numpy()
    return flat, flat.nbytes


def _bytes_to_tensor(data: bytes | np.ndarray, torch_dtype: str, shape: list) -> torch.Tensor:
    dtype = getattr(torch, torch_dtype.replace("torch.", ""))
    t = torch.frombuffer(bytearray(data), dtype=torch.uint8).view(dtype)
    return t.reshape(shape)


def _should_use_shm(tensor: torch.Tensor) -> bool:
    """Check if a tensor should be transferred via SHM instead of inline."""
    if not isinstance(tensor, torch.Tensor):
        return False
    return tensor.nelement() * tensor.element_size() > _SHM_TENSOR_THRESHOLD


def _extract_tensors(
    val: object,
    field_name: str,
    tensor_fields: list,
) -> object:
    """Walk a value, extract large tensors for SHM transfer.

    Replaces each large tensor with a ``__shm_placeholder__`` dict so the
    consumer can restore them in order.  Small tensors and non-tensor values
    pass through unchanged so they are pickled inline through the MessageQueue.

    Supports the container shapes that pipelines return via
    ``DiffusionOutput.output``:

    * bare ``torch.Tensor``
    * dicts (e.g. Cosmos3 ``{"image": ..., "video": ...}``)
    * lists / tuples (e.g. LTX2 / DreamID ``(video, audio)``)

    Each extracted tensor is appended to ``tensor_fields`` as a
    ``(field_name, tensor, use_shm)`` tuple, matching the ordering contract
    that :func:`_restore_placeholders` consumes on the receiver side.
    """
    if isinstance(val, torch.Tensor):
        if _should_use_shm(val):
            tensor_fields.append((field_name, val, True))
            return {_SHM_PLACEHOLDER_KEY: True}
        # Small tensor: keep inline
        tensor_fields.append((field_name, val, False))
        return val
    if isinstance(val, dict):
        return {k: _extract_tensors(v, field_name, tensor_fields) for k, v in val.items()}
    if isinstance(val, list):
        return [_extract_tensors(v, field_name, tensor_fields) for v in val]
    if isinstance(val, tuple):
        return tuple(_extract_tensors(v, field_name, tensor_fields) for v in val)
    return val


def pack_diffusion_output_shm(output: object, result_mq: MessageQueue, ack_mq: MessageQueue | None = None) -> None:
    """Send a output through result_mq, streaming large tensors
    chunk-by-chunk via SHM.

    Small tensors (<= _SHM_TENSOR_THRESHOLD) are sent inline with the output
    object. Large tensors are streamed via SHM with chunk-by-chunk protocol.

    Supports container types in ``DiffusionOutput.output`` (dicts, lists,
    tuples of tensors) — each large tensor inside the container is replaced
    with a placeholder and streamed individually.

    Protocol:
        1. Send output object (small tensors inline, large tensors replaced
           with placeholders)
        2. Send header with field descriptors (only for SHM fields)
        3. For each SHM chunk:
           - Create SHM, write data, send chunk info
           - Wait for ACK from consumer
           - Unlink SHM
    """
    diff_output = output if isinstance(output, DiffusionOutput) else getattr(output, "result", None)
    if not isinstance(diff_output, DiffusionOutput):
        # Not a DiffusionOutput, send directly without SHM
        result_mq.enqueue(output)
        result_mq.enqueue({"__shm_fields__": []})
        return

    max_chunk_size = _get_max_shm_segment_size()

    # Walk every DiffusionOutput field, extracting large tensors for SHM.
    # tensor_fields entries: (field_name, tensor, use_shm)
    tensor_fields = []
    for field_name in ("output", "trajectory_latents", "trajectory_timesteps", "trajectory_log_probs"):
        val = getattr(diff_output, field_name, None)
        if val is not None:
            stripped = _extract_tensors(val, field_name, tensor_fields)
            setattr(diff_output, field_name, stripped)

    # Build header with only SHM field descriptors
    shm_fields = []
    for field_name, tensor, use_shm in tensor_fields:
        if use_shm:
            tensor_cpu = tensor.detach().cpu().contiguous()
            nbytes = tensor_cpu.nelement() * tensor_cpu.element_size()
            num_chunks = (nbytes + max_chunk_size - 1) // max_chunk_size
            shm_fields.append(
                {
                    "field": field_name,
                    "shape": list(tensor_cpu.shape),
                    "torch_dtype": str(tensor_cpu.dtype),
                    "total_nbytes": nbytes,
                    "num_chunks": num_chunks,
                    "_tensor_cpu": tensor_cpu,
                }
            )

    # Send output and header
    header = {"__shm_fields__": [{k: v for k, v in fd.items() if k != "_tensor_cpu"} for fd in shm_fields]}
    result_mq.enqueue(output)
    result_mq.enqueue(header)

    # If no SHM fields, we're done
    if not shm_fields:
        return

    # Allocate ONE reusable SHM segment at max_chunk_size and stream every
    # chunk through it.  The ACK protocol guarantees the consumer has finished
    # reading before the producer overwrites the buffer, so a single segment
    # is safe.  This eliminates the per-chunk shm_open + ftruncate + mmap +
    # munmap + shm_unlink syscall storm of the original implementation.
    shm_name = ""
    try:
        shm = shared_memory.SharedMemory(create=True, size=max_chunk_size)
        shm_name = shm.name
        shm.close()  # keep the segment alive by name, not by local handle
    except Exception as e:
        logger.warning(
            "SHM segment alloc failed (%d bytes): %s, falling back to inline for all chunks",
            max_chunk_size,
            e,
        )

    use_shm = bool(shm_name)

    try:
        for fd in shm_fields:
            flat, total_nbytes = _tensor_to_bytes(fd["_tensor_cpu"])
            offset = 0

            for i in range(fd["num_chunks"]):
                chunk_size = min(max_chunk_size, total_nbytes - offset)

                if use_shm:
                    # Open the reusable segment → write → close.
                    shm = shared_memory.SharedMemory(name=shm_name)
                    try:
                        np.copyto(
                            np.ndarray((chunk_size,), dtype=np.uint8, buffer=shm.buf),
                            flat[offset : offset + chunk_size],
                        )
                    finally:
                        shm.close()

                    result_mq.enqueue(
                        {
                            "__shm_chunk__": True,
                            "name": shm_name,
                            "size": chunk_size,
                            "field": fd["field"],
                            "chunk_index": i,
                        }
                    )

                    # Wait for ACK before overwriting the buffer.
                    if ack_mq is not None:
                        ack_mq.dequeue(timeout=_SHM_ACK_TIMEOUT)
                        logger.debug("Received ACK for %s chunk %d", fd["field"], i)
                else:
                    # No SHM — fall back to inline transfer.
                    logger.debug("Sending %s chunk %d inline (%d bytes)", fd["field"], i, chunk_size)
                    result_mq.enqueue(
                        {
                            "__shm_chunk__": True,
                            "__inline__": True,
                            "data": bytes(flat[offset : offset + chunk_size]),
                            "size": chunk_size,
                            "field": fd["field"],
                            "chunk_index": i,
                        }
                    )
                    if ack_mq is not None:
                        ack_mq.dequeue(timeout=_SHM_ACK_TIMEOUT)

                offset += chunk_size
    finally:
        if shm_name:
            try:
                shared_memory.SharedMemory(name=shm_name).unlink()
            except FileNotFoundError:
                pass


_DEQUEUE_TIMEOUT_S = 5.0


def _dequeue_with_failure_check(mq, timeout, is_failed_fn=None):
    import zmq

    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        if deadline is None:
            chunk_t = _DEQUEUE_TIMEOUT_S
        else:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("dequeue timed out")
            chunk_t = min(_DEQUEUE_TIMEOUT_S, remaining)
        try:
            return mq.dequeue(timeout=chunk_t)
        except (TimeoutError, zmq.error.Again):
            if is_failed_fn is not None and is_failed_fn():
                raise EngineDeadError()
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("dequeue timed out")


def _restore_placeholders(
    val: object,
    tensor_list: list,
    consumed: list,
) -> object:
    """Replace ``__shm_placeholder__`` markers with tensors in order.

    Consumers call this once per DiffusionOutput field with the list of
    tensors reconstructed for that field.  The function walks the value tree
    and substitutes each placeholder dict with the next unused tensor from
    ``tensor_list``, maintaining the ordering established by
    :func:`_extract_tensors`.

    *consumed* is a single-element list used as a mutable counter; it
    is advanced by 1 for each consumed tensor.
    """
    if isinstance(val, dict) and _SHM_PLACEHOLDER_KEY in val:
        tensor = tensor_list[consumed[0]]
        consumed[0] += 1
        return tensor
    if isinstance(val, dict):
        return {k: _restore_placeholders(v, tensor_list, consumed) for k, v in val.items()}
    if isinstance(val, list):
        return [_restore_placeholders(v, tensor_list, consumed) for v in val]
    if isinstance(val, tuple):
        return tuple(_restore_placeholders(v, tensor_list, consumed) for v in val)
    return val


def unpack_diffusion_output_shm(
    result_mq: MessageQueue,
    ack_mq: MessageQueue | None = None,
    timeout: float | None = None,
    is_failed_fn: Callable[[], bool] | None = None,
) -> object:
    """Receive a output from result_mq, reassembling streamed tensors.

    Small tensors are received inline with the output object.
    Large tensors are reassembled from SHM chunks.

    This function handles all dequeue operations internally, making it symmetric
    with pack_diffusion_output_shm which handles all enqueue operations.

    Protocol:
        1. Dequeue response object
        2. Dequeue header with SHM field descriptors
        3. For each SHM field:
           - For each chunk:
             - Dequeue chunk info
             - Read data from SHM
             - Close SHM
             - Send ACK to producer
           - Reconstruct tensor and set on output

    Args:
        result_mq: Result message queue to dequeue from
        ack_mq: Optional ACK message queue for flow control
        timeout: Optional timeout in seconds for the initial response dequeue.
            If None, blocks indefinitely.

    Returns:
        The received object (DiffusionOutput or other type)
    """
    # Dequeue response and header
    response = _dequeue_with_failure_check(result_mq, timeout, is_failed_fn)
    header = _dequeue_with_failure_check(result_mq, _SHM_ACK_TIMEOUT, is_failed_fn)

    # Check if header indicates any SHM fields
    if not isinstance(header, dict):
        return response

    shm_fields = header.get("__shm_fields__", [])
    if not shm_fields:
        # No SHM fields, all data is inline
        return response

    diff_output = response if isinstance(response, DiffusionOutput) else getattr(response, "result", None)
    if not isinstance(diff_output, DiffusionOutput):
        return response

    # Reconstruct every streamed tensor, keeping them ordered per field
    # so _restore_placeholders can walk the original value tree and insert
    # tensors at the correct positions.
    tensors_by_field: dict[str, list[torch.Tensor]] = {}
    for fd in shm_fields:
        buf = bytearray(fd["total_nbytes"])
        offset = 0

        for chunk_idx in range(fd["num_chunks"]):
            chunk_msg = _dequeue_with_failure_check(result_mq, _SHM_ACK_TIMEOUT, is_failed_fn)
            if not isinstance(chunk_msg, dict) or not chunk_msg.get("__shm_chunk__"):
                raise RuntimeError(f"Expected SHM chunk, got: {type(chunk_msg)}")

            # Inline fallback: data sent directly via MessageQueue
            size = chunk_msg["size"]
            if chunk_msg.get("__inline__"):
                buf[offset : offset + size] = chunk_msg["data"]
                logger.debug("Received inline chunk %d (%d bytes) for %s", chunk_idx, size, fd["field"])
            else:
                shm = shared_memory.SharedMemory(name=chunk_msg["name"])
                try:
                    buf[offset : offset + size] = shm.buf[:size]
                    logger.debug("Received SHM chunk %d (%d bytes) for %s", chunk_idx, size, fd["field"])
                finally:
                    shm.close()

            offset += size
            if ack_mq is not None:
                ack_mq.enqueue({"status": "chunk_processed"})

        tensor = _bytes_to_tensor(buf, fd["torch_dtype"], fd["shape"])
        tensors_by_field.setdefault(fd["field"], []).append(tensor)

    # Walk each field's value tree and substitute placeholders with the
    # reconstructed tensors (preserving container structure for Cosmos3,
    # LTX2, DreamID, etc.).
    for field_name, tensor_list in tensors_by_field.items():
        val = getattr(diff_output, field_name)
        restored = _restore_placeholders(val, tensor_list, [0])
        setattr(diff_output, field_name, restored)

    return response
