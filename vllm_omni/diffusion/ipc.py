# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""IPC utilities for transferring large tensors via POSIX shared memory.

Used by Hop1 (GPU worker <-> scheduler) to avoid pickling large video tensors
through the MessageQueue. Tensors above ``_SHM_TENSOR_THRESHOLD`` are copied
into a named shared-memory segment; only a lightweight metadata dict is
serialised through the queue.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from vllm_omni.diffusion.data import DiffusionOutput

_SHM_TENSOR_THRESHOLD = 1_000_000  # 1 MB
DIFFUSION_RPC_RESULT_ENVELOPE = "diffusion_rpc_result"


def _array_to_shm(array: np.ndarray) -> dict[str, Any]:
    """Copy a contiguous NumPy-compatible array into shared memory."""
    from multiprocessing import shared_memory

    if array.dtype.hasobject:
        raise TypeError("NumPy object arrays cannot be transferred through raw shared memory")

    array = np.ascontiguousarray(array)
    nbytes = array.nbytes
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    shm_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf[:nbytes])
    np.copyto(shm_array, array)
    handle = {
        "name": shm.name,
        "shape": list(array.shape),
        "numpy_dtype": str(array.dtype),
        "nbytes": nbytes,
    }
    shm.close()
    return handle


def _array_from_shm(handle: dict[str, Any]) -> np.ndarray:
    """Copy an array from shared memory, then close and unlink its segment."""
    from multiprocessing import shared_memory

    shm = shared_memory.SharedMemory(name=handle["name"])
    try:
        array = np.ndarray(
            handle["shape"],
            dtype=np.dtype(handle["numpy_dtype"]),
            buffer=shm.buf[: handle["nbytes"]],
        ).copy()
    finally:
        shm.close()
        shm.unlink()
    return array


def _tensor_to_shm(
    tensor: torch.Tensor,
    d2h_stream: torch.Stream | None = None,
) -> dict[str, Any]:
    """Copy a tensor into POSIX shared memory and return a metadata handle.

    The shared memory segment remains alive after this call (the local fd is
    closed, but the segment persists until ``_tensor_from_shm`` unlinks it).

    If *d2h_stream* is provided, the D2H copy uses ``copy_()`` with
    ``pin_memory=True`` on that stream instead of the synchronous ``.cpu()``
    path.  The caller must synchronize *d2h_stream* after all tensors are
    packed.
    """
    original_dtype = tensor.dtype
    if d2h_stream is not None:
        # Non-blocking D2H: copy on side stream to pinned CPU memory.
        old_stream = torch.accelerator.current_stream()
        torch.accelerator.set_stream(d2h_stream)
        try:
            t = tensor.detach()
            if original_dtype == torch.bfloat16:
                t = t.to(torch.float32)
            cpu = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
            cpu.copy_(t, non_blocking=True)
        finally:
            torch.accelerator.set_stream(old_stream)
        d2h_stream.synchronize()
        tensor = cpu
    else:
        tensor = tensor.detach().cpu().contiguous()
        if original_dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)
    handle = _array_to_shm(tensor.numpy())
    handle.update(
        {
            "__tensor_shm__": True,
            "torch_dtype": str(original_dtype),
        }
    )
    return handle


def _tensor_from_shm(handle: dict[str, Any]) -> torch.Tensor:
    """Reconstruct a tensor from a shared-memory handle and free the segment."""
    tensor = torch.from_numpy(_array_from_shm(handle))
    # Restore the original dtype if it differs from the numpy-compatible
    # dtype used for the SHM transfer (e.g. bfloat16 → float32 → bfloat16).
    torch_dtype_str = handle.get("torch_dtype", "")
    if torch_dtype_str:
        original_dtype = getattr(torch, torch_dtype_str.replace("torch.", ""), None)
        if original_dtype is not None and tensor.dtype != original_dtype:
            tensor = tensor.to(original_dtype)
    return tensor


def _pack_tensor_if_large(
    val: torch.Tensor,
    d2h_stream: torch.Stream | None = None,
) -> torch.Tensor | dict:
    """Replace a tensor with an SHM handle if it exceeds the threshold.

    Batch outputs are split into per-request views that share one storage.
    Pickle serializes a view's whole storage, so a batch of small views costs
    the full batch tensor once per request. Size the decision on the storage
    to keep those views off the wire; packing copies just the view.
    """
    view_bytes = val.nelement() * val.element_size()
    try:
        storage_bytes = val.untyped_storage().nbytes()
    except Exception:
        storage_bytes = view_bytes
    if max(view_bytes, storage_bytes) > _SHM_TENSOR_THRESHOLD:
        return _tensor_to_shm(val, d2h_stream=d2h_stream)
    return val


def _ndarray_to_shm(array: np.ndarray) -> dict[str, Any]:
    """Copy a contiguous NumPy array into POSIX shared memory."""
    handle = _array_to_shm(array)
    handle["__ndarray_shm__"] = True
    return handle


def _ndarray_from_shm(handle: dict[str, Any]) -> np.ndarray:
    """Reconstruct a NumPy array from shared memory and free the segment."""
    return _array_from_shm(handle)


def _pack_value_if_large(
    val: object,
    d2h_stream: torch.Stream | None = None,
) -> object:
    """Recursively replace large tensors with SHM handles.

    Walks the container shapes pipelines return as ``DiffusionOutput.output``:
    bare tensors, dicts (e.g. Cosmos3 ``{"image"/"video": ...}``), and
    tuples/lists (e.g. LTX2 and DreamID ``(video, audio)``). Other values pass
    through unchanged. ``_unpack_if_shm_handle`` must mirror these shapes — keep
    the two in sync.
    """
    if isinstance(val, torch.Tensor):
        return _pack_tensor_if_large(val, d2h_stream=d2h_stream)
    if isinstance(val, np.ndarray):
        return _ndarray_to_shm(val) if not val.dtype.hasobject and val.nbytes > _SHM_TENSOR_THRESHOLD else val
    if isinstance(val, dict):
        return {key: _pack_value_if_large(value, d2h_stream=d2h_stream) for key, value in val.items()}
    if isinstance(val, list):
        return [_pack_value_if_large(item, d2h_stream=d2h_stream) for item in val]
    if isinstance(val, tuple):
        return tuple(_pack_value_if_large(item, d2h_stream=d2h_stream) for item in val)
    return val


def _unpack_if_shm_handle(val: object) -> object:
    """Reconstruct tensors from SHM handles, mirroring ``_pack_value_if_large``."""
    if isinstance(val, dict) and val.get("__tensor_shm__"):
        return _tensor_from_shm(val)
    if isinstance(val, dict) and val.get("__ndarray_shm__"):
        return _ndarray_from_shm(val)
    if isinstance(val, dict):
        return {key: _unpack_if_shm_handle(value) for key, value in val.items()}
    if isinstance(val, list):
        return [_unpack_if_shm_handle(item) for item in val]
    if isinstance(val, tuple):
        return tuple(_unpack_if_shm_handle(item) for item in val)
    return val


def _pack_diffusion_fields(
    output: DiffusionOutput,
    d2h_stream: torch.Stream | None = None,
) -> DiffusionOutput:
    if output.output is not None:
        output.output = _pack_value_if_large(output.output, d2h_stream=d2h_stream)
    if output.trajectory_latents is not None and isinstance(output.trajectory_latents, torch.Tensor):
        output.trajectory_latents = _pack_tensor_if_large(output.trajectory_latents, d2h_stream=d2h_stream)
    if output.trajectory_timesteps is not None and isinstance(output.trajectory_timesteps, torch.Tensor):
        output.trajectory_timesteps = _pack_tensor_if_large(output.trajectory_timesteps, d2h_stream=d2h_stream)
    if output.trajectory_log_probs is not None and isinstance(output.trajectory_log_probs, torch.Tensor):
        output.trajectory_log_probs = _pack_tensor_if_large(output.trajectory_log_probs, d2h_stream=d2h_stream)
    return output


def _is_rpc_result_envelope(output: object) -> bool:
    return isinstance(output, dict) and output.get("type") == DIFFUSION_RPC_RESULT_ENVELOPE


def pack_diffusion_output_shm(
    output: object,
    d2h_stream: torch.Stream | None = None,
) -> object:
    """Replace large tensors in diffusion worker outputs with SHM handles.

    Supports a bare ``DiffusionOutput``, a wrapper object carrying one in
    ``.result`` (for example ``RunnerOutput``), an RPC result envelope carrying
    the diffusion output in ``["result"]``, a batch wrapper carrying
    ``RunnerOutput`` objects in ``.runner_outputs``, or a DP-tagged dict
    ``{"dp_rank": int, "output": DiffusionOutput}`` used by DP multi-concurrency.

    If *d2h_stream* is provided, D2H copies use that stream (non-blocking on
    the default stream).  The caller must synchronize *d2h_stream* afterward.
    """
    if isinstance(output, DiffusionOutput):
        return _pack_diffusion_fields(output, d2h_stream=d2h_stream)

    # DP multi-concurrency: {"dp_rank": int, "output": DiffusionOutput}
    if isinstance(output, dict) and "dp_rank" in output and "output" in output:
        inner = output["output"]
        if isinstance(inner, DiffusionOutput):
            output["output"] = _pack_diffusion_fields(inner, d2h_stream=d2h_stream)
        return output

    if _is_rpc_result_envelope(output):
        result = output.get("result")
        output["result"] = pack_diffusion_output_shm(result, d2h_stream=d2h_stream)
        return output

    result = getattr(output, "result", None)
    if isinstance(result, DiffusionOutput):
        output.result = _pack_diffusion_fields(result, d2h_stream=d2h_stream)

    runner_outputs = getattr(output, "runner_outputs", None)
    if isinstance(runner_outputs, list):
        for runner_output in runner_outputs:
            pack_diffusion_output_shm(runner_output, d2h_stream=d2h_stream)
    return output


def _unpack_diffusion_fields(output: DiffusionOutput) -> DiffusionOutput:
    output.output = _unpack_if_shm_handle(output.output)
    output.trajectory_latents = _unpack_if_shm_handle(output.trajectory_latents)
    output.trajectory_timesteps = _unpack_if_shm_handle(output.trajectory_timesteps)
    output.trajectory_log_probs = _unpack_if_shm_handle(output.trajectory_log_probs)
    return output


def unpack_diffusion_output_shm(output: object) -> object:
    """Reconstruct tensors from SHM handles in diffusion worker outputs."""
    if isinstance(output, DiffusionOutput):
        return _unpack_diffusion_fields(output)

    # DP multi-concurrency: {"dp_rank": int, "output": DiffusionOutput}
    if isinstance(output, dict) and "dp_rank" in output and "output" in output:
        inner = output["output"]
        if isinstance(inner, DiffusionOutput):
            output["output"] = _unpack_diffusion_fields(inner)
        return output

    if _is_rpc_result_envelope(output):
        result = output.get("result")
        output["result"] = unpack_diffusion_output_shm(result)
        return output

    result = getattr(output, "result", None)
    if isinstance(result, DiffusionOutput):
        output.result = _unpack_diffusion_fields(result)

    runner_outputs = getattr(output, "runner_outputs", None)
    if isinstance(runner_outputs, list):
        for runner_output in runner_outputs:
            unpack_diffusion_output_shm(runner_output)
    return output
