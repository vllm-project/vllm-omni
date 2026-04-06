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

import torch

from vllm_omni.diffusion.data import DiffusionOutput

_SHM_TENSOR_THRESHOLD = 0  # Always use SHM for CUDA tensor safety


def _tensor_to_shm(tensor: torch.Tensor) -> dict[str, Any]:
    """Copy a tensor into POSIX shared memory and return a metadata handle.

    The shared memory segment remains alive after this call (the local fd is
    closed, but the segment persists until ``_tensor_from_shm`` unlinks it).

    BFloat16 and other numpy-incompatible dtypes are stored as raw uint8 bytes
    and reconstructed using the stored ``torch_dtype``.
    """
    from multiprocessing import shared_memory

    import numpy as np

    orig_dtype = tensor.dtype
    tensor = tensor.detach().cpu().contiguous()
    # BFloat16 (and some other dtypes) are not natively supported by numpy.
    # Use a raw uint8 byte view so data can be round-tripped without precision loss.
    try:
        arr = tensor.numpy()
        use_raw_bytes = False
    except TypeError:
        arr = tensor.view(torch.uint8).numpy()
        use_raw_bytes = True
    nbytes = arr.nbytes
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf[:nbytes])
    np.copyto(shm_arr, arr)
    handle = {
        "__tensor_shm__": True,
        "name": shm.name,
        "shape": list(tensor.shape),
        "torch_dtype": str(orig_dtype),
        "numpy_dtype": str(arr.dtype),
        "nbytes": nbytes,
        "raw_bytes": use_raw_bytes,
    }
    shm.close()
    return handle


def _tensor_from_shm(handle: dict[str, Any]) -> torch.Tensor:
    """Reconstruct a tensor from a shared-memory handle and free the segment."""
    from multiprocessing import shared_memory

    import numpy as np

    shm = shared_memory.SharedMemory(name=handle["name"])
    try:
        np_dtype = np.dtype(handle["numpy_dtype"])
        if handle.get("raw_bytes"):
            # Data was stored as raw uint8 bytes (e.g. BFloat16 round-trip).
            byte_arr = np.ndarray(handle["nbytes"], dtype=np.uint8, buffer=shm.buf[: handle["nbytes"]])
            raw = torch.from_numpy(byte_arr.copy())
        else:
            arr = np.ndarray(handle["shape"], dtype=np_dtype, buffer=shm.buf[: handle["nbytes"]])
            raw = torch.from_numpy(arr.copy())
    finally:
        shm.close()
        shm.unlink()
    # Restore the original torch dtype (handles BF16 raw-byte round-trip).
    torch_dtype_str = handle["torch_dtype"].replace("torch.", "")
    torch_dtype = getattr(torch, torch_dtype_str)
    if raw.dtype != torch_dtype or handle.get("raw_bytes"):
        raw = raw.view(torch_dtype).reshape(handle["shape"])
    return raw


def _pack_tensor_if_large(val: torch.Tensor) -> torch.Tensor | dict:
    """Replace a tensor with an SHM handle if it exceeds the threshold."""
    if val.nelement() * val.element_size() > _SHM_TENSOR_THRESHOLD:
        return _tensor_to_shm(val)
    return val


def _unpack_if_shm_handle(val: object) -> object:
    """Reconstruct a tensor from an SHM handle dict, or return as-is."""
    if isinstance(val, dict) and val.get("__tensor_shm__"):
        return _tensor_from_shm(val)
    return val


def _pack_diffusion_fields(output: DiffusionOutput) -> DiffusionOutput:
    if output.output is not None and isinstance(output.output, torch.Tensor):
        output.output = _pack_tensor_if_large(output.output)
    if output.trajectory_latents is not None and isinstance(output.trajectory_latents, torch.Tensor):
        output.trajectory_latents = _pack_tensor_if_large(output.trajectory_latents)
    if output.trajectory_timesteps is not None and isinstance(output.trajectory_timesteps, torch.Tensor):
        output.trajectory_timesteps = _pack_tensor_if_large(output.trajectory_timesteps)
    if output.trajectory_log_probs is not None and isinstance(output.trajectory_log_probs, torch.Tensor):
        output.trajectory_log_probs = _pack_tensor_if_large(output.trajectory_log_probs)
    return output


def pack_diffusion_output_shm(output: object) -> object:
    """Replace large tensors in diffusion worker outputs with SHM handles.

    Supports either a bare ``DiffusionOutput`` or a wrapper object carrying one
    in ``.result`` (for example ``RunnerOutput``).
    """
    if isinstance(output, DiffusionOutput):
        return _pack_diffusion_fields(output)

    result = getattr(output, "result", None)
    if isinstance(result, DiffusionOutput):
        output.result = _pack_diffusion_fields(result)
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

    result = getattr(output, "result", None)
    if isinstance(result, DiffusionOutput):
        output.result = _unpack_diffusion_fields(result)
    return output
