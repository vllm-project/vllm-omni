# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Owned CUDA streams for graphs whose library workspaces must stay isolated."""

import weakref

import torch


def create_cuda_stream(device: torch.device) -> torch.cuda.Stream:
    """Create a nonblocking stream outside PyTorch's round-robin stream pool.

    Pooled capture streams can alias even while their Python objects are live.
    Graphs then share cuBLAS workspaces, which cannot be used concurrently or
    cleared while a peer graph still references them. Keep this stream alive
    for the lifetime of its graphs; its finalizer releases the CUDA handle.
    """
    from cuda.bindings import runtime

    from vllm_omni.platforms import current_omni_platform

    previous_device = current_omni_platform.current_device()
    current_omni_platform.set_device(device)
    try:
        error, handle = runtime.cudaStreamCreateWithFlags(runtime.cudaStreamNonBlocking)
        if error != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Cannot create CUDA stream: {error}")
        try:
            stream = torch.cuda.ExternalStream(int(handle), device=device)
        except BaseException:
            runtime.cudaStreamDestroy(handle)
            raise
        weakref.finalize(stream, runtime.cudaStreamDestroy, handle)
        return stream
    finally:
        current_omni_platform.set_device(torch.device(device.type, previous_device))
