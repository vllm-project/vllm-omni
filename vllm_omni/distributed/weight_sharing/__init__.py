# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent device storage sharing; importing this package starts no device.

Callers supply finalized read-only weights and matching rank/device identities.
They route opaque metadata over their existing trusted control plane and report
uncertain delivery/peer failures to the backend. No socket or thread is created.
Model binding and process supervision remain with callers; this package does
not replace the request KV connector.
"""

from .backend import ExportSession, ImportedWeights, TorchCudaIpcBackend, TorchIpcBackend
from .contracts import (
    CloseResult,
    CompatibilityError,
    Identity,
    Limits,
    SessionContext,
    SessionSnapshot,
    SessionState,
    TensorSpec,
    ThreadOwnershipError,
    WeightManifest,
    WeightSharingBackend,
    WeightSharingError,
    WeightSpec,
)
from .torch_cuda_ipc_codec import TorchCudaIpcCodec
from .torch_npu_ipc_codec import TorchNpuIpcCodec

__all__ = [
    "CloseResult",
    "CompatibilityError",
    "ExportSession",
    "Identity",
    "Limits",
    "ImportedWeights",
    "SessionContext",
    "SessionSnapshot",
    "SessionState",
    "TensorSpec",
    "ThreadOwnershipError",
    "TorchCudaIpcBackend",
    "TorchCudaIpcCodec",
    "TorchIpcBackend",
    "TorchNpuIpcCodec",
    "WeightManifest",
    "WeightSharingBackend",
    "WeightSharingError",
    "WeightSpec",
]
