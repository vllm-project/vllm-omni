# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent CUDA storage sharing; importing this package does not start CUDA.

Callers supply finalized read-only weights and matching rank/GPU identities.
They route opaque metadata over their existing trusted control plane and report
uncertain delivery/peer failures to the backend. No socket or thread is created.
Model binding and process supervision remain with callers; this package does
not replace the request KV connector.
"""

from .backend import ExportSession, ImportedWeights, TorchCudaIpcBackend
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
    "WeightManifest",
    "WeightSharingBackend",
    "WeightSharingError",
    "WeightSpec",
]
