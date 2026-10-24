# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent CUDA storage sharing; importing this package does not start CUDA.

Callers supply finalized read-only weights, matching rank/GPU identities, and
an endpoint in a private directory. Model binding and process supervision are
owned by callers. See the contracts and backend module docstrings for lifetime
requirements; this package does not replace the request KV connector.
"""

from .backend import ExportSession, ImportedWeights, TorchCudaIpcBackend
from .contracts import (
    CloseResult,
    CompatibilityError,
    Identity,
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
