# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Typed diffusion adapters for runtime-ready final-layout artifacts."""

from .contracts import (
    FinalLayoutArtifactSpec,
    FinalLayoutLoaderIdentity,
    FinalLayoutParallelIdentity,
    FinalLayoutRequest,
    ImplementationIdentity,
)
from .identity_adapter import (
    FinalLayoutIdentityContext,
    build_final_layout_identity,
)
from .producers import (
    FINAL_LAYOUT_BF16_POLICY,
    FINAL_LAYOUT_BF16_SPEC,
    FinalLayoutBF16Producer,
)
from .restorer import FinalLayoutTensorRestorer
from .source_identity import NodeSourceDigestCache, PreparedWeightSource, WeightSourceKind

__all__ = [
    "FINAL_LAYOUT_BF16_POLICY",
    "FINAL_LAYOUT_BF16_SPEC",
    "FinalLayoutArtifactSpec",
    "FinalLayoutBF16Producer",
    "FinalLayoutIdentityContext",
    "FinalLayoutLoaderIdentity",
    "FinalLayoutParallelIdentity",
    "FinalLayoutRequest",
    "FinalLayoutTensorRestorer",
    "ImplementationIdentity",
    "NodeSourceDigestCache",
    "PreparedWeightSource",
    "WeightSourceKind",
    "build_final_layout_identity",
]
