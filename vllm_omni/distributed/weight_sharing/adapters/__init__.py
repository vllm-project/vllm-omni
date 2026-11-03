# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-model adapters that bind the CUDA IPC backend to concrete model backbones.

Each adapter defines the shared backbone boundary ("fusion ABI"), the canonical
tensor name space used in the manifest, and the provider/consumer entry points.
"""

from .hunyuan_image3 import (
    BACKBONE_ATTR,
    LAYOUT_ID,
    attach_imported_weights,
    build_weight_spec,
    claim_backbone_export,
    collect_shared_bindings,
    filter_shared_checkpoint_weights,
    is_shared_checkpoint_name,
    is_shared_param,
    open_backbone_import,
    publish_backbone,
    shared_backbone_names,
    shared_param_full_names,
    validate_supported_layout,
)

__all__ = [
    "BACKBONE_ATTR",
    "LAYOUT_ID",
    "attach_imported_weights",
    "build_weight_spec",
    "claim_backbone_export",
    "collect_shared_bindings",
    "filter_shared_checkpoint_weights",
    "is_shared_checkpoint_name",
    "is_shared_param",
    "open_backbone_import",
    "publish_backbone",
    "shared_backbone_names",
    "shared_param_full_names",
    "validate_supported_layout",
]
