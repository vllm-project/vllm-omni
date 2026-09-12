# SPDX-License-Identifier: Apache-2.0
"""Cosmos-Dreams (Cosmos3-Interactive) diffusion model family."""

from vllm_omni.diffusion.models.cosmos_dreams.action_contract import CosmosDreamsActionSchema
from vllm_omni.diffusion.models.cosmos_dreams.config import CosmosDreamsManifest
from vllm_omni.diffusion.models.cosmos_dreams.control_contract import (
    CosmosDreamsActionConditioning,
    CosmosDreamsControlVideoConditioning,
)
from vllm_omni.diffusion.models.cosmos_dreams.geometry import (
    CosmosDreamsGeometry,
    CosmosDreamsResolutionPolicy,
    resolve_cosmos_dreams_geometry,
)

__all__ = [
    "CosmosDreamsActionSchema",
    "CosmosDreamsActionConditioning",
    "CosmosDreamsControlVideoConditioning",
    "CosmosDreamsGeometry",
    "CosmosDreamsManifest",
    "CosmosDreamsPipeline",
    "CosmosDreamsResolutionPolicy",
    "get_cosmos_dreams_post_process_func",
    "get_cosmos_dreams_pre_process_func",
    "resolve_cosmos_dreams_geometry",
]


def __getattr__(name: str):
    if name in {
        "CosmosDreamsPipeline",
        "get_cosmos_dreams_pre_process_func",
        "get_cosmos_dreams_post_process_func",
    }:
        from vllm_omni.diffusion.models.cosmos_dreams import pipeline_cosmos_dreams

        return getattr(pipeline_cosmos_dreams, name)
    raise AttributeError(name)
