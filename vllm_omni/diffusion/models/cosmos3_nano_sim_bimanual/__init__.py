# SPDX-License-Identifier: Apache-2.0
"""Cosmos3-Nano-Sim-Bimanual diffusion model family."""

from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.action_contract import Cosmos3NanoSimBimanualActionSchema
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.config import Cosmos3NanoSimBimanualManifest
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.control_contract import (
    Cosmos3NanoSimBimanualActionConditioning,
    Cosmos3NanoSimBimanualControlVideoConditioning,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.geometry import (
    Cosmos3NanoSimBimanualGeometry,
    Cosmos3NanoSimBimanualResolutionPolicy,
    resolve_cosmos3_nano_sim_bimanual_geometry,
)

__all__ = [
    "Cosmos3NanoSimBimanualActionSchema",
    "Cosmos3NanoSimBimanualActionConditioning",
    "Cosmos3NanoSimBimanualControlVideoConditioning",
    "Cosmos3NanoSimBimanualGeometry",
    "Cosmos3NanoSimBimanualManifest",
    "Cosmos3NanoSimBimanualPipeline",
    "Cosmos3NanoSimTransferPipeline",
    "Cosmos3NanoSimBimanualResolutionPolicy",
    "get_cosmos3_nano_sim_bimanual_post_process_func",
    "get_cosmos3_nano_sim_bimanual_pre_process_func",
    "resolve_cosmos3_nano_sim_bimanual_geometry",
]


def __getattr__(name: str):
    if name in {
        "Cosmos3NanoSimBimanualPipeline",
        "get_cosmos3_nano_sim_bimanual_pre_process_func",
        "get_cosmos3_nano_sim_bimanual_post_process_func",
    }:
        from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual import pipeline_cosmos3_nano_sim_bimanual

        return getattr(pipeline_cosmos3_nano_sim_bimanual, name)
    if name == "Cosmos3NanoSimTransferPipeline":
        from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual import pipeline_cosmos3_nano_sim_transfer

        return getattr(pipeline_cosmos3_nano_sim_transfer, name)
    raise AttributeError(name)
