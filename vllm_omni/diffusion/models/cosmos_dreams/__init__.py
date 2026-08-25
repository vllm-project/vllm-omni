# SPDX-License-Identifier: Apache-2.0
"""Cosmos-Dreams (Cosmos3-Interactive) diffusion model family."""

from vllm_omni.diffusion.models.cosmos_dreams.action_contract import CosmosDreamsActionSchema
from vllm_omni.diffusion.models.cosmos_dreams.config import CosmosDreamsManifest
from vllm_omni.diffusion.models.cosmos_dreams.control_contract import (
    CosmosDreamsActionConditioning,
    CosmosDreamsControlVideoConditioning,
)

__all__ = [
    "AgiBotKeyboardController",
    "AgiBotKeyboardResampler",
    "AgiBotSceneState",
    "Cosmos3InteractivePipeline",
    "CosmosDreamsActionSchema",
    "CosmosDreamsActionConditioning",
    "CosmosDreamsControlVideoConditioning",
    "CosmosDreamsManifest",
    "CosmosDreamsOmniPipeline",
    "CosmosDreamsPipeline",
    "CosmosDreamsTransferPipeline",
    "CosmosDreamsTransferOmniPipeline",
    "Cosmos3TransferInteractivePipeline",
    "CosmosDreamsTickRuntime",
    "get_cosmos_dreams_post_process_func",
    "get_cosmos_dreams_pre_process_func",
]


def __getattr__(name: str):
    if name in {"AgiBotKeyboardController", "AgiBotKeyboardResampler", "AgiBotSceneState"}:
        from vllm_omni.diffusion.models.cosmos_dreams import controller

        return getattr(controller, name)
    if name == "CosmosDreamsTickRuntime":
        from vllm_omni.diffusion.models.cosmos_dreams import runtime

        return runtime.CosmosDreamsTickRuntime
    if name in {
        "CosmosDreamsPipeline",
        "CosmosDreamsOmniPipeline",
        "Cosmos3InteractivePipeline",
        "get_cosmos_dreams_pre_process_func",
        "get_cosmos_dreams_post_process_func",
    }:
        from vllm_omni.diffusion.models.cosmos_dreams import pipeline_cosmos_dreams

        return getattr(pipeline_cosmos_dreams, name)
    if name in {
        "CosmosDreamsTransferPipeline",
        "CosmosDreamsTransferOmniPipeline",
        "Cosmos3TransferInteractivePipeline",
    }:
        from vllm_omni.diffusion.models.cosmos_dreams import pipeline_cosmos_dreams_transfer

        return getattr(pipeline_cosmos_dreams_transfer, name)
    raise AttributeError(name)
