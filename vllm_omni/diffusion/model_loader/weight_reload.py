# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Prepare diffusion pipelines for upstream layerwise weight reloading.

Pipelines opt in with ``supports_weight_reload`` after their ``load_weights``
path has been validated with parameter weight loaders and repeated updates.
Metadata is captured before checkpoint loading or weight postprocessing, with
the execution device selected explicitly even for CPU-constructed pipelines.
The upstream WTE owns the subsequent reload lifecycle.
"""

import torch
from torch import nn
from vllm.model_executor.model_loader.reload import record_metadata_for_reloading

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.offloader.config import offload_enabled


def _uses_merged_lora_backend(od_config: OmniDiffusionConfig) -> bool:
    wt_cfg = od_config.weight_transfer_config
    backend = wt_cfg.get("backend") if isinstance(wt_cfg, dict) else getattr(wt_cfg, "backend", None)
    return backend == "omni_lora_ipc" or getattr(od_config, "lora_application", "runtime") == "merged"


def prepare_model_for_weight_reload(
    model: nn.Module,
    od_config: OmniDiffusionConfig,
    device: torch.device,
) -> None:
    """Capture construction metadata for an explicitly supported WTE pipeline."""
    if od_config.weight_transfer_config is None:
        return
    if _uses_merged_lora_backend(od_config):
        from vllm_omni.lora.capability import resolve_merged_lora_spec

        resolve_merged_lora_spec(model, stage="diffusion", require_protocol=True)
    if not getattr(model, "supports_weight_reload", False):
        raise ValueError(f"{type(model).__name__} has not enabled upstream weight reloading")
    if od_config.quantization_config is not None:
        raise ValueError("Diffusion weight reloading currently requires unquantized weights")
    if od_config.parallel_config.use_hsdp or offload_enabled(od_config):
        raise ValueError("Diffusion weight reloading does not support HSDP or inference weight offloading")
    if od_config.cache_backend not in (None, "none"):
        raise ValueError("Diffusion weight reloading requires cache_backend='none'")
    with device:
        record_metadata_for_reloading(model)
    model._omni_weight_reload_prepared = True
