# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU offloading utilities for diffusion models."""

import torch
from torch import nn
from vllm.logger import init_logger

logger = init_logger(__name__)


def offload_to_cpu(module: nn.Module, pin_memory: bool = True) -> None:
    """Move module to CPU with optional pinned memory for fast transfer.
    
    Args:
        module: PyTorch module to offload
        pin_memory: If True, pin memory for faster GPU transfers
    """
    if module is None:
        return
    
    module.to("cpu")
    
    if pin_memory and torch.cuda.is_available():
        for p in module.parameters():
            if not p.data.is_pinned():
                p.data = p.data.pin_memory()
    
    logger.debug("Offloaded %s to CPU (pinned=%s)", module.__class__.__name__, pin_memory)


def move_to_device(module: nn.Module, device: torch.device) -> None:
    """Move module to device if not already there.
    
    Args:
        module: PyTorch module to move
        device: Target device
    """
    if module is None:
        return
    
    try:
        current = next(module.parameters()).device
        if current != device:
            module.to(device)
            logger.debug("Moved %s to %s", module.__class__.__name__, device)
    except StopIteration:
        pass


def apply_cpu_offload(model: nn.Module, od_config) -> None:
    """Apply CPU offloading to diffusion model components based on config.
    
    Args:
        model: Diffusion pipeline model
        od_config: OmniDiffusionConfig with offload flags
    """
    pin = getattr(od_config, "pin_cpu_memory", True)
    
    # Text encoder(s)
    if getattr(od_config, "text_encoder_cpu_offload", False):
        for attr in ["text_encoder", "text_encoder_2", "text_encoder_3"]:
            if hasattr(model, attr) and getattr(model, attr) is not None:
                offload_to_cpu(getattr(model, attr), pin)
                logger.info("CPU offloaded: %s", attr)
    
    # Image encoder (for img2img, inpainting)
    if getattr(od_config, "image_encoder_cpu_offload", False):
        if hasattr(model, "image_encoder") and model.image_encoder is not None:
            offload_to_cpu(model.image_encoder, pin)
            logger.info("CPU offloaded: image_encoder")
    
    # VAE
    if getattr(od_config, "vae_cpu_offload", False):
        if hasattr(model, "vae") and model.vae is not None:
            offload_to_cpu(model.vae, pin)
            logger.info("CPU offloaded: vae")
    
    # DiT/Transformer (usually keep on GPU for speed)
    if getattr(od_config, "dit_cpu_offload", False):
        for attr in ["transformer", "dit", "unet"]:
            if hasattr(model, attr) and getattr(model, attr) is not None:
                offload_to_cpu(getattr(model, attr), pin)
                logger.info("CPU offloaded: %s", attr)


class CPUOffloadMixin:
    """Mixin for pipelines to handle CPU offload during forward pass.
    
    Provides methods to temporarily move components to GPU for computation
    and offload back to CPU afterwards.
    """
    
    def _get_execution_device(self) -> torch.device:
        """Get the device where main computation happens."""
        # Try to find a module that's on GPU
        for attr in ["transformer", "dit", "unet"]:
            if hasattr(self, attr):
                module = getattr(self, attr)
                if module is not None:
                    try:
                        return next(module.parameters()).device
                    except StopIteration:
                        continue
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _should_offload(self) -> bool:
        """Check if offloading is enabled."""
        od_config = getattr(self, "od_config", None)
        if od_config is None:
            return False
        return any([
            getattr(od_config, "text_encoder_cpu_offload", False),
            getattr(od_config, "vae_cpu_offload", False),
            getattr(od_config, "image_encoder_cpu_offload", False),
        ])
    
    def _ensure_on_device(self, module: nn.Module) -> None:
        """Ensure module is on execution device."""
        if module is None:
            return
        device = self._get_execution_device()
        move_to_device(module, device)
    
    def _offload_after_use(self, module: nn.Module) -> None:
        """Offload module back to CPU after use."""
        if module is None or not self._should_offload():
            return
        od_config = getattr(self, "od_config", None)
        pin = getattr(od_config, "pin_cpu_memory", True) if od_config else True
        offload_to_cpu(module, pin)

