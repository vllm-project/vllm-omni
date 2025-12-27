# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU offloading utilities for diffusion models.

This module provides hook-based CPU offloading that works automatically
with any pipeline - no per-pipeline code changes needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


# =============================================================================
# Hook-Based Offloading (Recommended - Zero Pipeline Changes)
# =============================================================================


class OffloadHooks:
    """Manages CPU offload hooks for a module.

    Uses PyTorch's forward hooks to automatically:
    1. Move module to GPU before forward pass
    2. Move module back to CPU after forward pass

    This allows CPU offloading without modifying any pipeline code.
    """

    def __init__(self, module: nn.Module, execution_device: torch.device, pin_memory: bool = True):
        self.module = module
        self.execution_device = execution_device
        self.pin_memory = pin_memory
        self._handles: list = []
        self._name = module.__class__.__name__

    def _pre_forward_hook(self, module: nn.Module, args: tuple) -> None:
        """Move module to GPU before forward."""
        module.to(self.execution_device)
        logger.debug("Hook: moved %s to %s", self._name, self.execution_device)

    def _post_forward_hook(self, module: nn.Module, args: tuple, output: Any) -> Any:
        """Move module back to CPU after forward."""
        module.to("cpu")
        if self.pin_memory and torch.cuda.is_available():
            for p in module.parameters():
                if p.data.device.type == "cpu" and not p.data.is_pinned():
                    p.data = p.data.pin_memory()
        logger.debug("Hook: moved %s back to CPU", self._name)
        return output

    def register(self) -> OffloadHooks:
        """Register the hooks on the module."""
        h1 = self.module.register_forward_pre_hook(self._pre_forward_hook)
        h2 = self.module.register_forward_hook(self._post_forward_hook)
        self._handles = [h1, h2]
        logger.info("Registered offload hooks for %s", self._name)
        return self

    def remove(self) -> None:
        """Remove all hooks."""
        for h in self._handles:
            h.remove()
        self._handles = []
        logger.debug("Removed offload hooks for %s", self._name)


def _get_execution_device(model: nn.Module) -> torch.device:
    """Find the device where main computation happens (transformer/dit/unet)."""
    for attr in ["transformer", "dit", "unet"]:
        if hasattr(model, attr) and getattr(model, attr) is not None:
            try:
                return next(getattr(model, attr).parameters()).device
            except StopIteration:
                continue
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _wrap_method_with_offload(module: nn.Module, method_name: str, exec_device: torch.device, pin_memory: bool) -> None:
    """Wrap a method to move module to GPU before call and back to CPU after.

    This is needed for methods like VAE.decode() that don't go through forward().
    """
    original_method = getattr(module, method_name)

    def wrapped_method(*args, **kwargs):
        # Move to GPU
        module.to(exec_device)
        logger.debug("Method wrap: moved %s to %s for %s()", module.__class__.__name__, exec_device, method_name)
        try:
            result = original_method(*args, **kwargs)
        finally:
            # Move back to CPU
            module.to("cpu")
            if pin_memory and torch.cuda.is_available():
                for p in module.parameters():
                    if p.data.device.type == "cpu" and not p.data.is_pinned():
                        p.data = p.data.pin_memory()
            logger.debug("Method wrap: moved %s back to CPU", module.__class__.__name__)
        return result

    setattr(module, method_name, wrapped_method)
    logger.debug("Wrapped %s.%s with offload", module.__class__.__name__, method_name)


def apply_offload_hooks(model: nn.Module, od_config: OmniDiffusionConfig) -> list[OffloadHooks]:
    """Apply offload hooks to model components based on config.

    This is the main entry point for hook-based offloading.
    Call once after model creation - works for ALL pipelines automatically.

    Args:
        model: Diffusion pipeline model
        od_config: OmniDiffusionConfig with offload flags

    Returns:
        List of OffloadHooks objects (keep reference to remove later if needed)
    """
    hooks = []
    pin = getattr(od_config, "pin_cpu_memory", True)
    exec_device = _get_execution_device(model)

    # Text encoder(s)
    if getattr(od_config, "text_encoder_cpu_offload", False):
        for attr in ["text_encoder", "text_encoder_2", "text_encoder_3"]:
            if hasattr(model, attr) and getattr(model, attr) is not None:
                module = getattr(model, attr)
                module.to("cpu")  # Initial offload
                if pin and torch.cuda.is_available():
                    for p in module.parameters():
                        if not p.data.is_pinned():
                            p.data = p.data.pin_memory()
                hook = OffloadHooks(module, exec_device, pin).register()
                hooks.append(hook)
                logger.info("CPU offload (hooks): %s", attr)

    # VAE - needs method wrapping since decode/encode don't use forward()
    if getattr(od_config, "vae_cpu_offload", False):
        if hasattr(model, "vae") and model.vae is not None:
            model.vae.to("cpu")
            if pin and torch.cuda.is_available():
                for p in model.vae.parameters():
                    if not p.data.is_pinned():
                        p.data = p.data.pin_memory()
            # Wrap decode/encode methods since they don't go through forward()
            for method in ["decode", "encode"]:
                if hasattr(model.vae, method):
                    _wrap_method_with_offload(model.vae, method, exec_device, pin)
            # Also register forward hook for any direct forward() calls
            hook = OffloadHooks(model.vae, exec_device, pin).register()
            hooks.append(hook)
            logger.info("CPU offload (hooks+method wrap): vae")

    # Image encoder
    if getattr(od_config, "image_encoder_cpu_offload", False):
        if hasattr(model, "image_encoder") and model.image_encoder is not None:
            model.image_encoder.to("cpu")
            if pin and torch.cuda.is_available():
                for p in model.image_encoder.parameters():
                    if not p.data.is_pinned():
                        p.data = p.data.pin_memory()
            hook = OffloadHooks(model.image_encoder, exec_device, pin).register()
            hooks.append(hook)
            logger.info("CPU offload (hooks): image_encoder")

    # DiT/Transformer (usually keep on GPU, but support offload if requested)
    if getattr(od_config, "dit_cpu_offload", False):
        for attr in ["transformer", "dit", "unet"]:
            if hasattr(model, attr) and getattr(model, attr) is not None:
                module = getattr(model, attr)
                module.to("cpu")
                if pin and torch.cuda.is_available():
                    for p in module.parameters():
                        if not p.data.is_pinned():
                            p.data = p.data.pin_memory()
                # For dit, use CPU as exec device since it's offloaded
                hook = OffloadHooks(module, torch.device("cuda"), pin).register()
                hooks.append(hook)
                logger.info("CPU offload (hooks): %s", attr)

    return hooks


# =============================================================================
# Legacy Manual Offloading (kept for backward compatibility)
# =============================================================================


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


def apply_cpu_offload(model: nn.Module, od_config: OmniDiffusionConfig) -> None:
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
        return any(
            [
                getattr(od_config, "text_encoder_cpu_offload", False),
                getattr(od_config, "vae_cpu_offload", False),
                getattr(od_config, "image_encoder_cpu_offload", False),
            ]
        )

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
