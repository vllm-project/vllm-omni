# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU offloading utilities for diffusion models.

This module provides mutual-exclusion CPU offloading between DiT and encoders.
When enable_cpu_offload is enabled:
- Text encoders run on GPU while DiT is on CPU
- DiT runs on GPU while encoders are offloaded to CPU

This allows running large models on limited GPU memory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


class SequentialOffloader:
    """Sequential offloader: DiT and encoders take turns on GPU.

    Uses PyTorch's forward pre-hooks to automatically swap models:
    - Before encoder runs: move DiT modules to CPU, move encoder to GPU
    - Before DiT runs: move encoders to CPU, move active DiT to GPU

    This ensures only one large model group is on GPU at a time.
    """

    def __init__(
        self,
        dits: list[nn.Module],
        encoders: list[nn.Module],
        device: torch.device,
        pin_memory: bool = True,
    ):
        assert all(isinstance(m, nn.Module) for m in dits), "All dits must be nn.Module"
        assert all(isinstance(m, nn.Module) for m in encoders), "All encoders must be nn.Module"
        self.dits = dits
        self.encoders = encoders
        self.device = device
        self.pin_memory = pin_memory
        self._handles: list = []

    def _to_cpu(self, module: nn.Module) -> None:
        """Move module to CPU with optional memory pinning."""
        # Skip if already on CPU
        try:
            param = next(module.parameters())
            if param.device.type == "cpu":
                return
        except StopIteration:
            return

        previous_device = param.device
        module.to("cpu", non_blocking=True)

        # Release allocator blocks when tensors leave the GPU.
        if previous_device.type != "cpu":
            current_omni_platform.empty_cache()

        if self.pin_memory:
            for p in module.parameters():
                if p.data.device.type == "cpu" and not p.data.is_pinned():
                    p.data = p.data.pin_memory()

    def _to_gpu(self, module: nn.Module) -> None:
        """Move module to GPU."""
        # Skip if already on target device
        try:
            if next(module.parameters()).device == self.device:
                return
        except StopIteration:
            return

        module.to(self.device, non_blocking=True)

    def _dit_pre_hook(self, module: nn.Module, args: tuple) -> None:
        """Before DiT forward: offload encoders, load DiT."""
        for enc in self.encoders:
            self._to_cpu(enc)
        self._to_gpu(module)

        current_omni_platform.synchronize()

        logger.debug("Swapped: encoders -> CPU, DiT -> GPU")

    def _encoder_pre_hook(self, module: nn.Module, args: tuple) -> None:
        """Before encoder forward: offload DiT, load encoder."""
        for dit_mod in self.dits:
            self._to_cpu(dit_mod)
        self._to_gpu(module)

        current_omni_platform.synchronize()

        logger.debug("Swapped: DiT -> CPU, encoder -> GPU")

    def register(self) -> None:
        """Register forward pre-hooks on DiT and encoders."""
        # Hook on each DiT-like module
        for dit_mod in self.dits:
            h = dit_mod.register_forward_pre_hook(self._dit_pre_hook)
            self._handles.append(h)
            logger.debug("Registered offload hook for %s", dit_mod.__class__.__name__)

        # Hook on each encoder
        for enc in self.encoders:
            h = enc.register_forward_pre_hook(self._encoder_pre_hook)
            self._handles.append(h)
            logger.debug("Registered offload hook for %s", enc.__class__.__name__)

    def remove(self) -> None:
        """Remove all hooks."""
        for h in self._handles:
            h.remove()
        self._handles = []


def apply_offload_hooks(
    model: nn.Module,
    od_config: OmniDiffusionConfig,
    *,
    device: torch.device | None = None,
) -> None:
    """Apply mutual-exclusion offload hooks based on config.

    When enable_cpu_offload is enabled, DiT and encoders swap GPU access:
    - Encoders (text_encoder, text_encoder_2, text_encoder_3, image_encoder)
      run on GPU while DiT is on CPU
    - DiT runs on GPU while encoders are on CPU

    Args:
        model: Diffusion pipeline model
        od_config: OmniDiffusionConfig with offload settings
    """
    if not getattr(od_config, "enable_cpu_offload", False):
        return

    # Find DiT/transformer modules
    dit_modules: list[nn.Module] = []
    dit_names: list[str] = []
    candidate_attrs = ["transformer", "transformer_2", "dit"]
    for attr in candidate_attrs:
        if not hasattr(model, attr):
            continue
        module_obj = getattr(model, attr)
        if module_obj is None:
            continue

        assert isinstance(module_obj, nn.Module), f"Expected {attr} to be nn.Module, got {type(module_obj)!r}"

        if module_obj in dit_modules:
            continue

        dit_modules.append(module_obj)
        dit_names.append(attr)

    if not dit_modules:
        logger.warning("enable_cpu_offload enabled but no transformer/dit/unet found")
        return

    if device is None:
        try:
            device = next(dit_modules[0].parameters()).device
        except StopIteration:
            try:
                device = current_omni_platform.get_torch_device()
            except (NotImplementedError, AttributeError):
                device = torch.device("cpu")

    # Collect all encoders
    encoders: list[nn.Module] = []
    encoder_names: list[str] = []
    for attr in ["text_encoder", "text_encoder_2", "text_encoder_3", "image_encoder"]:
        if hasattr(model, attr) and getattr(model, attr) is not None:
            encoders.append(getattr(model, attr))
            encoder_names.append(attr)

    if not encoders:
        logger.warning("enable_cpu_offload enabled but no encoders found")
        return

    # Initial state: keep DiT modules on CPU (encoders typically run first)
    pin = getattr(od_config, "pin_cpu_memory", True)
    for dit_mod in dit_modules:
        dit_mod.to("cpu")

    current_omni_platform.empty_cache()

    if pin:
        for dit_mod in dit_modules:
            for p in dit_mod.parameters():
                if p.data.device.type == "cpu" and not p.data.is_pinned():
                    p.data = p.data.pin_memory()

    # Register sequential offload hooks
    SequentialOffloader(dit_modules, encoders, device, pin).register()

    logger.info(
        "CPU offload enabled: %s <-> %s (mutual exclusion)",
        ", ".join(dit_names),
        ", ".join(encoder_names),
    )
