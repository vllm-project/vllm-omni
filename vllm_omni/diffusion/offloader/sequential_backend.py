# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.hooks import HookRegistry, ModelHook
from vllm_omni.platforms import current_omni_platform

from .base import OffloadBackend, OffloadConfig
from .module_collector import ModuleDiscovery

logger = init_logger(__name__)


class SequentialOffloadHook(ModelHook):
    """Hook for sequential offloading with mutual exclusion on encoder and DiT modules.

    To be used as a model-level (or "component-level") of CPU offloading method;
    When a module's forward is called, this hook offloads target modules to CPU
    and loads the current module to GPU.
    """

    _HOOK_NAME = "sequential_offload"

    def __init__(
        self,
        offload_targets: list[nn.Module],
        device: torch.device,
        pin_memory: bool = True,
    ):
        # Modules to offload to CPU before this module runs
        self.offload_targets = offload_targets
        self.device = device
        self.pin_memory = pin_memory

    @staticmethod
    def _move_params(module: nn.Module, device: torch.device) -> None:
        """Move module parameters and buffers to device.

        This cls method specifically prevents recursion device movement,
        E.g., Cache-DiT CachedBlocks has attr `transformer` as a ref to original
        transformer blocks, thus `module.to(device)` will fail for recursion calling,
        refer to
        https://github.com/vipshop/cache-dit/blob/v1.2.3/src/cache_dit/caching/cache_blocks/__init__.py#L83
        """
        for p in module.parameters():
            if p.data.device != device:
                p.data = p.data.to(device, non_blocking=True)
        for b in module.buffers():
            if b.device != device:
                b.data = b.data.to(device, non_blocking=True)

    def _to_cpu(self, module: nn.Module) -> None:
        """Move module to CPU."""
        try:
            param = next(module.parameters())
        except StopIteration:
            return

        # Skip if already on CPU
        if param.device.type == "cpu":
            return

        self._move_params(module, torch.device("cpu"))
        current_omni_platform.empty_cache()

        if self.pin_memory:
            for p in module.parameters():
                if p.data.device.type == "cpu" and not p.data.is_pinned():
                    p.data = p.data.pin_memory()

    def _to_gpu(self, module: nn.Module) -> None:
        """Move module to GPU."""
        try:
            # Skip if already on target device
            if next(module.parameters()).device == self.device:
                return
        except StopIteration:
            return

        self._move_params(module, self.device)

    def pre_forward(self, module: nn.Module, *args, **kwargs) -> tuple[tuple, dict]:
        # Offload target modules to CPU
        for target in self.offload_targets:
            self._to_cpu(target)

        # Load current module to GPU
        self._to_gpu(module)
        current_omni_platform.synchronize()

        logger.debug(
            "Swapped: %s -> CPU, %s -> %s, free memory: %.4f GB",
            [t.__class__.__name__ for t in self.offload_targets],
            module.__class__.__name__,
            f"{self.device.type}:{self.device.index}",
            current_omni_platform.get_free_memory() / 1024 / 1024 / 1024,
        )

        return args, kwargs


def apply_sequential_offload(
    dit_modules: list[nn.Module],
    encoder_modules: list[nn.Module],
    device: torch.device,
    pin_memory: bool = True,
) -> None:
    """Apply sequential offloading hooks to DiT and encoder modules.

    Registers hooks on modules to implement mutual-exclusion GPU allocation.
        - Before DiT runs, encoders are offloaded to CPU.
        - Before encoders run, DiT is offloaded to CPU.

    Args:
        dit_modules: DiT/transformer modules to register hooks on
        encoder_modules: Encoder modules to register hooks on
        device: Target GPU device for loading
        pin_memory: Whether to pin CPU memory for faster transfers

    Example:
        >>> apply_sequential_offload(
        ...     dit_modules=[pipeline.transformer],
        ...     encoder_modules=[pipeline.text_encoder, pipeline.vae],
        ...     device=torch.device("cuda:0"),
        ... )
        >>> # Modules of pipeline now automatically swap between CPU and GPU
    """
    # Register hooks on DiT modules (offload encoders when DiT runs)
    for dit_mod in dit_modules:
        registry = HookRegistry.get_or_create(dit_mod)
        hook = SequentialOffloadHook(
            offload_targets=encoder_modules,
            device=device,
            pin_memory=pin_memory,
        )
        registry.register_hook(SequentialOffloadHook._HOOK_NAME, hook)
        logger.debug("Registered offload hook for %s", dit_mod.__class__.__name__)

    # Register hooks on encoders (offload DiTs when encoder runs)
    for enc in encoder_modules:
        registry = HookRegistry.get_or_create(enc)
        hook = SequentialOffloadHook(
            offload_targets=dit_modules,
            device=device,
            pin_memory=pin_memory,
        )
        registry.register_hook(SequentialOffloadHook._HOOK_NAME, hook)
        logger.debug("Registered offload hook for %s", enc.__class__.__name__)


def remove_sequential_offload(modules: list[nn.Module]) -> None:
    """Remove sequential offloading hooks from modules.

    Args:
        modules: Modules to remove hooks from

    Example:
        >>> all_modules = [*dit_modules, *encoder_modules]
        >>> remove_sequential_offload(all_modules)
    """
    for module in modules:
        registry: HookRegistry | None = getattr(module, "_hook_registry", None)
        if registry is not None:
            registry.remove_hook(SequentialOffloadHook._HOOK_NAME)
            logger.debug("Removed offload hook from %s", module.__class__.__name__)


class ModelLevelOffloadBackend(OffloadBackend):
    """Model-level (sequential) offloading backend.

    Uses SequentialOffloadHook registered via HookRegistry for automatic module swapping.
    """

    def __init__(self, config: OffloadConfig, device: torch.device):
        super().__init__(config, device)
        self._offload_modules: list[nn.Module] = []  # Track modules with hooks

    def enable(self, pipeline: nn.Module) -> None:
        if self.enabled:
            logger.warning("ModelLevelOffloadBackend already enabled")
            return

        modules = ModuleDiscovery.discover(pipeline)
        if not modules.dits:
            logger.warning("No DiT/transformer modules found, skipping model-level offloading")
            return
        if not modules.encoders:
            logger.warning("No encoder modules found, skipping model-level offloading")
            return

        # Move encoders to GPU
        for enc in modules.encoders:
            enc.to(self.device)

        # Move VAE to GPU if available
        if modules.vae is not None:
            try:
                modules.vae.to(self.device, non_blocking=True)
            except Exception as exc:
                logger.debug("Failed to move VAE to GPU: %s", exc)

        # Apply sequential offloading hooks
        apply_sequential_offload(
            dit_modules=modules.dits,
            encoder_modules=modules.encoders,
            device=self.device,
            pin_memory=self.config.pin_cpu_memory,
        )

        # Track modules for cleanup
        self._offload_modules = [*modules.dits, *modules.encoders]

        self.enabled = True

        logger.info(
            "Model-level offloading enabled: %s <-> %s (mutual exclusion)",
            ", ".join(modules.dit_names),
            ", ".join(modules.encoder_names),
        )

    def disable(self) -> None:
        if not self.enabled:
            return

        remove_sequential_offload(self._offload_modules)

        self._offload_modules.clear()
        self.enabled = False
        logger.info("Model-level offloading disabled")
