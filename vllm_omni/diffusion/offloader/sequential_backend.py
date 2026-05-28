# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn
from torch.distributed._tensor import DTensor  # type: ignore[attr-defined]
from vllm.logger import init_logger

from vllm_omni.diffusion.hooks import HookRegistry, ModelHook
from vllm_omni.platforms import current_omni_platform

from .base import OffloadBackend, OffloadConfig, OffloadGranularity
from .module_collector import ModuleDiscovery, PipelineModules

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
        use_hsdp: bool = False,
    ):
        # Modules to offload to CPU before this module runs
        self.offload_targets = offload_targets
        self.device = device
        self.pin_memory = pin_memory
        self.use_hsdp = use_hsdp

    @staticmethod
    def _move_params(
        module: nn.Module,
        target_device: torch.device,
        *,
        non_blocking: bool = False,
        pin_memory: bool = False,
    ) -> None:
        """Move module parameters and buffers to device.

        This cls method specifically prevents recursion device movement,
        E.g., Cache-DiT CachedBlocks has attr `transformer` as a ref to original
        transformer blocks, thus `module.to(device)` will fail for recursion calling,
        refer to
        https://github.com/vipshop/cache-dit/blob/v1.2.3/src/cache_dit/caching/cache_blocks/__init__.py#L83
        """
        for p in module.parameters():
            if p.data.device != target_device:
                data = p.data.to(target_device, non_blocking=non_blocking)
                if pin_memory and target_device.type == "cpu" and not isinstance(data, DTensor):
                    data = data.pin_memory()
                p.data = data
        for b in module.buffers():
            if b.device != target_device:
                data = b.data.to(target_device, non_blocking=non_blocking)
                if pin_memory and target_device.type == "cpu" and not isinstance(data, DTensor):
                    data = data.pin_memory()
                b.data = data

    def _to_cpu(self, module: nn.Module) -> None:
        try:
            param = next(module.parameters())
        except StopIteration:
            return

        if param.device.type == "cpu":
            return

        # XPU's allocator doesn't respect stream dependencies in empty_cache,
        # so non-blocking copies can race with cache eviction. Use blocking
        # copies on XPU to avoid NULL pointer errors during DMA.
        non_blocking = not self.use_hsdp and not current_omni_platform.is_xpu()
        self._move_params(
            module,
            torch.device("cpu"),
            non_blocking=non_blocking,
            pin_memory=self.pin_memory,
        )
        current_omni_platform.empty_cache()

    def _to_gpu(self, module: nn.Module) -> None:
        try:
            if next(module.parameters()).device == self.device:
                return
        except StopIteration:
            return

        self._move_params(module, self.device, non_blocking=False)

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
    pipeline: nn.Module,
    device: torch.device,
    *,
    pin_memory: bool = True,
    use_hsdp: bool = False,
) -> list[nn.Module]:
    """Install model-level CPU-offload hooks on ``pipeline``.

    Discovers offloadable submodules, reads the pipeline's
    ``_offload_granularity`` (defaults to :attr:`OffloadGranularity.GROUPED`
    -- see :class:`OffloadGranularity`), preloads/evicts modules as
    appropriate for that mode, and registers the per-module hooks.

    Returns the list of modules that now carry hooks. Pass it to
    :func:`remove_sequential_offload` to tear everything down.
    """
    modules = ModuleDiscovery.discover(pipeline)
    if not modules.dits:
        logger.warning("No DiT/transformer modules found, skipping model-level offloading")
        return []
    if not modules.encoders:
        logger.warning("No encoder modules found, skipping model-level offloading")
        return []

    # Resident modules are always pinned to GPU regardless of granularity.
    for res, name in zip(modules.resident_modules, modules.resident_names):
        try:
            res.to(device)
        except Exception as exc:
            logger.warning("Failed to move resident module '%s' to GPU: %s", name, exc)

    granularity = getattr(pipeline, "_offload_granularity", OffloadGranularity.GROUPED)
    if granularity is OffloadGranularity.STRICT:
        return _apply_strict_offload(modules, device, pin_memory, use_hsdp)
    return _apply_grouped_offload(modules, device, pin_memory, use_hsdp)


def remove_sequential_offload(modules: list[nn.Module]) -> None:
    """Remove sequential offloading hooks and unwraps any method-wraps from modules.

    Args:
        modules: Modules to remove hooks from

    Example:
        >>> all_modules = [*dit_modules, *encoder_modules]
        >>> remove_sequential_offload(all_modules)
    """
    for module in modules:
        registry: HookRegistry | None = getattr(module, "_hook_registry", None)
        if registry is None:
            continue
        registry.unwrap_all_methods()
        registry.remove_hook(SequentialOffloadHook._HOOK_NAME)
        logger.debug("Removed offload hook from %s", module.__class__.__name__)


def _apply_grouped_offload(
    modules: PipelineModules,
    device: torch.device,
    pin_memory: bool,
    use_hsdp: bool,
) -> list[nn.Module]:
    """2-group mutual exclusion: dits vs encoders; VAEs/auxiliaries resident on GPU."""
    for enc in modules.encoders:
        enc.to(device)
    for vae in modules.vaes:
        try:
            vae.to(device, non_blocking=True)
        except Exception as exc:
            logger.debug("Failed to move VAE to GPU: %s", exc)

    # DiT hooks evict encoders + sibling dits; encoder hooks evict dits only.
    for i, dit_mod in enumerate(modules.dits):
        other_dits = [d for j, d in enumerate(modules.dits) if j != i]
        _register_offload_hook(dit_mod, modules.encoders + other_dits, device, pin_memory, use_hsdp)
    for enc in modules.encoders:
        _register_offload_hook(enc, modules.dits, device, pin_memory, use_hsdp)

    logger.info(
        "Model-level offloading enabled (grouped): %s <-> %s%s",
        ", ".join(modules.dit_names),
        ", ".join(modules.encoder_names),
        f"; resident on GPU: {', '.join(modules.resident_names)}" if modules.resident_names else "",
    )
    return [*modules.dits, *modules.encoders]


def _apply_strict_offload(
    modules: PipelineModules,
    device: torch.device,
    pin_memory: bool,
    use_hsdp: bool,
) -> list[nn.Module]:
    """Full N-way mutual exclusion + VAE ``.decode``/``.encode`` method wrap."""
    all_modules: list[nn.Module] = [*modules.dits, *modules.encoders, *modules.auxiliaries, *modules.vaes]
    all_names: list[str] = [
        *modules.dit_names,
        *modules.encoder_names,
        *modules.auxiliary_names,
        *modules.vae_names,
    ]

    # Bulk-move every participant to CPU; only the active one will live on GPU.
    for mod in all_modules:
        try:
            SequentialOffloadHook._move_params(mod, torch.device("cpu"))
        except Exception as exc:
            logger.debug("Failed to move %s to CPU: %s", mod.__class__.__name__, exc)
    current_omni_platform.empty_cache()

    for module in all_modules:
        offload_targets = [m for m in all_modules if m is not module]
        _register_offload_hook(module, offload_targets, device, pin_memory, use_hsdp)

    # VAEs are invoked via .decode()/.encode() rather than __call__, so the
    # pre_forward hook never fires on them. Wrap those methods so the swap
    # runs before each call. Skipped silently if the method is absent.
    wrapped_vae_methods: list[str] = []
    for vae_mod, vae_name in zip(modules.vaes, modules.vae_names):
        registry = HookRegistry.get_or_create(vae_mod)
        for method_name in ("decode", "encode"):
            if hasattr(vae_mod, method_name):
                registry.wrap_method(method_name)
                wrapped_vae_methods.append(f"{vae_name}.{method_name}")

    logger.info(
        "Model-level offloading enabled (strict): %s%s%s",
        ", ".join(all_names),
        f"; VAE method-wrap: {', '.join(wrapped_vae_methods)}" if wrapped_vae_methods else "",
        f"; resident on GPU: {', '.join(modules.resident_names)}" if modules.resident_names else "",
    )
    return all_modules


def _register_offload_hook(
    module: nn.Module,
    offload_targets: list[nn.Module],
    device: torch.device,
    pin_memory: bool,
    use_hsdp: bool,
) -> None:
    registry = HookRegistry.get_or_create(module)
    registry.register_hook(
        SequentialOffloadHook._HOOK_NAME,
        SequentialOffloadHook(
            offload_targets=offload_targets,
            device=device,
            pin_memory=pin_memory,
            use_hsdp=use_hsdp,
        ),
    )
    logger.debug("Registered offload hook for %s", module.__class__.__name__)


class ModelLevelOffloadBackend(OffloadBackend):
    """Thin wrapper around :func:`apply_sequential_offload`.

    Granularity is selected per-pipeline via ``_offload_granularity``;
    see :class:`OffloadGranularity`.
    """

    def __init__(self, config: OffloadConfig, device: torch.device):
        super().__init__(config, device)
        self._offload_modules: list[nn.Module] = []

    def enable(self, pipeline: nn.Module) -> None:
        if self.enabled:
            logger.warning("ModelLevelOffloadBackend already enabled")
            return
        self._offload_modules = apply_sequential_offload(
            pipeline,
            self.device,
            pin_memory=self.config.pin_cpu_memory,
            use_hsdp=self.config.use_hsdp,
        )
        self.enabled = bool(self._offload_modules)

    def disable(self) -> None:
        if not self.enabled:
            return
        remove_sequential_offload(self._offload_modules)
        self._offload_modules.clear()
        self.enabled = False
        logger.info("Model-level offloading disabled")
