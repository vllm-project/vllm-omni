# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from itertools import chain
from typing import Any

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.hooks import HookRegistry, ModelHook
from vllm_omni.platforms import current_omni_platform

from .base import OffloadBackend, OffloadConfig, run_cleanup_steps
from .block_discovery import (
    get_blocks_attr_names,
    get_blocks_from_dit,
    set_blocks_attr_names,
)
from .component_utils import (
    clear_encoder_layerwise_state,
    get_encoder_block_groups,
    iter_streamable_dits,
    move_non_block_state_to_device,
    prepare_pipeline_components,
    set_encoder_layerwise_state,
)
from .config import DIT_COMPONENT
from .module_collector import ModuleDiscovery
from .offload_plan import OffloadPlan, get_offload_plan
from .tensor_utils import (
    clear_block_storage,
    clear_tensor_storage,
    describe_tensor_storage,
    flatten_physical_storage,
    group_named_tensors_by_dtype,
    is_materialized_tensor,
    materialization_probe,
    module_materialization_probe,
    restore_tensor_storage,
    set_tensor_storage,
    tensor_storage_metadata,
)

logger = init_logger(__name__)


class LayerwiseOffloadHook(ModelHook):
    """Hook for layerwise (transformer-block-wise) CPU offloading.

    The hook instance retains parameters for both the current registered block
    module and those for the next block, as well as flattened CPU tensors which
    record the parameters of the current block module, so that these parameters
    could be re-materialized on device in an overlapping way.
    This hook should be registered to each of the transformer blocks in DiT
    module(s) of the target pipeline.

    Based on implementations from:
    https://github.com/sgl-project/sglang/blob/v0.5.8/python/sglang/multimodal_gen/runtime/utils/layerwise_offload.py
    """

    _HOOK_NAME = "layerwise_offload"

    def __init__(
        self,
        next_block: nn.Module,
        device: torch.device,
        stream: current_omni_platform.Stream | None = None,
        pin_memory: bool = True,
        materialization_probe_tensor: torch.Tensor | None = None,
    ):
        assert isinstance(next_block, nn.Module), "transformer block must be type `torch.nn.Module`"

        self.next_block = next_block
        self.device = device
        self.copy_stream = stream or current_omni_platform.current_stream()
        self.pin_memory = pin_memory

        # Per-block synchronization primitive: set after H2D copy completes.
        self._prefetch_done: current_omni_platform.Event | None = None

        # Backward link to the hook that is responsible for prefetching *this* block's weights
        self._prev_hook: LayerwiseOffloadHook | None = None

        self.next_block_parameters: dict[str, nn.Parameter] = {}
        self.next_block_buffers: dict[str, torch.Tensor] = {}
        self.dtype_cpu_flattened_weights: dict[torch.dtype, torch.Tensor] = {}
        self.dtype_metadata: dict[torch.dtype, list[dict[str, Any]]] = {}
        self._materialization_probe = materialization_probe_tensor

    def initialize_hook(self, module: nn.Module) -> nn.Module:
        # This all happen during the hook instance being registered to hook registry;
        # the input module is kept intact
        module = super().initialize_hook(module)

        self.block_parameters: dict[str, nn.Parameter] = dict(module.named_parameters())
        self.block_buffers: dict[str, torch.Tensor] = dict(module.named_buffers())
        if self._materialization_probe is None:
            self._materialization_probe = materialization_probe(self.block_parameters, self.block_buffers)

        self.next_block_parameters: dict[str, nn.Parameter] = dict(self.next_block.named_parameters())
        self.next_block_buffers: dict[str, torch.Tensor] = dict(self.next_block.named_buffers())

        # Pre-allocate gpu tensors in a flattened way
        self.dtype_cpu_flattened_weights, self.dtype_metadata = LayerwiseOffloadHook._to_cpu(
            self.next_block_parameters,
            self.next_block_buffers,
            self.pin_memory,
        )

        return module

    @staticmethod
    def _to_cpu(
        params: dict[str, nn.Parameter],
        bufs: dict[str, torch.Tensor],
        pin_memory: bool = True,
    ) -> tuple[dict[torch.dtype, torch.Tensor], dict[torch.dtype, list[dict[str, Any]]]]:
        """Helper method to move block parameters and buffers to CPU, flattening by dtype.

        Consolidates parameters and buffers into contiguous CPU tensors grouped by dtype
        for GPU transfers. Replaces original tensors with empty placeholders.

        Returns:
            Tuple of
                flattened CPU tensors by dtype,
                metadata for reconstruction by dtype
        """
        dtype_cpu_flattened_weights: dict[torch.dtype, torch.Tensor] = {}
        # NOTE: order does matter
        dtype_metadata: dict[torch.dtype, list[dict[str, Any]]] = {}
        targets_to_offload: list[torch.Tensor] = []

        for dtype, named_weights in group_named_tensors_by_dtype(params, bufs).items():
            # total # of parameters + buffers
            specs = describe_tensor_storage(named_weights)
            total_numel = sum(spec.storage_numel for spec in specs)
            cpu_tensor = torch.empty(total_numel, dtype=dtype, device="cpu", pin_memory=pin_memory)

            current_offset = 0
            for spec in specs:
                flat_storage = flatten_physical_storage(spec.value, spec.storage_numel)
                cpu_tensor[current_offset : current_offset + spec.storage_numel].copy_(flat_storage)
                dtype_metadata.setdefault(dtype, []).append(
                    tensor_storage_metadata(spec, current_offset, include_device=True)
                )
                targets_to_offload.append(spec.target)
                current_offset += spec.storage_numel

            dtype_cpu_flattened_weights[dtype] = cpu_tensor

        # Do not mutate the module until every host master has been built.
        # Allocation/copy failures therefore leave the input module intact.
        clear_tensor_storage(targets_to_offload)

        return dtype_cpu_flattened_weights, dtype_metadata

    @property
    def is_materialized(self) -> bool:
        """Check whether this block's parameters hold real data on device."""
        return self._materialization_probe is None or is_materialized_tensor(self._materialization_probe)

    @torch.compiler.disable
    def prefetch_layer(self, non_blocking: bool = True) -> None:
        """Copy layer weights from CPU -> GPU.

        Pre-fetch target block in an asynchronous way with compute - memory copy overlap,
        with non_blocking set to True.
        """
        self.copy_stream.wait_stream(current_omni_platform.current_stream())

        layer_params = self.next_block_parameters
        layer_bufs = self.next_block_buffers

        evt = current_omni_platform.Event()
        gpu_weights: dict[torch.dtype, torch.Tensor] = {}

        with current_omni_platform.stream(self.copy_stream):
            for dtype, cpu_weight in self.dtype_cpu_flattened_weights.items():
                gpu_weight = torch.empty(cpu_weight.shape, dtype=dtype, device=self.device)
                gpu_weight.copy_(cpu_weight, non_blocking=non_blocking)
                gpu_weights[dtype] = gpu_weight

            evt.record(self.copy_stream)

        for dtype, ordered_metadata in self.dtype_metadata.items():
            # ordered_metadata: list[dict[str, Any]]
            gpu_weight = gpu_weights[dtype]

            for metadata in ordered_metadata:
                target_name = metadata["name"]
                target_param_or_buf = (
                    layer_params[target_name] if target_name in layer_params else layer_bufs[target_name]
                )

                set_tensor_storage(
                    target_param_or_buf,
                    torch.as_strided(
                        gpu_weight[metadata["offset"] : metadata["offset"] + metadata["numel"]],
                        size=metadata["shape"],
                        stride=metadata["stride"],
                    ),
                )

        self._prefetch_done = evt

    @torch.compiler.disable
    def offload_layer(self) -> None:
        """Free GPU memory for layer by replacing tensors with empty placeholders.
        This function does not actually offload weights from GPU back to CPU.
        """
        clear_block_storage(self.block_parameters, self.block_buffers, self._prefetch_done)
        self._prefetch_done = None

    @torch.compiler.disable
    def restore_next_block(self) -> None:
        """Detach the next block from this hook's host backing store."""
        for dtype, ordered_metadata in self.dtype_metadata.items():
            flat = self.dtype_cpu_flattened_weights[dtype]
            for metadata in ordered_metadata:
                value = torch.as_strided(
                    flat[metadata["offset"] : metadata["offset"] + metadata["numel"]],
                    size=metadata["shape"],
                    stride=metadata["stride"],
                )
                target_name = metadata["name"]
                target = (
                    self.next_block_parameters[target_name]
                    if target_name in self.next_block_parameters
                    else self.next_block_buffers[target_name]
                )
                restore_tensor_storage(target, value, device=metadata["device"])

    def pre_forward(self, module: nn.Module, *args: Any, **kwargs: Any) -> tuple[tuple, dict]:
        # if the previous hook was skipped and the weights are not on device,
        # (e.g. by cache-dit block caching), ask the previous hook to
        # synchronously prefetch *this* block's weights before computation
        if not self.is_materialized and self._prev_hook is not None:
            self._prev_hook.prefetch_layer(non_blocking=False)

        self.prefetch_layer(non_blocking=True)

        return args, kwargs

    def post_forward(self, module: nn.Module, output: Any) -> Any:
        self.offload_layer()

        return output


def apply_block_hook(
    module: nn.Module,
    next_block: nn.Module,
    device: torch.device,
    stream: current_omni_platform.Stream | None = None,
    pin_memory: bool = True,
    *,
    materialization_probe_tensor: torch.Tensor | None = None,
) -> LayerwiseOffloadHook:
    registry = HookRegistry.get_or_create(module)
    hook = LayerwiseOffloadHook(
        next_block,
        device,
        stream,
        pin_memory,
        materialization_probe_tensor,
    )
    registry.register_hook(LayerwiseOffloadHook._HOOK_NAME, hook)

    return hook


def remove_block_hook(module: nn.Module) -> None:
    registry: HookRegistry | None = getattr(module, "_hook_registry", None)
    if registry is not None:
        registry.remove_hook(LayerwiseOffloadHook._HOOK_NAME)
        logger.debug("Removed offload hook from %s", module.__class__.__name__)


def _install_layerwise_hook_group(
    blocks: list[nn.Module] | nn.ModuleList,
    device: torch.device,
    stream: Any,
    pin_memory: bool,
) -> list[LayerwiseOffloadHook]:
    """Install one circular hook ring and roll it back transactionally."""
    block_list = list(blocks)
    if len(block_list) <= 1:
        raise ValueError("A layerwise hook group requires at least two blocks")

    hooks: list[LayerwiseOffloadHook] = []
    hooked_blocks: list[nn.Module] = []
    probes = {id(block): module_materialization_probe(block) for block in block_list}
    try:
        for block, next_block in zip(
            chain((block_list[-1],), block_list[:-1]),
            block_list,
            strict=True,
        ):
            hooks.append(
                apply_block_hook(
                    block,
                    next_block,
                    device,
                    stream,
                    pin_memory,
                    materialization_probe_tensor=probes[id(block)],
                )
            )
            hooked_blocks.append(block)
    except BaseException:
        run_cleanup_steps(
            [
                *(("restoring a partially installed layerwise block", hook.restore_next_block) for hook in hooks),
                *(
                    ("removing a partially installed layerwise hook", lambda block=block: remove_block_hook(block))
                    for block in hooked_blocks
                ),
            ]
        )
        raise

    for index, hook in enumerate(hooks):
        hook._prev_hook = hooks[index - 1]
    return hooks


def enable_plan_encoder_layerwise_offload(
    module: nn.Module,
    name: str,
    plan: OffloadPlan | None,
    *,
    device: torch.device,
    stream: current_omni_platform.Stream,
    pin_memory: bool,
    stage_on_demand: bool = False,
    strict: bool = False,
) -> bool:
    """Apply rank-local layerwise hooks to plan-declared encoder stacks."""
    if getattr(module, "_omni_layerwise_enabled", False):
        return True

    hooks: list[LayerwiseOffloadHook] = []
    hooked_blocks: list[nn.Module] = []
    block_groups = get_encoder_block_groups(
        module,
        name,
        plan,
        strict=strict,
    )
    if not block_groups:
        return False
    try:
        for blocks in block_groups:
            group_hooks = _install_layerwise_hook_group(blocks, device, stream, pin_memory)
            hooks.extend(group_hooks)
            hooked_blocks.extend(blocks)
        if not stage_on_demand:
            move_non_block_state_to_device(module, block_groups, device)
    except BaseException:
        run_cleanup_steps(
            [
                *(("restoring a partially installed encoder block", hook.restore_next_block) for hook in hooks),
                *(
                    ("removing a partially installed encoder hook", lambda block=block: remove_block_hook(block))
                    for block in hooked_blocks
                ),
            ]
        )
        raise
    set_encoder_layerwise_state(
        module,
        hooks,
        block_groups,
    )
    logger.info(
        "Enabled rank-local layerwise offload for encoder %s (%d blocks across %d stacks)",
        name,
        sum(len(blocks) for blocks in block_groups),
        len(block_groups),
    )
    return True


def disable_plan_encoder_layerwise_offload(
    module: nn.Module,
    *,
    restore_weights: bool = True,
) -> None:
    """Remove hooks installed by :func:`enable_plan_encoder_layerwise_offload`."""
    if not getattr(module, "_omni_layerwise_enabled", False):
        return
    hooks = getattr(module, "_omni_layerwise_hooks", [])
    block_groups = getattr(module, "_omni_layerwise_block_groups", [])
    steps = []
    if restore_weights:
        steps.extend(("restoring an encoder block", hook.restore_next_block) for hook in hooks)
    steps.extend(
        ("removing an encoder hook", lambda block=block: remove_block_hook(block))
        for blocks in block_groups
        for block in blocks
    )
    cleanup_error = run_cleanup_steps(steps)
    if cleanup_error is not None:
        raise RuntimeError("Failed to fully disable encoder layerwise offload") from cleanup_error
    clear_encoder_layerwise_state(module)


class LayerWiseOffloadBackend(OffloadBackend):
    """Layer-wise (block-level) offloading backend.

    Implements sliding window offloading where only a small number of transformer
    blocks reside on GPU at a time. Blocks are prefetched asynchronously while
    previous blocks compute, and freed after use.
    """

    def __init__(self, config: OffloadConfig, device: torch.device):
        super().__init__(config, device)

        self.copy_stream = current_omni_platform.Stream()
        self._blocks: list[list[nn.Module]] = []
        self._dit_hooks: list[LayerwiseOffloadHook] = []
        self._hooked_dit_blocks: list[nn.Module] = []
        self._encoder_modules: list[nn.Module] = []
        self._staged_components: list[nn.Module] = []

    def enable(self, pipeline: nn.Module) -> None:
        try:
            self._enable(pipeline)
        except BaseException:
            try:
                self.disable()
            except BaseException:
                logger.exception("LayerWiseOffloadBackend cleanup failed while handling an enable failure")
            raise

    def _enable(self, pipeline: nn.Module) -> None:
        if self.enabled:
            logger.warning("LayerWiseOffloadBackend already enabled")
            return

        modules = ModuleDiscovery.discover(pipeline)
        plan = get_offload_plan(pipeline)
        if not modules.dits and self.config.offloads(DIT_COMPONENT):
            message = "No DiT/transformer modules found for selected DiT layerwise offload"
            if self.config.components is not None:
                raise ValueError(message)
            logger.warning(message)
            return

        def enable_encoder_blocks(
            module: nn.Module,
            name: str,
            component_plan: OffloadPlan | None,
            stage_on_demand: bool,
        ) -> bool:
            enabled = enable_plan_encoder_layerwise_offload(
                module,
                name,
                component_plan,
                device=self.device,
                stream=self.copy_stream,
                pin_memory=self.config.pin_cpu_memory,
                stage_on_demand=stage_on_demand,
                strict=self.config.components is not None,
            )
            if enabled:
                # Record each successful installation immediately so a later
                # component failure can remove these hooks transactionally.
                self._encoder_modules.append(module)
            return enabled

        prepare_pipeline_components(
            modules,
            self.config,
            plan,
            device=self.device,
            staged_components=self._staged_components,
            enable_encoder_blocks=enable_encoder_blocks,
        )

        if not self.config.offloads(DIT_COMPONENT):
            self.enabled = bool(self._encoder_modules or self._staged_components)
            if not self.enabled:
                raise ValueError(
                    "None of the selected layerwise offload components have "
                    "a model-declared streamable or on-demand plan"
                )
            return

        logger.info("Applying layer-wise offloading on %s", modules.dit_names)

        # Apply block-wise offloading hook for each of the blocks in DiT model(s)
        # Note that there might exist multiple DiT models in specific pipelines
        for dit_name, dit_module, blocks_attr_names, blocks in iter_streamable_dits(
            modules, self.config, self.device, plan
        ):
            num_blocks = len(blocks)
            if num_blocks <= 1:
                if self.config.components is not None:
                    raise ValueError(
                        f"Selected DiT {dit_name!r} requires at least two streamable layerwise-offload blocks"
                    )
                logger.warning(
                    "#Target layers (blocks) <= 1. Skipping offloading on %s (%s)",
                    dit_name,
                    dit_module.__class__.__name__,
                )
                dit_module.to(self.device)
                continue

            # Move non-block modules to GPU (they stay resident)
            for name, m in dit_module.named_children():
                if name not in blocks_attr_names:
                    m.to(self.device)
                    logger.debug(f"Moved {name} to device {self.device}")
                else:
                    logger.debug(f"Skipped blocks module {name}")

            # Move top-level params/buffers to GPU (dit_module's own, not sub-modules)
            for param in dit_module._parameters.values():
                if param is not None:
                    param.data = param.data.to(self.device, non_blocking=True)

            for buffer in dit_module._buffers.values():
                if buffer is not None:
                    buffer.data = buffer.data.to(self.device, non_blocking=True)

            block_hooks = _install_layerwise_hook_group(
                blocks,
                self.device,
                self.copy_stream,
                self.config.pin_cpu_memory,
            )
            self._dit_hooks.extend(block_hooks)
            self._hooked_dit_blocks.extend(blocks)

            # The last block owns block zero's host backing. Materialize block
            # zero once; later denoising iterations prefetch it from the ring.
            block_hooks[0].prefetch_layer(non_blocking=False)

            logger.info(f"Layer-wise offloading enabled on {num_blocks} layers (blocks)")

            # Track hooked blocks for cleanup
            self._blocks.append(blocks)

        self.enabled = bool(self._blocks or self._encoder_modules or self._staged_components)

    def _disable(self, *, restore_weights: bool) -> None:
        if not self.enabled and not (
            self._dit_hooks or self._hooked_dit_blocks or self._encoder_modules or self._staged_components
        ):
            return

        steps = []
        if restore_weights:
            steps.extend(("restoring a DiT block", hook.restore_next_block) for hook in self._dit_hooks)
        steps.extend(
            ("removing a DiT block hook", lambda block=block: remove_block_hook(block))
            for block in self._hooked_dit_blocks
        )
        steps.extend(
            (
                "disabling encoder layerwise offload",
                lambda module=module: disable_plan_encoder_layerwise_offload(
                    module,
                    restore_weights=restore_weights,
                ),
            )
            for module in self._encoder_modules
        )
        cleanup_error = run_cleanup_steps(steps)
        if cleanup_error is not None:
            # Keep every host-master reference so a transient teardown error
            # can be retried without losing weights behind placeholders.
            raise RuntimeError("Failed to fully disable layerwise offload") from cleanup_error
        self._blocks.clear()
        self._dit_hooks.clear()
        self._hooked_dit_blocks.clear()
        self._encoder_modules.clear()
        self._staged_components.clear()
        self.enabled = False
        logger.info("Layer-wise offloading disabled")

    def disable(self) -> None:
        self._disable(restore_weights=True)

    # Compatibility aliases for existing model integrations.
    get_blocks_attr_names = staticmethod(get_blocks_attr_names)
    set_blocks_attr_names = staticmethod(set_blocks_attr_names)
    get_blocks_from_dit = staticmethod(get_blocks_from_dit)
