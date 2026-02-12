# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from itertools import chain
from typing import Any

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.hooks import HookRegistry, ModelHook
from vllm_omni.platforms import current_omni_platform

from .base import OffloadBackend, OffloadConfig
from .module_collector import ModuleDiscovery

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
        stream: torch.cuda.Stream | None = None,
        pin_memory: bool = True,
    ):
        assert isinstance(next_block, nn.Module), "transformer block must be type `torch.nn.Module`"
        assert current_omni_platform.is_cuda(), "Layerwise offloading is only supported on cuda devices for now"

        self.next_block = next_block
        self.device = device
        self.copy_stream = stream or torch.cuda.current_stream()
        self.pin_memory = pin_memory

        # Per-block synchronization primitive: set after H2D copy completes.
        self._prefetch_done: torch.cuda.Event | None = None

        self.next_block_parameters: dict[str, nn.Parameter] = {}
        self.next_block_buffers: dict[str, torch.Tensor] = {}
        self.dtype_cpu_flattened_weights: dict[torch.dtype, torch.Tensor] = {}
        self.dtype_metadata: dict[torch.dtype, list[dict[str, Any]]] = {}

    def initialize_hook(self, module: nn.Module) -> nn.Module:
        # This all happen during the hook instance being registered to hook registry;
        # the input module is kept intact
        module = super().initialize_hook(module)

        self.block_parameters: dict[str, nn.Parameter] = dict(module.named_parameters())
        self.block_buffers: dict[str, torch.Tensor] = dict(module.named_buffers())

        self.next_block_parameters: dict[str, nn.Parameter] = dict(self.next_block.named_parameters())
        self.next_block_buffers: dict[str, torch.Tensor] = dict(self.next_block.named_buffers())

        # Pre-allocate gpu tensors in a flattened way
        self.dtype_cpu_flattened_weights, self.dtype_metadata = LayerwiseOffloadHook._to_cpu(
            self.next_block_parameters, self.next_block_buffers, self.device, self.pin_memory
        )

        return module

    @staticmethod
    def _to_cpu(
        params: dict[str, nn.Parameter],
        bufs: dict[str, torch.Tensor],
        device: torch.device,
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
        dtype_grouped_weights: dict[torch.dtype, dict[str, torch.Tensor]] = {}
        dtype_cpu_flattened_weights: dict[torch.dtype, torch.Tensor] = {}
        # NOTE: order does matter
        dtype_metadata: dict[torch.dtype, list[dict[str, Any]]] = {}

        for name, param_or_buf in chain(params.items(), bufs.items()):
            dtype = param_or_buf.dtype
            if dtype not in dtype_grouped_weights:
                dtype_grouped_weights[dtype] = {}
            dtype_grouped_weights[dtype][name] = param_or_buf

        for dtype, name2weights in dtype_grouped_weights.items():
            # total # of parameters + buffers
            total_numel = sum(t.numel() for _, t in name2weights.items())
            cpu_tensor = torch.empty(total_numel, dtype=dtype, device="cpu", pin_memory=pin_memory)

            current_offset = 0
            for name, param_or_buf in name2weights.items():
                numel = param_or_buf.numel()
                cpu_tensor[current_offset : current_offset + numel].copy_(param_or_buf.flatten())
                if dtype not in dtype_metadata:
                    dtype_metadata[dtype] = []
                dtype_metadata[dtype].append(
                    {
                        "name": name,
                        "offset": current_offset,
                        "numel": numel,
                        "shape": param_or_buf.shape,
                    }
                )

                param_or_buf.data = torch.empty((), device=device, dtype=dtype)
                current_offset += numel

            dtype_cpu_flattened_weights[dtype] = cpu_tensor

        return dtype_cpu_flattened_weights, dtype_metadata

    @torch.compiler.disable
    def prefetch_layer(self, non_blocking: bool = True) -> None:
        """Copy layer weights from CPU -> GPU.

        Pre-fetch target block in an asynchronous way with compute - memory copy overlap,
        with non_blocking set to True.
        """
        self.copy_stream.wait_stream(torch.cuda.current_stream())

        layer_params = self.next_block_parameters
        layer_bufs = self.next_block_buffers

        evt = torch.cuda.Event()
        gpu_weights: dict[torch.dtype, torch.Tensor] = {}

        with torch.cuda.stream(self.copy_stream):
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

                target_param_or_buf.data = gpu_weight[metadata["offset"] : metadata["offset"] + metadata["numel"]].view(
                    metadata["shape"]
                )

        self._prefetch_done = evt

    @torch.compiler.disable
    def offload_layer(self) -> None:
        """Free GPU memory for layer by replacing tensors with empty placeholders.
        This function does not actually offload weights from GPU back to CPU.
        """
        evt = self._prefetch_done
        if evt is not None:
            torch.cuda.current_stream().wait_event(evt)

        self._prefetch_done = None

        # free GPU residency
        for _, param in self.block_parameters.items():
            param.data = torch.empty((), device=self.device, dtype=param.dtype)
        for _, buf in self.block_buffers.items():
            buf.data = torch.empty((), device=self.device, dtype=buf.dtype)

    def pre_forward(self, module: nn.Module, *args: Any, **kwargs: Any) -> tuple[tuple, dict]:
        self.prefetch_layer(non_blocking=True)

        return args, kwargs

    def post_forward(self, module: nn.Module, output: Any) -> Any:
        self.offload_layer()

        return output


def apply_block_hook(
    module: nn.Module,
    next_block: nn.Module,
    device: torch.device,
    stream: torch.cuda.Stream | None = None,
    pin_memory: bool = True,
) -> LayerwiseOffloadHook:
    registry = HookRegistry.get_or_create(module)
    hook = LayerwiseOffloadHook(next_block, device, stream, pin_memory)
    registry.register_hook(LayerwiseOffloadHook._HOOK_NAME, hook)

    return hook


def remove_block_hook(module: nn.Module) -> None:
    registry: HookRegistry | None = getattr(module, "_hook_registry", None)
    if registry is not None:
        registry.remove_hook(LayerwiseOffloadHook._HOOK_NAME)
        logger.debug("Removed offload hook from %s", module.__class__.__name__)


class LayerWiseOffloadBackend(OffloadBackend):
    """Layer-wise (block-level) offloading backend.

    Implements sliding window offloading where only a small number of transformer
    blocks reside on GPU at a time. Blocks are prefetched asynchronously while
    previous blocks compute, and freed after use.
    """

    def __init__(self, config: OffloadConfig, device: torch.device):
        super().__init__(config, device)

        self.copy_stream = torch.cuda.Stream()
        self._blocks: list[list[nn.Module]] = []

    def enable(self, pipeline: nn.Module) -> None:
        if self.enabled:
            logger.warning("LayerWiseOffloadBackend already enabled")
            return

        modules = ModuleDiscovery.discover(pipeline)
        if not modules.dits:
            logger.warning("No DiT/transformer modules found, skipping layer-wise offloading")
            return

        # Move encoders to GPU (they stay resident)
        for enc in modules.encoders:
            enc.to(self.device)

        # Move VAE to GPU if available
        if modules.vae is not None:
            try:
                modules.vae.to(self.device, non_blocking=True)
            except Exception as exc:
                logger.debug("Failed to move VAE to GPU: %s", exc)

        logger.info("Applying layer-wise offloading on %s", modules.dit_names)

        # Apply block-wise offloading hook for each of the blocks in DiT model(s)
        # Note that there might exist multiple DiT models in specific pipelines
        for i, dit_module in enumerate(modules.dits):
            dit_name = modules.dit_names[i]
            logger.info(f"Applying hooks on {dit_name} ({dit_module.__class__.__name__})")

            blocks_attr_name = LayerWiseOffloadBackend.get_blocks_attr_name(dit_module)
            blocks = LayerWiseOffloadBackend.get_blocks_from_dit(dit_module)

            if not blocks_attr_name or not blocks:
                logger.warning(
                    "Target layers (blocks) not found. Skipping offloading on %s (%s)",
                    dit_name,
                    dit_module.__class__.__name__,
                )
                dit_module.to(self.device)
                continue

            num_blocks = len(blocks)
            if num_blocks <= 1:
                logger.warning(
                    "#Target layers (blocks) <= 1. Skipping offloading on %s (%s)",
                    dit_name,
                    dit_module.__class__.__name__,
                )
                dit_module.to(self.device)
                continue

            # Move non-block modules to GPU (they stay resident)
            for name, m in dit_module.named_children():
                if name == blocks_attr_name:
                    logger.debug(f"Skipped blocks module {name}")
                    continue
                m.to(self.device)
                logger.debug(f"Moved {name} to device {self.device}")

            # Pre-fetch the first layer by manually calling the hook function on the last layer;
            # For subsequent requests, the first layer/block will be pre-fetched
            # during the last layer compute of the previous request.
            last_block, first_block = blocks[-1], blocks[0]
            hook = apply_block_hook(last_block, first_block, self.device, self.copy_stream, self.config.pin_cpu_memory)
            hook.prefetch_layer(non_blocking=False)

            # Register hook for each of blocks
            for i, block in enumerate(blocks[:-1]):
                next_block = blocks[(i + 1) % num_blocks]
                apply_block_hook(block, next_block, self.device, self.copy_stream, self.config.pin_cpu_memory)

            logger.info(f"Layer-wise offloading enabled on {num_blocks} layers (blocks)")

            # Track hooked blocks for cleanup
            self._blocks.append(blocks)

        if len(self._blocks) > 0 and len(self._blocks[0]) > 0:
            self.enabled = True

    def disable(self) -> None:
        if not self.enabled:
            return

        for blocks in self._blocks:
            for block in blocks:
                remove_block_hook(block)

        self._blocks.clear()
        self.enabled = False
        logger.info("Layer-wise offloading disabled")

    @staticmethod
    def get_blocks_attr_name(model: nn.Module) -> str | None:
        """Retrieve blocks attribute name from provided DiT model"""
        return getattr(model.__class__, "_layerwise_offload_blocks_attr", None)

    @staticmethod
    def set_blocks_attr_name(model: nn.Module, name: str) -> None:
        if not hasattr(model.__class__, "_layerwise_offload_blocks_attr"):
            setattr(model.__class__, "_layerwise_offload_blocks_attr", name)

    @staticmethod
    def get_blocks_from_dit(model: nn.Module) -> list[nn.Module]:
        """
        Retrieve a list of blocks from provided DiT model. Blocks attribute name
        are found by `_layerwise_offload_blocks_attr` set to DiT models. For example,

        ```
        class WanTransformer3DModel(nn.Module):
            _layerwise_offload_blocks_attr = "blocks"
        ```
        """
        blocks_attr_name = LayerWiseOffloadBackend.get_blocks_attr_name(model)
        if blocks_attr_name is None:
            logger.warning(
                f"No _layerwise_offload_blocks_attr defined for {model.__class__.__name__}, "
                "skipping layerwise offloading"
            )
            return []

        _blocks = getattr(model, blocks_attr_name, None)
        if _blocks is None:
            logger.warning(
                f"Blocks (layers) '{blocks_attr_name}' not found on {model.__class__.__name__}, "
                "skipping layerwise offloading"
            )
            return []

        return list(_blocks)
