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

from functools import partial
from itertools import chain
from typing import TYPE_CHECKING, Any

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
            torch.cuda.empty_cache()

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


class LayerwiseOffloader:
    """Layer-wise CPU offloading for transformer blocks.

    Keeps only a sliding window of layers (blocks), by default a single layer, on GPU,
    prefetching the next block while the current block computes to approach compute - memcpy overlap.
    Unused blocks are freed on GPU.

    Based on implementations from:
    https://github.com/sgl-project/sglang/blob/v0.5.8/python/sglang/multimodal_gen/runtime/utils/layerwise_offload.py
    """

    def __init__(
        self,
        blocks: list[nn.Module],
        device: torch.device,
        pin_memory: bool = True,
        num_gpu_layers: int = 1,
    ):
        assert all(isinstance(m, nn.Module) for m in blocks), "All transformer blocks must be torch.nn.Module"
        assert current_omni_platform.is_cuda(), "Layerwise offloading is only supported on cuda devices for now"

        self.blocks = blocks
        self.device = device
        self.pin_memory = pin_memory
        self.num_gpu_layers = num_gpu_layers
        self.num_blocks = len(self.blocks)
        if self.num_blocks == 0:
            raise ValueError("LayerwiseOffloader requires at least one block, but found 0.")
        if not (1 <= self.num_gpu_layers <= self.num_blocks):
            raise ValueError(f"Invalid num_gpu_layers {self.num_gpu_layers} with {self.num_blocks} blocks")

        self._pre_hook_handles: list = []
        self._post_hook_handles: list = []

        self._copy_stream = torch.cuda.Stream()

        # Per-layer synchronization primitive: set after H2D copy completes.
        self._prefetch_done: list[torch.cuda.Event | None] = [None] * self.num_blocks

        # Simple state to avoid redundant work.
        self._resident: list[bool] = [False] * self.num_blocks

        # Pre-allocate gpu tensors
        # layer-id -> {dtype -> flattened aggregated cpu tensor}
        self.layer_cpu_weights: list[dict[torch.dtype, torch.Tensor]] = []
        self.layer_metadata: list[dict[torch.dtype, list[dict[str, Any]]]] = []

        self.block_parameters: dict[int, dict[str, nn.Parameter]] = {}
        self.block_buffers: dict[int, dict[str, torch.Tensor]] = {}
        for layer_idx, block in enumerate(self.blocks):
            self.block_parameters[layer_idx] = dict(block.named_parameters())
            self.block_buffers[layer_idx] = dict(block.named_buffers())

            dtype_cpu_flattened_weights, dtype_metadata = self._to_cpu(
                self.block_parameters[layer_idx], self.block_buffers[layer_idx]
            )
            self.layer_cpu_weights.append(dtype_cpu_flattened_weights)
            self.layer_metadata.append(dtype_metadata)

        if self.num_blocks != len(self.layer_cpu_weights):
            logger.error(
                f"Inconsistent block layers happened: # of blocks: {self.num_blocks}; "
                f"# of layer cpu weights: {len(self.layer_cpu_weights)}"
            )

        # Register pre and post forward hooks on each of the blocks
        self.register_block_hooks()

        # Pre-fetch the first layer
        # For subsequent requests, the first layer/block will be pre-fetched
        # during the last layer compute of the previous request.
        self.prefetch_layer(0, non_blocking=False)

    def _to_cpu(
        self, params: dict[str, nn.Parameter], bufs: dict[str, torch.Tensor]
    ) -> tuple[dict[torch.dtype, torch.Tensor], dict[torch.dtype, list[dict[str, Any]]]]:
        """Move block parameters and buffers to CPU, flattening by dtype.

        Consolidates parameters and buffers into contiguous CPU tensors grouped by dtype
        for GPU transfers. Replaces original tensors with empty placeholders.

        Returns:
            Tuple of
                flattened CPU tensors by dtype,
                metadata for reconstruction by dtype
        """
        dtype_grouped_weights: dict[torch.dtype, dict[str, torch.Tensor]] = {}
        dtype_cpu_flattened_weights: dict[torch.dtype, torch.Tensor] = {}
        # order does matter
        dtype_metadata: dict[torch.dtype, list[dict[str, Any]]] = {}

        for name, param_or_buf in chain(params.items(), bufs.items()):
            dtype = param_or_buf.dtype
            if dtype not in dtype_grouped_weights:
                dtype_grouped_weights[dtype] = {}
            dtype_grouped_weights[dtype][name] = param_or_buf

        for dtype, name2weights in dtype_grouped_weights.items():
            # total # of parameters + buffers
            total_numel = sum(t.numel() for _, t in name2weights.items())
            cpu_tensor = torch.empty(total_numel, dtype=dtype, device="cpu", pin_memory=self.pin_memory)

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

                param_or_buf.data = torch.empty((), device=self.device, dtype=dtype)
                current_offset += numel

            dtype_cpu_flattened_weights[dtype] = cpu_tensor

        return dtype_cpu_flattened_weights, dtype_metadata

    def register_block_hooks(self) -> None:
        """Register forward hooks on blocks for prefetching and offloading."""

        def _pre_hook(module: nn.Module, args: tuple, *, layer_idx: int) -> None:
            # For the last block / layer, prefetch layer 0 (the first layer)
            next_id = (layer_idx + 1) % self.num_blocks
            self.prefetch_layer(next_id, non_blocking=True)

        def _post_hook(module: nn.Module, args: tuple, output: tuple, *, layer_idx: int) -> None:
            self.offload_layer(layer_idx)
            self._resident[layer_idx] = False
            self._prefetch_done[layer_idx] = None

        for i, layer in enumerate(self.blocks):
            pre_hook_fn = partial(_pre_hook, layer_idx=i)
            handle = layer.register_forward_pre_hook(pre_hook_fn)
            self._pre_hook_handles.append(handle)

            post_hook_fn = partial(_post_hook, layer_idx=i)
            handle = layer.register_forward_hook(post_hook_fn)
            self._post_hook_handles.append(handle)

    @torch.compiler.disable
    def prefetch_layer(self, layer_idx: int, non_blocking: bool = True) -> None:
        """Copy layer weights from CPU -> GPU.

        Pre-fetch target layer in an asynchronous way with compute - memory copy overlap,
        with non_blocking set to True.
        """
        if layer_idx >= self.num_blocks or layer_idx < 0:
            logger.warning(f"Invalid layer id specified: {layer_idx}")
            return

        self._copy_stream.wait_stream(torch.cuda.current_stream())

        layers_to_fetch = [(layer_idx + i) % self.num_blocks for i in range(self.num_gpu_layers)]

        for idx in layers_to_fetch:
            if self._resident[idx]:
                continue

            layer_params = self.block_parameters[idx]
            layer_bufs = self.block_buffers[idx]

            evt = torch.cuda.Event()
            gpu_weights: dict[torch.dtype, torch.Tensor] = {}

            with torch.cuda.stream(self._copy_stream):
                for dtype, cpu_weight in self.layer_cpu_weights[idx].items():
                    gpu_weight = torch.empty(cpu_weight.shape, dtype=dtype, device=self.device)
                    gpu_weight.copy_(cpu_weight, non_blocking=non_blocking)
                    gpu_weights[dtype] = gpu_weight

                evt.record(self._copy_stream)

            for dtype in self.layer_metadata[idx]:
                ordered_metadata: list[dict[str, Any]] = self.layer_metadata[idx][dtype]

                gpu_weight = gpu_weights[dtype]

                for metadata in ordered_metadata:
                    target_name = metadata["name"]
                    target_param_or_buf = (
                        layer_params[target_name] if target_name in layer_params else layer_bufs[target_name]
                    )

                    target_param_or_buf.data = gpu_weight[
                        metadata["offset"] : metadata["offset"] + metadata["numel"]
                    ].view(metadata["shape"])

            self._prefetch_done[idx] = evt
            self._resident[idx] = True

    @torch.compiler.disable
    def offload_layer(self, layer_idx: int) -> None:
        """Free GPU memory for layer by replacing tensors with empty placeholders."""
        if layer_idx >= self.num_blocks or layer_idx < 0:
            logger.warning(f"Invalid layer id specified: {layer_idx}")
            return
        if not self._resident[layer_idx]:
            logger.warning(f"{layer_idx} is not residing on GPU")
            return

        evt = self._prefetch_done[layer_idx]
        if evt is not None:
            torch.cuda.current_stream().wait_event(evt)

        # free GPU residency
        for _, param in self.block_parameters[layer_idx].items():
            param.data = torch.empty((), device=self.device, dtype=param.dtype)
        for _, buf in self.block_buffers[layer_idx].items():
            buf.data = torch.empty((), device=self.device, dtype=buf.dtype)

    def remove_all_hooks(self) -> None:
        """Remove all hooks."""
        for h in self._pre_hook_handles:
            h.remove()
        for h in self._post_hook_handles:
            h.remove()
        self._pre_hook_handles.clear()
        self._post_hook_handles.clear()

    @staticmethod
    def get_blocks_attr_name(model: nn.Module) -> str | None:
        """Retrieve blocks attribute name from provided DiT model"""
        return getattr(model.__class__, "_layerwise_offload_blocks_attr", None)

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
        blocks_attr_name = LayerwiseOffloader.get_blocks_attr_name(model)
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
    enable_cpu_offload = getattr(od_config, "enable_cpu_offload", False)
    enable_layerwise_offload = getattr(od_config, "enable_layerwise_offload", False)
    pin_cpu_memory = getattr(od_config, "pin_cpu_memory", True)

    if not enable_cpu_offload and not enable_layerwise_offload:
        return
    if enable_cpu_offload and enable_layerwise_offload:
        # NOTE: Model-wise and layerwise cpu offloading are not supported together at this moment,
        # consider layerwise offloading has higher priority than model-wise offloading
        enable_cpu_offload = False
        logger.info(
            "Model-wise and layer-wise CPU offloading are not supported together at this moment. "
            "Automatically disabled model-wise offloading."
        )
    # For now, model-wise and layer-wise (block-wise) offloading
    # are functioning as expected when cuda device is available
    if not current_omni_platform.is_cuda() or current_omni_platform.get_device_count() < 1:
        logger.info("CPU Offloading requires cuda devices available. Skipping for now...")
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
                logger.error("Fail to get device of pipeline. Skipping applying offloading hooks")
                return

    # Collect all encoders
    encoders: list[nn.Module] = []
    encoder_names: list[str] = []
    for attr in ["text_encoder", "text_encoder_2", "text_encoder_3", "image_encoder"]:
        if hasattr(model, attr) and getattr(model, attr) is not None:
            encoders.append(getattr(model, attr))
            encoder_names.append(attr)
    if not encoders and enable_cpu_offload:
        logger.warning("enable_cpu_offload enabled but no encoders found")
        return
    for enc in encoders:
        enc.to(device)

    # Collect VAE
    for name in ["vae"]:
        module = getattr(model, name, None)
        if module is None:
            continue
        try:
            module.to(device, non_blocking=True)
        except Exception as exc:
            logger.debug("Failed to move %s to GPU: %s", name, exc)

    if enable_cpu_offload:
        # Initial state: keep DiT modules on CPU (encoders typically run first)
        for dit_mod in dit_modules:
            dit_mod.to("cpu")

        torch.cuda.empty_cache()

        if pin_cpu_memory:
            for dit_mod in dit_modules:
                for p in dit_mod.parameters():
                    if p.data.device.type == "cpu" and not p.data.is_pinned():
                        p.data = p.data.pin_memory()

        # Register sequential offload hooks
        SequentialOffloader(dit_modules, encoders, device, pin_cpu_memory).register()
        logger.info(
            "CPU offload enabled: %s <-> %s (mutual exclusion)",
            ", ".join(dit_names),
            ", ".join(encoder_names),
        )
    elif enable_layerwise_offload:
        logger.info(f"Applying offloading hooks on {dit_names}")

        for i, dit_module in enumerate(dit_modules):
            logger.info(f"Applying hook on {dit_names[i]} ({dit_module.__class__.__name__})")
            blocks_attr_name = LayerwiseOffloader.get_blocks_attr_name(dit_module)
            blocks = LayerwiseOffloader.get_blocks_from_dit(dit_module)

            if not blocks_attr_name or not blocks:
                logger.warning(
                    "Target layers (blocks) are not found. "
                    f"Skipping offloading on {dit_names[i]} ({dit_module.__class__.__name__})"
                )
                continue

            # move modules other than blocks to gpu and keep them on gpu
            for name, m in dit_module.named_children():
                # Skip the blocks module (layers to be offloaded)
                if name == blocks_attr_name:
                    logger.debug(f"Skipped module {name}")
                    continue

                m.to(device)
                logger.debug(f"Moved {name} to device {device}")

            # set to the module (transformer)
            offloader = LayerwiseOffloader(blocks, device, pin_cpu_memory, od_config.layerwise_num_gpu_layers)
            setattr(dit_module, "_layerwise_offloader", offloader)

            logger.info(
                f"Layerwise offloading enabled on {len(blocks)} layers (blocks), "
                f"with {od_config.layerwise_num_gpu_layers} kept on device"
            )
