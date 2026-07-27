# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Distributed Layerwise Offload backend with H2D + AllGather overlap.

This module implements the RFC-1 "Distributed Layerwise Offload" mechanism that:

* Shards model weights across DP ranks and stores only the local shard
  (1/DP_size of full model) in host pinned memory per rank.
* Uses a fixed double-buffer scheme that keeps only two layers' worth of
  weights on each device at any time.
* Asynchronously pipelines both H2D transfers and AllGather communications
  on dedicated streams, fully overlapping them with computation.
* Is hardware-agnostic, supporting both NVIDIA GPU (CUDA) and Ascend NPU
  (CANN) platforms via vLLM-Omni's platform abstraction layer.
"""

from __future__ import annotations

import os
from itertools import chain
from typing import Any

import torch
import torch.distributed
from torch import nn
from torch.distributed.tensor import DTensor
from vllm.logger import init_logger

from vllm_omni.diffusion.hooks import HookRegistry, ModelHook
from vllm_omni.platforms import current_omni_platform

from .base import OffloadBackend, OffloadConfig
from .module_collector import ModuleDiscovery

logger = init_logger(__name__)


def _dtype_size(dtype: torch.dtype) -> int:
    """Return element size in bytes for a torch.dtype."""
    return torch.empty(1, dtype=dtype).element_size()


class DistributedLayerwiseOffloadHook(ModelHook):
    """Hook for distributed layerwise offloading with fixed double-buffer.

    Each rank stores only a shard of each block's weights on host memory.
    Two device slots alternate: one holds current weights, one holds next weights.
    H2D and AllGather run asynchronously on dedicated streams, overlapped
    with computation.

    Supports both NVIDIA GPU (CUDA) and Ascend NPU (CANN) platforms.
    """

    _HOOK_NAME = "distributed_layerwise_offload"

    def __init__(
        self,
        next_block: nn.Module,
        device: torch.device,
        dp_group: torch.distributed.ProcessGroup | None,
        dp_size: int,
        rank: int,
        copy_stream: Any | None = None,
        comm_stream: Any | None = None,
        pin_memory: bool = True,
        shared_buffers: list[dict[torch.dtype, torch.Tensor] | None] | None = None,
    ):
        assert isinstance(next_block, nn.Module), "transformer block must be type `torch.nn.Module`"

        self.next_block = next_block
        self.device = device
        self.dp_group = dp_group
        self.dp_size = dp_size
        self.rank = rank
        self.pin_memory = pin_memory

        self.copy_stream = copy_stream or current_omni_platform.Stream()
        self.comm_stream = comm_stream or current_omni_platform.Stream()

        # Double buffers: either shared (from backend) or self-allocated (lazy)
        if shared_buffers is not None:
            self.gpu_buffers: list[dict[torch.dtype, torch.Tensor] | None] = shared_buffers
            self._owns_buffers = False
        else:
            self.gpu_buffers = [None, None]
            self._owns_buffers = True
        self.ready_events: list[Any | None] = [None, None]

        # Sharded host weights for the next block, keyed by dtype
        self.cpu_shards: dict[torch.dtype, torch.Tensor] = {}
        self.metadata: dict[torch.dtype, list[dict[str, Any]]] = {}

        # Current slot index (0 or 1)
        self.current_slot = 0

        # Backward link to previous hook for fallback (cache-dit skip)
        self._prev_hook: DistributedLayerwiseOffloadHook | None = None

        # Marks the first hook in a shared-buffer group.  When multiple DiT
        # groups share the same 2 GPU buffers, another group may have
        # overwritten this group's slot between forwards.  The first block
        # must always sync-prefetch on entry to ensure it loads the correct
        # weights, even if is_materialized sees a non-empty tensor left by
        # the other group.
        self._is_group_first: bool = False

        # Parameters/buffers of the current and next blocks
        self.block_parameters: dict[str, nn.Parameter] = {}
        self.block_buffers: dict[str, torch.Tensor] = {}
        self.next_block_parameters: dict[str, nn.Parameter] = {}
        self.next_block_buffers: dict[str, torch.Tensor] = {}

        # Per-block synchronization primitive: set after H2D copy completes.
        self._prefetch_done: Any | None = None

        # Shared shard (AllGather input) buffers — assigned by backend.
        self.gpu_shard_buffers: list[dict[torch.dtype, torch.Tensor] | None] = [None, None]

        # Pending async AllGather work (prevent GC before completion).
        self._pending_work: Any | None = None
        self._cached_repoint: list | None = None

    # ------------------------------------------------------------------ #
    #  DTensor helpers (shared with LayerwiseOffloadHook)                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_dtensor(t: torch.Tensor) -> bool:
        return isinstance(t, DTensor)

    @staticmethod
    def _set_tensor_storage(target: torch.Tensor, value: torch.Tensor) -> None:
        if DistributedLayerwiseOffloadHook._is_dtensor(target):
            target._local_tensor = value
        else:
            target.data = value

    @staticmethod
    def _make_offload_placeholder(tensor: torch.Tensor) -> torch.Tensor:
        if DistributedLayerwiseOffloadHook._is_dtensor(tensor):
            local_shape = tuple(tensor.to_local().shape)
            return torch.empty(local_shape, device="meta", dtype=tensor.dtype)
        return torch.empty((0,), device=tensor.device, dtype=tensor.dtype)

    @staticmethod
    def _is_materialized_tensor(t: torch.Tensor) -> bool:
        if DistributedLayerwiseOffloadHook._is_dtensor(t):
            local_t = t.to_local()
            return not local_t.is_meta
        return not t.is_meta and t.data.numel() > 0

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    def initialize_hook(self, module: nn.Module) -> nn.Module:
        module = super().initialize_hook(module)

        self.block_parameters = dict(module.named_parameters())
        self.block_buffers = dict(module.named_buffers())

        self.next_block_parameters = dict(self.next_block.named_parameters())
        self.next_block_buffers = dict(self.next_block.named_buffers())

        # Shard next block's weights and store local shard in pinned CPU memory
        self.cpu_shards, self.metadata = self._shard_and_pin(
            self.next_block_parameters,
            self.next_block_buffers,
            self.dp_size,
            self.rank,
            self.pin_memory,
        )

        # Allocate device buffers only if not using shared buffers from backend
        if self._owns_buffers:
            self._allocate_device_buffers()

        # Cache parameter re-pointing metadata to avoid per-layer dict lookups.
        self._cached_repoint = []
        for slot in range(2):
            repoint = []
            for dtype, metas in self.metadata.items():
                for m in metas:
                    target = (
                        self.next_block_parameters[m["name"]]
                        if m["name"] in self.next_block_parameters
                        else self.next_block_buffers[m["name"]]
                    )
                    repoint.append((target, dtype, m["offset"], m["numel"], m["shape"]))
            self._cached_repoint.append(repoint)

        # Pre-compute AG output sizes (avoid sum() per layer).
        self._ag_output_sizes: dict[torch.dtype, int] = {}
        for dtype, metas in self.metadata.items():
            total_numel = sum(m["numel"] for m in metas)
            shard_numel = self.cpu_shards[dtype].numel()
            self._ag_output_sizes[dtype] = shard_numel * self.dp_size if self.dp_size > 1 else total_numel

        return module

    @staticmethod
    def _shard_and_pin(
        params: dict[str, nn.Parameter],
        bufs: dict[str, torch.Tensor],
        dp_size: int,
        rank: int,
        pin_memory: bool,
    ) -> tuple[dict[torch.dtype, torch.Tensor], dict[torch.dtype, list[dict[str, Any]]]]:
        """Flatten params+buffers by dtype, split into DP shards, store local shard.

        Each rank stores only ``1/dp_size`` of the total weights. The full
        tensor is reconstructed at runtime via AllGather.
        """
        dtype_grouped: dict[torch.dtype, dict[str, torch.Tensor]] = {}
        dtype_metadata: dict[torch.dtype, list[dict[str, Any]]] = {}

        for name, param_or_buf in chain(params.items(), bufs.items()):
            dtype = param_or_buf.dtype
            if dtype not in dtype_grouped:
                dtype_grouped[dtype] = {}
            dtype_grouped[dtype][name] = param_or_buf

        cpu_shards: dict[torch.dtype, torch.Tensor] = {}

        for dtype, name2weights in dtype_grouped.items():
            # Resolve local tensors (handle DTensor via to_local)
            weights_with_local = []
            for name, t in name2weights.items():
                local_t = t.to_local() if hasattr(t, "to_local") else t
                weights_with_local.append((name, t, local_t))

            total_numel = sum(local.numel() for _, _, local in weights_with_local)

            # Equal-sized shards (ceil division) for all_gather_into_tensor
            shard_size = (total_numel + dp_size - 1) // dp_size  # ceil
            shard_start = rank * shard_size
            shard_end = min(shard_start + shard_size, total_numel)

            # Allocate ONLY the shard (1/dp_size), zero-padded to ceil.
            # Avoids materialising the full block on CPU.
            shard = torch.zeros(shard_size, dtype=dtype, device="cpu")

            current_offset = 0
            for name, original_tensor, local_tensor in weights_with_local:
                numel = local_tensor.numel()
                if dtype not in dtype_metadata:
                    dtype_metadata[dtype] = []
                # Offsets remain relative to the FULL flattened buffer
                # (needed for correct AllGather reconstruction).
                dtype_metadata[dtype].append(
                    {
                        "name": name,
                        "offset": current_offset,
                        "numel": numel,
                        "shape": local_tensor.shape,
                    }
                )

                # Copy ONLY the portion within [shard_start, shard_end)
                overlap_start = max(current_offset, shard_start)
                overlap_end = min(current_offset + numel, shard_end)
                if overlap_start < overlap_end:
                    src_start = overlap_start - current_offset
                    src_end = overlap_end - current_offset
                    dst_start = overlap_start - shard_start
                    dst_end = overlap_end - shard_start
                    shard[dst_start:dst_end].copy_(local_tensor.flatten()[src_start:src_end])

                # Replace original tensor with placeholder (frees CPU storage)
                DistributedLayerwiseOffloadHook._set_tensor_storage(
                    original_tensor,
                    DistributedLayerwiseOffloadHook._make_offload_placeholder(original_tensor),
                )
                current_offset += numel

            if pin_memory:
                shard = shard.pin_memory()

            cpu_shards[dtype] = shard

        return cpu_shards, dtype_metadata

    def _allocate_device_buffers(self) -> None:
        """Pre-allocate exactly two device buffers (one per slot)."""
        for slot in range(2):
            gpu_weights: dict[torch.dtype, torch.Tensor] = {}
            for dtype, metas in self.metadata.items():
                total_numel = sum(m["numel"] for m in metas)
                # AllGather output = dp_size * shard_size (padded)
                padded = total_numel
                if self.dp_size > 1:
                    shard_sz = (total_numel + self.dp_size - 1) // self.dp_size
                    padded = shard_sz * self.dp_size
                gpu_weights[dtype] = torch.empty(padded, dtype=dtype, device=self.device)
            self.gpu_buffers[slot] = gpu_weights

    @property
    def is_materialized(self) -> bool:
        """Check whether this block's parameters hold real data on device."""
        for param in self.block_parameters.values():
            return DistributedLayerwiseOffloadHook._is_materialized_tensor(param)
        return True

    # ------------------------------------------------------------------ #
    #  Prefetch: H2D + AllGather (overlapped on dedicated streams)      #
    # ------------------------------------------------------------------ #

    @torch.compiler.disable
    def prefetch_layer(self, slot: int, non_blocking: bool = True) -> None:
        """Prepare next block's weights into the shared device buffer for *slot*.

        Uses the pre-allocated ``self.gpu_buffers[slot]`` instead of
        allocating fresh tensors every layer.  This enforces the fixed
        double-buffer memory bound (exactly 2 blocks on device).
        """
        self.copy_stream.wait_stream(current_omni_platform.current_stream())

        evt = current_omni_platform.Event()
        gpu_weights = self.gpu_buffers[slot]
        assert gpu_weights is not None, f"gpu_buffers[{slot}] not allocated"

        if self.dp_size <= 1 or self.dp_group is None:
            with current_omni_platform.stream(self.copy_stream):
                for dtype, cpu_shard in self.cpu_shards.items():
                    gw = gpu_weights[dtype]
                    gw[: cpu_shard.numel()].copy_(cpu_shard, non_blocking=non_blocking)
                evt.record(self.copy_stream)
        else:
            gpu_shards: dict[torch.dtype, torch.Tensor] = {}
            shard_bufs = self.gpu_shard_buffers[slot]
            assert shard_bufs is not None, f"gpu_shard_buffers[{slot}] not allocated"
            with current_omni_platform.stream(self.copy_stream):
                for dtype, cpu_shard in self.cpu_shards.items():
                    gpu_shard = shard_bufs[dtype][: cpu_shard.numel()]
                    gpu_shard.copy_(cpu_shard, non_blocking=non_blocking)
                    gpu_shards[dtype] = gpu_shard

            self.comm_stream.wait_stream(self.copy_stream)
            with current_omni_platform.stream(self.comm_stream):
                for dtype, local_shard in gpu_shards.items():
                    gw = gpu_weights[dtype]
                    torch.distributed.all_gather_into_tensor(
                        gw,
                        local_shard,
                        group=self.dp_group,
                    )
                evt.record(self.comm_stream)

        self.ready_events[slot] = evt
        self._prefetch_done = evt

        # Re-point using cached metadata (avoids per-layer dict lookups).
        for target, dtype, offset, numel, shape in self._cached_repoint[slot]:
            DistributedLayerwiseOffloadHook._set_tensor_storage(
                target,
                gpu_weights[dtype][offset : offset + numel].view(shape),
            )

    def get_weights(self, slot: int) -> dict[torch.dtype, torch.Tensor] | None:
        """Wait for AllGather completion and return full weights for the slot.

        The ready event for this slot was set by the *previous* hook's
        prefetch_layer (which prefetched THIS block's weights into the
        shared buffer).  This hook's own ready_events[slot] is None because
        it never prefetched into this slot itself.  Fall back to the
        previous hook's event to ensure the compute stream waits for the
        AllGather (or H2D) to complete before reading the buffer.
        """
        evt = self.ready_events[slot]
        if evt is None and self._prev_hook is not None:
            evt = self._prev_hook.ready_events[slot]
        if evt is not None:
            current_omni_platform.current_stream().wait_event(evt)
        return self.gpu_buffers[slot]

    # ------------------------------------------------------------------ #
    #  Offload: free device memory for current block                     #
    # ------------------------------------------------------------------ #

    @torch.compiler.disable
    def offload_layer(self) -> None:
        """Free GPU memory for current block by replacing tensors with placeholders."""
        evt = self._prefetch_done
        if evt is not None:
            current_omni_platform.current_stream().wait_event(evt)
        self._prefetch_done = None

        for _, param in self.block_parameters.items():
            DistributedLayerwiseOffloadHook._set_tensor_storage(
                param, DistributedLayerwiseOffloadHook._make_offload_placeholder(param)
            )
        for _, buf in self.block_buffers.items():
            DistributedLayerwiseOffloadHook._set_tensor_storage(
                buf, DistributedLayerwiseOffloadHook._make_offload_placeholder(buf)
            )

    # ------------------------------------------------------------------ #
    #  ModelHook interface                                                #
    # ------------------------------------------------------------------ #

    def pre_forward(self, module: nn.Module, *args: Any, **kwargs: Any) -> tuple[tuple, dict]:
        # If this is the first block in a shared-buffer group, always
        # sync-prefetch — another group may have overwritten our slot.
        # A non-empty view into the shared buffer does NOT prove the
        # weights belong to this group.
        if self._is_group_first and self._prev_hook is not None:
            self._prev_hook.prefetch_layer(self.current_slot, non_blocking=False)
            self._prev_hook.get_weights(self.current_slot)
        elif not self.is_materialized and self._prev_hook is not None:
            # Previous hook was skipped (e.g. by cache-dit), sync-prefetch
            self._prev_hook.prefetch_layer(self.current_slot, non_blocking=False)
            self._prev_hook.get_weights(self.current_slot)

        # Prefetch next layer into the other slot (overlapped with compute).
        # No explicit get_weights() here — offload_layer() in the previous
        # hook's post_forward already enqueued wait_event for the current
        # layer's H2D/AllGather, which is sufficient to ensure compute_stream
        # waits for weights before reading them.  The redundant wait_event
        # caused cascading NPU stream synchronization stalls (~10ms/layer).
        next_slot = 1 - self.current_slot
        self.prefetch_layer(next_slot, non_blocking=True)

        return args, kwargs

    def post_forward(self, module: nn.Module, output: Any) -> Any:
        self.offload_layer()
        # Do NOT swap current_slot here.  The slot assignment is fixed at
        # init time (i % 2) so that the circular prefetch (last block →
        # first block) writes to the correct slot.  Swapping would make
        # block 0 read slot 1 in step 2 while block 63 prefetched into
        # slot 0 — causing garbled output.
        return output


# ---------------------------------------------------------------------- #
#  Module-level helpers                                                   #
# ---------------------------------------------------------------------- #


def apply_distributed_block_hook(
    module: nn.Module,
    next_block: nn.Module,
    device: torch.device,
    dp_group: torch.distributed.ProcessGroup | None,
    dp_size: int,
    rank: int,
    copy_stream: Any | None = None,
    comm_stream: Any | None = None,
    pin_memory: bool = True,
    shared_buffers: list[dict[torch.dtype, torch.Tensor] | None] | None = None,
) -> DistributedLayerwiseOffloadHook:
    """Register a DistributedLayerwiseOffloadHook on *module*."""
    registry = HookRegistry.get_or_create(module)
    hook = DistributedLayerwiseOffloadHook(
        next_block=next_block,
        device=device,
        dp_group=dp_group,
        dp_size=dp_size,
        rank=rank,
        copy_stream=copy_stream,
        comm_stream=comm_stream,
        pin_memory=pin_memory,
        shared_buffers=shared_buffers,
    )
    registry.register_hook(DistributedLayerwiseOffloadHook._HOOK_NAME, hook)
    return hook


def remove_distributed_block_hook(module: nn.Module) -> None:
    """Remove the distributed layerwise offload hook from *module*."""
    registry: HookRegistry | None = getattr(module, "_hook_registry", None)
    if registry is not None:
        registry.remove_hook(DistributedLayerwiseOffloadHook._HOOK_NAME)
        logger.debug("Removed distributed offload hook from %s", module.__class__.__name__)


# ---------------------------------------------------------------------- #
#  Backend                                                                #
# ---------------------------------------------------------------------- #


class DistributedLayerwiseOffloadBackend(OffloadBackend):
    """Distributed layer-wise (block-level) offloading backend.

    Supports both GPU (CUDA) and NPU (CANN) platforms.
    Device type is determined by the device passed to enable().

    Each rank stores only a shard of each block's weights on host memory.
    Two device slots alternate: one holds current weights, one holds next
    weights. H2D and AllGather run asynchronously on dedicated streams,
    overlapped with computation.
    """

    def __init__(self, config: OffloadConfig, device: torch.device):
        super().__init__(config, device)

        self.copy_stream = current_omni_platform.Stream()
        self.comm_stream = current_omni_platform.Stream()
        self.dp_group: torch.distributed.ProcessGroup | None = None
        self.dp_size = config.dp_size
        self.rank = 0
        self._blocks: list[list[nn.Module]] = []
        self._all_hook_groups: list[list[DistributedLayerwiseOffloadHook]] = []

    def _load_weights_via_mmap(self, pipeline: nn.Module, modules) -> None:
        """Load DiT weights from safetensors via mmap views (no RSS).

        When the transformer is created on meta device, this method
        replaces meta params with mmap views of the checkpoint files.
        The views point to OS page cache (shared across ranks), so
        no private copy is created.  _shard_and_pin then copies only
        the 1/dp_size shard from the mmap view to a private buffer.

        Non-DiT modules (VAE, encoders) are NOT affected — they were
        created on CPU with real weights via from_pretrained.
        """
        import glob
        import json

        from safetensors import safe_open

        model_path = self.config.model_path
        if not model_path:
            logger.warning("No model_path for mmap weight loading, skipping")
            return

        # Resolve HF repo ID to local snapshot path.
        # If model_path is a local directory, use it directly; otherwise
        # download the safetensors index + weight files via HuggingFace Hub.
        if not os.path.isdir(model_path):
            from vllm.model_executor.model_loader.weight_utils import (
                download_weights_from_hf,
            )

            logger.info("model_path %s is not local, downloading from HF", model_path)
            model_path = download_weights_from_hf(
                model_name_or_path=model_path,
                cache_dir=None,
                allow_patterns=["*.safetensors", "*.safetensors.index.json"],
            )

        # Build {checkpoint_key: file_path} from safetensors index
        weight_map: dict[str, str] = {}
        for idx_file in glob.glob(os.path.join(model_path, "**", "*.safetensors.index.json"), recursive=True):
            idx_dir = os.path.dirname(idx_file)
            with open(idx_file) as f:
                idx = json.load(f)
            for key, filename in idx.get("weight_map", {}).items():
                weight_map[key] = os.path.join(idx_dir, filename)

        if not weight_map:
            logger.warning("No safetensors index found at %s", model_path)
            return

        logger.info("Loading DiT weights via mmap (meta → page cache, no RSS): %d keys", len(weight_map))

        # Build reverse mapping {model_param_name: (ckpt_key, file_path)} using
        # the pipeline's _remap_ckpt_key (if available).  This handles the
        # model-specific checkpoint key remapping (e.g., Cosmos3's
        # layers.0.mlp_moe_gen.gate_proj → transformer.gen_layers.0.mlp.gate_proj).
        remap_fn = getattr(type(pipeline), "_remap_ckpt_key", None)
        model_to_ckpt: dict[str, tuple[str, str]] = {}
        if remap_fn is not None:
            for ckpt_key, file_path in weight_map.items():
                model_name = remap_fn(ckpt_key)
                if model_name is not None:
                    model_to_ckpt[model_name] = (ckpt_key, file_path)
            logger.info(
                "Built reverse remap: %d model params from %d checkpoint keys", len(model_to_ckpt), len(weight_map)
            )
        else:
            # No remap function — use checkpoint keys directly
            for ckpt_key, file_path in weight_map.items():
                model_to_ckpt[ckpt_key] = (ckpt_key, file_path)

        # Collect all meta params in DiT modules
        dit_ids = set()
        for dit_module in modules.dits:
            dit_ids.update(id(m) for m in dit_module.modules())

        # Cache open file handles
        file_cache: dict[str, Any] = {}
        loaded = 0
        skipped = 0

        # Discover blocks first so we can distinguish block params (deferred
        # to _shard_and_pin) from non-block params (loaded here as mmap views).
        block_module_ids: set[int] = set()
        for dit_module in modules.dits:
            blocks_attr_names, blocks = self.get_blocks_from_dit(dit_module)
            for block in blocks:
                block_module_ids.update(id(m) for m in block.modules())

        for name, param in pipeline.named_parameters():
            if not (hasattr(param, "is_meta") and param.is_meta):
                continue

            # Check if this param is inside a block (will be handled by
            # _shard_and_pin via mmap views in section 2 below).
            parent = pipeline
            parts = name.split(".")
            is_block_param = False
            for part in parts[:-1]:
                try:
                    if part.isdigit():
                        parent = parent[int(part)]
                    else:
                        parent = getattr(parent, part)
                except (AttributeError, IndexError, TypeError):
                    break
                if id(parent) in block_module_ids:
                    is_block_param = True
                    break

            if is_block_param:
                # Block param — will be loaded as mmap view in section 2
                continue

            # Non-block param (merger, patch_embed, norm, proj_out, etc.)
            # Look up in reverse remap mapping (handles Cosmos3 key remapping).
            entry = model_to_ckpt.get(name)
            if entry is None:
                skipped += 1
                continue

            ckpt_key, file_path = entry
            if file_path not in file_cache:
                file_cache[file_path] = safe_open(file_path, framework="pt", device="cpu")
            f = file_cache[file_path]
            tensor = f.get_tensor(ckpt_key)

            # Replace meta param with mmap view (no copy!)
            parent = pipeline
            for part in parts[:-1]:
                if part.isdigit():
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)
            parent._parameters[parts[-1]] = torch.nn.Parameter(tensor, requires_grad=param.requires_grad)
            loaded += 1

        logger.info("Mmap weight loading: %d non-block params loaded, %d skipped", loaded, skipped)

        # Now load DiT block params from safetensors using mmap views.
        # Each block param is assigned an mmap view (no RSS).
        # _shard_and_pin will later copy the shard portion to a private buffer.
        block_loaded = 0
        for dit_idx, dit_module in enumerate(modules.dits):
            dit_name = modules.dit_names[dit_idx]
            blocks_attr_names, blocks = self.get_blocks_from_dit(dit_module)
            if not blocks:
                continue

            # Determine the blocks_attr for this DiT module (e.g. "layers" or "gen_layers")
            blocks_attr = None
            for attr_name in blocks_attr_names:
                if hasattr(dit_module, attr_name):
                    candidate = getattr(dit_module, attr_name)
                    if isinstance(candidate, nn.ModuleList) and len(candidate) > 1:
                        blocks_attr = attr_name
                        break

            for block_idx, block in enumerate(blocks):
                # Build the full model param prefix for this block
                # e.g. "transformer.language_model.layers.0" or "transformer.gen_layers.0"
                block_full_prefix = f"{dit_name}.{blocks_attr}.{block_idx}"

                for bname, bparam in block.named_parameters():
                    if not (hasattr(bparam, "is_meta") and bparam.is_meta):
                        continue

                    # Use reverse remap to find checkpoint key
                    full_param_name = f"{block_full_prefix}.{bname}"
                    entry = model_to_ckpt.get(full_param_name)
                    if entry is None:
                        skipped += 1
                        continue

                    ckpt_key, file_path = entry
                    if file_path not in file_cache:
                        file_cache[file_path] = safe_open(file_path, framework="pt", device="cpu")
                    f = file_cache[file_path]
                    tensor = f.get_tensor(ckpt_key)

                    # Replace meta param with mmap view
                    parent = block
                    bparts = bname.split(".")
                    for part in bparts[:-1]:
                        if part.isdigit():
                            parent = parent[int(part)]
                        else:
                            parent = getattr(parent, part)
                    parent._parameters[bparts[-1]] = torch.nn.Parameter(tensor, requires_grad=bparam.requires_grad)
                    block_loaded += 1

        logger.info("Mmap block loading: %d params loaded, %d skipped", block_loaded, skipped)

        # Keep file handles open — _shard_and_pin will read from the mmap views.
        # They will be released after _shard_and_pin completes (when params are
        # replaced with offload placeholders).
        self._mmap_file_cache = file_cache
        self._mmap_model_to_ckpt = model_to_ckpt

        # The regular loader runs model-specific post-load transforms after
        # assigning checkpoint tensors. The mmap path bypasses that loader, so
        # preserve the same lifecycle for transforms such as Cosmos3's fp32
        # timestep embedder.
        for dit_module in modules.dits:
            post_load_weights = getattr(dit_module, "post_load_weights", None)
            if callable(post_load_weights):
                post_load_weights()

    def _load_module_weights_from_mmap(self, module: nn.Module, dit_name: str, child_name: str) -> None:
        """Load a non-block DiT submodule's weights from safetensors via mmap."""
        from safetensors import safe_open

        if not hasattr(self, "_mmap_file_cache") or not self._mmap_file_cache:
            return

        file_cache = self._mmap_file_cache
        model_to_ckpt = getattr(self, "_mmap_model_to_ckpt", {})
        loaded = 0

        for pname, param in module.named_parameters():
            if not (hasattr(param, "is_meta") and param.is_meta):
                continue

            # Build full model param name and look up in reverse remap
            full_name = f"{dit_name}.{child_name}.{pname}"
            entry = model_to_ckpt.get(full_name)
            if entry is None:
                # Try without dit_name prefix
                full_name = f"{child_name}.{pname}"
                entry = model_to_ckpt.get(full_name)
            if entry is None:
                continue

            ckpt_key, file_path = entry
            if file_path not in file_cache:
                file_cache[file_path] = safe_open(file_path, framework="pt", device="cpu")
            f = file_cache[file_path]
            tensor = f.get_tensor(ckpt_key)

            # Replace meta param with mmap view
            parent = module
            parts = pname.split(".")
            for part in parts[:-1]:
                if part.isdigit():
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)
            parent._parameters[parts[-1]] = torch.nn.Parameter(tensor, requires_grad=param.requires_grad)
            loaded += 1

        if loaded > 0:
            logger.info("Loaded %d params for submodule %s.%s via mmap", loaded, dit_name, child_name)

    def _init_dp_group(self) -> None:
        """Reuse the process group initialized by parallel_state.

        When DP > 1, uses the DP group (handles strided groups with SP/CFG/PP).
        When DP = 1 but SP > 1 (and dp_size was set to sp_size in OffloadConfig),
        uses the SP group so weights are sharded across SP ranks.
        """
        if self.dp_size <= 1:
            logger.info("Distributed layerwise offload: dp_size=1, running without AllGather")
            self.dp_group = None
            return

        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "torch.distributed is not initialized. "
                "Distributed layerwise offload with dp_size > 1 requires "
                "an initialized process group."
            )

        from vllm_omni.diffusion.distributed.parallel_state import (
            get_data_parallel_world_size,
            get_dp_group,
        )

        # Determine which parallel group to use for sharding.
        # When data_parallel_size > 1, use the DP group.
        # When data_parallel_size == 1 but dp_size > 1 (set from sp_size in
        # OffloadConfig), use the SP group for weight sharding.
        dp_world = get_data_parallel_world_size()
        if dp_world > 1:
            coord = get_dp_group()
        else:
            from vllm_omni.diffusion.distributed.parallel_state import get_sp_group

            coord = get_sp_group()
            logger.info(
                "Distributed layerwise offload: DP=1, using SP group (world_size=%d) for weight sharding",
                coord.world_size,
            )

        self.dp_group = coord.device_group
        self.rank = coord.rank_in_group
        self.dp_size = coord.world_size

        logger.info(
            "Distributed layerwise offload: dp_size=%d, rank_in_group=%d, global_rank=%d, group_ranks=%s",
            self.dp_size,
            self.rank,
            coord.rank,
            coord.ranks,
        )

    def _register_on_demand_hook(self, module: nn.Module, label: str) -> None:
        """Move *module* to GPU permanently.

        Previously, on-demand hooks moved the module to GPU before forward
        and back to CPU after forward (with synchronize + empty_cache).
        However, _post_forward's global synchronize() and empty_cache()
        disrupt the async prefetch pipeline (copy_stream / comm_stream)
        used by distributed layerwise offload, causing garbled output.

        Moving the module to GPU permanently avoids this issue at the
        cost of slightly higher HBM usage (VAE/encoders are small).
        """
        module.to(self.device)
        logger.info("Moved %s (%s) to GPU (resident)", label, module.__class__.__name__)

    def _try_layerwise_offload_submodule(self, module: nn.Module, name: str) -> bool:
        """Try to apply layerwise offload to a large submodule's blocks.

        Searches for common block-list attributes (layers, blocks, h).
        If found, applies the same layerwise streaming hooks used for the DiT,
        so only 2 layers reside on GPU at a time instead of the full module.

        Returns True if layerwise offload was applied, False otherwise.
        """
        from operator import attrgetter

        blocks = None
        blocks_attr = None
        for attr_name in ("layers", "blocks", "h", "model.layers"):
            try:
                candidate = attrgetter(attr_name)(module)
            except AttributeError:
                continue
            if isinstance(candidate, nn.ModuleList) and len(candidate) > 1:
                blocks = candidate
                blocks_attr = attr_name
                break

        if blocks is None:
            return False

        num_blocks = len(blocks)
        logger.info(
            "Distributed layerwise offload for submodule '%s.%s' (%d blocks, %.0f MB total, dp_size=%d)",
            name,
            blocks_attr,
            num_blocks,
            sum(p.nelement() * p.element_size() for p in module.parameters()) / 1048576,
            self.dp_size,
        )

        # Move non-block parts of the submodule to GPU (small: embeddings, norms)
        for child_name, child in module.named_children():
            if child_name != blocks_attr:
                child.to(self.device)

        # Apply distributed hooks (1/4 sharding + AllGather, same as DiT)
        # Pass shared_buffers=[None,None] to defer per-hook allocation
        # (prevents OOM on large models where N hooks × 2 buffers >> HBM)
        last_block, first_block = blocks[-1], blocks[0]
        last_hook = apply_distributed_block_hook(
            last_block,
            first_block,
            self.device,
            self.dp_group,
            self.dp_size,
            self.rank,
            self.copy_stream,
            self.comm_stream,
            self.config.pin_cpu_memory,
            shared_buffers=[None, None],
        )
        sub_hooks = [last_hook]
        for i, block in enumerate(blocks[:-1]):
            next_block = blocks[(i + 1) % num_blocks]
            hook = apply_distributed_block_hook(
                block,
                next_block,
                self.device,
                self.dp_group,
                self.dp_size,
                self.rank,
                self.copy_stream,
                self.comm_stream,
                self.config.pin_cpu_memory,
                shared_buffers=[None, None],
            )
            sub_hooks.append(hook)

        # Wire backward references + slot alternation
        for i in range(len(sub_hooks)):
            sub_hooks[i]._prev_hook = sub_hooks[i - 1]
        for i, hook in enumerate(sub_hooks):
            hook.current_slot = i % 2
        # Mark first hook in this group — it must always sync-prefetch on
        # entry because another group may have overwritten the shared slot.
        if sub_hooks:
            sub_hooks[0]._is_group_first = True

        # Defer buffer allocation and prefetch to enable() unified allocation
        self._all_hook_groups.append(sub_hooks)
        self._blocks.append(blocks)
        return True

    def enable(self, pipeline: nn.Module) -> None:
        if self.enabled:
            logger.warning("DistributedLayerwiseOffloadBackend already enabled")
            return

        self._on_demand_shard_infos: list[dict] = []
        self._on_demand_handles: list[Any] = []

        # Initialize DP group (if not already done by early init)
        if self.dp_group is None and self.dp_size > 1:
            self._init_dp_group()

        modules = ModuleDiscovery.discover(pipeline)
        if not modules.dits:
            logger.warning("No DiT/transformer modules found, skipping distributed layer-wise offloading")
            return

        # If transformer was created on meta device, load weights from
        # safetensors via mmap views (no RSS — data stays in page cache).
        # This avoids O(dp_size × model_size) peak RSS during load_model.
        _has_meta = any(
            (p.is_meta if hasattr(p, "is_meta") else False) for dit in modules.dits for p in dit.parameters()
        )
        if _has_meta:
            self._load_weights_via_mmap(pipeline, modules)

        # Keep VAE/encoders on CPU; move to GPU on-demand via hooks.
        # This saves ~4.3 GB HBM per card (VAE 1.3 + encoder 1.1 + sound 1.9)
        # during the DiT forward pass.  They are only needed briefly for
        # text-encoding (before DiT) and VAE-decode (after DiT).
        for enc in modules.encoders:
            self._register_on_demand_hook(enc, "encoder")
        for vae in modules.vaes:
            self._register_on_demand_hook(vae, "vae")

        # Move resident modules to GPU (small modules needed every forward)
        for name, module in zip(modules.resident_names, modules.resident_modules):
            try:
                module.to(self.device)
            except Exception as exc:
                logger.debug("Failed to move resident module %s to GPU: %s", name, exc)

        logger.info("Applying distributed layer-wise offloading on %s", modules.dit_names)

        # Collect all DiT module objects to detect submodules that are
        # already handled as a separate DiT module (avoids duplicate hooks).
        all_dit_modules = set(id(m) for m in modules.dits)

        # Apply hooks for each DiT module
        for i, dit_module in enumerate(modules.dits):
            dit_name = modules.dit_names[i]
            logger.info(f"Applying hooks on {dit_name} ({dit_module.__class__.__name__})")

            blocks_attr_names, blocks = DistributedLayerwiseOffloadBackend.get_blocks_from_dit(dit_module)

            if not blocks:
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

            # Move non-block modules to GPU (they stay resident).
            # Large modules (> 1 GB) use on-demand hooks instead — they are
            # only needed briefly (e.g. language_model for text encoding)
            # and keeping them resident wastes HBM during the DiT loop.
            _ON_DEMAND_THRESHOLD = 1024  # MB
            for name, m in dit_module.named_children():
                if name not in blocks_attr_names:
                    _has_meta = any((p.is_meta if hasattr(p, "is_meta") else False) for p in m.parameters())
                    if _has_meta:
                        # Meta params: load from safetensors before moving to GPU
                        self._load_module_weights_from_mmap(m, dit_name, name)
                    _mb = (
                        sum(
                            (p.nelement() * p.element_size() if not (hasattr(p, "is_meta") and p.is_meta) else 0)
                            for p in m.parameters()
                        )
                        / 1048576
                    )
                    if _mb > _ON_DEMAND_THRESHOLD:
                        # Skip if this submodule is already a separate DiT
                        # module (e.g. transformer.language_model when both
                        # "transformer.language_model" and "transformer" are
                        # in _dit_modules) — hooks already applied.
                        if id(m) in all_dit_modules:
                            logger.info("Submodule '%s' is already a DiT module, skipping layerwise offload", name)
                        elif self._try_layerwise_offload_submodule(m, name):
                            pass  # layerwise hooks applied
                        else:
                            self._register_on_demand_hook(m, name)
                    else:
                        try:
                            m.to(self.device)
                        except (NotImplementedError, RuntimeError):
                            # Meta params: load from safetensors via mmap, then move to GPU
                            self._load_module_weights_from_mmap(m, dit_name, name)
                            # Materialize any remaining meta buffers (e.g. RoPE
                            # inv_freq, timestep freqs) on device and
                            # recompute them from their original formulas.
                            # These are non-persistent buffers not present in
                            # the checkpoint — they must be reconstructed, not
                            # left as torch.empty.
                            # NOTE: Do NOT call reset_parameters() here — it
                            # would re-initialize ALL parameters (including
                            # mmap-loaded weights) with random values.
                            _has_meta_buf = any(hasattr(b, "is_meta") and b.is_meta for b in m.buffers(recurse=True))
                            if _has_meta_buf:
                                # Save mmap-loaded param values before
                                # reset_parameters (which re-initializes
                                # ALL params with random values).
                                saved_params = {
                                    n: p.data.clone()
                                    for n, p in m.named_parameters()
                                    if not (hasattr(p, "is_meta") and p.is_meta)
                                }
                                m.to_empty(device=self.device)
                                for submod in m.modules():
                                    if hasattr(submod, "reset_parameters"):
                                        submod.reset_parameters()
                                # Restore mmap-loaded weights
                                for n, p in m.named_parameters():
                                    if n in saved_params:
                                        p.data.copy_(saved_params[n])
                            try:
                                m.to(self.device)
                            except Exception:
                                logger.warning("Module %s still has meta params after mmap load", name)
                else:
                    logger.debug(f"Skipped blocks module {name}")

            # Move top-level params/buffers to GPU
            for param in dit_module._parameters.values():
                if param is not None and not (hasattr(param, "is_meta") and param.is_meta):
                    param.data = param.data.to(self.device, non_blocking=True)
            for buffer in dit_module._buffers.values():
                if buffer is not None:
                    buffer.data = buffer.data.to(self.device, non_blocking=True)

            # Register hooks in a circular sliding window:
            # last block prefetches first block, block i prefetches block (i+1)
            # All hooks share 2 global device buffers (RFC: "exactly two layers on device")
            last_block, first_block = blocks[-1], blocks[0]

            # Pass 1: create hooks with deferred buffer allocation
            # (shared_buffers=[None,None] prevents per-hook OOM on large models)
            last_hook = apply_distributed_block_hook(
                last_block,
                first_block,
                self.device,
                self.dp_group,
                self.dp_size,
                self.rank,
                self.copy_stream,
                self.comm_stream,
                self.config.pin_cpu_memory,
                shared_buffers=[None, None],
            )

            block_hooks: list[DistributedLayerwiseOffloadHook] = [last_hook]
            for i, block in enumerate(blocks[:-1]):
                next_block = blocks[(i + 1) % num_blocks]
                hook = apply_distributed_block_hook(
                    block,
                    next_block,
                    self.device,
                    self.dp_group,
                    self.dp_size,
                    self.rank,
                    self.copy_stream,
                    self.comm_stream,
                    self.config.pin_cpu_memory,
                    shared_buffers=[None, None],
                )
                block_hooks.append(hook)

            # Wire backward references for cache-dit fallback
            for i in range(len(block_hooks)):
                block_hooks[i]._prev_hook = block_hooks[i - 1]

            # Initialize alternating slots
            for i, hook in enumerate(block_hooks):
                hook.current_slot = i % 2

            # Mark first hook in this group — shared buffers may be
            # overwritten by other groups between forwards.
            if block_hooks:
                block_hooks[0]._is_group_first = True

            # Defer buffer allocation — collected for unified allocation below
            self._all_hook_groups.append(block_hooks)
            self._blocks.append(blocks)

        if not self._all_hook_groups:
            return

        # Unified allocation: 2 shared output buffers + 2 shared shard buffers
        # sized to the max block across ALL module groups (gen_layers +
        # language_model).  Groups execute sequentially, so 2 buffers suffice.
        all_hooks: list[DistributedLayerwiseOffloadHook] = []
        for group in self._all_hook_groups:
            all_hooks.extend(group)

        unified_buffers = self._allocate_shared_buffers(all_hooks)
        unified_shard_buffers = None
        if self.dp_size > 1:
            unified_shard_buffers = self._allocate_shared_shard_buffers(all_hooks)
        for hook in all_hooks:
            hook.gpu_buffers = unified_buffers
            hook._owns_buffers = False
            if unified_shard_buffers is not None:
                hook.gpu_shard_buffers = unified_shard_buffers

        # Prefetch first block of the FIRST module group only.
        # Subsequent groups share the same 2 device buffers; prefetching
        # them now would overwrite the first group's data in the shared
        # buffer (both groups default to slot 0).  Instead, subsequent
        # groups' first blocks remain as meta placeholders, and their
        # pre_forward will sync-prefetch on-demand via the is_materialized
        # check — by which point the first group's forward has completed
        # and its buffer slots are free.
        if self._all_hook_groups:
            group = self._all_hook_groups[0]
            first_slot = group[0].current_slot
            group[-1].prefetch_layer(slot=first_slot, non_blocking=False)
            group[-1].get_weights(first_slot)

        total_blocks = sum(len(b) for b in self._blocks)
        logger.info(
            f"Distributed layer-wise offloading enabled on {total_blocks} blocks "
            f"across {len(self._all_hook_groups)} group(s), dp_size={self.dp_size}, "
            f"unified shared_buffers=2"
        )

        self.enabled = True

        # Release mmap file handles — _shard_and_pin has copied all shards,
        # params now point to offload placeholders (not mmap views).
        self._release_mmap_handles()

        # Assign GPU buffers to sharded on-demand modules (VAE/encoders).
        # Each module gets a dedicated input (shard-sized) and output
        # (full-sized) buffer.  VAE/encoders run before/after DiT, never
        # concurrently, so peak HBM = max(DiT, VAE) not sum.
        for si in self._on_demand_shard_infos:
            out_bufs: dict[torch.dtype, torch.Tensor] = {}
            in_bufs: dict[torch.dtype, torch.Tensor] = {}
            for dtype, shard in si["cpu_shards"].items():
                # AllGather output = dp_size * shard_size
                full_size = shard.numel() * self.dp_size if self.dp_size > 1 else shard.numel()
                out_bufs[dtype] = torch.empty(full_size, dtype=dtype, device=self.device)
                in_bufs[dtype] = torch.empty(shard.shape, dtype=dtype, device=self.device)
            si["gpu_output"] = out_bufs
            si["gpu_input"] = in_bufs
            _mb = sum(t.nelement() * t.element_size() for t in out_bufs.values()) / 1048576
            logger.info("Allocated %.0f MB GPU buffer for sharded on-demand module", _mb)

        self._cleanup_after_loading()

    def _release_mmap_handles(self) -> None:
        """Release safetensors mmap file handles."""
        if hasattr(self, "_mmap_file_cache"):
            self._mmap_file_cache.clear()
            del self._mmap_file_cache
            if hasattr(self, "_mmap_model_to_ckpt"):
                del self._mmap_model_to_ckpt
            logger.info("Released safetensors mmap file handles")

    def _cleanup_after_loading(self) -> None:
        """Synchronize and release freed device/CPU memory after sharding."""
        current_omni_platform.synchronize()
        current_omni_platform.empty_cache()
        import ctypes as _ctypes
        import gc as _gc

        _gc.collect()
        try:
            _ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    def disable(self) -> None:
        if not self.enabled:
            return

        for blocks in self._blocks:
            for block in blocks:
                remove_distributed_block_hook(block)

        for h in getattr(self, "_on_demand_handles", []):
            h.remove()
        self._on_demand_handles = []
        self._on_demand_shard_infos = []

        self._blocks.clear()
        self._all_hook_groups.clear()
        self.enabled = False
        logger.info("Distributed layer-wise offloading disabled")

    @staticmethod
    def _allocate_shared_buffers(
        hooks: list[DistributedLayerwiseOffloadHook],
    ) -> list[dict[torch.dtype, torch.Tensor] | None]:
        """Allocate exactly 2 shared device buffers sized to the largest block.

        All hooks share these 2 buffers. At any time, slot 0 holds the current
        layer's weights and slot 1 holds the next layer's weights (or vice
        versa). This ensures only 2 layers' worth of weights reside on device,
        regardless of the total number of blocks.
        """
        max_sizes: dict[torch.dtype, int] = {}
        for hook in hooks:
            dp = hook.dp_size
            for dtype, metas in hook.metadata.items():
                total = sum(m["numel"] for m in metas)
                # AllGather output = dp * ceil(total/dp) (padded for equal shards)
                if dp > 1:
                    total = ((total + dp - 1) // dp) * dp
                if dtype not in max_sizes or total > max_sizes[dtype]:
                    max_sizes[dtype] = total

        device = hooks[0].device
        shared_buffers: list[dict[torch.dtype, torch.Tensor] | None] = [None, None]
        for slot in range(2):
            gpu_weights: dict[torch.dtype, torch.Tensor] = {}
            for dtype, total_numel in max_sizes.items():
                gpu_weights[dtype] = torch.empty(total_numel, dtype=dtype, device=device)
            shared_buffers[slot] = gpu_weights

        logger.info(
            "Allocated 2 shared device buffers (max block size: %s)",
            {str(k): f"{v * _dtype_size(k) / 1024 / 1024:.1f}MB" for k, v in max_sizes.items()},
        )
        return shared_buffers

    @staticmethod
    def _allocate_shared_shard_buffers(
        hooks: list[DistributedLayerwiseOffloadHook],
    ) -> list[dict[torch.dtype, torch.Tensor] | None]:
        """Allocate 2 shared shard (AllGather input) buffers sized to the
        largest per-rank shard.

        All hooks share these 2 input buffers — the same device address is
        reused for every layer's AllGather so HCCL can reuse its internal
        communication buffers (preventing rank-dependent HBM imbalance).
        Cost: 2 × max_shard_size (~184 MB) instead of 2 × N_hooks × shard.
        """
        max_shard_sizes: dict[torch.dtype, int] = {}
        for hook in hooks:
            for dtype, shard in hook.cpu_shards.items():
                numel = shard.numel()
                if dtype not in max_shard_sizes or numel > max_shard_sizes[dtype]:
                    max_shard_sizes[dtype] = numel

        device = hooks[0].device
        shared_shard_buffers: list[dict[torch.dtype, torch.Tensor] | None] = [None, None]
        for slot in range(2):
            shard_bufs: dict[torch.dtype, torch.Tensor] = {}
            for dtype, numel in max_shard_sizes.items():
                shard_bufs[dtype] = torch.empty(numel, dtype=dtype, device=device)
            shared_shard_buffers[slot] = shard_bufs

        logger.info(
            "Allocated 2 shared shard buffers (max shard size: %s)",
            {str(k): f"{v * _dtype_size(k) / 1024 / 1024:.1f}MB" for k, v in max_shard_sizes.items()},
        )
        return shared_shard_buffers

    # ------------------------------------------------------------------ #
    #  Block discovery (reuses LayerWiseOffloadBackend logic)            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_blocks_attr_names(model: nn.Module) -> list[str]:
        """Get block attribute names from model class."""
        attrs: list[str] = getattr(model.__class__, "_layerwise_offload_blocks_attrs", [])

        if not attrs:
            old_attr = getattr(model.__class__, "_layerwise_offload_blocks_attr", None)
            if old_attr is not None:
                logger.warning(
                    "'_layerwise_offload_blocks_attr' is deprecated, "
                    "please use '_layerwise_offload_blocks_attrs' instead. "
                    "Example: _layerwise_offload_blocks_attrs = ['blocks']"
                )
                attrs = [old_attr] if isinstance(old_attr, str) else list(old_attr)

        return attrs

    @staticmethod
    def set_blocks_attr_names(model: nn.Module, names: list[str]) -> None:
        if not hasattr(model.__class__, "_layerwise_offload_blocks_attrs"):
            setattr(model.__class__, "_layerwise_offload_blocks_attrs", names)

    @staticmethod
    def get_blocks_from_dit(model: nn.Module) -> tuple[list[str], list[nn.Module]]:
        """Retrieve blocks and attribute names from provided DiT model."""
        blocks_attr_names = DistributedLayerwiseOffloadBackend.get_blocks_attr_names(model)
        if not blocks_attr_names:
            logger.warning(
                f"No _layerwise_offload_blocks_attrs defined for {model.__class__.__name__}, "
                "skipping distributed layerwise offloading"
            )
            return [], []

        blocks: list[nn.Module] = []
        for name in blocks_attr_names:
            attr = getattr(model, name, None)
            if attr is None:
                raise AttributeError(
                    f"Attribute '{name}' declared in _layerwise_offload_blocks_attrs "
                    f"does not exist on model {model.__class__.__name__}"
                )
            try:
                attr_iter = iter(attr)
            except TypeError:
                if isinstance(attr, nn.Module):
                    logger.warning(
                        "Attribute '%s' on %s is not iterable; treating it as one block.",
                        name,
                        model.__class__.__name__,
                    )
                    blocks.append(attr)
                    continue

                logger.warning(
                    "Attribute '%s' on %s is not iterable (got %s); skipping it.",
                    name,
                    model.__class__.__name__,
                    type(attr).__name__,
                )
            else:
                blocks.extend(attr_iter)

        if not blocks:
            logger.warning(
                "No blocks found in %s for %s, skipping distributed layerwise offloading",
                blocks_attr_names,
                model.__class__.__name__,
            )
            return [], []

        return blocks_attr_names, blocks
