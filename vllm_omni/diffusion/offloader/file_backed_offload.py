# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared file-backed BF16 block transport; adapters own weight mappings."""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Generator
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from io import FileIO
from pathlib import Path
from traceback import clear_frames

import torch
from torch import nn

_CHECKPOINT_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32}


def bind_parameter(module: nn.Module, name: str, storage: torch.Tensor) -> nn.Parameter:
    """Replace storage across meta/real devices, retaining vLLM loader metadata."""
    template = module.get_parameter(name)
    parameter = template.__class__.__new__(template.__class__, storage)
    parameter.__dict__.update(template.__dict__)
    parameter.requires_grad_(template.requires_grad)
    module.register_parameter(name, parameter)
    return parameter


class FileBackedBlockOffloader:
    """Manage bounded staging, prefetch and cleanup for a model-owned plan."""

    def __init__(
        self,
        model: nn.Module,
        layers: nn.ModuleList,
        manifest: dict,
        device: torch.device,
        *,
        num_slots: int = 2,
    ) -> None:
        """Describe storage without allocating CPU/GPU weight slots.

        The manifest supplies model_path, shards, blocks and resident_tensors.
        Adapters validate source ranges and tensor layouts before construction.
        Entries describe BF16 block weights or BF16/F32 resident state using
        runtime_name, shape, dtype, bytes and sources. CPU buffers already
        materialized by the adapter are retained for use between runs. Model
        parameters must be on meta; the original objects are reused on offload.
        Invalid slot counts or layer descriptions raise ValueError.
        """
        if type(num_slots) is not int or num_slots <= 0:
            raise ValueError("num_slots must be a positive integer")
        self.blocks = manifest["blocks"]
        if len(self.blocks) != len(layers) or not self.blocks:
            raise ValueError("The manifest must describe every nonempty layer")
        for block in self.blocks:
            if type(block["bytes"]) is not int or block["bytes"] <= 0 or block["bytes"] % 2:
                raise ValueError("Block byte counts must be positive BF16-aligned integers")
        self.max_bytes = max(block["bytes"] for block in self.blocks)
        self.num_slots = num_slots
        self.device = device
        self.root = Path(manifest["model_path"])
        self.manifest = manifest
        self.layers = layers
        self.resident = manifest["resident_tensors"]
        self.entries = self.resident + [entry for block in self.blocks for entry in block["parameters"]]
        self.targets = {}
        self._offloaded_tensors = {}
        for entry in self.entries:
            parent, _, name = entry["runtime_name"].rpartition(".")
            module = model.get_submodule(parent)
            is_buffer = entry.get("buffer", False)
            original = module.get_buffer(name) if is_buffer else module.get_parameter(name)
            if not original.is_meta and (not is_buffer or original.device.type != "cpu"):
                raise ValueError(f"File-backed parameters need meta and buffers need CPU/meta: {entry['runtime_name']}")
            self.targets[entry["runtime_name"]] = (module, name)
            self._offloaded_tensors[entry["runtime_name"]] = original
        self.cpu_slots: list[torch.Tensor] = []
        self.gpu_slots: list[torch.Tensor] = []
        self.handles: dict[str, FileIO] = {}
        self.copy_stream: torch.Stream | None = None
        self.executor: ThreadPoolExecutor | None = None
        self.futures: dict[int, Future[torch.Event]] = {}
        self.h2d_done: list[torch.Event | None] = [None] * self.num_slots
        self.compute_done: list[torch.Event | None] = [None] * self.num_slots
        self.stats = {"bytes_read": 0, "blocks_read": 0, "read_seconds": 0.0}

    def _read_into(self, source: dict, target: memoryview, offset: int = 0) -> None:
        # Release every view on failure too: a traceback must not keep a whole
        # pinned slot alive after deactivate() discards the owning tensors.
        with target:
            shard = source["shard"]
            if shard not in self.handles:
                self.handles[shard] = FileIO(self.root / shard, "rb")
            handle = self.handles[shard]
            handle.seek(source["file_offset"] + offset)
            done = 0
            started = time.perf_counter()
            while done < len(target):
                with target[done:] as remaining:
                    count = handle.readinto(remaining)
                if not count:
                    raise EOFError(f"Checkpoint ended while reading {source['source_name']}")
                done += count
                self.stats["bytes_read"] += count
            self.stats["read_seconds"] += time.perf_counter() - started

    def _bind_tensor(self, entry: dict, storage: torch.Tensor) -> None:
        module, name = self.targets[entry["runtime_name"]]
        if entry.get("buffer", False):
            module._buffers[name] = storage
        elif isinstance(storage, nn.Parameter):
            module.register_parameter(name, storage)
        else:
            bind_parameter(module, name, storage)

    def _restore_offloaded_tensors(self, entries: list[dict]) -> None:
        for entry in entries:
            self._bind_tensor(entry, self._offloaded_tensors[entry["runtime_name"]])

    def _convert_weight_layout(self, entry: dict, storage: torch.Tensor) -> None:
        """Apply an adapter's in-place layout conversion on the copy stream."""

    def _load_resident_tensors(self) -> None:
        assert self.copy_stream is not None
        copy_pending = False
        # Resident tensors can exceed one slot; reuse staging in bounded chunks.
        with self.copy_stream, memoryview(self.cpu_slots[0].view(torch.uint8).numpy()) as staging:
            for entry in self.resident:
                storage = torch.empty(entry["shape"], dtype=_CHECKPOINT_DTYPES[entry["dtype"]], device=self.device)
                try:
                    row_bytes = math.prod(entry["shape"][1:]) * storage.element_size()
                    for source in entry["sources"]:
                        destination_offset = source["concat_offset"] * row_bytes
                        for offset in range(0, source["bytes"], self.max_bytes):
                            size = min(self.max_bytes, source["bytes"] - offset)
                            # The previous H2D copy must finish before staging is overwritten.
                            if copy_pending:
                                self.copy_stream.synchronize()
                            self._read_into(source, staging[:size], offset)
                            storage.view(torch.uint8).view(-1)[
                                destination_offset + offset : destination_offset + offset + size
                            ].copy_(self.cpu_slots[0].view(torch.uint8)[:size], non_blocking=True)
                            copy_pending = True
                    self._convert_weight_layout(entry, storage)
                    self._bind_tensor(entry, storage)
                finally:
                    del storage
        if copy_pending:
            self.copy_stream.synchronize()

    def _validate_checkpoint(self) -> None:
        for shard in self.manifest["shards"]:
            path = self.root / shard["shard"]
            with path.open("rb") as handle:
                header = handle.read(shard["header_bytes"])
            if (
                path.stat().st_size != shard["file_bytes"]
                or int.from_bytes(header[:8], "little") != shard["header_bytes"] - 8
                or hashlib.sha256(header[8:]).hexdigest() != shard["header_sha256"]
            ):
                raise ValueError(f"Checkpoint manifest does not match shard header: {path.name}")

    def activate(self) -> None:
        if self.executor is not None:
            return
        self._validate_checkpoint()
        try:
            self.copy_stream = torch.Stream(device=self.device)
            for _ in range(self.num_slots):
                self.cpu_slots.append(
                    torch.empty(self.max_bytes // 2, dtype=torch.bfloat16, device="cpu", pin_memory=True)
                )
            with self.copy_stream:
                for _ in range(self.num_slots):
                    self.gpu_slots.append(torch.empty(self.max_bytes // 2, dtype=torch.bfloat16, device=self.device))
            self._load_resident_tensors()
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="file-backed-io")
        except BaseException:
            self.deactivate()
            raise

    @torch.inference_mode()
    def _prefetch(self, index: int, compute_done: torch.Event | None) -> torch.Event:
        assert self.copy_stream is not None
        slot = index % self.num_slots
        h2d_done = self.h2d_done[slot]
        if h2d_done is not None:
            h2d_done.synchronize()
        block = self.blocks[index]
        with memoryview(self.cpu_slots[slot].view(torch.uint8).numpy()) as staging:
            for entry in block["parameters"]:
                row_bytes = math.prod(entry["shape"][1:]) * 2
                for source in entry["sources"]:
                    start = entry["buffer_offset"] + source["concat_offset"] * row_bytes
                    self._read_into(source, staging[start : start + source["bytes"]])
        ready = torch.Event(device=self.device)
        with self.copy_stream:
            if compute_done is not None:
                self.copy_stream.wait_event(compute_done)
            size = block["bytes"] // 2
            self.gpu_slots[slot][:size].copy_(self.cpu_slots[slot][:size], non_blocking=True)
            for entry in block["parameters"]:
                self._convert_weight_layout(entry, self._block_tensor(entry, slot))
            ready.record(self.copy_stream)
        self.h2d_done[slot] = ready
        self.stats["blocks_read"] += 1
        return ready

    def _submit(self, index: int) -> None:
        assert self.executor is not None
        self.futures[index] = self.executor.submit(self._prefetch, index, self.compute_done[index % self.num_slots])

    def _drain_prefetch(self) -> None:
        pending = list(self.futures.values())
        for future in pending:
            future.cancel()
        wait(pending)
        self.futures.clear()

    @contextmanager
    def run(self) -> Generator[None, None, None]:
        self.activate()
        try:
            for index in range(min(self.num_slots, len(self.blocks))):
                self._submit(index)
            with torch.inference_mode():
                yield
        except BaseException as error:
            # Attention traceback locals can own views of an entire GPU slot.
            clear_frames(error.__traceback__)
            self.deactivate()
            raise
        else:
            self._drain_prefetch()

    def _block_tensor(self, entry: dict, slot: int) -> torch.Tensor:
        """View a runtime tensor within a BF16 device slot."""
        start = entry["buffer_offset"] // 2
        return self.gpu_slots[slot][start : start + entry["bytes"] // 2].view(entry["shape"])

    @contextmanager
    def layer(self, index: int) -> Generator[nn.Module, None, None]:
        ready = self.futures.pop(index).result()
        stream = torch.accelerator.current_stream(self.device)
        stream.wait_event(ready)
        slot = index % self.num_slots
        entries = self.blocks[index]["parameters"]
        for entry in entries:
            self._bind_tensor(entry, self._block_tensor(entry, slot))
        try:
            yield self.layers[index]
        finally:
            done = torch.Event(device=self.device)
            done.record(stream)
            self.compute_done[slot] = done
            self._restore_offloaded_tensors(entries)
        if index + self.num_slots < len(self.blocks):
            self._submit(index + self.num_slots)

    def deactivate(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
        self.futures.clear()
        try:
            if self.copy_stream is not None:
                torch.accelerator.synchronize(self.device)
        finally:
            self._restore_offloaded_tensors(self.entries)
            self.cpu_slots.clear()
            self.gpu_slots.clear()
            self.h2d_done = [None] * self.num_slots
            self.compute_done = [None] * self.num_slots
            self.copy_stream = None
            for handle in self.handles.values():
                handle.close()
            self.handles.clear()
            # Return idle pinned buffers before the next model stage allocates.
            empty_host_cache = getattr(torch._C, "_host_emptyCache", None)
            if callable(empty_host_cache):
                empty_host_cache()
