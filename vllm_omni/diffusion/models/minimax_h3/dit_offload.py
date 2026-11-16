# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native H3 weights through two CPU slots and two GPU slots."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from vllm_omni.diffusion.offloader.base import OffloadBackend, OffloadConfig
from vllm_omni.diffusion.offloader.file_backed_offload import FileBackedBlockOffloader

from .minimax_h3_transformer import _reorder_grouped_qkv_to_qkv
from .offload_manifest import make_tensor_entry, read_weight_sources


class MiniMaxH3DiTOffloader(FileBackedBlockOffloader, OffloadBackend):
    def __init__(
        self,
        model: torch.nn.Module,
        root: str | Path,
        device: torch.device,
        *,
        num_slots: int = 2,
    ) -> None:
        OffloadBackend.__init__(self, OffloadConfig.from_od_config(model.od_config), device)
        self.model = model
        root = Path(root)
        sources, shards = read_weight_sources(root)
        blocks = [dict(parameters=[], bytes=0) for _ in model.blocks]
        resident = []
        tensors = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        tensors.update(buffers)
        for name, tensor in tensors.items():
            source = sources[name]
            entry = make_tensor_entry(name, tensor, [source], buffer=name in buffers)
            parent, _, local_name = name.rpartition(".")
            if name.startswith("blocks."):
                if tensor.dtype != torch.bfloat16:
                    raise ValueError(f"DiT streaming blocks require BF16: {name}")
                block = blocks[int(name.split(".")[1])]
                entry["buffer_offset"] = block["bytes"]
                block["parameters"].append(entry)
                block["bytes"] += entry["bytes"]
            else:
                resident.append(entry)
            if entry["buffer"]:
                # RoPE is needed while preparing inputs, before the denoise scope.
                with (root / source["shard"]).open("rb") as handle:
                    handle.seek(source["file_offset"])
                    data = bytearray(handle.read(source["bytes"]))
                value = torch.frombuffer(data, dtype=tensor.dtype).view(tensor.shape)
                model.get_submodule(parent)._buffers[local_name] = value

        manifest = {"model_path": str(root), "shards": shards, "blocks": blocks, "resident_tensors": resident}
        FileBackedBlockOffloader.__init__(self, model, model.blocks, manifest, device, num_slots=num_slots)
        self._hook_handles, self._layer_contexts = [], {}
        self._run_context = None

    def _convert_weight_layout(self, entry: dict, storage: torch.Tensor) -> None:
        if entry["runtime_name"].endswith(".attn.qkv_proj.weight"):
            storage.copy_(
                _reorder_grouped_qkv_to_qkv(
                    storage,
                    num_query_groups=self.model.arch.num_attention_heads,
                    heads_per_group=1,
                    head_dim=self.model.arch.attention_head_dim,
                )
            )

    def _load_resident_tensors(self) -> None:
        super()._load_resident_tensors()
        self.model.post_load_weights()

    def _start_run(self, module, args) -> None:
        context = self.run()
        context.__enter__()
        self._run_context = context

    def _finish_run(self, module, args, output) -> None:
        if self._run_context is not None:
            context, self._run_context = self._run_context, None
            context.__exit__(*(sys.exc_info() if output is None else (None, None, None)))

    def _start_layer(self, index: int) -> None:
        context = self.layer(index)
        context.__enter__()
        self._layer_contexts[index] = context

    def _finish_layer(self, index: int, output) -> None:
        context = self._layer_contexts.pop(index, None)
        if context is not None:
            context.__exit__(*(sys.exc_info() if output is None else (None, None, None)))

    def enable(self, pipeline: torch.nn.Module) -> None:
        if self.enabled:
            return
        self._hook_handles.append(self.model.register_forward_pre_hook(self._start_run))
        self._hook_handles.append(self.model.register_forward_hook(self._finish_run, always_call=True))
        for index, block in enumerate(self.model.blocks):
            self._hook_handles.append(
                block.register_forward_pre_hook(lambda module, args, index=index: self._start_layer(index))
            )
            self._hook_handles.append(
                block.register_forward_hook(
                    lambda module, args, output, index=index: self._finish_layer(index, output), always_call=True
                )
            )
        self.enabled = True

    def disable(self) -> None:
        self.deactivate()
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self.enabled = False
