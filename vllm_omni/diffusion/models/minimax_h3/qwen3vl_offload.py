# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3-VL model and checkpoint adapter for file-backed block offload."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from vllm_omni.diffusion.offloader.file_backed_offload import FileBackedBlockOffloader

from .offload_manifest import make_tensor_entry, read_weight_sources

if TYPE_CHECKING:
    from .encoder import MiniMaxH3Qwen3VLEncoder


class Qwen3VLBlockOffloader(FileBackedBlockOffloader):
    def __init__(
        self,
        encoder: MiniMaxH3Qwen3VLEncoder,
        root: str | Path,
        *,
        num_slots: int = 3,
    ) -> None:
        from .encoder import _fused_source_shards

        device = encoder.device_target
        if device.type != "cuda" or encoder.tp_size != 1 or encoder.quant_config is not None:
            raise ValueError("Qwen block offload requires CUDA, TP=1 and unquantized BF16")
        root = Path(root)
        sources, shards = read_weight_sources(root)
        parameters = dict(encoder.named_parameters())
        mapped_sources: dict[str, dict[str | int | None, dict]] = {name: {} for name in parameters}
        for source_name, source in sources.items():
            mapped = encoder._map_weight_name(source_name)
            if mapped is None:
                continue
            name, shard_id = mapped
            if shard_id in mapped_sources[name]:
                raise ValueError(f"Duplicate Qwen source shard: {source_name}")
            mapped_sources[name][shard_id] = source

        blocks = [dict(parameters=[], bytes=0) for _ in encoder.text_model.layers]
        resident = []
        for name, parameter in parameters.items():
            if parameter.dtype != torch.bfloat16:
                raise ValueError(f"Qwen block offload requires BF16: {name}")
            parent, _, _ = name.rpartition(".")
            fused_shards = _fused_source_shards(encoder.get_submodule(parent))
            shard_ids = tuple(shard_id for _, shard_id in fused_shards) if fused_shards is not None else (None,)
            parts = mapped_sources[name]
            if set(parts) != set(shard_ids):
                raise ValueError(f"Qwen checkpoint does not cover tensor: {name}")
            entry = make_tensor_entry(name, parameter, [parts[shard_id] for shard_id in shard_ids])
            if name.startswith("text_model.layers."):
                block = blocks[int(name.split(".")[2])]
                entry["buffer_offset"] = block["bytes"]
                block["parameters"].append(entry)
                block["bytes"] += entry["bytes"]
            else:
                resident.append(entry)

        manifest = {"model_path": str(root), "shards": shards, "blocks": blocks, "resident_tensors": resident}
        super().__init__(encoder, encoder.text_model.layers, manifest, device, num_slots=num_slots)
