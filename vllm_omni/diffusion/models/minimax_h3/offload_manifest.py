# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Checkpoint metadata shared by the H3 file-backed offload adapters."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch


def read_weight_sources(root: Path) -> tuple[dict[str, dict], list[dict]]:
    """Read safetensors headers without materializing checkpoint tensors."""
    index_path = next(root.glob("*.safetensors.index.json"))
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    sources, shards = {}, []
    for filename in sorted(set(weight_map.values())):
        path = root / filename
        with path.open("rb") as handle:
            header_size = int.from_bytes(handle.read(8), "little")
            header_bytes = handle.read(header_size)
        header = json.loads(header_bytes)
        file_bytes = path.stat().st_size
        shards.append(
            dict(
                shard=filename,
                header_bytes=8 + header_size,
                header_sha256=hashlib.sha256(header_bytes).hexdigest(),
                file_bytes=file_bytes,
            )
        )
        for name, source in header.items():
            if name == "__metadata__" or weight_map.get(name) != filename:
                continue
            start, end = source["data_offsets"]
            if not 0 <= start <= end <= file_bytes - 8 - header_size:
                raise ValueError(f"Invalid checkpoint tensor byte range: {name}")
            sources[name] = dict(
                source_name=name,
                shard=filename,
                file_offset=8 + header_size + start,
                bytes=end - start,
                shape=source["shape"],
                dtype=source["dtype"],
            )
    return sources, shards


def make_tensor_entry(name: str, tensor: torch.Tensor, sources: list[dict], *, buffer: bool = False) -> dict:
    """Validate one meta tensor and describe its ordered source slices."""
    dtype = {torch.bfloat16: "BF16", torch.float32: "F32"}.get(tensor.dtype)
    parts = [dict(source) for source in sources]
    shape = list(parts[0]["shape"])
    rows = 0
    for part in parts:
        if (
            part["dtype"] != dtype
            or part["shape"][1:] != shape[1:]
            or part["bytes"] != math.prod(part["shape"]) * tensor.element_size()
        ):
            raise ValueError(f"Checkpoint sources do not match tensor: {name}")
        part["concat_offset"] = rows
        rows += part["shape"][0] if shape else 1
    if shape:
        shape[0] = rows
    size = sum(part["bytes"] for part in parts)
    if not tensor.is_meta or shape != list(tensor.shape) or size != tensor.numel() * tensor.element_size():
        raise ValueError(f"Checkpoint does not match meta tensor: {name}")
    return dict(
        runtime_name=name,
        shape=shape,
        dtype=dtype,
        bytes=size,
        sources=parts,
        buffer=buffer,
    )
