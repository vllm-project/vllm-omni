# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Read one namespaced safetensors checkpoint without unpacking stage weights."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file as _load_file

FORMAT = "drvexo_unified_safetensors_v1"
VIEW_FORMAT = "drvexo_unified_tensor_view_v1"


def _inside(root, relative):
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError(f"Checkpoint path escapes its directory: {relative}")
    return path


class UnifiedCheckpoint:
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.index = json.loads((self.root / "model.safetensors.index.json").read_text())
        if self.index["metadata"].get("format") != FORMAT:
            raise ValueError("Not a DRVEXO unified checkpoint")
        self.components = json.loads((self.root / "components.json").read_text())
        if self.components["format"] != FORMAT:
            raise ValueError("Invalid component manifest")
        self.weight_map = self.index["weight_map"]
        for name, filename in self.weight_map.items():
            if not isinstance(name, str) or not name or Path(filename).name != filename:
                raise ValueError("Invalid tensor index entry")
            if not filename.endswith(".safetensors"):
                raise ValueError("Only safetensors shards are allowed")
            _inside(self.root, filename)

    def iter_tensors(self, component, device="cpu"):
        specification = self.components["components"][component]
        prefix = specification["prefix"]
        groups = defaultdict(list)
        for name, filename in self.weight_map.items():
            if name.startswith(prefix):
                groups[filename].append(name)
        if sum(map(len, groups.values())) != specification["tensor_count"]:
            raise ValueError(f"Incomplete component: {component}")
        for filename, names in groups.items():
            with safe_open(_inside(self.root, filename), framework="pt", device=device) as source:
                for name in names:
                    yield name.removeprefix(prefix), source.get_tensor(name)

    def state_dict(self, component, device="cpu"):
        return dict(self.iter_tensors(component, device))


def open_view(path):
    path = Path(path).resolve(strict=True)
    view = json.loads(path.read_text())
    if view.get("format") != VIEW_FORMAT:
        raise ValueError(f"Not a unified tensor view: {path}")
    index = (path.parent / view["index"]).resolve(strict=True)
    if index.name != "model.safetensors.index.json" or not path.is_relative_to(index.parent):
        raise ValueError("Tensor view escapes its checkpoint")
    checkpoint = UnifiedCheckpoint(index.parent)
    if view["component"] not in checkpoint.components["components"]:
        raise ValueError("Unknown checkpoint component")
    return checkpoint, view


def iter_view(path, device="cpu"):
    checkpoint, view = open_view(path)
    yield from checkpoint.iter_tensors(view["component"], device)


def load_tensor_file(path, device="cpu"):
    if str(path).endswith(".json"):
        return dict(iter_view(path, device))
    return _load_file(str(path), device=device)


def load_checkpoint(path):
    """Reconstruct legacy payload *in memory*, never serialize pickle files."""
    checkpoint, view = open_view(path)
    component = checkpoint.components["components"][view["component"]]
    state = checkpoint.state_dict(view["component"])
    if component["payload_kind"] == "tensor":
        if set(state) != {"value"}:
            raise ValueError("Invalid standalone tensor component")
        return state["value"]
    if component["payload_kind"] == "state_dict":
        return {**component.get("metadata", {}), "state_dict": state}
    return state


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()
