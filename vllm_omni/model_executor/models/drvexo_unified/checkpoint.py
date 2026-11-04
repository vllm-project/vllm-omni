# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Strict component views into a complete, already-merged BF16 checkpoint."""

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from safetensors import safe_open

if TYPE_CHECKING:
    import torch

_COMPONENTS = {
    "language_model",
    "vision_model",
    "audio_encoder",
    "connectors",
    "speech_model",
    "speech_codec",
    "speaker",
    "general_decision_head",
}
_FORMAT = "drvexo_unified_safetensors_v1"


class UnifiedBF16Checkpoint:
    """Keep component namespaces and all stored BF16/FP32 values intact.

    Construction verifies headers and index coverage without loading tensors.
    It does not instantiate any network or establish inference compatibility.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve(strict=True)
        config = self._read_object("config.json")
        if config.get("model_type") != "drvexo_speech_qwen35_4b":
            raise ValueError("Expected a DRVEXO Qwen3.5 4B unified checkpoint")
        cognition = config.get("drvexo_cognition", {})
        if (
            not isinstance(cognition, dict)
            or cognition.get("language_LoRA_merged") is not True
            or cognition.get("embedded_v12_branch_tensors") != 0
        ):
            raise ValueError("Expected physically merged cognition weights without an adapter branch")
        if config.get("quantization_config"):
            raise ValueError("This reader accepts BF16/FP32 source weights, not quantized checkpoints")
        index = self._read_object("model.safetensors.index.json")
        manifest = self._read_object("components.json")
        metadata = index.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("format") != _FORMAT or manifest.get("format") != _FORMAT:
            raise ValueError("Invalid DRVEXO unified checkpoint format")
        self.components = manifest.get("components")
        if not isinstance(self.components, dict) or set(self.components) != _COMPONENTS:
            raise ValueError("A complete eight-component model is required")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Expected a nonempty checkpoint weight_map")
        self.weight_map: dict[str, str] = weight_map
        for name, row in self.components.items():
            if not isinstance(row, dict) or type(row.get("tensor_count")) is not int or row["tensor_count"] <= 0:
                raise ValueError(f"Invalid component tensor count: {name}")
        prefixes = {name: row.get("prefix") for name, row in self.components.items()}
        if any(not isinstance(value, str) or not value for value in prefixes.values()):
            raise ValueError("Invalid component prefix")
        counts: Counter[str] = Counter()
        self.owners: dict[str, str] = {}
        for key, shard in self.weight_map.items():
            if not isinstance(key, str) or "lora_" in key.lower() or "__v12_adapter__" in key:
                raise ValueError("Invalid or unmerged tensor key")
            self._shard_path(shard)
            owners = [name for name, prefix in prefixes.items() if key.startswith(prefix)]
            if len(owners) != 1:
                raise ValueError(f"Tensor must have exactly one component owner: {key}")
            self.owners[key] = owners[0]
            counts[owners[0]] += 1
        for name, row in self.components.items():
            if not counts[name] or counts[name] != row["tensor_count"]:
                raise ValueError(f"Component tensor count mismatch: {name}")
        observed = set()
        self.dtypes: Counter[str] = Counter()
        for shard in sorted(set(self.weight_map.values())):
            with safe_open(self._shard_path(shard), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key in observed or self.weight_map.get(key) != shard:
                        raise ValueError(f"Shard/index coverage mismatch: {key}")
                    dtype = handle.get_slice(key).get_dtype()
                    if dtype not in {"BF16", "F32"}:
                        raise ValueError(f"Unsupported source precision: {key}: {dtype}")
                    observed.add(key)
                    self.dtypes[dtype] += 1
        if observed != set(self.weight_map):
            raise ValueError("The checkpoint index references missing tensors")

    def _read_object(self, filename: str) -> dict:
        value = json.loads((self.root / filename).read_text())
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object: {filename}")
        return value

    def _shard_path(self, filename: str) -> Path:
        if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".safetensors"):
            raise ValueError("Invalid checkpoint shard filename")
        path = (self.root / filename).resolve(strict=True)
        if path.parent != self.root:
            raise ValueError("Checkpoint shard escapes the model directory")
        return path

    def iter_component(self, component: str) -> Iterator[tuple[str, "torch.Tensor"]]:
        """Yield original component-relative names and original tensor dtypes."""
        prefix = self.components[component]["prefix"]
        for shard in sorted(set(self.weight_map.values())):
            keys = sorted(
                key for key, filename in self.weight_map.items() if filename == shard and self.owners[key] == component
            )
            if not keys:
                continue
            with safe_open(self._shard_path(shard), framework="pt", device="cpu") as handle:
                for key in keys:
                    yield key.removeprefix(prefix), handle.get_tensor(key)

    def iter_thinker_weights(self) -> Iterator[tuple[str, "torch.Tensor"]]:
        """Expose HF language and visual names to the native Thinker loader."""
        yield from self.iter_component("language_model")
        yield from self.iter_component("vision_model")
