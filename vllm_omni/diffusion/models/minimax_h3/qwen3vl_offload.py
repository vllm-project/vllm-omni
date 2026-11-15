# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3-VL model and checkpoint adapter for file-backed block offload."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from vllm_omni.diffusion.offloader.file_backed_offload import FileBackedBlockOffloader

if TYPE_CHECKING:
    from .encoder import MiniMaxH3Qwen3VLEncoder


class Qwen3VLBlockOffloader(FileBackedBlockOffloader):
    def __init__(
        self,
        encoder: MiniMaxH3Qwen3VLEncoder,
        manifest_path: str | Path,
        *,
        num_slots: int = 6,
    ) -> None:
        device = encoder.device_target
        if device.type != "cuda" or encoder.tp_size != 1 or encoder.quant_config is not None:
            raise ValueError("Qwen block offload requires CUDA, TP=1 and unquantized BF16")
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if manifest["schema_version"] != 2:
            raise ValueError("Qwen block offload requires the schema 2 weight manifest")
        root = Path(manifest["model_path"])
        for filename, digest_key in (
            ("config.json", "config_sha256"),
            ("model.safetensors.index.json", "index_sha256"),
        ):
            if hashlib.sha256((root / filename).read_bytes()).hexdigest() != manifest[digest_key]:
                raise ValueError(f"Qwen manifest does not match {filename}")
        blocks = manifest["blocks"]
        entries = manifest["resident_tensors"] + [entry for block in blocks for entry in block["parameters"]]
        parameters = dict(encoder.named_parameters())
        if {entry["runtime_name"] for entry in entries} != parameters.keys():
            raise ValueError("Qwen manifest does not cover the runtime parameters")
        for entry in entries:
            name = entry["runtime_name"]
            parameter = parameters[name]
            if parameter.dtype != torch.bfloat16 or list(parameter.shape) != entry["shape"]:
                raise ValueError(f"Qwen block offload needs a matching meta BF16 parameter: {name}")
        super().__init__(encoder, encoder.text_model.layers, manifest, device, num_slots=num_slots)
