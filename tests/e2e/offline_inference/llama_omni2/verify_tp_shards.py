# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors import safe_open

_PARAMETER_NAMES = {
    "qkv": "language_model.model.layers.0.self_attn.qkv_proj.weight",
    "o": "language_model.model.layers.0.self_attn.o_proj.weight",
    "gate_up": "language_model.model.layers.0.mlp.gate_up_proj.weight",
    "down": "language_model.model.layers.0.mlp.down_proj.weight",
}


def _shard(tensor: torch.Tensor, *, dim: int, rank: int, world_size: int) -> torch.Tensor:
    return tensor.chunk(world_size, dim=dim)[rank].contiguous()


def build_expected_local_parameters(
    tensors: Mapping[str, torch.Tensor],
    *,
    source_prefix: str,
    tp_rank: int,
    tp_world_size: int,
) -> dict[str, torch.Tensor]:
    def get(suffix: str) -> torch.Tensor:
        return tensors[f"{source_prefix}{suffix}"]

    q = _shard(
        get("self_attn.q_proj.weight"),
        dim=0,
        rank=tp_rank,
        world_size=tp_world_size,
    )
    k = _shard(
        get("self_attn.k_proj.weight"),
        dim=0,
        rank=tp_rank,
        world_size=tp_world_size,
    )
    v = _shard(
        get("self_attn.v_proj.weight"),
        dim=0,
        rank=tp_rank,
        world_size=tp_world_size,
    )
    gate = _shard(
        get("mlp.gate_proj.weight"),
        dim=0,
        rank=tp_rank,
        world_size=tp_world_size,
    )
    up = _shard(
        get("mlp.up_proj.weight"),
        dim=0,
        rank=tp_rank,
        world_size=tp_world_size,
    )
    return {
        _PARAMETER_NAMES["qkv"]: torch.cat([q, k, v], dim=0),
        _PARAMETER_NAMES["o"]: _shard(
            get("self_attn.o_proj.weight"),
            dim=1,
            rank=tp_rank,
            world_size=tp_world_size,
        ),
        _PARAMETER_NAMES["gate_up"]: torch.cat([gate, up], dim=0),
        _PARAMETER_NAMES["down"]: _shard(
            get("mlp.down_proj.weight"),
            dim=1,
            rank=tp_rank,
            world_size=tp_world_size,
        ),
    }


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _load_checkpoint_tensors(model_path: Path) -> dict[str, torch.Tensor]:
    prefixes = (
        "model.layers.0.",
        "speech_generator.model.model.layers.0.",
    )
    tensors = {}
    for checkpoint_file in sorted(model_path.glob("*.safetensors")):
        with safe_open(checkpoint_file, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name.startswith(prefixes):
                    tensors[name] = handle.get_tensor(name)
    return tensors


def verify_tp_shards(
    model_path: Path,
    result_path: Path,
) -> dict[str, object]:
    payload = json.loads(result_path.read_text())
    actual_stages = payload["tp_shards"]
    checkpoint_tensors = _load_checkpoint_tensors(model_path)
    source_prefixes = {
        "0": "model.layers.0.",
        "1": "speech_generator.model.model.layers.0.",
    }
    verified = []
    for stage_id, source_prefix in source_prefixes.items():
        workers = actual_stages[stage_id][0]
        for worker in workers:
            rank = int(worker["tp_rank"])
            world_size = int(worker["tp_world_size"])
            expected = build_expected_local_parameters(
                checkpoint_tensors,
                source_prefix=source_prefix,
                tp_rank=rank,
                tp_world_size=world_size,
            )
            for parameter_name, expected_tensor in expected.items():
                actual = worker["parameters"][parameter_name]
                expected_digest = _digest(expected_tensor)
                assert actual["shape"] == list(expected_tensor.shape)
                assert actual["sha256"] == expected_digest
                verified.append(
                    {
                        "stage_id": int(stage_id),
                        "tp_rank": rank,
                        "parameter": parameter_name,
                        "shape": actual["shape"],
                        "sha256": expected_digest,
                    }
                )
    return {
        "verified": True,
        "checks": len(verified),
        "parameters": verified,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("result_path", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_tp_shards(args.model_path, args.result_path), sort_keys=True))


if __name__ == "__main__":
    main()
