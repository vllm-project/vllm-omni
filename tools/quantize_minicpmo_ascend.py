# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Export the MiniCPM-o 4.5 Thinker to dynamic INT8 W8A8 for Ascend."""

import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_INDEX = "model.safetensors.index.json"


def quantize_shard(source: Path, destination: Path) -> tuple[int, int]:
    tensors = {}
    linear_layers = 0
    with safe_open(str(source), framework="pt", device="cpu") as shard:
        for name in shard.keys():
            tensor = shard.get_tensor(name)
            if name.startswith("llm.model.layers.") and name.endswith(".weight") and tensor.ndim == 2:
                weight = tensor.float()
                scale = weight.abs().amax(dim=1, keepdim=True) / 127
                scale[scale == 0] = 1
                tensor = (weight / scale).round().clamp_(-128, 127).to(torch.int8)
                tensors[f"{name}_scale"] = scale
                tensors[f"{name}_offset"] = torch.zeros_like(scale, dtype=torch.int8)
                linear_layers += 1
            tensors[name] = tensor
        save_file(tensors, destination, metadata={"format": "pt"})
    return sum(tensor.nbytes for tensor in tensors.values()), linear_layers


def export_checkpoint(source: Path, output: Path) -> None:
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "minicpmo" or config.get("version") != "4.5" or "quantization_config" in config:
        raise ValueError("Expected the original, unquantized MiniCPM-o 4.5 checkpoint")

    index = json.loads((source / _INDEX).read_text())
    shards = sorted(set(index["weight_map"].values()))
    total_size = linear_layers = 0
    for shard in shards:
        size, layers = quantize_shard(source / shard, output / shard)
        total_size += size
        linear_layers += layers
        print(f"{shard}: quantized {layers} Thinker projections", flush=True)
    if linear_layers != config["num_hidden_layers"] * 7:
        raise ValueError(f"Unexpected Thinker projection count: {linear_layers}")

    for asset in source.iterdir():
        if asset.is_file() and not asset.name.startswith(".") and asset.name not in {*shards, _INDEX, "config.json"}:
            shutil.copyfile(asset, output / asset.name)
    if (source / "assets").is_dir():
        shutil.copytree(source / "assets", output / "assets")
    config["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "config_groups": {
            "group_0": {
                "weights": {
                    "num_bits": 8,
                    "type": "int",
                    "strategy": "channel",
                    "symmetric": True,
                    "dynamic": False,
                    "observer": "minmax",
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "int",
                    "strategy": "token",
                    "dynamic": True,
                    "symmetric": True,
                    "observer": None,
                },
                "format": "int-quantized",
                "targets": ["Linear"],
            },
        },
        "ignore": [
            "apm",
            "vpm",
            "resampler",
            "audio_projection_layer",
            "tts",
            "talker",
            "code2wav",
            "token2wav",
            "estimator",
            "detector",
            "audio_tokenizer",
            "llm.lm_head",
            "llm.model.embed_tokens",
        ],
        "global_group": None,
    }
    # vLLM discovers scale and offset tensors in the indexed weight shards.
    index["metadata"]["total_size"] = total_size
    (output / _INDEX).write_text(json.dumps(index, indent=2), encoding="utf-8")
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Local snapshot of openbmb/MiniCPM-o-4_5")
    parser.add_argument("output", type=Path, help="New destination directory")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as staging:
        staged = Path(staging)
        export_checkpoint(source, staged)
        staged.rename(output)


if __name__ == "__main__":
    main()
