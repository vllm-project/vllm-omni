# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Prepare the static-scale FLUX.2-dev-NVFP4 checkpoint beside local base components."""

import argparse
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from vllm_omni.diffusion.models.flux2.nvfp4_checkpoint import map_bfl_weight, quantized_layer_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.base_model.resolve()
    config = (base / "transformer/config.json").read_text()
    with safe_open(args.checkpoint, framework="pt", device="cpu") as checkpoint:
        metadata = json.loads(checkpoint.metadata()["_quantization_metadata"])
        layers = sorted(quantized_layer_names(metadata))
        tensors = dict(map_bfl_weight(name, checkpoint.get_tensor(name)) for name in checkpoint.keys())
        for name in layers:
            for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
                if name + "." + suffix not in tensors:
                    raise ValueError(f"Missing {name}.{suffix}; mixed/dynamic checkpoints are not supported yet")
        args.output.mkdir()
        for component in base.iterdir():
            if component.name != "transformer":
                (args.output / component.name).symlink_to(component)
        transformer = args.output / "transformer"
        transformer.mkdir()
        (transformer / "config.json").write_text(config)
        save_file(tensors, transformer / "diffusion_pytorch_model.safetensors", metadata={"format": "pt"})
        (transformer / "quantization_config.json").write_text(
            json.dumps({"method": "comfy_nvfp4", "quantized_layers": layers}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
