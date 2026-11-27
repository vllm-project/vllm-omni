# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run the pinned Modular Diffusers pipeline in the reference interpreter."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from diffusers import ComponentsManager, MiniMaxH3ImageModularPipeline, ModularPipeline
from diffusers.hooks import apply_group_offloading


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    output = args.request.parent
    model = Path(request["model"]).resolve()
    index = json.loads((model / "modular_model_index.json").read_text())
    for name, spec in index.items():
        if name.startswith("_") or not isinstance(spec, list) or len(spec) != 3 or not isinstance(spec[2], dict):
            continue
        source = spec[2]
        if "subfolder" in source:
            assert (model / source["subfolder"]).is_dir(), name
            source["pretrained_model_name_or_path"] = str(model)
    device_type = request["device_type"]
    device = f"{device_type}:0"
    if device_type == "npu":
        import torch_npu  # noqa: F401

        torch.npu.config.allow_internal_format = True
    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device=device)
    # Resolve every component from the shared snapshot, including the converted
    # transformer/VAE. Keep the Hub cache immutable.
    with TemporaryDirectory() as directory:
        (Path(directory) / "modular_model_index.json").write_text(json.dumps(index))
        pipe = ModularPipeline.from_pretrained(directory, components_manager=manager)
        assert isinstance(pipe, MiniMaxH3ImageModularPipeline)
        print(f"Diffusers {version('diffusers')}, torch {torch.__version__}, snapshot {model}", flush=True)
        if args.preflight:
            return
        pipe.load_components(dtype=torch.bfloat16, local_files_only=True)
        if device_type == "npu":
            # ComponentsManager leaves group-offloaded modules in charge of their own placement.
            apply_group_offloading(pipe.text_encoder, onload_device=device, offload_type="leaf_level")
            apply_group_offloading(
                pipe.transformer, onload_device=device, offload_type="block_level", num_blocks_per_group=1
            )
        case = request["cases"]["t2i"]
        width, height = map(int, case["size"].split("x"))
        image = pipe(
            prompt=case["prompt"],
            height=height,
            width=width,
            min_generation_area=request["extra_params"]["min_generation_area"],
            generator=torch.Generator(device="cpu").manual_seed(case["seed"]),
            output="images",
        )[0]
        assert image.size == (width, height)
        rgb = image.convert("RGB")
        assert any(low != high for low, high in rgb.getextrema())
        rgb.save(output / "diffusers-t2i.png")


if __name__ == "__main__":
    main()
