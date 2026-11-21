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
from PIL import Image


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
    if device_type == "npu":
        import torch_npu  # noqa: F401

        torch.npu.config.allow_internal_format = True
    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device=f"{device_type}:0")
    # Resolve every component from the pinned snapshot, including the converted
    # transformer/VAE. Keep the Hub cache immutable.
    with TemporaryDirectory() as directory:
        (Path(directory) / "modular_model_index.json").write_text(json.dumps(index))
        pipe = ModularPipeline.from_pretrained(directory, components_manager=manager)
        assert isinstance(pipe, MiniMaxH3ImageModularPipeline)
        print(f"Diffusers {version('diffusers')}, torch {torch.__version__}, snapshot {model}", flush=True)
        if args.preflight:
            return
        pipe.load_components(dtype=torch.bfloat16, local_files_only=True)
        width, height = map(int, request["size"].split("x"))
        with Image.open(output / "edit-input.png") as source_image:
            reference = source_image.convert("RGB")
        for task, prompt in request["prompts"].items():
            kwargs = {"image": reference} if task == "edit" else {}
            for draw in range(2):
                image = pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    generator=torch.Generator(device="cpu").manual_seed(request["seed"]),
                    output="images",
                    **kwargs,
                )[0]
                assert image.size == (width, height)
                rgb = image.convert("RGB")
                assert any(low != high for low, high in rgb.getextrema())
                rgb.save(output / f"diffusers-{task}-{draw}.png")


if __name__ == "__main__":
    main()
