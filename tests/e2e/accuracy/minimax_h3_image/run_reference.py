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
from diffusers.modular_pipelines.minimax_h3_image.before_denoise import get_image_canvas_size
from diffusers.modular_pipelines.minimax_h3_image.before_encoder import bound_short_edge
from PIL import Image, ImageOps


def edit_noises(
    pipe: MiniMaxH3ImageModularPipeline,
    reference: Image.Image,
    width: int,
    height: int,
    min_generation_area: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    canvas_h, canvas_w = get_image_canvas_size(height, width, min_generation_area, pipe.canvas_multiple)
    ref_w, ref_h = bound_short_edge(
        *reference.size,
        pipe.config.reference_min_short_edge,
        pipe.config.reference_max_short_edge,
        pipe.canvas_multiple,
    )
    ratio = pipe.vae_spatial_compression_ratio
    _, patch_h, patch_w = pipe.patch_size
    ref_latent_h = ref_h // ratio // patch_h * patch_h
    ref_latent_w = ref_w // ratio // patch_w * patch_w
    channels = pipe.vae_latent_channels
    # Match Omni's CPU draw order. Diffusers uses the advanced generator for the
    # VAE posterior and consumes the supplied target and anchor noise as-is.
    target = torch.randn(
        (1, channels, 1, canvas_h // ratio, canvas_w // ratio), generator=generator, dtype=torch.float32
    )
    anchor = torch.randn((1, channels, 1, ref_latent_h, ref_latent_w), generator=generator, dtype=torch.float32)
    return target.squeeze(2), anchor.squeeze(2)


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
        with Image.open(output / "edit-input.png") as source_image:
            reference = ImageOps.exif_transpose(source_image).convert("RGB")
        for task, case in request["cases"].items():
            width, height = map(int, case["size"].split("x"))
            min_generation_area = case.get("extra_params", {}).get("min_generation_area", 2048 * 2048)
            generator = torch.Generator(device="cpu").manual_seed(case["seed"])
            kwargs = {}
            if task == "edit":
                target, anchor = edit_noises(pipe, reference, width, height, min_generation_area, generator)
                kwargs = {"image": reference, "latents": target, "reference_noise": [anchor]}
            image = pipe(
                prompt=case["prompt"],
                height=height,
                width=width,
                min_generation_area=min_generation_area,
                generator=generator,
                output="images",
                **kwargs,
            )[0]
            assert image.size == (width, height)
            rgb = image.convert("RGB")
            assert any(low != high for low, high in rgb.getextrema())
            rgb.save(output / f"diffusers-{task}.png")


if __name__ == "__main__":
    main()
