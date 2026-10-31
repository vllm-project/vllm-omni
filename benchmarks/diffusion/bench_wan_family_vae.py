# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Benchmark and exactness check for the Wan-family VAE fast path on any registered family (encode and decode).

Loads only the checkpoint's ``vae/`` (the class comes from its ``config.json``: ``AutoencoderKLWan``,
``AutoencoderKLQwenImage``, ``AutoencoderKLQwenImage21``), encodes a real image resized to each ``--size`` (repeated
over ``--frames``) and decodes the resulting latent at each ``--fast-path`` level. Reports the median encode and decode
time, and against ``off`` bitwise equality of both and the PSNR of the decode. The latent comes from the ``off``
encoder, so every level decodes the same input.

    python benchmarks/diffusion/bench_wan_family_vae.py --model Qwen/Qwen-Image --image input.png --size 1024
    python benchmarks/diffusion/bench_wan_family_vae.py --model /path/to/Qwen-Image-2.1 --image input.png \\
        --size 1024 --size 2048 --fast-path off,lossless,channels_last
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics

import numpy as np
import torch
from PIL import Image

from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import (
    find_family,
    install_wan_vae_encoder_fastpath,
    install_wan_vae_fastpath,
    uninstall_wan_vae_encoder_fastpath,
    uninstall_wan_vae_fastpath,
)
from vllm_omni.platforms import current_omni_platform


def _vae_class(name: str):
    if ":" in name:  # an explicit "module:Class", e.g. the vLLM-Omni Qwen-Image-Layered copy
        import importlib

        module, _, attr = name.partition(":")
        return getattr(importlib.import_module(module), attr)
    if name == "AutoencoderKLQwenImage21":
        from vllm_omni.diffusion.models.qwen_image_21.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21

        return AutoencoderKLQwenImage21
    import diffusers

    return getattr(diffusers, name)


def _vae_class_name(model: str) -> str:
    """``_class_name`` of the checkpoint's ``vae/config.json``, for a local directory or a Hub repo id."""
    if os.path.isdir(model):
        config_path = os.path.join(model, "vae", "config.json")
    else:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(model, "config.json", subfolder="vae")  # only the config, not the weights
    with open(config_path) as f:
        name = json.load(f).get("_class_name")
    if not name:
        raise ValueError(f"{config_path} has no _class_name; cannot tell which VAE class to load")
    return name


def _timed(fn, device: torch.device, warmup: int, iters: int) -> tuple[torch.Tensor, list[float]]:
    out = None
    for _ in range(warmup):
        out = fn()
    times = []
    for _ in range(iters):
        torch.accelerator.synchronize(device)
        start = torch.Event(device=device, enable_timing=True)
        end = torch.Event(device=device, enable_timing=True)
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return out, times


def _peak_gib(device: torch.device) -> float:
    return torch.accelerator.max_memory_allocated(device) / 2**30


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = ((a.float() - b.float()) ** 2).mean().item() / 4.0  # [-1, 1] -> [0, 1]
    return math.inf if mse == 0 else 10 * math.log10(1.0 / mse)


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    view = torch.int16 if a.element_size() == 2 else torch.int32
    return a.shape == b.shape and torch.equal(a.view(view), b.view(view))


def _set_level(vae: torch.nn.Module, level: str) -> None:
    uninstall_wan_vae_fastpath(vae)
    uninstall_wan_vae_encoder_fastpath(vae)
    if level == "off":
        return
    for install in (install_wan_vae_fastpath, install_wan_vae_encoder_fastpath):
        report = install(vae, level=level)
        if not report.installed:
            print(f"  note: {install.__name__}({level}) not installed: {report.reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Checkpoint directory or repo id (the VAE is read from vae/)")
    parser.add_argument("--image", required=True, help="Input image; resized to each --size")
    parser.add_argument(
        "--vae-class",
        help="VAE class as module:Class instead of vae/config.json's _class_name, e.g. "
        "vllm_omni.diffusion.models.qwen_image.autoencoder_kl_qwenimage:AutoencoderKLQwenImage (Qwen-Image-Layered)",
    )
    parser.add_argument("--size", type=int, action="append", help="Square image size, repeatable")
    parser.add_argument("--frames", type=int, default=1, help="Video frames (the image repeated); 1 for images")
    parser.add_argument(
        "--fast-path", default="off,off,lossless,channels_last", help="The second `off` checks that off is repeatable"
    )
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--json", help="Write the results to this file")
    args = parser.parse_args()

    levels = args.fast_path.split(",")
    if levels[0] != "off":
        levels.insert(0, "off")  # the reference every other level is compared with; repeat it to check repeatability
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device(current_omni_platform.device_type, 0)
    class_name = args.vae_class or _vae_class_name(args.model)
    vae = _vae_class(class_name).from_pretrained(args.model, subfolder="vae", torch_dtype=dtype).to(device).eval()
    family = find_family(vae)
    config = vae.config
    patch = config.get("patch_size") or 1
    channels = config.get("in_channels", 3) // (patch * patch)
    print(f"{class_name}: family={family.name if family else None}, image channels={channels}, dtype={args.dtype}")
    source = Image.open(args.image).convert("RGBA" if channels == 4 else "RGB")

    results = []
    print(
        f"{'size':>6} {'level':13} {'encode ms':>10} {'decode ms':>10} {'encode exact':>12} {'decode exact':>12} "
        f"{'decode PSNR':>12} {'peak GiB enc/dec':>17}"
    )
    with torch.no_grad():
        for size in args.size or [1024]:
            pixels = np.asarray(source.resize((size, size), Image.LANCZOS), dtype=np.float32) / 127.5 - 1.0
            x = torch.from_numpy(pixels).permute(2, 0, 1)[None, :, None].repeat(1, 1, args.frames, 1, 1)
            x = x.to(device, dtype)
            ref_latent = ref_image = None
            for level in levels:
                _set_level(vae, level)
                # No empty_cache() here: on XPU the bf16 reference decode at some shapes (e.g. a 166x166 latent)
                # changes with allocator state, which would make "bitwise equal to off" meaningless.
                torch.accelerator.reset_peak_memory_stats(device)
                latent, enc_ms = _timed(lambda: vae.encode(x).latent_dist.mode(), device, args.warmup, args.iters)
                if ref_latent is None:
                    ref_latent = latent.clone()  # a copy: later levels must not be able to alias it
                enc_peak = _peak_gib(device)
                torch.accelerator.reset_peak_memory_stats(device)
                image, dec_ms = _timed(lambda: vae.decode(ref_latent).sample, device, args.warmup, args.iters)
                dec_peak = _peak_gib(device)
                if ref_image is None:
                    ref_image = image.clone()
                row = dict(
                    size=size,
                    level=level,
                    encode_ms=statistics.median(enc_ms),
                    decode_ms=statistics.median(dec_ms),
                    encode_exact=_same_bits(latent, ref_latent),
                    decode_exact=_same_bits(image, ref_image),
                    decode_psnr=_psnr(image, ref_image),
                    encode_peak_gib=enc_peak,
                    decode_peak_gib=dec_peak,
                )
                results.append(row)
                print(
                    f"{size:>5}² {level:13} {row['encode_ms']:10.1f} {row['decode_ms']:10.1f} "
                    f"{str(row['encode_exact']):>12} {str(row['decode_exact']):>12} {row['decode_psnr']:10.1f} dB "
                    f"{enc_peak:8.2f}/{dec_peak:.2f}",
                    flush=True,
                )
            _set_level(vae, "off")
    if args.json:
        json.dump(
            dict(model=args.model, vae=class_name, dtype=args.dtype, frames=args.frames, results=results),
            open(args.json, "w"),
            indent=1,
        )


if __name__ == "__main__":
    main()
