# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU-only quality comparison of ABot saved video tensors.

Lossless RGB encodes are made after inference, never inside latency timings.
PSNR/SSIM use the repository's existing ffmpeg similarity helpers.
"""

import argparse
import json
import math
import subprocess
from pathlib import Path

import torch

from tests.e2e.accuracy.helpers import parse_psnr_score, parse_ssim_score, run_ffmpeg_similarity


def load(path):
    return torch.load(path, weights_only=True, map_location="cpu").float()


def video(frames, path):
    raw = frames.mul(255).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous().numpy()
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-n",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            "832x512",
            "-framerate",
            "24",
            "-i",
            "pipe:0",
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "bgr0",
            str(path),
        ],
        input=raw.tobytes(),
        check=True,
    )


def main(root, suffix, output):
    output = output or root / f"quality{suffix}"
    output.mkdir(exist_ok=False)
    rows = []
    for backend in ("wan", "taew2_2"):
        offline = load(root / f"{backend}-offline{suffix}/clip-1.pt")
        start = 0
        for index in range(10):
            count = 9 if index == 0 else 12
            frames = {
                "typed": load(root / f"{backend}-typed{suffix}/tick-{index:03d}.pt"),
                "stepwise": load(root / f"{backend}-stepwise{suffix}/tick-{index:03d}.pt"),
                "offline": offline[start : start + count],
            }
            paths = {}
            for mode, tensor in frames.items():
                assert tensor.shape == (count, 3, 512, 832)
                paths[mode] = output / f"{backend}-{mode}-{index:03d}.mkv"
                video(tensor, paths[mode])
            if backend == "taew2_2":
                frames["full_wan"] = load(root / f"wan-typed{suffix}/tick-{index:03d}.pt")
                paths["full_wan"] = output / f"wan-typed-{index:03d}.mkv"
            for other in tuple(key for key in frames if key != "typed"):
                reference, generated = frames[other], frames["typed"]
                mse = (reference - generated).square().mean().item()
                ssim_raw = run_ffmpeg_similarity("ssim", paths[other], paths["typed"])
                psnr_raw = run_ffmpeg_similarity("psnr", paths[other], paths["typed"])
                (output / f"{backend}-typed-vs-{other}-{index:03d}.txt").write_text(ssim_raw + "\n" + psnr_raw)
                # The shared parser only accepts finite numeric PSNR. Identical
                # RGB videos legitimately produce average:inf in ffmpeg.
                psnr = math.inf if "average:inf" in psnr_raw else parse_psnr_score(psnr_raw)
                rows.append(
                    {
                        "backend": backend,
                        "comparison": f"typed_vs_{other}",
                        "tick": index,
                        "float_psnr_db": -10 * math.log10(mse) if mse else "inf",
                        "ffmpeg_psnr_db": psnr if math.isfinite(psnr) else "inf",
                        "ffmpeg_ssim": parse_ssim_score(ssim_raw),
                        "max_abs_error": (reference - generated).abs().max().item(),
                        "exact": torch.equal(reference, generated),
                    }
                )
            start += count
    (output / "quality.json").write_text(
        json.dumps(
            {
                "input_control": "same image/prompt/seed=42, no motion actions, 512x832, first 117 frames",
                "reference": (
                    "native cache offline generation, same per-frame VAE decode path; not original upstream numerics"
                ),
                "encoding": "lossless FFV1, 8-bit RGB; float PSNR also retained",
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    for backend in ("wan", "taew2_2"):
        for mode in ("offline", "stepwise"):
            subset = [r for r in rows if r["backend"] == backend and r["comparison"] == f"typed_vs_{mode}"]
            print(
                backend,
                mode,
                "SSIM min",
                min(r["ffmpeg_ssim"] for r in subset),
                "float PSNR",
                [r["float_psnr_db"] for r in subset],
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--suffix", default="", help="Optional run directory suffix, e.g. -v3")
    parser.add_argument("--output", type=Path, help="New output directory; existing evidence is never overwritten")
    args = parser.parse_args()
    main(args.root, args.suffix, args.output)
