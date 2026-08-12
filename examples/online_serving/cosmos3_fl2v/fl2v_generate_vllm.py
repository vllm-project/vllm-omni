#!/usr/bin/env python3
"""First-last boundary to video (FL2V) with Cosmos3 on the vLLM-Omni runtime.

WHAT THIS DOES
  Pins a Cosmos3 clip to a START boundary and an END boundary, and fills the
  middle from a text/JSON prompt. Each boundary may be either:
    - a single image (frame conditioning), or
    - a short video clip (clip conditioning)

  That gives four combinations in one script:
    frame+frame | clip+frame | frame+clip | clip+clip

  This is a pure HTTP client for `POST /v1/videos/sync`. No vLLM-Omni source
  changes, no model changes, no fork. Everything happens through documented
  request fields plus `extra_params`.

  The server does need one upstream vLLM bug fix, applied by the companion
  script `patch_vllm_shm.py`. FL2V reference videos are large enough to trip a
  shared-memory deadlock that ordinary Cosmos3 requests never reach, and the
  symptom is a permanent silent hang. Run the patch once per environment; this
  script warns if it is missing. See README.md, "Required vLLM patch".

HOW FL2V IS EXPRESSED ON THIS RUNTIME
  Stock Cosmos3 V2V locks a clean PREFIX of latents (usually [0, 1]). FL2V locks
  a contiguous range at each end. On vLLM-Omni only one piece is client-side,
  because the Cosmos3 V2V path already implements the other two server-side:

    1) Boundary latent indexes -> NATIVE.
       `extra_params.condition_frame_indexes_vision` accepts any latent indexes,
       so [0..k] + [T_lat-j..T_lat-1] is a plain request field. Stock V2V
       default is [0, 1].

    2) Clean head/tail for the causal VAE -> OURS (this script).
       The server encodes the first `max(indexes)*4+1` pixel frames of the
       reference video, so we must hand it a video shaped
       [start head][filler...][end tail] — stills repeated, or real clip frames.

    3) Re-inject clean latents every step -> NATIVE.
       The V2V denoise loop already runs
       `latents = velocity_mask * latents + (1 - velocity_mask) * condition_latents`
       after every scheduler step.

QUICKSTART (demo package: frame + frame)
  # Server (separate shell, needs the GPU). Super recommended for quality;
  # Nano also works — serve nvidia/Cosmos3-Nano and pass matching --model.
  #   vllm-omni serve nvidia/Cosmos3-Super --omni --port 8000 --init-timeout 1800 --no-guardrails
  python fl2v_generate_vllm.py \\
    --output testdata/fl2v_from_cosmos_v2v/outputs/generated_jar_diversion_framecond_vllm.mp4

YOUR OWN EXAMPLE
  # Frame + frame
  python fl2v_generate_vllm.py \\
    --start /path/to/start.png --end /path/to/end.png \\
    --prompt event.json --negative negative.json --output out.mp4

  # Clip + clip (extensions select the mode)
  python fl2v_generate_vllm.py \\
    --start /path/to/start_clip.mp4 --end /path/to/end_clip.mp4 \\
    --prompt event.json --negative negative.json --output out.mp4 \\
    --head-frames 9 --tail-frames 9

KNOWN LIMITS
  The reference video travels as an encoded MP4, so boundary frames go through
  one extra codec round-trip versus feeding raw frames in-process.
  We write it near-losslessly (yuv444p, -qp 0) to keep that negligible.
  Boundary clip lengths should be VAE-aligned: k*4+1 (e.g. 1, 5, 9, 13).

  See README.md for setup, the CLI table, and how FL2V works.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import imageio.v2 as imageio
import numpy as np
import requests
from PIL import Image

VAE_TEMPORAL = 4  # Wan-style causal VAE temporal compression factor
SCRIPT_DIR = Path(__file__).resolve().parent
PKG_RELATIVE = Path("testdata") / "fl2v_from_cosmos_v2v"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".gif"}

# vLLM ships requests to the diffusion worker through a shared-memory ring
# buffer with chunks this size; anything larger takes the overflow path that
# patch_vllm_shm.py fixes.
SHM_CHUNK_BYTES = 24 * 1024 * 1024
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def default_package() -> Path:
    """Locate the demo package shipped with this runtime."""
    return SCRIPT_DIR / PKG_RELATIVE


def is_image_path(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def is_video_path(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


def latent_count_for_frames(num_frames: int) -> int:
    """Pixel length T -> latent count with Wan temporal factor 4: (T-1)/4 + 1."""
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if (num_frames - 1) % VAE_TEMPORAL != 0:
        raise ValueError(
            f"frame count {num_frames} is not VAE-aligned; need (n - 1) divisible by "
            f"{VAE_TEMPORAL} (e.g. 1, 5, 9, 13, 189)."
        )
    return (num_frames - 1) // VAE_TEMPORAL + 1


def align_frame_count(n: int) -> int:
    """Largest VAE-aligned length (k*4+1) that is <= n, at least 1."""
    if n < 1:
        raise ValueError(f"need at least 1 frame, got {n}")
    aligned = ((n - 1) // VAE_TEMPORAL) * VAE_TEMPORAL + 1
    return max(1, aligned)


def load_rgb(path: Path, width: int, height: int) -> Image.Image:
    """Load an image at exactly the generation size."""
    img = Image.open(path).convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height), Image.LANCZOS)
    return img


def load_video_frames(path: Path, width: int, height: int) -> list[Image.Image]:
    raw = imageio.mimread(str(path), memtest=False)
    if not raw:
        raise ValueError(f"no frames read from video: {path}")
    frames: list[Image.Image] = []
    for arr in raw:
        img = Image.fromarray(np.asarray(arr)).convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), Image.LANCZOS)
        frames.append(img)
    return frames


def prepare_boundary(
    path: Path,
    side: str,
    pad_frames: int,
    width: int,
    height: int,
) -> tuple[list[Image.Image], int, Image.Image, str]:
    """Prepare one boundary (start or end).

    Returns:
      frames: pixel frames placed at that end of the condition video
      n_lock: how many latents to lock at that end
      ref: image used for boundary PSNR (first start frame / last end frame)
      mode: "frame" or "clip"
    """
    if is_image_path(path):
        img = load_rgb(path, width, height)
        if pad_frames < 1:
            raise ValueError(f"--{'head' if side == 'start' else 'tail'}-frames must be >= 1")
        return [img] * pad_frames, 1, img, "frame"

    if not is_video_path(path):
        raise ValueError(
            f"{side} path has unsupported extension '{path.suffix}'. "
            f"Use an image ({sorted(IMAGE_EXTS)}) or video ({sorted(VIDEO_EXTS)})."
        )

    video = load_video_frames(path, width, height)
    take = min(pad_frames, len(video))
    chunk = video[:take] if side == "start" else video[-take:]
    aligned = align_frame_count(len(chunk))
    if aligned != len(chunk):
        print(
            f"{side} clip: snapped {len(chunk)} frames -> {aligned} "
            f"(VAE grid k*{VAE_TEMPORAL}+1)"
        )
        chunk = chunk[:aligned] if side == "start" else chunk[-aligned:]
    n_lock = latent_count_for_frames(len(chunk))
    ref = chunk[0] if side == "start" else chunk[-1]
    return chunk, n_lock, ref, "clip"


def build_conditioning_frames(
    start_frames: list[Image.Image],
    end_frames: list[Image.Image],
    num_frames: int,
) -> list[Image.Image]:
    """Assemble [start head][filler][end tail] for the reference MP4 upload.

    The server slices `video[:max(indexes)*4+1]` and encodes it with the VAE.
    """
    head = len(start_frames)
    tail = len(end_frames)
    if head + tail > num_frames:
        raise ValueError(
            f"start frames ({head}) + end frames ({tail}) exceed num_frames ({num_frames}). "
            f"Shorten --head-frames/--tail-frames or lengthen --num-frames."
        )
    frames: list[Image.Image] = []
    mid = num_frames - head - tail
    filler_a = start_frames[-1]
    filler_b = end_frames[0]
    for i in range(num_frames):
        if i < head:
            frames.append(start_frames[i])
        elif i >= num_frames - tail:
            frames.append(end_frames[i - (num_frames - tail)])
        else:
            frames.append(filler_a if (i - head) < mid // 2 else filler_b)
    return frames


def write_reference_video(
    frames: list[Image.Image],
    path: Path,
    fps: float,
    codec: str,
    pix_fmt: str,
    qp: str,
) -> None:
    """Encode the conditioning frames as the MP4 we upload as input_reference.

    Near-lossless on purpose: these pixels become the locked boundary latents,
    so codec error here shows up directly as lower start/end PSNR.
    """
    writer = imageio.get_writer(
        str(path),
        fps=float(fps),
        codec=codec,
        macro_block_size=1,
        pixelformat=pix_fmt,
        ffmpeg_params=["-qp", qp],
    )
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    finally:
        writer.close()


def load_prompt(path: Path) -> str:
    """Load a prompt file: JSON object/string, or plain text."""
    text = path.read_text().strip()
    if path.suffix.lower() == ".json":
        obj = json.loads(text)
        if isinstance(obj, dict):
            # Repo bookkeeping key; not part of the model caption schema.
            obj.pop("conditioning", None)
            return json.dumps(obj)
        return json.dumps(obj) if not isinstance(obj, str) else obj
    return text


def warn_if_vllm_unpatched(url: str, payload_bytes: int) -> None:
    """Warn when this request is big enough to hit the unfixed vLLM deadlock.

    The server decodes our reference video to raw pixels before handing it to
    the diffusion worker. Past the shared-memory chunk size that transfer takes
    a path which, unpatched, hangs forever instead of failing — so it is worth
    catching before the user waits on a dead request.

    Only checked for a server on this machine: a remote server has its own
    install that we cannot inspect from here.
    """
    if payload_bytes <= SHM_CHUNK_BYTES:
        return
    if (urlparse(url).hostname or "") not in LOCAL_HOSTS:
        return
    try:
        spec = importlib.util.find_spec("vllm")
    except (ImportError, ValueError):
        return
    if spec is None or not spec.submodule_search_locations:
        return
    shm = (
        Path(next(iter(spec.submodule_search_locations)))
        / "distributed"
        / "device_communicators"
        / "shm_broadcast.py"
    )
    try:
        if "FL2V patch" in shm.read_text():
            return
    except OSError:
        return
    print(
        f"WARNING: this request ships ~{payload_bytes / 2**20:.0f} MB to the diffusion "
        f"worker, over the {SHM_CHUNK_BYTES // 2**20} MB shared-memory chunk size,\n"
        "         and this environment's vLLM is missing the FL2V fix for that path.\n"
        "         The server will most likely hang with no error and no GPU activity.\n"
        "         Fix it with:  python patch_vllm_shm.py   (then restart the server)\n"
        "         See README.md, \"Required vLLM patch\".",
        file=sys.stderr,
    )


def psnr(a, b) -> float:
    a = np.asarray(a, dtype=np.float32) / 255.0
    b_img = b if isinstance(b, Image.Image) else Image.fromarray(np.asarray(b))
    b = np.asarray(b_img.resize(a.shape[1::-1]), dtype=np.float32) / 255.0
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else 10.0 * float(np.log10(1.0 / mse))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--url", default="http://localhost:8000", help="vLLM-Omni base URL")
    p.add_argument(
        "--model",
        default="nvidia/Cosmos3-Super",
        help="Served model name (Super recommended for quality; "
        "nvidia/Cosmos3-Nano also works)",
    )

    p.add_argument(
        "--package",
        type=Path,
        default=None,
        help="Demo package dir with assets/, fl2v_prompt.json, negative_prompt.json",
    )
    p.add_argument(
        "--start",
        type=Path,
        default=None,
        help="Start boundary: image OR video (mode auto-detected by extension)",
    )
    p.add_argument(
        "--end",
        type=Path,
        default=None,
        help="End boundary: image OR video (mode auto-detected by extension)",
    )
    p.add_argument("--prompt", type=Path, default=None, help="Event prompt .json or .txt")
    p.add_argument("--negative", type=Path, default=None, help="Negative prompt .json or .txt")
    p.add_argument("--output", type=Path, default=None, help="Output .mp4 path")

    p.add_argument("--num-frames", type=int, default=189, help="Optional temporal length")
    p.add_argument("--fps", type=float, default=24.0, help="Optional FPS metadata / export rate")
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--guidance", type=float, default=8.0)
    p.add_argument("--flow-shift", type=float, default=15.0)
    p.add_argument("--max-sequence-length", type=int, default=4096)
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument(
        "--head-frames",
        type=int,
        default=5,
        help="Start boundary length: still repeats (frame mode) or frames taken from "
        "the start of the video (clip mode). Prefer k*4+1 for clips (1, 5, 9, ...).",
    )
    p.add_argument(
        "--tail-frames",
        type=int,
        default=9,
        help="End boundary length: still repeats (frame mode) or frames taken from "
        "the end of the video (clip mode). Prefer k*4+1 for clips (1, 5, 9, ...).",
    )

    p.add_argument("--guardrails", action="store_true", help="Enable Cosmos safety checker")
    p.add_argument(
        "--resolution-template",
        action="store_true",
        help="Let the server inject resolution/duration templates (off: our JSON prompt already encodes them)",
    )
    p.add_argument("--codec", default="libx264", help="Reference-video codec")
    p.add_argument("--pix-fmt", default="yuv444p", help="Reference pixel format (use yuv420p if 4:4:4 fails to decode)")
    p.add_argument("--qp", default="0", help="Reference quantizer; 0 is lossless")
    p.add_argument(
        "--keep-reference",
        type=Path,
        default=None,
        help="Also save the generated conditioning video here (debugging)",
    )
    p.add_argument("--timeout", type=float, default=1800.0, help="HTTP timeout in seconds")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the reference video and print the request, but do not POST",
    )
    args = p.parse_args()

    pkg = args.package or default_package()
    start_path = args.start or (pkg / "assets" / "seed_start.png")
    end_path = args.end or (pkg / "assets" / "seed_end.png")
    prompt_path = args.prompt or (pkg / "fl2v_prompt.json")
    negative_path = args.negative or (pkg / "negative_prompt.json")
    out_path = args.output or (pkg / "outputs" / "generated_jar_diversion_framecond_vllm.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for path, label in [
        (start_path, "start"),
        (end_path, "end"),
        (prompt_path, "prompt"),
        (negative_path, "negative"),
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} file: {path}")

    if (args.num_frames - 1) % VAE_TEMPORAL != 0:
        raise ValueError(
            f"num_frames={args.num_frames} is invalid for Wan VAE factor {VAE_TEMPORAL}; "
            f"need (num_frames - 1) divisible by {VAE_TEMPORAL} (e.g. 189, 93, 49)."
        )
    t_lat = latent_count_for_frames(args.num_frames)

    start_frames, n_start_lock, start_ref, start_mode = prepare_boundary(
        start_path, "start", args.head_frames, args.width, args.height
    )
    end_frames, n_end_lock, end_ref, end_mode = prepare_boundary(
        end_path, "end", args.tail_frames, args.width, args.height
    )

    if n_start_lock + n_end_lock > t_lat:
        raise ValueError(
            f"start locks {n_start_lock} latents + end locks {n_end_lock} latents "
            f"exceeds total {t_lat}. Shorten boundary clips / head-tail pads."
        )

    # Trick 1: lock a contiguous range at each end (frame mode => single index).
    cond_idx = list(range(n_start_lock)) + list(range(t_lat - n_end_lock, t_lat))

    # The server encodes exactly this many reference pixel frames, so our
    # head/tail content must live inside a video of this length.
    condition_pixel_frames = max(cond_idx) * VAE_TEMPORAL + 1

    prompt = load_prompt(prompt_path)
    negative = load_prompt(negative_path)

    print(f"start={start_path} ({start_mode}, {len(start_frames)} frames, lock {n_start_lock} latents)")
    print(f"end={end_path} ({end_mode}, {len(end_frames)} frames, lock {n_end_lock} latents)")
    print(f"prompt={prompt_path}")
    print(f"num_frames={args.num_frames} -> latent_frames={t_lat}, conditioning latents {cond_idx}")
    print(f"reference video frames={condition_pixel_frames}")

    frames = build_conditioning_frames(start_frames, end_frames, condition_pixel_frames)

    extra_params = {
        "use_resolution_template": args.resolution_template,
        "use_duration_template": args.resolution_template,
        "guardrails": args.guardrails,
        # Trick 1 (stock V2V uses [0, 1]); trick 3 is applied server-side for
        # every index listed here.
        "condition_frame_indexes_vision": cond_idx,
        # Our reference video is exactly condition_pixel_frames long, so "first"
        # and "last" select the same frames; pinned for reproducibility.
        "condition_video_keep": "first",
    }
    data = {
        "model": args.model,
        "prompt": prompt,
        "negative_prompt": negative,
        "size": f"{args.width}x{args.height}",
        "num_frames": str(args.num_frames),
        "fps": str(args.fps),
        "num_inference_steps": str(args.steps),
        "guidance_scale": str(args.guidance),
        "flow_shift": str(args.flow_shift),
        "max_sequence_length": str(args.max_sequence_length),
        "seed": str(args.seed),
        "extra_params": json.dumps(extra_params),
    }

    with tempfile.TemporaryDirectory() as tmp:
        ref_path = Path(tmp) / "fl2v_reference.mp4"
        write_reference_video(frames, ref_path, args.fps, args.codec, args.pix_fmt, args.qp)
        if args.keep_reference is not None:
            args.keep_reference.parent.mkdir(parents=True, exist_ok=True)
            args.keep_reference.write_bytes(ref_path.read_bytes())
            print(f"reference video saved to {args.keep_reference}")

        endpoint = args.url.rstrip("/") + "/v1/videos/sync"
        print(f"extra_params={data['extra_params']}")
        if args.dry_run:
            print(f"dry run: would POST {endpoint} with reference {ref_path.stat().st_size} bytes")
            return

        # The MP4 upload is small; what matters is the raw frames the server
        # decodes it into and forwards to the worker.
        warn_if_vllm_unpatched(args.url, condition_pixel_frames * args.height * args.width * 3)

        try:
            with ref_path.open("rb") as ref:
                response = requests.post(
                    endpoint,
                    headers={"Accept": "video/mp4"},
                    data=data,
                    files={"input_reference": ("fl2v_reference.mp4", ref, "video/mp4")},
                    timeout=args.timeout,
                )
        except requests.exceptions.ConnectionError:
            # Model load can take a long time; a refused connection here
            # usually means the server is still starting, not misconfigured.
            print(f"cannot reach {endpoint}", file=sys.stderr)
            print("is the vLLM-Omni server up and finished loading? see README.md", file=sys.stderr)
            raise SystemExit(1) from None

    if response.status_code != 200:
        print(f"request failed: HTTP {response.status_code}", file=sys.stderr)
        print(response.text[:4000], file=sys.stderr)
        raise SystemExit(1)
    if "video" not in response.headers.get("Content-Type", ""):
        print("server did not return video bytes:", file=sys.stderr)
        print(response.text[:4000], file=sys.stderr)
        raise SystemExit(1)

    out_path.write_bytes(response.content)
    video = imageio.mimread(str(out_path), memtest=False)
    print(f"wrote {out_path} ({len(video)} frames)")

    # SIL-relevant check: do the locked ends still match the boundary refs?
    print(f"start-frame PSNR vs boundary: {psnr(video[0], start_ref):.2f} dB")
    print(f"end-frame   PSNR vs boundary: {psnr(video[-1], end_ref):.2f} dB")


if __name__ == "__main__":
    main()
