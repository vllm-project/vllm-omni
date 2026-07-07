"""Unified Gradio demo for Lance × vllm-omni — all 7 tasks.

Style mirrors upstream Lance's lance_gradio.py:
  * header with Lance + vllm-omni logos + shield badges
  * radio task selector (Video Generation / Video Edit / Video Understanding /
    Image-to-Video / Image Generation / Image Edit / Image Understanding)
  * aspect-ratio + resolution dropdowns (not free H/W sliders)
  * two-column layout (inputs left, output right)
  * advanced-params accordion at the bottom

Backend: routes to one of two Omni instances based on task:
  * Lance_3B          (t2i, image_edit, x2t_image)
  * Lance_3B_Video    (t2v, i2v, video_edit, x2t_video)

Run:
    CUDA_VISIBLE_DEVICES=6,7 \\
      /home/zjy/code/vllm-src/.venv/bin/python \\
      examples/offline_inference/lance/gradio_demo.py \\
      --model /path/to/Lance/snapshots/<hash>
"""

from __future__ import annotations

import argparse
import base64
import os
import tempfile
from pathlib import Path

import gradio as gr
import imageio.v3 as iio
import numpy as np
from PIL import Image

# --------------------------------------------------------- constants -----

# Mirror upstream Lance's resolution/aspect tables.
#
# Per ``data/datasets_custom/validation_dataset.py``, upstream restricts
# aspect-ratio buckets to **6 fixed ratios** and computes the per-bucket
# canonical (W, H) via:
#
#   max_area = resolution_vae ** 2
#   for each ar = w/h:
#       w1 = round(sqrt(max_area * ar) / 16) * 16
#       h1 = round(w1 / ar / 16) * 16
#       (… pick the candidate whose actual ratio is closer to ``ar``)
#
# with ``resolution_vae = 480`` for ``video_360p``, ``640`` for
# ``video_480p``, ``768`` for ``image_768res``.  The numbers below are
# computed exactly that way so this gradio demo lands on the same canvas
# upstream would for the same input image — no aspect-ratio drift.
ASPECT_RATIO_CHOICES = ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"]
DEFAULT_VIDEO_AR = "16:9"
DEFAULT_IMAGE_AR = "1:1"

VIDEO_360P_AR_TO_SIZE = {
    "21:9": (752, 320),
    "16:9": (624, 352),
    "4:3": (560, 416),
    "1:1": (480, 480),
    "3:4": (416, 560),
    "9:16": (352, 624),
}
VIDEO_480P_AR_TO_SIZE = {
    "21:9": (976, 416),
    "16:9": (848, 480),
    "4:3": (752, 560),
    "1:1": (640, 640),
    "3:4": (560, 752),
    "9:16": (480, 848),
}
IMAGE_AR_TO_SIZE = {
    "21:9": (1152, 496),
    "16:9": (1024, 576),
    "4:3": (896, 672),
    "1:1": (768, 768),
    "3:4": (672, 896),
    "9:16": (576, 1024),
}

VIDEO_RES_CHOICES = [("360p", "video_360p"), ("480p", "video_480p")]
VIDEO_RES_MAP = {"video_360p": VIDEO_360P_AR_TO_SIZE, "video_480p": VIDEO_480P_AR_TO_SIZE}
DEFAULT_VIDEO_RES = "video_480p"
DEFAULT_TIMESTEPS = 30  # upstream: --validation_num_timesteps 30
DEFAULT_TIMESTEP_SHIFT = 3.5  # upstream: --validation_timestep_shift 3.5
DEFAULT_CFG = 4.0  # upstream: --cfg_text_scale 4.0 (NOT 8.0)
DEFAULT_SEED = 42  # upstream: --validation_data_seed 42
DEFAULT_VIDEO_DURATION_SEC = 5  # 5s × 12 fps + 1 = 61 frames — snappy for interactive demo
MAX_VIDEO_DURATION_SEC = 10
# Per-task duration defaults — quality knobs (cfg, steps, ts_shift) are
# locked to upstream values, but the **demo duration** is intentionally
# shorter than upstream's per-script ``num_frames`` to keep gradio
# responsive (upstream uses 121 frames for t2v/i2v = 10s ≈ 2 min/req on
# B300; demo defaults give the same per-frame fidelity in ~60 s).
# Users can extend up to 10s with the slider when they want
# byte-identical reproduction of upstream's official output.
DURATION_DEFAULT_PER_TASK: dict[str, int] = {
    "t2v": 5,  # ~61 frames; upstream uses 121
    "i2v": 5,  # ~61 frames; upstream uses 121
    "video_edit": 4,  # ~49 frames; matches upstream's 50
}

# Task radio choices (label, internal_key)
TASK_VIDEO_GEN = "Video Generation"
TASK_I2V_LABEL = "Image to Video"
TASK_VIDEO_EDIT = "Video Edit"
TASK_VIDEO_UND = "Video Understanding"
TASK_IMAGE_GEN = "Image Generation"
TASK_IMAGE_EDIT = "Image Edit"
TASK_IMAGE_UND = "Image Understanding"

LABEL_TO_TASK = {
    TASK_VIDEO_GEN: "t2v",
    TASK_I2V_LABEL: "i2v",
    TASK_VIDEO_EDIT: "video_edit",
    TASK_VIDEO_UND: "x2t_video",
    TASK_IMAGE_GEN: "t2i",
    TASK_IMAGE_EDIT: "image_edit",
    TASK_IMAGE_UND: "x2t_image",
}

LANCE_HOMEPAGE_URL = "https://lance-project.github.io/"
LANCE_PAPER_URL = "http://arxiv.org/abs/2605.18678"
LANCE_HF_URL = "https://huggingface.co/bytedance-research/Lance"
LANCE_GITHUB_URL = "https://github.com/bytedance/Lance"
VLLM_OMNI_GITHUB_URL = "https://github.com/vllm-project/vllm-omni"


# --------------------------------------------------- module handles -----

_OMNI_IMAGE = None
_OMNI_VIDEO = None
_PARAMS_IMAGE = None
_PARAMS_VIDEO = None


# ----------------------------------------------------------- args -----


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Parent snapshot dir containing Lance_3B and Lance_3B_Video subdirs.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    p.add_argument(
        "--examples-root",
        default="/home/zjy/code/lance-upstream",
        help=(
            "Path to a Lance upstream checkout containing ``config/examples/*.json`` "
            "and the referenced media assets.  When present, each task tab gets a "
            "clickable list of official examples that auto-fill the prompt + media. "
            "Set to '' to disable."
        ),
    )
    p.add_argument(
        "--ulysses-degree",
        type=int,
        default=1,
        help=(
            "Ulysses sequence-parallel degree for *each* Omni instance "
            "(image and video).  Needs ``2 * ulysses_degree`` GPUs exposed via "
            "CUDA_VISIBLE_DEVICES.  NOTE: Lance currently lacks an ``_sp_plan`` "
            "registry and the i2v first-frame pin is not SP-aware, so SP >1 is "
            "experimental — t2i/t2v/edit/x2t work but i2v crashes.  Prefer "
            "``--replicas-per-omni`` for throughput scaling instead."
        ),
    )
    p.add_argument(
        "--replicas-per-omni",
        type=int,
        default=1,
        help=(
            "Spawn ``N`` parallel replicas of each Omni (image + video) across "
            "``2*N`` GPUs.  Default ``1`` = the P0 layout (2 GPUs total, image "
            "on GPU 0, video on GPU 1).  Set ``4`` to use all 8 GPUs for "
            "throughput; each replica loads a fresh model copy so startup time "
            "and memory scale linearly."
        ),
    )
    return p.parse_args()


# -------------------------------------------------- model loading -----


def _resolve_ckpts(snapshot: str):
    img = Path(snapshot) / "Lance_3B"
    vid = Path(snapshot) / "Lance_3B_Video"
    if not img.is_dir():
        raise FileNotFoundError(f"Lance_3B not found under {snapshot}")
    if not vid.is_dir():
        raise FileNotFoundError(f"Lance_3B_Video not found under {snapshot}")
    return str(img), str(vid)


def _init_models(
    snapshot: str,
    ulysses_degree: int = 1,
    replicas_per_omni: int = 1,
):
    """Load the two Lance checkpoints onto distinct GPU sets so image
    and video tasks can run truly in parallel, optionally with Ulysses
    sequence parallel inside each Omni.

    Device layout (with ``CUDA_VISIBLE_DEVICES=$DEVICES``):

    - ``ulysses_degree=1`` (P0): image Omni → logical 0, video Omni →
      logical 1.  Concurrent throughput = 2x serial; needs 2 GPUs total.
    - ``ulysses_degree=2`` (P1): image Omni → logical ``0,1``, video
      Omni → logical ``2,3``.  Each Omni internally splits the DiT
      denoising sequence across 2 GPUs (Ulysses SP).  Total = 4 GPUs;
      single-request t2v/i2v ~1.5-1.8x faster *and* image+video still
      concurrent.
    - ``ulysses_degree=N``: image → ``0..N-1``, video → ``N..2N-1``.
      Needs ``2*N`` GPUs.

    Falls back gracefully when there aren't enough GPUs: prints a
    warning and downgrades to the largest feasible degree.
    """
    global _OMNI_IMAGE, _OMNI_VIDEO, _PARAMS_IMAGE, _PARAMS_VIDEO
    from vllm_omni.entrypoints.omni import Omni

    img_ckpt, vid_ckpt = _resolve_ckpts(snapshot)

    import os

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible = [d for d in cvd.split(",") if d.strip()] if cvd else []
    n_visible = len(visible)

    # Resolve ulysses & replicas against available device count.
    if ulysses_degree < 1:
        ulysses_degree = 1
    if replicas_per_omni < 1:
        replicas_per_omni = 1
    # Per-Omni GPU count = ulysses_degree × replicas_per_omni
    # (ulysses splits a single replica across GPUs; replicas multiply by GPU).
    per_omni_gpus = ulysses_degree * replicas_per_omni
    needed = 2 * per_omni_gpus
    if needed > max(n_visible, 1):
        # Downgrade replicas first (cheaper than losing SP), then ulysses.
        feasible_replicas = max(1, n_visible // (2 * ulysses_degree))
        if feasible_replicas < replicas_per_omni:
            print(
                f"[unified] WARN: --replicas-per-omni={replicas_per_omni} × "
                f"ulysses_degree={ulysses_degree} needs {needed} GPUs but only "
                f"{n_visible} are visible. Falling back to replicas={feasible_replicas}.",
                flush=True,
            )
            replicas_per_omni = feasible_replicas
            per_omni_gpus = ulysses_degree * replicas_per_omni
        if 2 * per_omni_gpus > n_visible:
            feasible_sp = max(1, n_visible // 2)
            if ulysses_degree > feasible_sp:
                print(
                    f"[unified] WARN: --ulysses-degree={ulysses_degree} needs "
                    f"{2 * ulysses_degree} GPUs. Falling back to ulysses_degree={feasible_sp}.",
                    flush=True,
                )
                ulysses_degree = feasible_sp
                per_omni_gpus = ulysses_degree * replicas_per_omni

    can_split = n_visible >= 2
    use_sp = ulysses_degree > 1
    use_replicas = replicas_per_omni > 1

    print(
        f"[unified] CUDA_VISIBLE_DEVICES={cvd!r}  n_visible={n_visible}  "
        f"can_split={can_split}  ulysses_degree={ulysses_degree}  "
        f"replicas_per_omni={replicas_per_omni}  per_omni_gpus={per_omni_gpus}",
        flush=True,
    )

    # Build per-Omni device strings.  Image takes the lower half, video
    # the upper half.
    if per_omni_gpus > 1:
        img_devices = ",".join(str(i) for i in range(per_omni_gpus))
        vid_devices = ",".join(str(i) for i in range(per_omni_gpus, 2 * per_omni_gpus))
    elif can_split:
        img_devices, vid_devices = "0", "1"
    else:
        img_devices, vid_devices = "0", "0"

    print(
        f"[unified] Loading Lance_3B (image weights) on logical devices {img_devices!r}"
        f"  replicas={replicas_per_omni}...",
        flush=True,
    )
    # Lance single-stage engine defaults — mirrors the offline ``end2end.py``
    # flat-kwargs path so this demo does not depend on ``deploy/lance.yaml``.
    # ``pipeline="lance"`` is the explicit selector (Lance HF config has no
    # ``model_type`` field, so auto-detect from the model dir alone fails).
    lance_engine_defaults = dict(
        pipeline="lance",
        max_num_batched_tokens=32768,
        max_num_seqs=1,
        enforce_eager=True,
        trust_remote_code=True,
        enable_prefix_caching=False,
        async_chunk=False,
    )

    img_kwargs: dict = {"model": img_ckpt, **lance_engine_defaults}
    if img_devices != "0":
        img_kwargs["stage_0_devices"] = img_devices
    if use_sp:
        # Per the post-#3483 API, parallel knobs flow through *flat*
        # ``ulysses_degree`` / ``ring_degree`` kwargs on the Omni
        # constructor; ``_apply_diffusion_parallel_runtime_overrides``
        # moves them into a nested ``parallel_config`` dict for the stage.
        img_kwargs["ulysses_degree"] = ulysses_degree
    if use_replicas:
        img_kwargs["stage_0_num_replicas"] = replicas_per_omni
    _OMNI_IMAGE = Omni(**img_kwargs)
    _PARAMS_IMAGE = _OMNI_IMAGE.default_sampling_params_list

    print(
        f"[unified] Loading Lance_3B_Video (video weights) on logical devices {vid_devices!r}"
        f"  replicas={replicas_per_omni}...",
        flush=True,
    )
    vid_kwargs: dict = {"model": vid_ckpt, **lance_engine_defaults}
    vid_kwargs["stage_0_devices"] = vid_devices
    if use_sp:
        vid_kwargs["ulysses_degree"] = ulysses_degree
    if use_replicas:
        vid_kwargs["stage_0_num_replicas"] = replicas_per_omni
    _OMNI_VIDEO = Omni(**vid_kwargs)
    _PARAMS_VIDEO = _OMNI_VIDEO.default_sampling_params_list
    print("[unified] Both models ready.", flush=True)


def _omni_for(task: str):
    if task in {"t2v", "i2v", "video_edit", "x2t_video"}:
        return _OMNI_VIDEO, _PARAMS_VIDEO
    return _OMNI_IMAGE, _PARAMS_IMAGE


# ------------------------------------------ size resolution helpers -----

_AR_RATIOS: dict[str, float] = {
    # Mirror upstream's 6 buckets exactly — *no* 3:2 / 2:3, those are
    # snapped to 4:3 / 3:4 by upstream's bucket selector.  Must match
    # ``ASPECT_RATIO_CHOICES`` or gradio will reject the auto-picked
    # value as "not in the list of choices".
    "21:9": 21 / 9,
    "16:9": 16 / 9,
    "4:3": 4 / 3,
    "1:1": 1.0,
    "3:4": 3 / 4,
    "9:16": 9 / 16,
}


def _closest_aspect_ratio(width: int, height: int) -> str:
    """Snap an arbitrary ``W×H`` to the nearest preset in
    ``ASPECT_RATIO_CHOICES`` (used to auto-select the dropdown when the
    user uploads / clicks an i2v example image).  Distance is computed
    in log-space so 4:3 isn't unfairly favored over 1:1 just because of
    linear-scale proximity."""
    import math

    if not width or not height:
        return DEFAULT_VIDEO_AR
    img_log = math.log(width / height)
    best = min(_AR_RATIOS.items(), key=lambda kv: abs(math.log(kv[1]) - img_log))
    return best[0]


def get_size_for_task(task: str, ar: str, resolution: str):
    """Return (width, height) for a task + aspect-ratio + resolution preset."""
    if task in {"t2v", "i2v", "video_edit"}:
        m = VIDEO_RES_MAP.get(resolution, VIDEO_480P_AR_TO_SIZE)
        return m.get(ar, m["16:9"])
    elif task in {"t2i", "image_edit"}:
        return IMAGE_AR_TO_SIZE.get(ar, IMAGE_AR_TO_SIZE["1:1"])
    return (640, 352)


def _format_size_md(task: str, w: int, h: int) -> str:
    unit = "frames" if task in {"t2v", "i2v", "video_edit"} else "px"
    return f"{w} × {h} {unit}"


# ------------------------------------------ per-task prompt formats -----


def _vision_block(kind: str = "video"):
    from vllm_omni.diffusion.models.lance.prompts import (
        IMAGE_PAD,
        VIDEO_PAD,
        VISION_END,
        VISION_START,
    )

    pad = VIDEO_PAD if kind == "video" else IMAGE_PAD
    return f"{VISION_START}{pad}{VISION_END}"


def _format(task, prompt, image, video, num_frames, height, width, cfg, ts_shift):
    from vllm_omni.diffusion.models.lance.prompts import render_lance_prompt

    if task == "t2i":
        return [
            {
                "prompt": render_lance_prompt("t2i", prompt),
                "modalities": ["image"],
                "extra_args": {"height": int(height), "width": int(width), "cfg_text_scale": float(cfg)},
            }
        ]
    if task == "image_edit":
        return [
            {
                "prompt": render_lance_prompt("image_edit", prompt, vision_token=_vision_block("image")),
                "multi_modal_data": {"img2img": image},
                "modalities": ["image"],
                "extra_args": {"cfg_text_scale": float(cfg)},
            }
        ]
    if task == "x2t_image":
        q = (prompt or "").strip() or "Describe this image in detail."
        # Upstream's official x2t_image examples carry a QA-style
        # per-example system prompt ("Look at the image carefully and
        # answer the question.").  Without it, Lance's caption-default
        # system prompt makes the model describe the image instead of
        # answering — matching that here so gradio out-of-the-box agrees
        # with the upstream CLI on VQA cases.
        return [
            {
                "prompt": render_lance_prompt(
                    "x2t_image",
                    q,
                    vision_token=_vision_block("image"),
                    system_prompt="Look at the image carefully and answer the question.",
                ),
                "multi_modal_data": {"image": image},
                "modalities": ["text"],
                "extra_args": {"max_think_tokens": 500},
            }
        ]
    if task == "t2v":
        return [
            {
                "prompt": render_lance_prompt("t2v", prompt),
                "modalities": ["video"],
                "extra_args": {
                    "num_frames": int(num_frames),
                    "video_height": int(height),
                    "video_width": int(width),
                    "cfg_text_scale": float(cfg),
                    "timestep_shift": float(ts_shift),
                },
            }
        ]
    if task == "i2v":
        return [
            {
                "prompt": render_lance_prompt("i2v", prompt, vision_token=_vision_block("video")),
                "multi_modal_data": {"first_frame": image},
                "modalities": ["video"],
                "extra_args": {
                    "num_frames": int(num_frames),
                    "video_height": int(height),
                    "video_width": int(width),
                    "cfg_text_scale": float(cfg),
                    "timestep_shift": float(ts_shift),
                },
            }
        ]
    if task == "video_edit":
        return [
            {
                "prompt": render_lance_prompt("video_edit", prompt, vision_token=_vision_block("video")),
                "multi_modal_data": {"video": video},
                "modalities": ["video"],
                "extra_args": {
                    "num_frames": int(num_frames),
                    "video_height": int(height),
                    "video_width": int(width),
                    "cfg_text_scale": float(cfg),
                    "timestep_shift": float(ts_shift),
                },
            }
        ]
    if task == "x2t_video":
        q = (prompt or "").strip() or "Describe this video in detail."
        return [
            {
                "prompt": render_lance_prompt(
                    "x2t_video",
                    q,
                    vision_token=_vision_block("video"),
                    system_prompt="Watch the video carefully and answer the question.",
                ),
                "multi_modal_data": {"video": video},
                "modalities": ["text"],
                "extra_args": {"max_think_tokens": 500},
            }
        ]
    raise ValueError(f"Unknown task: {task}")


def _extract_demo_sources(root: str, cache_dir: str = "/tmp/lance_demo_sources") -> list[tuple[str, str]]:
    """Split upstream Lance's ``video-editing-demo-*.mp4`` and the
    ``multi-turn-editing-demo-01.mp4`` filmstrip into runnable source
    videos (the *input* half of each side-by-side demo) so they can be
    surfaced as clickable ``gr.Examples`` for ``video_edit``.

    Upstream's 8 video-editing demos are **horizontally split**
    ``(source | edit)`` — left half is the input video, right half is
    the edited result.  (A naïve top↔bottom RGB diff comparison is
    misleading: the demos are landscape scenes where natural sky/ground
    variance dominates the metric.  Confirmed visually that the split
    runs vertically through the middle.)  The multi-turn-editing demo
    is a horizontal filmstrip ``(panel0 | panel1 | … | panelN)`` where
    panel 0 is the source and each subsequent panel adds one more edit
    step.

    Returns ``[(source_video_path, suggested_prompt), ...]``.  Generated
    files are cached under ``cache_dir``; if the cache already has them
    we skip re-encoding.
    """
    from pathlib import Path as _Path

    import imageio.v3 as _iio
    import numpy as _np

    if not root or not _Path(root).is_dir():
        return []
    root_p = _Path(root)
    cache_p = _Path(cache_dir)
    cache_p.mkdir(parents=True, exist_ok=True)
    out: list[tuple[str, str]] = []

    # Hand-curated prompts inferred by visually diffing each demo's left
    # (source) vs right (edit) half — upstream's README ships the result
    # videos but doesn't publish the prompts that produced them, so these
    # are best-effort descriptions of the visible transformation.
    demo_prompts = {
        1: "Add a roaring bonfire with bright orange flames behind the bear.",
        2: "Change the lush green tropical forest to a dry autumn forest with warm sunlight.",
        3: "Add several colorful balloons floating around the giraffe.",
        4: "Change her hair from red to natural chestnut brown and have her gently touch it with one hand.",
        5: "Replace the dog with a tabby cat, keeping the same red collar and car window.",
        6: "Remove the black square patch from her cheek.",
        7: "Change the boy to a girl with long black hair, blue eyes, and a black dress.",
        8: "Place her in a snowy forest and add a red striped scarf around her neck.",
    }

    # ---- 8 horizontally-split video-editing demos ----
    for i in range(1, 9):
        src_demo = root_p / "assets" / "video-editing" / "videos" / f"video-editing-demo-{i:02d}.mp4"
        if not src_demo.exists():
            continue
        out_path = cache_p / f"video-edit-demo-{i:02d}-source.mp4"
        if not out_path.exists():
            arr = _iio.imread(str(src_demo), plugin="pyav")
            # Horizontal split: take left half (width axis).
            left = arr[:, :, : arr.shape[2] // 2, :]
            left = _np.ascontiguousarray(left)
            _iio.imwrite(str(out_path), left, fps=12, codec="libx264", quality=8)
        out.append((str(out_path), demo_prompts.get(i, "Describe the edit you'd like to apply.")))

    # ---- multi-turn editing filmstrip → first panel as source ----
    mt_src = root_p / "assets" / "multi-turn-editing" / "videos" / "multi-turn-editing-demo-01.mp4"
    if mt_src.exists():
        out_path = cache_p / "multi-turn-edit-source.mp4"
        if not out_path.exists():
            arr = _iio.imread(str(mt_src), plugin="pyav")
            T, H, W, _C = arr.shape
            # The filmstrip is roughly ``N`` square-ish panels of width H
            # each; pick the first panel as the source.  W/H ≈ 11.7 on the
            # shipped clip → panel_width ≈ H to keep aspect.
            panel_w = H
            first = arr[:, :, :panel_w, :]
            first = _np.ascontiguousarray(first)
            _iio.imwrite(str(out_path), first, fps=12, codec="libx264", quality=8)
        out.append(
            (
                str(out_path),
                "Apply a multi-turn consistency edit — try sequential edits like "
                "'change clothing color, then change background, then add accessories'.",
            )
        )

    return out


def _load_lance_showcase(root: str) -> dict[str, list[str]]:
    """Collect upstream Lance's README showcase videos as visual
    reference (full side-by-side ``before|after`` demos).  Distinct from
    :func:`_extract_demo_sources`, which crops just the source half so
    they can be run.

    Surfaces:
      * ``video_edit``: 8 ``video-editing-demo-{01..08}.mp4`` + the
        ``multi-turn-editing-demo-01.mp4`` filmstrip.
    """
    from pathlib import Path as _Path

    if not root or not _Path(root).is_dir():
        return {}
    root_p = _Path(root)
    out: dict[str, list[str]] = {}

    edit_videos: list[str] = []
    for i in range(1, 9):
        p = root_p / "assets" / "video-editing" / "videos" / f"video-editing-demo-{i:02d}.mp4"
        if p.exists():
            edit_videos.append(str(p))
    mt = root_p / "assets" / "multi-turn-editing" / "videos" / "multi-turn-editing-demo-01.mp4"
    if mt.exists():
        edit_videos.append(str(mt))
    if edit_videos:
        out["video_edit"] = edit_videos

    return out


def _load_lance_examples(root: str) -> dict[str, list[list]]:
    """Parse upstream Lance's ``config/examples/*.json`` into clickable
    presets keyed by task name.

    Each value is the list of rows that :class:`gr.Examples` expects, with
    one row per official sample.  The row column order matches the
    ``inputs`` we'll bind below: ``[prompt, image_path | None, video_path | None]``.

    Path resolution: example JSONs reference media via repo-relative paths
    (``config/examples/...``).  We join against ``root`` so the gradio
    process can serve the files directly.  Examples whose media is
    missing on disk are skipped silently so a partial upstream checkout
    still works.
    """
    import json as _json
    from pathlib import Path as _Path

    if not root or not _Path(root).is_dir():
        return {}
    root_p = _Path(root)
    out: dict[str, list[list]] = {}

    def _resolve(rel: str) -> str | None:
        if not isinstance(rel, str):
            return None
        p = root_p / rel
        return str(p) if p.exists() else None

    def _safe_json(name: str):
        f = root_p / "config" / "examples" / name
        if not f.exists():
            return None
        try:
            return _json.loads(f.read_text())
        except Exception:  # malformed json → skip
            return None

    # ---- t2i / t2v: ``{filename: prompt}`` maps; only prompt is useful as
    # an input preset (the filename is the *expected output*, not an
    # input). ----
    for task, fname in (("t2i", "t2i_example.json"), ("t2v", "t2v_example.json")):
        data = _safe_json(fname)
        if isinstance(data, dict):
            out[task] = [[prompt, None, None] for _, prompt in data.items() if isinstance(prompt, str)]

    # ---- i2v / video_edit: ``interleave_array = [prompt, media_path]`` ----
    # NOTE: image_edit handled separately below because upstream ships
    # only one source image with 5 different prompts; we substitute
    # diverse source images so each gradio example is visually distinct.
    for task, fname, modality in (
        ("i2v", "i2v_example.json", "image"),
        ("video_edit", "video_edit_example.json", "video"),
    ):
        data = _safe_json(fname)
        if not isinstance(data, dict):
            continue
        rows: list[list] = []
        for _, sample in data.items():
            ia = sample.get("interleave_array") if isinstance(sample, dict) else None
            if not (isinstance(ia, list) and len(ia) >= 2):
                continue
            prompt, media = ia[0], ia[1]
            media_path = _resolve(media)
            if media_path is None:
                continue
            row = [prompt, media_path, None] if modality == "image" else [prompt, None, media_path]
            rows.append(row)
        if rows:
            out[task] = rows

    # ---- image_edit: curated visually-distinct pairs ----
    # Upstream ``image_edit_example.json`` ships 5 *different prompts* against
    # the *same* source image (``edit_img.jpg``), which makes for a confusing
    # gradio gallery (5 thumbnails that look identical).  We substitute 5
    # visually-distinct sources from the i2v frame pool — same upstream
    # repo, just paired differently — and write a fresh edit prompt for
    # each so users see what kinds of transforms image_edit supports.
    image_edit_pairs = [
        # (relative source path, prompt)
        ("config/examples/image_edit_examples/edit_img.jpg", "Remove the hat from the painting."),
        (
            "config/examples/text_image_to_video_examples/00001.png",
            "Replace the icy glacier background with a sandy desert.",
        ),
        (
            "config/examples/text_image_to_video_examples/00005.png",
            "Change her hair to bright red curly hair and add red lipstick.",
        ),
        ("config/examples/text_image_to_video_examples/00004.png", "Convert the scene into a 3D cartoon render style."),
        (
            "config/examples/text_image_to_video_examples/00003.png",
            "Add a vivid sunset sky with orange and purple clouds behind the penguin.",
        ),
    ]
    ie_rows: list[list] = []
    for rel, prompt in image_edit_pairs:
        path = _resolve(rel)
        if path is not None:
            ie_rows.append([prompt, path, None])
    if ie_rows:
        out["image_edit"] = ie_rows

    # ---- Append the cropped video-editing showcase demos to ``video_edit`` ----
    # ``_extract_demo_sources`` slices the *source* half out of upstream's
    # vertically-stacked side-by-side clips so they're runnable as
    # ``video_edit`` examples (the bare side-by-side mp4s have the edit
    # output baked in and would re-edit the wrong frames).
    demo_rows = _extract_demo_sources(root)
    if demo_rows:
        existing = out.get("video_edit", [])
        for src_path, prompt in demo_rows:
            existing.append([prompt, None, src_path])
        out["video_edit"] = existing

    # ---- x2t_image / x2t_video: ``interleave_array = [media_path, [system, question, answer]]`` ----
    for task, fname, modality in (
        ("x2t_image", "x2t_image_example.json", "image"),
        ("x2t_video", "x2t_video_example.json", "video"),
    ):
        data = _safe_json(fname)
        if not isinstance(data, dict):
            continue
        rows: list[list] = []
        for _, sample in data.items():
            ia = sample.get("interleave_array") if isinstance(sample, dict) else None
            if not (isinstance(ia, list) and len(ia) >= 2 and isinstance(ia[1], list) and len(ia[1]) >= 2):
                continue
            media_path = _resolve(ia[0])
            question = ia[1][1]
            if media_path is None or not isinstance(question, str):
                continue
            row = [question, media_path, None] if modality == "image" else [question, None, media_path]
            rows.append(row)
        if rows:
            out[task] = rows

    return out


def _to_pil(x):
    if x is None:
        return None
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, np.ndarray):
        return Image.fromarray(x).convert("RGB")
    if isinstance(x, str):
        return Image.open(x).convert("RGB")
    raise ValueError(f"Unsupported image type {type(x)}")


def _save_video(frames):
    arr = np.stack([np.asarray(f) for f in frames], axis=0).astype(np.uint8)
    out = Path(tempfile.mkdtemp(prefix="lance_grad_")) / "out.mp4"
    iio.imwrite(str(out), arr, fps=12, codec="libx264", quality=8)
    return str(out)


def _save_image(img):
    out = Path(tempfile.mkdtemp(prefix="lance_grad_")) / "out.png"
    if isinstance(img, np.ndarray):
        Image.fromarray(img).save(out)
    elif isinstance(img, Image.Image):
        img.save(out)
    else:
        raise ValueError(type(img))
    return str(out)


# ---------------------------------------------------- generation -----


def run_task(
    task_label,
    prompt,
    image,
    video,
    aspect_ratio,
    resolution,
    video_duration_sec,
    steps,
    cfg,
    ts_shift,
    seed,
    num_frames_override,
):
    task = LABEL_TO_TASK[task_label]
    omni, params = _omni_for(task)

    image = _to_pil(image) if image is not None else None

    # Resolve W/H from aspect + resolution
    width, height = get_size_for_task(task, aspect_ratio, resolution)

    # Frames: for video tasks, use duration × fps
    if task in {"t2v", "i2v", "video_edit"}:
        fps = 12
        if num_frames_override and num_frames_override > 0:
            num_frames = int(num_frames_override)
        else:
            num_frames = int(video_duration_sec * fps + 1)  # +1 for first frame
    else:
        num_frames = 1

    sp = params[0]
    if seed is not None:
        try:
            sp.seed = int(seed)
        except Exception:
            sp.seed = DEFAULT_SEED
    sp.num_inference_steps = int(steps)

    formatted = _format(task, prompt or "", image, video, num_frames, height, width, cfg, ts_shift)
    print(f"[unified] task={task} {width}x{height} frames={num_frames} cfg={cfg} steps={steps} seed={seed}", flush=True)

    outputs = list(omni.generate(prompts=formatted, sampling_params_list=params))
    custom = getattr(outputs[0], "custom_output", None) or {}
    # Lance/Bagel return text at outputs[0].output (string) or
    # custom_output["text_output"].
    text_out = None
    if task in {"x2t_image", "x2t_video"}:
        text_out = custom.get("text_output") or getattr(outputs[0], "output", None)
        if not isinstance(text_out, str):
            text_out = None

    # NOTE: previously this returned a 4-tuple ending in a status string
    # rendered into a ``gr.Markdown`` slot below the outputs.  That slot
    # left an empty grey bar in the UI before the first run and was
    # dropped — Gradio's built-in error toast already surfaces failures,
    # and successful runs are self-evident from the output panel.
    if task in {"x2t_image", "x2t_video"}:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=str(text_out or "(empty)"), visible=True),
        )
    if task in {"t2v", "i2v", "video_edit"}:
        frames = custom.get("video_frames")
        if frames is None:
            raise gr.Error("No video frames in output.")
        return (gr.update(visible=False), gr.update(value=_save_video(frames), visible=True), gr.update(visible=False))
    # image tasks — orchestrator strips ``outputs[0].output`` across the
    # IPC boundary but preserves ``custom_output``, so prefer the latter.
    img_out = custom.get("image") or getattr(outputs[0], "output", None)
    if img_out is None:
        raise gr.Error("No image in output.")
    return (gr.update(value=_save_image(img_out), visible=True), gr.update(visible=False), gr.update(visible=False))


# -------------------------------------------------- UI building -----

# Custom CSS — minimal, matches Lance branding (rounded panels + dark header).
APP_CSS = """
.lance-hero {
  display: flex; flex-direction: column; align-items: center;
  gap: 12px; padding: 18px 12px; margin-bottom: 8px;
}
.lance-hero .logos {
  display: flex; align-items: center; gap: 24px; justify-content: center;
}
.lance-hero .logos img { height: 64px; }
.lance-hero .logos .x { font-size: 28px; color: #888; font-weight: 300; }
.lance-hero .vllm-logo { height: 44px; }
.lance-hero .lance-title {
  margin: 6px 0 0 0; font-size: 1.45rem; font-weight: 600;
  text-align: center; color: #475569;
}
.lance-hero .lance-badges { display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; }
.lance-panel { padding: 12px; border-radius: 12px; }
.task-selector .wrap { display: flex; flex-wrap: wrap; gap: 6px; }
.lance-run-button { font-size: 1.05rem !important; padding: 12px 18px !important; }
/* Reserve a single line of vertical space so the layout doesn't jump
   when status text appears/disappears — but ``:empty`` collapses to 0
   so a placeholder ``gr.Markdown("")`` (e.g. the run-status slot before
   the first generation) doesn't render as a phantom grey bar. */
.lance-run-status { text-align: center; color: #475569; min-height: 1.4em; }
.lance-run-status:empty,
.lance-run-status .prose:empty,
.lance-run-status .prose p:empty { min-height: 0 !important; padding: 0 !important; margin: 0 !important; }
.lance-display-frame video, .lance-display-frame img { max-height: 560px; }
"""


def _embed_logo(path: str, mime: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_header_html() -> str:
    lance_uri = _embed_logo("/home/zjy/code/lance-upstream/assets/logo/lance-logo.webp", "image/webp")
    vllm_uri = _embed_logo("/home/zjy/code/vllm-omni/docs/source/logos/vllm-omni-logo.png", "image/png")
    logos_html = ""
    if lance_uri:
        logos_html += f'<img src="{lance_uri}" alt="Lance">'
    if vllm_uri:
        logos_html += '<span class="x">×</span>'
        logos_html += f'<img class="vllm-logo" src="{vllm_uri}" alt="vllm-omni">'
    return f"""
    <div class="lance-hero">
      <div class="logos">{logos_html}</div>
      <h1 class="lance-title">Lance × vllm-omni · Unified Multimodal Inference</h1>
      <div class="lance-badges">
        <a href="{LANCE_HOMEPAGE_URL}" target="_blank" rel="noopener"><img alt="Homepage" src="https://img.shields.io/badge/Homepage-Lance-2563eb?style=flat&labelColor=475569"></a>
        <a href="{LANCE_PAPER_URL}" target="_blank" rel="noopener"><img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv-2563eb?style=flat&labelColor=475569&logo=arxiv"></a>
        <a href="{LANCE_HF_URL}" target="_blank" rel="noopener"><img alt="HF" src="https://img.shields.io/badge/Model-HuggingFace-2563eb?style=flat&labelColor=475569&logo=huggingface"></a>
        <a href="{LANCE_GITHUB_URL}" target="_blank" rel="noopener"><img alt="Lance GitHub" src="https://img.shields.io/badge/Lance-GitHub-2563eb?style=flat&labelColor=475569&logo=github"></a>
        <a href="{VLLM_OMNI_GITHUB_URL}" target="_blank" rel="noopener"><img alt="vllm-omni GitHub" src="https://img.shields.io/badge/vllm--omni-GitHub-2563eb?style=flat&labelColor=475569&logo=github"></a>
      </div>
    </div>
    """


def build_ui(
    examples_by_task: dict[str, list[list]] | None = None,
    showcase_by_task: dict[str, list[str]] | None = None,
):
    examples_by_task = examples_by_task or {}
    showcase_by_task = showcase_by_task or {}
    with gr.Blocks(title="Lance × vllm-omni", css=APP_CSS) as demo:
        gr.HTML(build_header_html())

        # ── Task selector ──
        task_radio = gr.Radio(
            label="Task",
            show_label=False,
            choices=list(LABEL_TO_TASK.keys()),
            value=TASK_I2V_LABEL,
            elem_classes=["task-selector"],
        )

        with gr.Row():
            # ─── LEFT: inputs ───
            with gr.Column(scale=1):
                with gr.Group(elem_classes=["lance-panel"]):
                    prompt_in = gr.Textbox(
                        label="Prompt",
                        lines=5,
                        placeholder="Describe the desired output...",
                    )
                    # Aspect ratio is **not user-configurable** — upstream
                    # Lance always snaps to one of 6 hard-coded buckets based
                    # on the input image / video.  We keep the component in
                    # the tree (so existing event wiring still works) but
                    # render it invisibly; ``_auto_aspect_from_image`` /
                    # ``_auto_aspect_from_video`` are the only writers.
                    aspect_in = gr.Dropdown(
                        label="Aspect Ratio (auto)",
                        choices=ASPECT_RATIO_CHOICES,
                        value=DEFAULT_VIDEO_AR,
                        visible=False,
                        interactive=False,
                    )
                    with gr.Row():
                        resolution_in = gr.Dropdown(
                            label="Resolution",
                            choices=VIDEO_RES_CHOICES,
                            value=DEFAULT_VIDEO_RES,
                            visible=True,
                        )
                        duration_in = gr.Slider(
                            1,
                            MAX_VIDEO_DURATION_SEC,
                            value=DEFAULT_VIDEO_DURATION_SEC,
                            step=1,
                            label="Duration (sec)",
                        )
                    size_md = gr.Markdown(_format_size_md("i2v", 848, 480), elem_classes=["lance-run-status"])

                image_in = gr.Image(label="Input Image", type="pil", visible=True, elem_classes=["lance-display-frame"])
                video_in = gr.Video(label="Input Video", visible=False, elem_classes=["lance-display-frame"])

                with gr.Accordion("Advanced Parameters", open=False):
                    seed_in = gr.Number(label="Seed", value=DEFAULT_SEED, precision=0)
                    steps_in = gr.Slider(5, 80, value=DEFAULT_TIMESTEPS, step=5, label="Denoise steps")
                    with gr.Row():
                        cfg_in = gr.Number(label="CFG text scale", value=DEFAULT_CFG)
                        ts_shift_in = gr.Number(label="Timestep shift", value=DEFAULT_TIMESTEP_SHIFT)
                    num_frames_override = gr.Number(
                        label="Override num_frames (0 = derive from duration)",
                        value=0,
                        precision=0,
                    )

                run_btn = gr.Button("🚀 Generate", variant="primary", elem_classes=["lance-run-button"])

            # ─── RIGHT: output ───
            with gr.Column(scale=1):
                with gr.Group(elem_classes=["lance-panel"]):
                    image_out = gr.Image(label="Output Image", visible=False, elem_classes=["lance-display-frame"])
                    video_out = gr.Video(
                        label="Output Video",
                        autoplay=True,
                        loop=True,
                        visible=True,
                        elem_classes=["lance-display-frame"],
                    )
                    text_out = gr.Textbox(label="Output Caption", lines=6, visible=False)
                # Status text was previously rendered as ``gr.Markdown("",
                # ...)`` which left an empty grey bar in the UI before the
                # first run.  Dropped — Gradio's built-in toast / error
                # bubble already surfaces success / failure, so the extra
                # placeholder added no signal but stole vertical real estate.

        # ─── Official Lance examples per task ───
        # Click an example row → prompt + image/video auto-fill.  Each
        # task has its own ``gr.Examples`` block; only the active task's
        # block is shown.  Order matches LABEL_TO_TASK (radio order).
        _initial_task = LABEL_TO_TASK[TASK_I2V_LABEL]  # default selection
        _example_groups: dict[str, gr.Group] = {}
        if examples_by_task:
            # Per-task input bindings — Gradio's ``gr.Video`` doesn't reliably
            # update from a programmatic example click when the Examples row
            # also includes ``None`` placeholders for ``Image`` (the player
            # stays blank).  Binding each task's Examples block to *only* the
            # components it actually fills sidesteps that issue.
            task_input_components: dict[str, list] = {
                "t2i": [prompt_in],
                "t2v": [prompt_in],
                "i2v": [prompt_in, image_in],
                "image_edit": [prompt_in, image_in],
                "x2t_image": [prompt_in, image_in],
                "video_edit": [prompt_in, video_in],
                "x2t_video": [prompt_in, video_in],
            }

            def _project_row(task: str, row: list) -> list:
                # ``examples_by_task`` rows are always ``[prompt, image, video]``
                # (3 columns).  Project down to whatever this task binds.
                prompt, img, vid = row
                comps = task_input_components[task]
                out = []
                for c in comps:
                    if c is prompt_in:
                        out.append(prompt)
                    elif c is image_in:
                        out.append(img)
                    elif c is video_in:
                        out.append(vid)
                return out

            for task in ("t2i", "t2v", "i2v", "image_edit", "video_edit", "x2t_image", "x2t_video"):
                rows = examples_by_task.get(task)
                if not rows:
                    continue
                projected_rows = [_project_row(task, r) for r in rows]
                with gr.Group(visible=(task == _initial_task), elem_classes=["lance-panel"]) as grp:
                    gr.Markdown(f"### 📁 Official Lance examples — `{task}`  (click to load)")
                    # Bind to only the components this task actually populates.
                    gr.Examples(
                        examples=projected_rows,
                        inputs=task_input_components[task],
                        label="",
                        examples_per_page=4,
                        cache_examples=False,
                    )
                    # NOTE: Upstream's 8 ``video-editing-demo-*.mp4`` + the
                    # ``multi-turn-editing-demo-01.mp4`` filmstrip are merged
                    # into the runnable examples list above via
                    # ``_extract_demo_sources`` (left-half / first-panel crops
                    # become the source video, paired with a generic editable
                    # prompt).  No separate showcase gallery is needed.
                _example_groups[task] = grp

        gr.Markdown("""---
💡 **Tips:**
* **Video Generation**: text → 5-second video at 480p
* **Image to Video**: upload first frame + describe motion; for subtle facial expressions try CFG 10-15
* **Image Edit**: upload image + describe the edit; outputs a new image
* **Video Edit**: upload video + describe the edit; outputs a modified video
* **Understanding tasks**: upload media; outputs caption text""")

        # ─────────────── UI logic ───────────────

        # Fixed task order for the example-group visibility tail of on_task's
        # outputs — used both to build the return list and to register the
        # corresponding output components below.  Must stay in sync with the
        # group-creation loop above.
        _example_task_order = [
            t
            for t in ("t2i", "t2v", "i2v", "image_edit", "video_edit", "x2t_image", "x2t_video")
            if t in _example_groups
        ]

        def on_task(label):
            task = LABEL_TO_TASK[label]
            is_video_task = task in {"t2v", "i2v", "video_edit"}
            is_image_task = task in {"t2i", "image_edit"}
            is_understanding = task in {"x2t_image", "x2t_video"}
            # Understanding tasks don't need a prompt — system prompt handles
            # captioning. Generation/edit tasks need a prompt.
            need_text = not is_understanding
            need_image = task in {"i2v", "image_edit", "x2t_image"}
            need_video = task in {"video_edit", "x2t_video"}
            default_ar = DEFAULT_VIDEO_AR if (is_video_task or is_understanding) else DEFAULT_IMAGE_AR
            w, h = get_size_for_task(task, default_ar, DEFAULT_VIDEO_RES)
            base = [
                gr.update(visible=need_text),  # prompt_in
                gr.update(visible=need_image, value=None),  # image_in
                gr.update(visible=need_video, value=None),  # video_in
                gr.update(value=default_ar, visible=False),  # aspect (always hidden, auto-detected)
                gr.update(visible=is_video_task, value=DEFAULT_VIDEO_RES),  # resolution
                gr.update(
                    visible=is_video_task,
                    value=DURATION_DEFAULT_PER_TASK.get(task, DEFAULT_VIDEO_DURATION_SEC),
                ),  # duration — matches upstream's per-script num_frames default
                gr.update(value=_format_size_md(task, w, h)),  # size_md
                gr.update(visible=is_image_task and not is_video_task),  # image_out (only show for image-gen tasks)
                gr.update(visible=is_video_task),  # video_out
                gr.update(visible=is_understanding),  # text_out
            ]
            return base + [gr.update(visible=(t == task)) for t in _example_task_order]

        task_radio.change(
            on_task,
            [task_radio],
            [
                prompt_in,
                image_in,
                video_in,
                aspect_in,
                resolution_in,
                duration_in,
                size_md,
                image_out,
                video_out,
                text_out,
            ]
            + [_example_groups[t] for t in _example_task_order],
        )

        def on_aspect_or_res(task_label, ar, resolution):
            task = LABEL_TO_TASK[task_label]
            w, h = get_size_for_task(task, ar, resolution)
            return gr.update(value=_format_size_md(task, w, h))

        aspect_in.change(on_aspect_or_res, [task_radio, aspect_in, resolution_in], [size_md])
        resolution_in.change(on_aspect_or_res, [task_radio, aspect_in, resolution_in], [size_md])

        # ── Auto-pick aspect ratio from the input media ──
        # For tasks where the model is conditioned on an input
        # image/video (image_edit, i2v, video_edit), the output should
        # preserve the source aspect ratio — otherwise the model has to
        # squeeze / stretch the scene to fill a different frame, and the
        # geometry drifts (e.g. the snow-leopard "stepping into the
        # void" case: a 3:2 input rendered into a 16:9 output pushed the
        # landing platform off-frame).  When the user uploads or clicks
        # an example, snap the aspect dropdown to whichever preset is
        # closest to the source.
        def _auto_aspect_from_image(task_label, img):
            task = LABEL_TO_TASK[task_label]
            if task not in {"image_edit", "i2v"} or img is None:
                return gr.update(), gr.update()
            try:
                pil = _to_pil(img)
                w, h = pil.size
            except Exception:
                return gr.update(), gr.update()
            ar = _closest_aspect_ratio(w, h)
            new_w, new_h = get_size_for_task(task, ar, DEFAULT_VIDEO_RES)
            return (
                gr.update(value=ar),
                gr.update(value=_format_size_md(task, new_w, new_h)),
            )

        def _auto_aspect_from_video(task_label, vid_path):
            task = LABEL_TO_TASK[task_label]
            if task != "video_edit" or not vid_path:
                return gr.update(), gr.update()
            try:
                import imageio.v3 as _iio

                meta = _iio.immeta(vid_path, plugin="pyav")
                # ``immeta`` ships ``size = (w, h)`` for video readers.
                if "size" in meta:
                    w, h = meta["size"]
                else:
                    # Fall back to a single-frame read.
                    arr = _iio.imread(vid_path, plugin="pyav", index=0)
                    h, w = arr.shape[:2]
            except Exception:
                return gr.update(), gr.update()
            ar = _closest_aspect_ratio(w, h)
            new_w, new_h = get_size_for_task(task, ar, DEFAULT_VIDEO_RES)
            return (
                gr.update(value=ar),
                gr.update(value=_format_size_md(task, new_w, new_h)),
            )

        image_in.change(
            _auto_aspect_from_image,
            [task_radio, image_in],
            [aspect_in, size_md],
        )
        video_in.change(
            _auto_aspect_from_video,
            [task_radio, video_in],
            [aspect_in, size_md],
        )

        run_btn.click(
            run_task,
            inputs=[
                task_radio,
                prompt_in,
                image_in,
                video_in,
                aspect_in,
                resolution_in,
                duration_in,
                steps_in,
                cfg_in,
                ts_shift_in,
                seed_in,
                num_frames_override,
            ],
            outputs=[image_out, video_out, text_out],
        )

    return demo


def main():
    args = parse_args()
    # Lance aligns better with upstream when using FLASH_ATTN
    # (upstream uses flash_attn_varlen_func; SDPA accumulates ~5x more
    # numerical drift through the 36-layer Qwen2 stack).  Default to
    # flash_attn unless the user explicitly set a backend.
    os.environ.setdefault("DIFFUSION_ATTENTION_BACKEND", "FLASH_ATTN")
    print("=" * 60, flush=True)
    print("[unified] Lance × vllm-omni demo", flush=True)
    print(f"  snapshot:      {args.model}", flush=True)
    print(f"  gradio:        {args.host}:{args.port} (share={args.share})", flush=True)
    print("=" * 60, flush=True)
    _init_models(
        args.model,
        ulysses_degree=args.ulysses_degree,
        replicas_per_omni=args.replicas_per_omni,
    )
    examples_by_task = _load_lance_examples(args.examples_root)
    if examples_by_task:
        total = sum(len(rows) for rows in examples_by_task.values())
        per_task = ", ".join(f"{t}={len(r)}" for t, r in examples_by_task.items())
        print(f"[unified] Loaded {total} official examples ({per_task}) from {args.examples_root}", flush=True)
    else:
        print(f"[unified] No official examples loaded (examples_root={args.examples_root!r})", flush=True)
    demo = build_ui(examples_by_task=examples_by_task)
    # Concurrency budget = (image replicas + video replicas) so all the
    # available GPU slots can run in parallel.  P0 (1 replica each) = 2;
    # ``--replicas-per-omni 4`` on 8 GPUs gives 8 concurrent slots.
    queue_limit = 2 * max(1, args.replicas_per_omni)
    print(f"[unified] Gradio queue concurrency limit = {queue_limit}", flush=True)
    demo.queue(default_concurrency_limit=queue_limit)
    # Whitelist the upstream checkout so gr.Examples can stage media that
    # lives outside the gradio working dir / temp.  Without this, clicks
    # raise ``InvalidPathError`` and the example silently fails to load.
    launch_kwargs: dict = dict(server_name=args.host, server_port=args.port, share=args.share)
    allowed = []
    if args.examples_root and Path(args.examples_root).is_dir():
        allowed.append(str(Path(args.examples_root).resolve()))
    # ``_extract_demo_sources`` caches cropped video-editing demo sources under
    # ``/tmp/lance_demo_sources`` so they survive across launches; whitelist it
    # too or the click-to-load will hit ``InvalidPathError``.
    demo_cache = Path("/tmp/lance_demo_sources")
    if demo_cache.is_dir():
        allowed.append(str(demo_cache.resolve()))
    if allowed:
        launch_kwargs["allowed_paths"] = allowed
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
