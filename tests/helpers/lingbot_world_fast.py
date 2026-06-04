# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared L3-fixture helpers for Lingbot World Fast expansion tests."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants — single source of truth across the two expansion tests
# ---------------------------------------------------------------------------

# Mirrors ``examples/offline_inference/lingbot_world_fast/end2end.py`` and
# ``examples/online_serving/lingbot_world_fast/openai_client.py``.
GREAT_WALL_PROMPT = "A sweeping cinematic journey along the Great Wall of China, winding through golden autumn hills under a brilliant blue sky — stone pathways stretch into the distance, watchtowers stand sentinel, and vibrant foliage blankets the mountainsides as the camera glides smoothly forward, capturing the grandeur and timeless majesty of this ancient wonder."
SEED = 42
WIDTH = 832
HEIGHT = 480
FPS = 16

SHORT_NUM_FRAMES = 25
LONG_NUM_FRAMES = 81

EXTENSION_WARMUP_DROP = 4

SSIM_THRESHOLD = 0.95


@dataclass(frozen=True)
class LingbotWorldFastAssets:
    """All external assets a real-weight Lingbot World Fast test needs."""

    weights_path: Path
    camera_dir: Path
    image_path: Path


# ---------------------------------------------------------------------------
# Resolution helpers (no pytest imports here — callers decide whether to skip)
# ---------------------------------------------------------------------------


def _hf_cache_root() -> Path:
    return Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))


def _hf_model_snapshot_dirs(repo_id: str) -> list[Path]:
    snapshots = _hf_cache_root() / "hub" / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not snapshots.exists():
        return []
    return sorted(
        (p for p in snapshots.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _repo_root() -> Path:
    # tests/helpers/lingbot_world_fast.py → repo root
    return Path(__file__).resolve().parents[2]


def _example_checkpoint_root() -> Path:
    return (
        _repo_root()
        / "examples"
        / "offline_inference"
        / "lingbot_world_fast"
        / "lingbot_world"
        / "lingbot-world-base-cam"
        / "Lingbot-World-Fast"
    )


def _example_camera_root_candidates() -> list[Path]:
    base = (
        _repo_root()
        / "examples"
        / "offline_inference"
        / "lingbot_world_fast"
        / "lingbot_world"
        / "lingbot-world-base-cam"
    )
    return [base / "examples" / "04", base / "04", base]


def find_lingbot_world_fast_weights() -> Path | None:
    """Return a path to the Lingbot World Fast model directory, or ``None``.

    Resolution order: ``LINGBOT_WORLD_FAST_PATH`` env var → committed example
    checkpoint → HF cache snapshot of ``robbyant/lingbot-world-base-cam``.
    A *path is real* only when the required ``config.json`` plus at least
    one ``model-*.safetensors`` shard are present, so a half-pulled snapshot
    doesn't masquerade as a usable checkpoint.
    """
    override = os.environ.get("LINGBOT_WORLD_FAST_PATH")
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))

    candidates.append(_example_checkpoint_root())

    for snapshot in _hf_model_snapshot_dirs("robbyant/lingbot-world-base-cam"):
        candidates.append(snapshot / "Lingbot-World-Fast")

    for path in candidates:
        if not path.exists() or not path.is_dir():
            continue
        if not (path / "config.json").exists():
            continue
        if not any(path.glob("model-*.safetensors")):
            continue
        return path
    return None


def find_lingbot_world_fast_camera_dir() -> Path | None:
    """Locate a directory with ``poses.npy`` + ``intrinsics.npy``."""
    override = os.environ.get("LINGBOT_WORLD_FAST_CAMERA_PATH")
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))
    candidates.extend(_example_camera_root_candidates())
    for path in candidates:
        if path.exists() and (path / "poses.npy").exists() and (path / "intrinsics.npy").exists():
            return path
    return None


def find_lingbot_world_fast_image() -> Path | None:
    """Locate the example input image used by ``run_fast.sh`` case 2."""
    override = os.environ.get("LINGBOT_WORLD_FAST_IMAGE")
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))
    for camera_root in _example_camera_root_candidates():
        for name in ("image.jpg", "image.png", "input.jpg", "input.png"):
            candidates.append(camera_root / name)
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def find_lingbot_world_fast_assets() -> LingbotWorldFastAssets | None:
    """Resolve weights + camera trajectory + image in one call, or return None
    if any of the three is missing — callers can then ``pytest.skip`` with a
    single specific reason."""
    weights = find_lingbot_world_fast_weights()
    camera = find_lingbot_world_fast_camera_dir()
    image = find_lingbot_world_fast_image()
    if not (weights and camera and image):
        return None
    return LingbotWorldFastAssets(weights_path=weights, camera_dir=camera, image_path=image)


def golden_frames_dir() -> Path:
    return _repo_root() / "tests" / "data" / "lingbot_world_fast"


# ---------------------------------------------------------------------------
# Per-call payload helpers
# ---------------------------------------------------------------------------


def load_camera_trajectory(camera_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    poses = np.load(camera_dir / "poses.npy")
    intrinsics = np.load(camera_dir / "intrinsics.npy")
    return poses, intrinsics


def slice_camera_chunk(
    poses: np.ndarray,
    intrinsics: np.ndarray,
    *,
    call_index: int,
    chunk_stride: int = SHORT_NUM_FRAMES,
) -> dict[str, np.ndarray]:
    """Mirrors the slicing in ``examples/online_serving/lingbot_world_fast/openai_client.py``.

    Each call consumes ``chunk_stride`` poses. The model will floor internally if the slice has fewer
    poses than requested.
    """
    start = call_index * chunk_stride
    end = start + chunk_stride
    poses_slice = poses[start:end]
    intrinsics_slice = intrinsics[start:end] if intrinsics.ndim > 2 else intrinsics
    return {"poses": poses_slice, "intrinsics": intrinsics_slice}


# ---------------------------------------------------------------------------
# Frame post-processing helpers
# ---------------------------------------------------------------------------


def reassemble_chunked_video(
    per_call_frames: list[np.ndarray],
    *,
    drop_warmup: int = EXTENSION_WARMUP_DROP,
) -> np.ndarray:
    """Concatenate per-call frame chunks, dropping ``drop_warmup`` leading
    frames on every extension call (call index >= 1)."""
    assembled: list[np.ndarray] = []
    for i, frames in enumerate(per_call_frames):
        clip = frames[drop_warmup:] if i > 0 else frames
        assembled.append(clip)
    return np.concatenate(assembled, axis=0)


def normalize_to_uint8_rgb(frames: np.ndarray) -> np.ndarray:
    """Coerce a generated frames tensor to ``[N, H, W, 3]`` ``uint8``.

    The diffusion engine emits either an unsigned-int chunk (pre-encoded by
    ``_normalize_frames``) or a float tensor in ``[-1, 1]``. We accept both
    so the SSIM helper sees a single canonical shape.
    """
    arr = frames
    if arr.dtype.kind == "f":
        arr = np.clip((arr + 1.0) * 0.5, 0.0, 1.0)
        arr = (arr * 255.0).round().astype(np.uint8)
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def frame_ssim(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Per-frame SSIM with ``data_range=1``. Uses ``torchmetrics`` (already a
    transitive dep) and accepts ``[H, W, 3]`` uint8 arrays.
    """
    import torch
    from torchmetrics.image import StructuralSimilarityIndexMeasure

    pred_t = (torch.from_numpy(prediction.astype(np.float32) / 255.0)).permute(2, 0, 1).unsqueeze(0)
    ref_t = (torch.from_numpy(reference.astype(np.float32) / 255.0)).permute(2, 0, 1).unsqueeze(0)
    metric = StructuralSimilarityIndexMeasure(data_range=1.0)
    return float(metric(pred_t, ref_t).item())
