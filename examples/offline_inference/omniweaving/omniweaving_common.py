from __future__ import annotations

import json
import math
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image


@dataclass
class BenchmarkCase:
    name: str
    prompt: str
    width: int
    height: int
    num_frames: int
    num_inference_steps: int
    guidance_scale: float = 6.0
    negative_prompt: str | None = None
    image_path: str | None = None
    image_paths: list[str] | None = None
    flow_shift: float | None = None
    fps: int = 8
    notes: str | None = None
    # If set, passed to official `generate.py --aspect_ratio` instead of inferring from width/height.
    aspect_ratio: str | None = None
    # If set, passed to official `generate.py --pipeline_config` (e.g. `omniweaving2` for flow_shift=5.0 T2V).
    pipeline_config: str | None = None


@dataclass
class BenchmarkResult:
    name: str
    status: str
    latency_s: float | None = None
    peak_memory_gib: float | None = None
    output_path: str | None = None
    stage_durations: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_cuda_available() -> None:
    if torch.cuda.is_available():
        return
    raise RuntimeError(
        "OmniWeaving video generation requires a CUDA GPU. "
        "This machine is currently in no-GPU mode, so local execution is expected "
        "to stop at the CUDA availability check."
    )


def query_max_gpu_memory_gib() -> float | None:
    """Max GPU memory used (MiB) across devices, converted to GiB (via 1024).

    Matches `bench_omniweaving_baseline.py`: `nvidia-smi --query-gpu=memory.used`.
    """
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    values = [int(line.strip()) for line in proc.stdout.splitlines() if line.strip()]
    if not values:
        return None
    return max(values) / 1024.0


def nvidia_smi_peak_memory_delta_gib(
    fn: Callable[[], Any],
    *,
    poll_interval_s: float = 0.2,
) -> tuple[Any, float]:
    """Run ``fn()`` while polling `nvidia-smi`; return (result, delta_gib).

    ``delta_gib`` is ``max(observed max across GPUs) - baseline`` before ``fn``,
    floored at 0 — the same convention as `bench_omniweaving_baseline.py` for the
    baseline subprocess (incremental VRAM attributable to the measured window).

    Note: For a subprocess-based baseline, the window includes model load in the
    child. For an in-process ``Omni.generate()`` call, the window is usually
    **with the model already resident**; document which window you use.
    """
    baseline = query_max_gpu_memory_gib() or 0.0
    peak = baseline
    stop = threading.Event()

    def poll_worker() -> None:
        nonlocal peak
        while not stop.is_set():
            v = query_max_gpu_memory_gib()
            if v is not None and v > peak:
                peak = v
            time.sleep(poll_interval_s)

    worker = threading.Thread(target=poll_worker, daemon=True)
    worker.start()
    try:
        result = fn()
    finally:
        stop.set()
        worker.join(timeout=10.0)
    delta = max(peak - baseline, 0.0)
    return result, delta


def build_prompt_payload(
    prompt: str,
    *,
    negative_prompt: str | None = None,
    image_path: str | None = None,
    image_paths: list[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"prompt": prompt}
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    resolved_image_paths = image_paths or ([image_path] if image_path else [])
    if resolved_image_paths:
        images = [Image.open(path).convert("RGB") for path in resolved_image_paths]
        payload["multi_modal_data"] = {"image": images[0] if len(images) == 1 else images}
    return payload


def parse_prompt_text(text: str) -> tuple[str, str | None]:
    prompt_lines: list[str] = []
    negative_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("negative:"):
            negative_lines.append(line.split(":", 1)[1].strip().strip("\"'"))
            continue
        prompt_lines.append(line.strip().strip("\"'"))

    prompt = " ".join(prompt_lines).strip()
    if not prompt:
        raise ValueError("Prompt text is empty after stripping blank lines and Negative: entries.")

    negative_prompt = " ".join(part for part in negative_lines if part).strip() or None
    return prompt, negative_prompt


def load_prompt_file(path: str | Path) -> tuple[str, str | None]:
    return parse_prompt_text(Path(path).read_text(encoding="utf-8"))


def extract_stage_metrics(output: Any) -> tuple[dict[str, float], float | None]:
    stage_durations = dict(getattr(output, "stage_durations", {}) or {})
    peak_memory_mb = getattr(output, "peak_memory_mb", 0.0) or 0.0
    if peak_memory_mb <= 0 and torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
    peak_memory_gib = peak_memory_mb / 1024 if peak_memory_mb > 0 else None
    return stage_durations, peak_memory_gib


def _unwrap_payload(obj: Any) -> Any:
    seen: set[int] = set()
    current = obj
    for _ in range(12):
        obj_id = id(current)
        if obj_id in seen:
            break
        seen.add(obj_id)

        if hasattr(current, "data"):
            current = current.data
            continue
        if hasattr(current, "video"):
            current = current.video
            continue
        if hasattr(current, "multimodal_output") and isinstance(current.multimodal_output, dict):
            multimodal_output = current.multimodal_output
            for key in ("video", "frames", "image", "images", "latents"):
                if key in multimodal_output and multimodal_output[key] is not None:
                    current = multimodal_output[key]
                    break
            else:
                if hasattr(current, "images") and current.images:
                    current = current.images
                    continue
                if hasattr(current, "outputs") and current.outputs:
                    current = current.outputs[0]
                    continue
                if hasattr(current, "request_output") and current.request_output is not None:
                    current = current.request_output
                    continue
                break
            continue
        if hasattr(current, "request_output") and current.request_output is not None:
            current = current.request_output
            continue
        if hasattr(current, "images") and current.images:
            current = current.images[0]
            continue
        if hasattr(current, "outputs") and current.outputs:
            current = current.outputs[0]
            continue
        if isinstance(current, dict):
            for key in ("video", "frames", "data", "tensor", "images"):
                if key in current and current[key] is not None:
                    current = current[key]
                    break
            else:
                break
            continue
        if isinstance(current, (list, tuple)) and len(current) == 1:
            current = current[0]
            continue
        break
    return current


def _to_numpy(payload: Any) -> np.ndarray:
    if isinstance(payload, np.ndarray):
        return payload
    if hasattr(payload, "detach") and hasattr(payload, "cpu"):
        return payload.detach().cpu().float().numpy()
    if isinstance(payload, Image.Image):
        return np.asarray(payload)
    if isinstance(payload, (list, tuple)):
        if payload and isinstance(payload[0], Image.Image):
            return np.stack([np.asarray(frame) for frame in payload], axis=0)
        return np.asarray(payload)
    array = np.asarray(payload)
    if array.dtype == object and array.ndim == 0:
        inner = array.item()
        if inner is payload:
            raise TypeError(f"Unsupported payload type: {type(payload)!r}")
        return _to_numpy(inner)
    return array


def extract_video_array(output: Any) -> np.ndarray:
    payload = _unwrap_payload(output)
    array = _to_numpy(payload)
    if array.ndim == 5 and array.shape[0] == 1:
        array = np.squeeze(array, axis=0)
    if array.ndim != 4:
        raise ValueError(f"Expected a 4D video tensor, but received shape={array.shape!r}.")
    return array


def video_payload_to_uint8(payload: Any) -> np.ndarray:
    video = extract_video_array(payload)
    if not np.isfinite(video).all():
        video = np.nan_to_num(video, nan=0.0, posinf=1.0, neginf=0.0)

    if video.shape[-1] in (1, 3, 4):
        video_fhwc = video
    elif video.shape[0] in (1, 3, 4):
        video_fhwc = np.transpose(video, (1, 2, 3, 0))
    elif video.shape[1] in (1, 3, 4):
        video_fhwc = np.transpose(video, (0, 2, 3, 1))
    else:
        raise ValueError(f"Unable to infer video channel layout from shape={video.shape!r}.")

    if video_fhwc.shape[-1] == 1:
        video_fhwc = np.repeat(video_fhwc, 3, axis=-1)
    elif video_fhwc.shape[-1] == 4:
        video_fhwc = video_fhwc[..., :3]

    if np.issubdtype(video_fhwc.dtype, np.integer):
        return np.clip(video_fhwc, 0, 255).astype(np.uint8)

    if video_fhwc.max() <= 1.0 and video_fhwc.min() >= 0.0:
        return np.clip(video_fhwc * 255.0, 0, 255).astype(np.uint8)

    v_min = float(video_fhwc.min())
    v_max = float(video_fhwc.max())
    if math.isclose(v_min, v_max):
        return np.zeros_like(video_fhwc, dtype=np.uint8)
    normalized = (video_fhwc - v_min) / (v_max - v_min)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def save_video(
    video_uint8: np.ndarray,
    output_path: str | Path,
    *,
    fps: int,
    codec: str | None = "libx264",
    bitrate: str | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, Any] = {"fps": fps}
    if codec:
        save_kwargs["codec"] = codec
    if bitrate:
        save_kwargs["bitrate"] = bitrate
    imageio.mimsave(path, video_uint8, **save_kwargs)
    return path


def infer_aspect_ratio(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must both be positive integers.")
    gcd = math.gcd(width, height)
    return f"{width // gcd}:{height // gcd}"


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


def render_benchmark_markdown(results: Iterable[BenchmarkResult]) -> str:
    rows = list(results)
    lines = [
        "# OmniWeaving Benchmark Summary",
        "",
        "| Case | Status | Latency (s) | Peak VRAM (GiB) | Output | Notes |",
        "|------|--------|-------------|-----------------|--------|-------|",
    ]
    for row in rows:
        latency = f"{row.latency_s:.2f}" if row.latency_s is not None else "N/A"
        peak_memory = f"{row.peak_memory_gib:.2f}" if row.peak_memory_gib is not None else "N/A"
        output = row.output_path or "-"
        notes = row.notes or row.error or "-"
        lines.append(f"| {row.name} | {row.status} | {latency} | {peak_memory} | {output} | {notes} |")
    return "\n".join(lines) + "\n"


def load_video_frames(video_path: str | Path) -> np.ndarray:
    reader = imageio.get_reader(video_path)
    frames = [np.asarray(frame)[..., :3] for frame in reader]
    if not frames:
        raise ValueError(f"No frames found in video: {video_path}")
    return np.stack(frames, axis=0)


def compute_video_similarity(
    candidate_path: str | Path,
    reference_path: str | Path,
) -> dict[str, float | int]:
    try:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    except ImportError as exc:
        raise ImportError(
            "scikit-image is required for accuracy comparison. Install it with `pip install scikit-image`."
        ) from exc

    candidate = load_video_frames(candidate_path)
    reference = load_video_frames(reference_path)
    if candidate.shape != reference.shape:
        raise ValueError(
            f"Video shapes do not match for comparison: candidate={candidate.shape}, reference={reference.shape}"
        )

    psnr_scores: list[float] = []
    ssim_scores: list[float] = []
    for cand_frame, ref_frame in zip(candidate, reference, strict=True):
        psnr_scores.append(float(peak_signal_noise_ratio(ref_frame, cand_frame, data_range=255)))
        ssim_scores.append(float(structural_similarity(ref_frame, cand_frame, channel_axis=2, data_range=255)))

    return {
        "frames": int(candidate.shape[0]),
        "psnr_mean_db": float(np.mean(psnr_scores)),
        "psnr_min_db": float(np.min(psnr_scores)),
        "ssim_mean": float(np.mean(ssim_scores)),
        "ssim_min": float(np.min(ssim_scores)),
    }
