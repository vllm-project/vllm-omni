# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Video Depth Anything: temporally consistent depth for every frame of a video.

A request carries a video in ``multi_modal_data["video"]`` (a file path, a
``[T, H, W, 3]`` uint8 array or PIL frames) and optionally ``input_size`` and
``max_res`` in ``extra_args``. The pipeline returns
``DiffusionOutput(output={"video_depth": ...})`` with one FP16 depth map per
frame at the (``max_res``-bounded) video resolution: relative inverse depth
for the relative models, depth in meters for the metric ones.

Long videos follow the reference ``infer_video_depth``: 32-frame windows that
overlap by 10 frames, the overlap carrying keyframes of the previous window,
and for relative models a least-squares scale and shift that aligns each
window to the previous one before an 8-frame cross-fade.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, ClassVar

import av
import cv2
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from torch import nn
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.video_depth_anything.checkpoints import checkpoint_variant, resolve_checkpoint
from vllm_omni.diffusion.models.video_depth_anything.vda_model import (
    PATCH_SIZE,
    VideoDepthAnything,
    VideoDepthAnythingConfig,
)
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

# Fixed by the released models.
INFER_LEN = 32
OVERLAP = 10
KEYFRAMES = (0, 12, 24, 25, 26, 27, 28, 29, 30, 31)
INTERP_LEN = 8
# FP64 like the reference's Python-list normalization; the result is cast to FP32.
IMAGE_MEAN = np.array((0.485, 0.456, 0.406))
IMAGE_STD = np.array((0.229, 0.224, 0.225))

DEFAULT_INPUT_SIZE = 518
DEFAULT_MAX_RES = 1280
MAX_INPUT_SIZE = 1022  # 73 patches; larger sizes leave the trained position-embedding range far behind.
MAX_VIDEO_FRAMES = 2048
# Depth maps cross the worker boundary in FP16: half the bytes, about three significant digits.
DEPTH_DTYPE = np.float16


def _too_many_frames() -> ValueError:
    return ValueError(f"Video Depth Anything accepts at most {MAX_VIDEO_FRAMES} frames")


def _int_option(extra: dict[str, Any], key: str, default: int) -> int:
    value = extra.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Video Depth Anything {key} must be an integer, got {value!r}")
    return value


def parse_options(extra: dict[str, Any]) -> tuple[int, int]:
    """``(input_size, max_res)``; ``max_res`` -1 keeps the source resolution."""
    input_size = _int_option(extra, "input_size", DEFAULT_INPUT_SIZE)
    if input_size % PATCH_SIZE or not PATCH_SIZE <= input_size <= MAX_INPUT_SIZE:
        raise ValueError(
            f"Video Depth Anything input_size must be a multiple of {PATCH_SIZE} in "
            f"[{PATCH_SIZE}, {MAX_INPUT_SIZE}], got {input_size}"
        )
    max_res = _int_option(extra, "max_res", DEFAULT_MAX_RES)
    if max_res != -1 and max_res < 2:
        raise ValueError(f"Video Depth Anything max_res must be -1 or at least 2, got {max_res}")
    return input_size, max_res


def _video_frames(video: Any) -> tuple[Iterator[np.ndarray], float | None]:
    """RGB uint8 [H, W, 3] frames from a file path or decoded frames, plus the frame rate."""
    if isinstance(video, np.ndarray) and video.ndim == 4:
        if len(video) > MAX_VIDEO_FRAMES:
            raise _too_many_frames()
        return iter(np.ascontiguousarray(frame) for frame in video), None
    if isinstance(video, (list, tuple)):
        if len(video) > MAX_VIDEO_FRAMES:
            raise _too_many_frames()
        return iter(
            np.asarray(frame.convert("RGB")) if isinstance(frame, PIL.Image.Image) else np.ascontiguousarray(frame)
            for frame in video
        ), None
    if not isinstance(video, str):
        raise ValueError(f"Video Depth Anything cannot read a video of type {type(video).__name__}")
    container = av.open(video)
    try:
        if not container.streams.video:
            raise ValueError("Video Depth Anything input has no video stream")
        stream = container.streams.video[0]
        # MP4 reports the frame count; Matroska/WebM report 0 and are capped while decoding.
        if stream.frames > MAX_VIDEO_FRAMES:
            raise _too_many_frames()
    except BaseException:
        container.close()
        raise
    fps = float(stream.average_rate) if stream.average_rate else None

    def decode() -> Iterator[np.ndarray]:
        with container:
            for index, frame in enumerate(container.decode(stream)):
                if index >= MAX_VIDEO_FRAMES:
                    raise _too_many_frames()
                yield frame.to_ndarray(format="rgb24")

    return decode(), fps


def _bounded(frames: Iterator[np.ndarray], max_res: int) -> Iterator[np.ndarray]:
    """Scale frames whose long edge exceeds ``max_res``, as the reference reader does."""
    for frame in frames:
        height, width = frame.shape[:2]
        if max_res > 0 and max(height, width) > max_res:
            scale = max_res / max(height, width)
            frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
        yield frame


def network_size(height: int, width: int, input_size: int) -> tuple[int, int]:
    """The reference ``Resize``: short edge at least ``input_size``, aspect kept, both multiples of 14."""
    ratio = max(height, width) / min(height, width)
    if ratio > 1.78:  # The reference shrinks wide inputs to bound memory.
        input_size = round(int(input_size * 1.777 / ratio) / PATCH_SIZE) * PATCH_SIZE
    scale = max(input_size / height, input_size / width)

    def multiple(value: float) -> int:
        size = int(np.round(value / PATCH_SIZE) * PATCH_SIZE)
        return size if size >= input_size else int(np.ceil(value / PATCH_SIZE) * PATCH_SIZE)

    return multiple(scale * height), multiple(scale * width)


def preprocess(frame: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    """Normalized float32 [3, h, w], resized with OpenCV's bicubic filter like the reference."""
    image = cv2.resize(frame.astype(np.float32) / 255.0, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    return torch.from_numpy(np.ascontiguousarray(((image - IMAGE_MEAN) / IMAGE_STD).transpose(2, 0, 1), np.float32))


def scale_and_shift(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Least-squares ``s, t`` with ``s * prediction + t ~ target``; computed like the reference, in FP32 NumPy."""
    prediction, target = prediction.astype(np.float32), target.astype(np.float32)
    a_00, a_01, a_11 = np.sum(prediction * prediction), np.sum(prediction), np.float32(prediction.size)
    b_0, b_1 = np.sum(prediction * target), np.sum(target)
    det = a_00 * a_11 - a_01 * a_01
    if det == 0:
        return 1.0, 0.0
    return (a_11 * b_0 - a_01 * b_1) / det, (-a_01 * b_0 + a_00 * b_1) / det


class VideoDepthAnythingPipeline(nn.Module):
    """Serves one released checkpoint; the encoder and depth type come from its file name."""

    # Startup warmup sends a text prompt, and this pipeline only accepts videos.
    dummy_run_num_frames: ClassVar[int] = 0

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        filename = resolve_checkpoint(od_config.model)
        if filename is None:
            raise ValueError(f"{od_config.model!r} is not a Video Depth Anything checkpoint")
        encoder, metric = checkpoint_variant(filename)
        self.config = VideoDepthAnythingConfig(encoder, metric)
        # The weights stay FP32, as in the reference; see ``_autocast_dtype`` for the compute precision.
        self.model = VideoDepthAnything(self.config, num_frames=INFER_LEN).float()
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder=None,
                revision=od_config.revision,
                prefix="model.",
                allow_patterns_overrides=[filename],
            )
        ]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        targets = dict(self.named_parameters()) | dict(self.named_buffers())
        loaded = set()
        for name, tensor in weights:
            if name not in targets:
                raise ValueError(f"Unexpected Video Depth Anything weight {name!r}")
            default_weight_loader(targets[name], tensor)
            loaded.add(name)
        return loaded

    @property
    def _device(self) -> torch.device:
        return self.model.head.scratch.output_conv1.weight.device

    @property
    def _autocast_dtype(self) -> torch.dtype | None:
        """FP16 autocast, the reference default, unless the server runs FP32.

        BF16 (the server default) drifts about 10x further from the FP32 output
        than FP16 does: 3e-3 relative for relative depth and 3e-2 for metric
        depth, against 3e-4 and 3e-3. So any reduced precision runs as FP16.
        """
        return None if getattr(self.od_config, "dtype", torch.float32) == torch.float32 else torch.float16

    def _window_depth(self, window: torch.Tensor, height: int, width: int) -> np.ndarray:
        """[1, 32, 3, h, w] -> 32 FP32 depth maps at the frame resolution."""
        dtype = self._autocast_dtype
        with torch.autocast(device_type=self._device.type, dtype=dtype or torch.float16, enabled=dtype is not None):
            depth = self.model(window)
        depth = F.interpolate(
            depth.float().flatten(0, 1).unsqueeze(1), size=(height, width), mode="bilinear", align_corners=True
        )
        return depth[:, 0].cpu().numpy()

    @torch.inference_mode()
    def estimate(self, video: Any, input_size: int = DEFAULT_INPUT_SIZE, max_res: int = DEFAULT_MAX_RES) -> dict:
        frames, fps = _video_frames(video)
        frames = _bounded(frames, max_res)
        first = next(frames, None)
        if first is None:
            raise ValueError("Video Depth Anything input has no frames")
        height, width = first.shape[:2]
        size = network_size(height, width, input_size)
        step = INFER_LEN - OVERLAP

        def stream() -> Iterator[np.ndarray]:
            yield first
            yield from frames

        source = stream()
        windows: list[np.ndarray] = []  # Per window, 32 raw depth maps.
        num_frames = 0
        last = first
        previous: torch.Tensor | None = None
        # Window k covers frames [22k, 22k + 32) for every 22k below the clip length. Its first 10
        # inputs are keyframes of window k - 1, so only frames [22k + 10, 22k + 32) are new, and the
        # clip is padded with its last frame to whole windows, as in the reference.
        while previous is None or num_frames > len(windows) * step:
            needed = INFER_LEN if previous is None else step
            batch = []
            for frame in source:
                batch.append(frame)
                last = frame
                if len(batch) == needed:
                    break
            num_frames += len(batch)
            batch += [last] * (needed - len(batch))
            inputs = torch.stack([preprocess(frame, size) for frame in batch]).to(self._device).unsqueeze(0)
            if previous is not None:
                inputs = torch.cat([previous[:, list(KEYFRAMES)], inputs], dim=1)
            windows.append(self._window_depth(inputs, height, width))
            previous = inputs
        return {
            "depth": self._align(windows, num_frames).astype(DEPTH_DTYPE),
            "fps": fps,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "metric": self.config.metric,
        }

    def _align(self, windows: list[np.ndarray], num_frames: int) -> np.ndarray:
        """Stitch windows: align each to the previous one, cross-fade the overlap (reference ``infer_video_depth``)."""
        align_len = OVERLAP - INTERP_LEN
        align_keyframes = KEYFRAMES[:align_len]
        aligned = list(windows[0])
        reference = [windows[0][index] for index in align_keyframes]
        for window in windows[1:]:
            if self.config.metric:
                scale, shift = 1.0, 0.0
            else:
                scale, shift = scale_and_shift(np.concatenate(window[:align_len]), np.concatenate(reference))

            def adjusted(depth: np.ndarray) -> np.ndarray:
                depth = depth * scale + shift
                depth[depth < 0] = 0
                return depth

            # The last INTERP_LEN aligned frames fade into this window's view of the same frames.
            # Python-float weights keep the maps FP32, as in the reference.
            step = 1.0 / (INTERP_LEN - 1)
            weights = [0.0] + [index * step for index in range(1, INTERP_LEN - 1)] + [1.0]
            post = [adjusted(window[index]) for index in range(align_len, OVERLAP)]
            for offset, (pre, new) in enumerate(zip(aligned[-INTERP_LEN:], post)):
                aligned[len(aligned) - INTERP_LEN + offset] = pre * (1 - weights[offset]) + new * weights[offset]
            aligned.extend(adjusted(window[index]) for index in range(OVERLAP, INFER_LEN))
            reference = reference[:1] + [adjusted(window[index]) for index in align_keyframes[1:]]
        return np.stack(aligned[:num_frames])

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch, **kwargs) -> DiffusionOutput:
        if len(req.prompts) != 1:
            return DiffusionOutput(
                error="Video Depth Anything takes exactly one video per request.", error_status_code=400
            )
        prompt = req.prompts[0]
        data = {} if isinstance(prompt, str) else (prompt.get("multi_modal_data") or {})
        if data.get("video") is None:
            return DiffusionOutput(error="Video Depth Anything requires an input video.", error_status_code=400)
        try:
            input_size, max_res = parse_options(dict(getattr(req.sampling_params, "extra_args", None) or {}))
            result = self.estimate(data["video"], input_size, max_res)
        except (TypeError, ValueError, av.error.FFmpegError) as error:
            return DiffusionOutput(error=str(error), error_status_code=400)
        return DiffusionOutput(output={"video_depth": result})
