# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Closed-loop evaluation for LIBERO using the Action HTTP inference server.

# Single-view example (agentview camera):
python -m vllm_omni_cosmos3_client \
  --server_url http://localhost:8000 \
  --task_suite libero_10 \
  --num_trials_per_task 10 \
  --action_horizon 16 \
  --camera agentview \
  --save_gifs --gif_fps 20 \
  --action_space frame_wise_relative \
  --rotation_space 6d \
  --action_dim 10 \
  --output_dir results/libero_closed_loop_10_single_view

# Multi-view example (agentview + wrist cameras):
python -m vllm_omni_cosmos3_client \
  --server_url http://localhost:8000 \
  --task_suite libero_goal \
  --num_trials_per_task 2 \
  --action_horizon 16 \
  --camera agentview,wrist \
  --save_gifs --gif_fps 20 \
  --action_space frame_wise_relative \
  --rotation_space 6d \
  --action_dim 10 \
  --output_dir results/libero_closed_loop_goal_multiview
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import queue
import random
import sys
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

# Load LLVM before SciPy / Torch native libraries. In the native Ascend client,
# the inverse load order crashes Numba's level-3 optimizer on mat2quat.
import llvmlite.binding  # noqa: F401
import numpy as np
import requests
from PIL import Image
from scipy.spatial.transform import Rotation as SciRotation

from .eval_metrics import (
    build_report,
    latency_stats,
    make_request_record,
    report_text,
    summarize_request_timings,
    write_report,
)
from .timeline import TIMELINE

# Match the LIBERO conventions used by Cosmos Framework without importing its
# training/data package into this standalone benchmark client.
CONCAT_VIEW_PROMPT = "This video contains concatenated views from multiple camera perspectives."
LIBERO_ACTION_DIMS = {"3d": 7, "6d": 10, "9d": 13}


def libero_rotation_space_from_action_dim(action_dim: int) -> str:
    for rotation_space, width in LIBERO_ACTION_DIMS.items():
        if width == action_dim:
            return rotation_space
    raise ValueError(f"Unable to infer rotation_space from action_dim={action_dim}.")


benchmark: Any
get_libero_path: Any
OffScreenRenderEnv: Any


TASK_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


_CAMERA_PROMPT_NAMES: dict[str, str] = {
    "agentview": "third-person view",
    "wrist": "wrist-mounted camera",
}


def _append_prompt_sentence(prompt: str, sentence: str) -> str:
    """Append one metadata sentence using the same separator convention as training augmentors."""
    if sentence in prompt:
        return prompt
    prompt = prompt.rstrip()
    if not prompt:
        return sentence.rstrip()
    separator = " " if prompt.rstrip().endswith(".") else ". "
    return prompt + separator + sentence.rstrip()


def _concat_view_layout_description(cameras: list[str]) -> str:
    """Describe the horizontal camera layout sent by ``ActionEnvironmentClient``."""
    camera_names = [_CAMERA_PROMPT_NAMES[camera] for camera in cameras]
    if len(camera_names) == 2:
        return f"The left half shows the {camera_names[0]}; the right half shows the {camera_names[1]}."
    layout = ", ".join(camera_names)
    return f"The views are concatenated horizontally from left to right as: {layout}."


def _augment_task_prompt_with_viewpoint(task_description: str, cameras: list[str]) -> str:
    """Concat-view caption augmentation for closed-loop LIBERO eval."""
    if len(cameras) <= 1:
        return task_description
    prompt = _append_prompt_sentence(task_description, CONCAT_VIEW_PROMPT)
    return _append_prompt_sentence(prompt, _concat_view_layout_description(cameras))


def _rotation_repr_to_mat(rotation: np.ndarray, rotation_space: str) -> np.ndarray:
    """Convert a single LIBERO rotation block to a 3x3 rotation matrix."""
    values = np.asarray(rotation, dtype=np.float32)
    if rotation_space == "3d" and values.shape == (3,):
        return SciRotation.from_rotvec(values).as_matrix().astype(np.float32)
    if rotation_space == "6d" and values.shape == (6,):
        col0, col1 = values[:3], values[3:]
        candidate = np.stack((col0, col1, np.cross(col0, col1)), axis=-1)
    elif rotation_space == "9d" and values.shape == (9,):
        candidate = values.reshape(3, 3)
    else:
        raise ValueError(f"Invalid {rotation_space!r} rotation shape: {values.shape}")
    u, _, vt = np.linalg.svd(candidate)
    matrix = u @ vt
    if np.linalg.det(matrix) < 0:
        u[:, -1] *= -1
        matrix = u @ vt
    return matrix.astype(np.float32)


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    error: str | None
    actions: list[list[float]]


class ActionEnvironmentClient:
    """Client for interacting with the Action model server."""

    server_url: str
    domain_name: str
    prompt: str
    image_size: int
    timeout: float
    action_chunk_size: int | None

    def __init__(
        self,
        server_url: str,
        domain_name: str,
        prompt: str,
        image_size: int,
        timeout: float,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.domain_name = domain_name
        self.prompt = prompt
        self.image_size = image_size
        self.timeout = timeout
        self.action_chunk_size = None
        self.request_timings: list[dict[str, Any]] = []
        self._timing_lock = threading.Lock()
        self._request_sequence = 0
        self._timing_epoch = time.perf_counter()
        self.timing_warmup_requests = 2
        self.execution_window = 0  # 0 consumes the full valid returned chunk.
        self.timing_path: Path | None = None
        self.task_id: int | None = None
        self.episode: int | None = None
        self.run_id = uuid.uuid4().hex
        self.trace_id: str | None = None
        self._window_sequence = 0

    def begin_window(self) -> None:
        self._window_sequence += 1
        self.trace_id = f"{self.run_id}:{self._window_sequence}"

    def span(self, stage: str, **metadata):
        return TIMELINE.span(stage, trace_id=self.trace_id, task_id=self.task_id, episode=self.episode, **metadata)

    def _record_request_timing(
        self,
        *,
        endpoint: str,
        batch_size: int,
        http_total_ms: float,
        result: dict[str, Any],
        request_started_at: float,
        client_preprocess_ms: float,
        client_ttfa_ms: float | None = None,
        partial_events: int = 0,
        action_timings: list[dict[str, Any]] | None = None,
    ) -> None:
        with self._timing_lock:
            self._request_sequence += 1
            row = make_request_record(
                request=self._request_sequence,
                endpoint=endpoint,
                batch_size=batch_size,
                http_total_ms=http_total_ms,
                result=result,
                started_at_s=request_started_at - self._timing_epoch,
                client_preprocess_ms=client_preprocess_ms,
                client_ttfa_ms=client_ttfa_ms,
                partial_events=partial_events,
                action_timings=action_timings or [],
                warmup=self._request_sequence <= self.timing_warmup_requests,
                execution_window=self.execution_window,
            )
            row.update(task_id=self.task_id, episode=self.episode)
            row.update(
                trace_id=self.trace_id,
                server_request_id=(result.get("performance") or {}).get("request_id"),
                absolute_start_s=request_started_at,
                absolute_end_s=request_started_at + http_total_ms / 1000,
                clock_domain=TIMELINE.clock_domain,
            )
            TIMELINE.interval(
                "client.request",
                request_started_at,
                request_started_at + http_total_ms / 1000,
                trace_id=self.trace_id,
                task_id=self.task_id,
                episode=self.episode,
                warmup=row.get("warmup", False),
                endpoint=endpoint,
            )
            TIMELINE.interval(
                "client.preprocess",
                request_started_at - client_preprocess_ms / 1000,
                request_started_at,
                trace_id=self.trace_id,
                task_id=self.task_id,
                episode=self.episode,
            )
            server_timeline = (result.get("performance") or {}).get("timeline", {})
            row["server_timeline"] = server_timeline
            self.request_timings.append(row)
            if self.timing_path is not None:
                with self.timing_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")

    def check_health(self) -> bool:
        """Check if the model server is healthy."""
        try:
            resp = requests.get(f"{self.server_url}/", timeout=5.0)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def get_info(self) -> dict[str, Any]:
        """Get model server info."""
        resp = requests.get(f"{self.server_url}/info", timeout=5.0)
        resp.raise_for_status()
        return resp.json()

    def synthetic_warmup(self, *, requests_count: int, streaming: bool, batch_size: int) -> dict[str, Any]:
        response = requests.post(
            f"{self.server_url}/warmup",
            json={"requests": requests_count, "streaming": streaming, "batch_size": batch_size},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def notify_next_episode(self) -> None:
        """Notify server to advance to next episode (used with dataset action server)."""
        try:
            requests.post(
                f"{self.server_url}/next_episode",
                json={"prompt": self.prompt},
                timeout=5.0,
            )
        except requests.RequestException:
            pass

    def encode_image(self, image: np.ndarray) -> str:
        """Encode a numpy image (H, W, 3) uint8 to base64 PNG, resizing to image_size."""
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255.0).round().astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        pil_img = Image.fromarray(image)
        if pil_img.size != (self.image_size, self.image_size):
            pil_img = pil_img.resize(
                (self.image_size, self.image_size),
                resample=Image.Resampling.BILINEAR,
            )
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def encode_image_raw(self, image: np.ndarray) -> str:
        """Encode a numpy image (H, W, 3) uint8 to base64 PNG without resizing."""
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255.0).round().astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        pil_img = Image.fromarray(image)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to model input size."""
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255.0).round().astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        pil_img = Image.fromarray(image)
        if pil_img.size != (self.image_size, self.image_size):
            pil_img = pil_img.resize(
                (self.image_size, self.image_size),
                resample=Image.Resampling.BILINEAR,
            )
        return np.array(pil_img)

    def concatenate_images(self, images: list[np.ndarray]) -> np.ndarray:
        """Resize each image and concatenate horizontally (side-by-side).

        Args:
            images: List of images with shape (H, W, 3).

        Returns:
            Concatenated image with shape (image_size, image_size*num_views, 3).
        """
        resized = [self.resize_image(img) for img in images]
        return np.concatenate(resized, axis=1)

    def predict(self, observation: np.ndarray | list[np.ndarray]) -> dict[str, Any]:
        """Send observation(s) to model server and get predicted actions.

        Args:
            observation: Single image as np.ndarray or list of images for multi-view.
                For multi-view, images are resized and concatenated horizontally before sending.
        """
        preprocess_start = time.perf_counter()
        if isinstance(observation, list):
            # Multi-view: resize each, concatenate horizontally, and send as single image
            concatenated = self.concatenate_images(observation)
            encoded = self.encode_image_raw(concatenated)
        else:
            # Single view: send single image
            encoded = self.encode_image(observation)

        payload = {
            "trace_id": self.trace_id,
            "image": encoded,
            "prompt": self.prompt,
            "domain_name": self.domain_name,
            "image_size": self.image_size,
        }

        request_t0 = time.perf_counter()
        resp = requests.post(
            f"{self.server_url}/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        resp.raise_for_status()

        result = resp.json()
        if "error" in result and result["error"]:
            raise RuntimeError(f"Model server error: {result['error']}")
        self._record_request_timing(
            endpoint="/predict",
            batch_size=1,
            http_total_ms=(time.perf_counter() - request_t0) * 1000.0,
            result=result,
            request_started_at=request_t0,
            client_preprocess_ms=(request_t0 - preprocess_start) * 1000,
        )
        return result

    def predict_stream(
        self,
        observation: np.ndarray | list[np.ndarray],
        on_actions_ready: Callable[[list[int], list[list[float]]], None],
    ) -> dict[str, Any]:
        """Receive progressively denoised actions from ``/predict_stream``.

        The callback runs on the caller's thread once per NDJSON partial event.
        Call this method from a worker thread when actions should be executed
        concurrently with the server finishing the rest of the chunk.
        """
        preprocess_start = time.perf_counter()
        if isinstance(observation, list):
            encoded = self.encode_image_raw(self.concatenate_images(observation))
        else:
            encoded = self.encode_image(observation)
        payload = {
            "trace_id": self.trace_id,
            "image": encoded,
            "prompt": self.prompt,
            "domain_name": self.domain_name,
            "image_size": self.image_size,
        }
        request_t0 = time.perf_counter()
        first_partial_ms: float | None = None
        partial_events = 0
        action_timings = []
        with requests.post(
            f"{self.server_url}/predict_stream",
            json=payload,
            headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"},
            timeout=self.timeout,
            stream=True,
        ) as resp:
            resp.raise_for_status()
            final: dict[str, Any] | None = None
            # The server flushes every NDJSON event. Read complete lines from
            # http.client's buffered socket reader: iter_lines(chunk_size=1)
            # spends ~130 ms of Python CPU per final response and starves the
            # simulator thread despite the model actions already being ready.
            # readline() returns each flushed partial without waiting for final.
            lines = _stream_lines(resp)
            diagnose_cpu = os.environ.get("COSMOS3_DIAG_THREAD_CPU") == "1"
            if diagnose_cpu:
                def timed_lines():
                    while True:
                        try:
                            with TIMELINE.span("client.stream_line_read", trace_id=self.trace_id):
                                line = next(lines)
                        except StopIteration:
                            return
                        yield line

                incoming = timed_lines()
            else:
                incoming = lines
            for raw_line in incoming:
                if not raw_line:
                    continue
                with (TIMELINE.span("client.stream_line_decode", trace_id=self.trace_id)
                      if diagnose_cpu else nullcontext()):
                    event = json.loads(raw_line.decode("utf-8"))
                event_type = event.get("type")
                if event_type == "partial":
                    partial_events += 1
                    ready_ms = (time.perf_counter() - request_t0) * 1000
                    indices = event.get("action_indices", [])
                    if indices and first_partial_ms is None:
                        first_partial_ms = ready_ms
                    action_timings.extend(
                        {"sample_index": 0, "action_index": index, "ready_ms": ready_ms, "step": event.get("step")}
                        for index in indices
                    )
                    on_actions_ready(event.get("action_indices", []), event.get("actions", []))
                elif event_type == "error":
                    raise RuntimeError(f"Model server error: {event.get('error', 'unknown streaming error')}")
                elif event_type == "final":
                    final = event
            if final is None:
                raise RuntimeError("Model server closed the stream without a final event")
            final.pop("type", None)
            self._record_request_timing(
                endpoint="/predict_stream",
                batch_size=1,
                http_total_ms=(time.perf_counter() - request_t0) * 1000.0,
                client_ttfa_ms=first_partial_ms,
                partial_events=partial_events,
                result=final,
                request_started_at=request_t0,
                client_preprocess_ms=(request_t0 - preprocess_start) * 1000,
                action_timings=action_timings,
            )
            return final

    def predict_batch_stream(
        self,
        observations: list[list[np.ndarray]],
        on_actions_ready: Callable[[list[dict[str, Any]]], None],
    ) -> dict[str, Any]:
        """Receive a multiplexed stream for a batch of environments.

        Partial events contain ``items`` with ``sample_index`` so the caller
        can route each action to the corresponding environment queue.
        """
        preprocess_start = time.perf_counter()
        items = []
        for obs_imgs in observations:
            concat = self.concatenate_images(obs_imgs) if len(obs_imgs) > 1 else self.resize_image(obs_imgs[0])
            items.append(
                {
                    "image": self.encode_image_raw(concat),
                    "prompt": self.prompt,
                    "domain_name": self.domain_name,
                    "image_size": self.image_size,
                }
            )
        request_t0 = time.perf_counter()
        first_partial_ms: float | None = None
        partial_events = 0
        action_timings = []
        with requests.post(
            f"{self.server_url}/predict_batch_stream",
            json={"items": items, "trace_id": self.trace_id},
            headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"},
            timeout=max(self.timeout, 300.0),
            stream=True,
        ) as resp:
            resp.raise_for_status()
            final: dict[str, Any] | None = None
            for raw_line in _stream_lines(resp):
                if not raw_line:
                    continue
                event = json.loads(raw_line.decode("utf-8"))
                event_type = event.get("type")
                if event_type == "partial":
                    partial_events += 1
                    ready_ms = (time.perf_counter() - request_t0) * 1000
                    for item in event.get("items", []):
                        indices = item.get("action_indices", [])
                        if indices and first_partial_ms is None:
                            first_partial_ms = ready_ms
                        action_timings.extend(
                            {
                                "sample_index": item["sample_index"],
                                "action_index": index,
                                "ready_ms": ready_ms,
                                "step": event.get("step"),
                            }
                            for index in indices
                        )
                    on_actions_ready(event.get("items", []))
                elif event_type == "error":
                    raise RuntimeError(f"Model server error: {event.get('error', 'unknown streaming error')}")
                elif event_type == "final":
                    final = event
            if final is None:
                raise RuntimeError("Model server closed the batch stream without a final response")
            final.pop("type", None)
            self._record_request_timing(
                endpoint="/predict_batch_stream",
                batch_size=len(items),
                http_total_ms=(time.perf_counter() - request_t0) * 1000.0,
                client_ttfa_ms=first_partial_ms,
                partial_events=partial_events,
                result=final,
                request_started_at=request_t0,
                client_preprocess_ms=(request_t0 - preprocess_start) * 1000,
                action_timings=action_timings,
            )
            return final

    def predict_batch(self, observations: list[list[np.ndarray]]) -> list[list[list[float]]]:
        """Batched inference: a list of per-env multi-view observations -> ONE
        POST /predict_batch -> a list of action chunks (one per env). Used by the
        vectorized eval so N parallel envs share a single diffusion forward."""
        preprocess_start = time.perf_counter()
        items = []
        for obs_imgs in observations:
            concat = self.concatenate_images(obs_imgs) if len(obs_imgs) > 1 else self.resize_image(obs_imgs[0])
            items.append(
                {
                    "image": self.encode_image_raw(concat),
                    "prompt": self.prompt,
                    "domain_name": self.domain_name,
                    "image_size": self.image_size,
                }
            )
        request_t0 = time.perf_counter()
        resp = requests.post(
            f"{self.server_url}/predict_batch",
            json={"items": items, "trace_id": self.trace_id},
            headers={"Content-Type": "application/json"},
            timeout=max(self.timeout, 300.0),
        )
        resp.raise_for_status()
        result = resp.json()
        if "error" in result and result["error"]:
            raise RuntimeError(f"Model server error: {result['error']}")
        self._record_request_timing(
            endpoint="/predict_batch",
            batch_size=len(items),
            http_total_ms=(time.perf_counter() - request_t0) * 1000.0,
            result=result,
            request_started_at=request_t0,
            client_preprocess_ms=(request_t0 - preprocess_start) * 1000,
        )
        return result["actions"]




def _stream_lines(resp: requests.Response):
    """Read flushed NDJSON lines without a Python byte-at-a-time loop.

    urllib3 may buffer an entire response when iter_lines has a large chunk
    size; chunk_size=1 delivers partial actions promptly but burns CPU during
    the final diagnostic payload. The underlying HTTPResponse exposes the
    socket's buffered reader, whose readline returns on each flushed newline.
    """
    http_response = resp.raw._fp
    if http_response is None or http_response.fp is None:
        raise RuntimeError("Streaming HTTP response has no buffered socket reader")
    return iter(http_response.fp.readline, b"")


class _OrderedActionBuffer:
    """Reassemble indexed stream events into one strictly ordered action queue."""

    _END = object()

    def __init__(self, expected_count: int, label: str) -> None:
        self.expected_count = expected_count
        self.label = label
        self.queue: queue.Queue[list[float] | object] = queue.Queue()
        self._pending: dict[int, list[float]] = {}
        self._seen: set[int] = set()
        self._next_index = 0
        self._closed = False

    def add(self, index: Any, action: list[float]) -> None:
        if self._closed:
            raise RuntimeError(f"{self.label}: received action after stream closed")
        if isinstance(index, bool):
            raise RuntimeError(f"{self.label}: invalid boolean action index {index!r}")
        try:
            action_index = int(index)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{self.label}: invalid action index {index!r}") from exc
        if action_index < 0:
            raise RuntimeError(f"{self.label}: negative action index {action_index}")
        if self.expected_count > 0 and action_index >= self.expected_count:
            # The caller may intentionally execute only a prefix of the chunk.
            return
        if action_index in self._seen:
            raise RuntimeError(f"{self.label}: duplicate action index {action_index}")

        self._seen.add(action_index)
        self._pending[action_index] = action
        while self._next_index in self._pending:
            self.queue.put(self._pending.pop(self._next_index))
            self._next_index += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._pending:
            missing = self._next_index
            available = sorted(self._pending)
            raise RuntimeError(
                f"{self.label}: stream ended with missing action index {missing}; buffered later indices={available}"
            )
        if self.expected_count > 0 and self._next_index < self.expected_count:
            raise RuntimeError(
                f"{self.label}: stream ended after {self._next_index}/{self.expected_count} ordered actions"
            )

    @property
    def emitted_count(self) -> int:
        return self._next_index


class _StreamingActionRequest:
    """Run one streaming HTTP request while the environment consumes actions."""

    def __init__(
        self,
        client: ActionEnvironmentClient,
        observation: np.ndarray | list[np.ndarray],
        action_limit: int,
    ) -> None:
        self._client = client
        self._observation = observation
        self._action_limit = action_limit if action_limit > 0 else int(client.action_chunk_size or 0)
        self._buffer = _OrderedActionBuffer(self._action_limit, "stream sample 0")
        self._error: BaseException | None = None
        self._final: dict[str, Any] | None = None
        self._received = 0
        self._consumed = 0
        self._thread = threading.Thread(target=self._run, name="cosmos-action-stream", daemon=True)
        self._thread.start()

    def _on_actions_ready(self, indices: list[int], actions: list[list[float]]) -> None:
        if len(indices) != len(actions):
            raise RuntimeError("Streaming server returned mismatched action indices and actions")
        for index, action in zip(indices, actions, strict=True):
            before = self._buffer.emitted_count
            self._buffer.add(index, action)
            if self._buffer.emitted_count > before:
                self._received += self._buffer.emitted_count - before

    def _run(self) -> None:
        try:
            self._final = self._client.predict_stream(self._observation, self._on_actions_ready)
            self._buffer.close()
        except BaseException as exc:  # noqa: BLE001 - forwarded to environment thread
            self._error = exc
        finally:
            self._buffer.queue.put(self._buffer._END)

    def next_action(self) -> list[float] | None:
        item = self._buffer.queue.get(timeout=self._client.timeout)
        if item is self._buffer._END:
            if self._error is not None:
                raise RuntimeError(f"Streaming inference failed: {self._error}") from self._error
            return None
        self._consumed += 1
        return item  # type: ignore[return-value]

    @property
    def reached_action_limit(self) -> bool:
        return self._action_limit > 0 and self._consumed >= self._action_limit

    @property
    def received_actions(self) -> int:
        return self._received

    def finish(self) -> dict[str, Any]:
        self._thread.join(timeout=self._client.timeout)
        if self._thread.is_alive():
            raise TimeoutError("Timed out waiting for the streaming inference request to finish")
        if self._error is not None:
            raise RuntimeError(f"Streaming inference failed: {self._error}") from self._error
        if self._final is None:
            raise RuntimeError("Streaming inference produced no final response")
        return self._final


class _BatchStreamingActionRequest:
    """Multiplex one /predict_batch_stream request into per-env queues."""

    def __init__(
        self,
        client: ActionEnvironmentClient,
        observations: list[list[np.ndarray]],
        action_limit: int,
    ) -> None:
        self._client = client
        self._observations = observations
        self._action_limit = action_limit if action_limit > 0 else int(client.action_chunk_size or 0)
        self._buffers = [
            _OrderedActionBuffer(self._action_limit, f"batch sample {sample_index}")
            for sample_index in range(len(observations))
        ]
        self._error: BaseException | None = None
        self._final: dict[str, Any] | None = None
        self._thread = threading.Thread(target=self._run, name="cosmos-action-batch-stream", daemon=True)
        self._thread.start()

    def _on_actions_ready(self, items: list[dict[str, Any]]) -> None:
        for item in items:
            sample_index = int(item["sample_index"])
            if not 0 <= sample_index < len(self._buffers):
                raise RuntimeError(f"Invalid streaming sample_index={sample_index}")
            indices = item.get("action_indices", [])
            actions = item.get("actions", [])
            if len(indices) != len(actions):
                raise RuntimeError("Streaming server returned mismatched action indices and actions")
            for index, action in zip(indices, actions, strict=True):
                self._buffers[sample_index].add(index, action)

    def _run(self) -> None:
        try:
            self._final = self._client.predict_batch_stream(self._observations, self._on_actions_ready)
            for action_buffer in self._buffers:
                action_buffer.close()
        except BaseException as exc:  # noqa: BLE001
            self._error = exc
        finally:
            for action_buffer in self._buffers:
                action_buffer.queue.put(action_buffer._END)

    def next_action(self, sample_index: int) -> list[float] | None:
        item = self._buffers[sample_index].queue.get(timeout=self._client.timeout)
        if item is self._buffers[sample_index]._END:
            if self._error is not None:
                raise RuntimeError(f"Batch streaming inference failed: {self._error}") from self._error
            return None
        return item  # type: ignore[return-value]

    def next_ready_actions(self, sample_indices: list[int]) -> dict[int, list[float]]:
        """Consume ready samples without waiting for another environment's job.

        Worker jobs can run sequentially. A lockstep read of every sample's
        first action would prevent earlier samples from using their stream.
        """
        deadline = time.monotonic() + self._client.timeout
        while True:
            ready = {}
            for index in sample_indices:
                buffer = self._buffers[index]
                try:
                    item = buffer.queue.get_nowait()
                except queue.Empty:
                    continue
                if item is buffer._END:
                    if self._error is not None:
                        raise RuntimeError(f"Batch streaming inference failed: {self._error}") from self._error
                    raise RuntimeError(f"Batch stream ended before required action for sample_index={index}")
                ready[index] = item
            if ready:
                return ready
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for a ready streamed action")
            time.sleep(0.001)

    def finish(self) -> dict[str, Any]:
        self._thread.join(timeout=max(self._client.timeout, 300.0))
        if self._thread.is_alive():
            raise TimeoutError("Timed out waiting for the batch streaming inference request to finish")
        if self._error is not None:
            raise RuntimeError(f"Batch streaming inference failed: {self._error}") from self._error
        if self._final is None:
            raise RuntimeError("Batch streaming inference produced no final response")
        return self._final


def _find_accessible_dri_nodes() -> list[Path]:
    dri_path = Path("/dev/dri")
    if not dri_path.exists():
        return []
    nodes = list(dri_path.glob("renderD*")) + list(dri_path.glob("card*"))
    return [node for node in nodes if os.access(node, os.R_OK | os.W_OK)]


def _resolve_mujoco_backend(requested_backend: str) -> tuple[str, str]:
    requested_backend = requested_backend.lower()
    if requested_backend != "auto":
        return requested_backend, "requested"

    env_backend = os.environ.get("MUJOCO_GL")
    if env_backend:
        return env_backend.lower(), "env"

    if _find_accessible_dri_nodes():
        return "egl", "auto-gpu"
    return "osmesa", "auto-cpu"


def _configure_mujoco_env(requested_backend: str) -> str:
    backend, source = _resolve_mujoco_backend(requested_backend)
    if backend not in {"egl", "osmesa", "glfw"}:
        raise ValueError(f"Unsupported MuJoCo GL backend: {backend!r}. Use auto, egl, osmesa, or glfw.")

    os.environ["MUJOCO_GL"] = backend
    if backend == "egl":
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(Path(__file__).with_name("egl_vendor.json")))
    elif backend == "osmesa":
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    return f"{backend} ({source})"


def _import_libero() -> None:
    global benchmark, get_libero_path, OffScreenRenderEnv
    try:
        from vllm_omni_cosmos3_client._vendor.libero import benchmark as libero_benchmark
        from vllm_omni_cosmos3_client._vendor.libero import get_libero_path as libero_get_libero_path
        from vllm_omni_cosmos3_client._vendor.libero.envs import OffScreenRenderEnv as LiberoOffScreenRenderEnv
    except ImportError as exc:  # pragma: no cover - environment-specific dependency
        raise RuntimeError(
            "Failed to import the bundled LIBERO runtime. Install plugins/cosmos3_client in the client environment. "
            f"python={sys.executable!r}, import_error={exc!r}"
        ) from exc

    benchmark = libero_benchmark
    get_libero_path = libero_get_libero_path
    OffScreenRenderEnv = LiberoOffScreenRenderEnv


def _wait_for_server(client: ActionEnvironmentClient, timeout_s: float) -> None:
    start = time.perf_counter()
    while time.perf_counter() - start < timeout_s:
        if client.check_health():
            return
        time.sleep(1.0)
    raise RuntimeError(f"Timed out waiting for server at {client.server_url}")


def _get_libero_env(
    task: Any,
    *,
    resolution: int,
    seed: int,
    render_gpu_device_id: int,
    cameras: list[str] | None = None,
) -> tuple[Any, str]:
    task_description = str(task.language)
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "render_gpu_device_id": render_gpu_device_id,
        "use_camera_obs": False,
    }
    from .rendering.observation import ObservationRenderEnv

    env = ObservationRenderEnv(OffScreenRenderEnv(**env_args), cameras or ["agentview", "wrist"], resolution)
    env.seed(seed)
    return env, task_description


def _get_libero_dummy_action() -> list[float]:
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def _get_libero_image(
    obs: dict[str, Any],
    camera: str,
    *,
    flip_images: bool,
    rotate_180: bool,
) -> np.ndarray:
    if camera == "agentview":
        image = obs["agentview_image"]
    elif camera == "wrist":
        image = obs["robot0_eye_in_hand_image"]
    else:
        raise ValueError(f"Unsupported camera={camera!r}. Use 'agentview' or 'wrist'.")

    if rotate_180:
        image = image[::-1, ::-1]
    if flip_images:
        image = np.flipud(image)
    return image


def _get_libero_images(
    obs: dict[str, Any],
    cameras: list[str],
    *,
    flip_images: bool,
    rotate_180: bool,
) -> list[np.ndarray]:
    """Get images from multiple cameras."""
    return [_get_libero_image(obs, camera, flip_images=flip_images, rotate_180=rotate_180) for camera in cameras]


def _ensure_uint8_image(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255.0).round().astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    return image


def _save_gif(frames: list[Image.Image], output_path: Path, fps: int) -> None:
    if not frames:
        return
    duration_ms = int(1000 / fps) if fps > 0 else 100
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first, *rest = frames
    first.save(
        output_path,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
    )


def _decode_b64_frames(b64_frames: list[str]) -> list[Image.Image]:
    """Decode a list of base64-encoded PNG strings into PIL Images."""
    images: list[Image.Image] = []
    for b64 in b64_frames:
        raw = base64.b64decode(b64)
        images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
    return images


def _save_comparison_gif(
    comparison_windows: list[tuple[list[Image.Image], list[Image.Image]]],
    output_path: Path,
    fps: int,
    target_height: int = 256,
    separator_width: int = 4,
) -> None:
    """Create and save a side-by-side comparison GIF (Action prediction | env rollout).

    Each window is a (action_frames, env_frames) pair from one prediction call.
    Frames are paired index-by-index; the conditioning frame (index 0) of
    subsequent windows is skipped to avoid duplicating the boundary frame.
    """
    from PIL import ImageDraw

    combined_frames: list[Image.Image] = []
    banner_h = 16

    for window_idx, (action_frames, env_frames) in enumerate(comparison_windows):
        n = min(len(action_frames), len(env_frames))
        start = 1 if window_idx > 0 else 0
        for i in range(start, n):
            action_img = action_frames[i]
            env_img = env_frames[i]

            action_w = int(action_img.width * target_height / action_img.height)
            env_w = int(env_img.width * target_height / env_img.height)
            action_resized = action_img.resize((action_w, target_height), Image.Resampling.BILINEAR)
            env_resized = env_img.resize((env_w, target_height), Image.Resampling.BILINEAR)

            total_w = action_w + separator_width + env_w
            total_h = target_height + banner_h
            combined = Image.new("RGB", (total_w, total_h), color=0)

            draw = ImageDraw.Draw(combined)
            draw.rectangle([(0, 0), (action_w, banner_h)], fill=(30, 30, 60))
            draw.rectangle([(action_w + separator_width, 0), (total_w, banner_h)], fill=(30, 60, 30))
            draw.text((4, 1), "Action Prediction", fill=(100, 180, 255))
            draw.text((action_w + separator_width + 4, 1), "Environment", fill=(100, 255, 100))

            combined.paste(action_resized, (0, banner_h))
            combined.paste(env_resized, (action_w + separator_width, banner_h))
            combined_frames.append(combined)

    if combined_frames:
        _save_gif(combined_frames, output_path, fps)


def _select_action_chunk(actions: list[list[float]], action_horizon: int) -> list[list[float]]:
    if action_horizon <= 0 or action_horizon >= len(actions):
        return actions
    return actions[:action_horizon]


def _format_action(action: list[float], action_dim: int) -> list[float]:
    if len(action) < action_dim:
        raise ValueError(f"Action dimension {len(action)} smaller than expected {action_dim}")
    return action[:action_dim]


def _remap_gripper(action: list[float], mode: str) -> list[float]:
    """Map the model's gripper command to the LIBERO env's [-1, 1] (negative = open).

    The right mapping depends on the gripper convention of the dataset the policy
    was trained on (the server denormalizes back to that raw convention):

    * ``zero_one`` (NVIDIA LIBERO_LeRobot_v3): raw gripper in [0, 1]; the env wants
      [-1, 1] with negative=open. The i4/cosmos-rl reference BINARIZES this to hard
      {-1, +1} via ``-sign(2g - 1)`` (not the continuous ``1 - 2g`` from issue #50).
      For a confident policy the two agree (g~0/1), but an undertrained policy emits
      g~0.5 where continuous ``1-2g``~0 never actuates the gripper -> grasps fail.
      Binarizing matches the reference and is robust to weak checkpoints.
    * ``pm_one`` (community ``lerobot/libero_*``): raw gripper already in {-1, +1}
      (robosuite convention) -> pass through (clamped).
    * ``pm_one_flip``: {-1, +1} but with inverted open/close sign.
    """
    action = list(action)  # avoid mutating the caller's list
    g = action[-1]
    if mode == "zero_one":
        action[-1] = max(-1.0, min(1.0, g * 2.0 - 1.0)) * -1.0  # [0,1] -> [-1,1], negative=open (issue #50)
    elif mode == "pm_one":
        action[-1] = max(-1.0, min(1.0, g))
    elif mode == "pm_one_flip":
        action[-1] = max(-1.0, min(1.0, -g))
    else:
        raise ValueError(f"Unknown gripper_mode={mode!r}. Use zero_one/pm_one/pm_one_flip.")
    return action


def _infer_rotation_space(action_dim: int, rotation_space: str) -> str:
    if rotation_space != "auto":
        return rotation_space
    return libero_rotation_space_from_action_dim(action_dim)


def _obs_to_pose(obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
    rotation = SciRotation.from_quat(quat).as_matrix()
    return position, rotation


def _anchored_action_to_delta(
    anchored_action: np.ndarray,
    base_pose: tuple[np.ndarray, np.ndarray],
    current_pose: tuple[np.ndarray, np.ndarray],
    rotation_space: str,
) -> np.ndarray:
    anchored_translation = anchored_action[:3]
    rotation_dim = anchored_action.shape[0] - 4
    anchored_rotation = anchored_action[3 : 3 + rotation_dim]
    gripper = anchored_action[3 + rotation_dim : 4 + rotation_dim]

    base_pos, base_rot = base_pose
    current_pos, current_rot = current_pose

    if rotation_space == "3d":
        anchored_rot = SciRotation.from_rotvec(anchored_rotation).as_matrix()
    elif rotation_space == "6d":
        anchored_rot = _rotation_repr_to_mat(anchored_rotation, rotation_space)
    elif rotation_space == "9d":
        anchored_rot = anchored_rotation.reshape(3, 3)
    else:
        raise ValueError(f"Unsupported rotation_space={rotation_space!r}. Use 3d/6d/9d.")
    target_rot = base_rot @ anchored_rot
    target_pos = base_pos + base_rot @ anchored_translation
    delta_pos = target_pos - current_pos
    delta_rot = target_rot @ current_rot.T
    delta_rotvec = SciRotation.from_matrix(delta_rot).as_rotvec()

    return np.concatenate([delta_pos, delta_rotvec, gripper], axis=0)


def _framewise_action_to_delta(
    framewise_action: np.ndarray,
    rotation_space: str,
) -> np.ndarray:
    """Convert a frame-wise policy action to LIBERO's 7D simulator command.

    Frame-wise actions are already per-step deltas in the LIBERO controller's
    convention (see ``LiberoDataset`` with ``action_space='frame_wise_relative'``),
    so the only conversion required is decoding the chosen rotation
    representation back to a rotation vector. No anchor/current pose is needed.
    """
    if rotation_space == "3d":
        return framewise_action

    translation = framewise_action[:3]
    rotation_dim = framewise_action.shape[0] - 4
    rotation_repr = framewise_action[3 : 3 + rotation_dim]
    gripper = framewise_action[3 + rotation_dim : 4 + rotation_dim]
    rotation_delta = _rotation_repr_to_mat(rotation_repr, rotation_space)

    delta_pos = translation
    delta_rotvec = SciRotation.from_matrix(rotation_delta).as_rotvec()
    return np.concatenate([delta_pos, delta_rotvec, gripper], axis=0)

# 一个回合
def _run_episode(
    env: Any,
    client: ActionEnvironmentClient,
    *,
    cameras: list[str],
    flip_images: bool,
    rotate_180: bool,
    action_horizon: int,
    action_dim: int,
    action_space: str,
    rotation_space: str,
    gripper_mode: str,
    max_steps: int,
    warmup_steps: int,
    initial_state: np.ndarray | None,
    gif_path: Path | None,
    gif_fps: int,
    comparison_path: Path | None = None,
    streaming: bool = False,
) -> EpisodeResult:
    with TIMELINE.span("client.env_reset", task_id=client.task_id, episode=client.episode):
        env.reset()
    if initial_state is not None:
        with TIMELINE.span("client.set_init_state", task_id=client.task_id, episode=client.episode):
            obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    action_queue: list[list[float]] = []
    stream_request: _StreamingActionRequest | None = None
    base_pose: tuple[np.ndarray, np.ndarray] | None = None
    step = 0
    success = False
    gif_frames: list[Image.Image] = []
    action_log: list[list[float]] = []
    is_multi_view = len(cameras) > 1
    resolved_rotation_space = _infer_rotation_space(action_dim, rotation_space)

    comparison_windows: list[tuple[list[Image.Image], list[Image.Image]]] = []

    def record_frame(current_obs: dict[str, Any]) -> None:
        if gif_path is None:
            return
        image = _get_libero_image(
            current_obs,
            cameras[0],
            flip_images=flip_images,
            rotate_180=rotate_180,
        )
        image = _ensure_uint8_image(image)
        gif_frames.append(Image.fromarray(image).convert("RGB"))

    def capture_comparison_frame(current_obs: dict[str, Any]) -> Image.Image:
        """Capture an env frame matching Action's input view (multi-view concatenated if applicable)."""
        if is_multi_view:
            imgs = _get_libero_images(current_obs, cameras, flip_images=flip_images, rotate_180=rotate_180)
            concat = client.concatenate_images(imgs)
            return Image.fromarray(_ensure_uint8_image(concat)).convert("RGB")
        img = _get_libero_image(current_obs, cameras[0], flip_images=flip_images, rotate_180=rotate_180)
        return Image.fromarray(_ensure_uint8_image(img)).convert("RGB")

    if gif_path is not None:
        record_frame(env.policy_observation)
    # 预热
    if warmup_steps > 0:
        with TIMELINE.span("client.dummy_stabilization", task_id=client.task_id, episode=client.episode):
            while step < min(warmup_steps, max_steps):
                dummy = _get_libero_dummy_action()
                obs, _, _, _ = env.step(dummy)
                action_log.append(dummy)
                step += 1
                if gif_path is not None:
                    record_frame(env.policy_observation)

    while step < max_steps:
        if not action_queue and stream_request is None:
            client.begin_window()
            window_action_index = 0
            with client.span("client.render_observation"):
                obs = env.policy_observation
            if is_multi_view:
                observation_imgs = _get_libero_images(
                    obs,
                    cameras,
                    flip_images=flip_images,
                    rotate_180=rotate_180,
                )
                observation: np.ndarray | list[np.ndarray] = observation_imgs
            else:
                observation = _get_libero_image(
                    obs,
                    cameras[0],
                    flip_images=flip_images,
                    rotate_180=rotate_180,
                )

            if streaming:
                stream_request = _StreamingActionRequest(client, observation, action_horizon)
            else:
                with client.span("client.wait_action", environment_step=step):
                    result = client.predict(observation)
                actions = result.get("action", [])
                if not actions:
                    return EpisodeResult(False, step, "Empty action chunk from server", action_log)
                action_queue = _select_action_chunk(actions, action_horizon)

                if comparison_path is not None:
                    action_video_b64 = result.get("video", [])
                    if action_video_b64:
                        action_frames = _decode_b64_frames(action_video_b64)
                        env_comparison_frames = [capture_comparison_frame(obs)]
                        comparison_windows.append((action_frames, env_comparison_frames))

            if action_space == "relative":
                base_pose = _obs_to_pose(obs)

        if streaming:
            assert stream_request is not None
            with client.span("client.wait_action", environment_step=step, action_index=window_action_index):
                streamed_action = stream_request.next_action()
            if streamed_action is None:
                received_actions = stream_request.received_actions
                stream_request.finish()
                stream_request = None
                if received_actions == 0:
                    return EpisodeResult(False, step, "Empty action stream from server", action_log)
                continue
            raw_action = _format_action(streamed_action, action_dim)
        else:
            raw_action = _format_action(action_queue.pop(0), action_dim)
        if action_space == "relative":
            if base_pose is None:
                raise RuntimeError("Missing base pose for relative action conversion")
            current_pose = _obs_to_pose(obs)
            action = _anchored_action_to_delta(
                np.asarray(raw_action, dtype=np.float32),
                base_pose,
                current_pose,
                resolved_rotation_space,
            )
            action_list = action.tolist()
        else:
            action = _framewise_action_to_delta(
                np.asarray(raw_action, dtype=np.float32),
                resolved_rotation_space,
            )
            action_list = action.tolist()

        # Map the model's gripper command to the env's [-1, 1] per the dataset convention.
        action_list = _remap_gripper(action_list, gripper_mode)

        action_log.append(action_list)
        with client.span("client.execute_action", environment_step=step, action_index=window_action_index):
            obs, _, done, info = env.step(action_list)
        window_action_index += 1
        step += 1
        if gif_path is not None or (comparison_path is not None and comparison_windows):
            obs = env.policy_observation
        record_frame(obs)

        if comparison_path is not None and comparison_windows:
            comparison_windows[-1][1].append(capture_comparison_frame(obs))

        if stream_request is not None and stream_request.reached_action_limit:
            with client.span("client.wait_final"):
                stream_request.finish()
            stream_request = None

        if isinstance(info, dict) and info.get("success"):
            success = True
            break
        if done:
            success = True if not isinstance(info, dict) else bool(info.get("success", True))
            break

    if stream_request is not None:
        with client.span("client.wait_final"):
            stream_request.finish()
    if gif_path is not None:
        _save_gif(gif_frames, gif_path, gif_fps)
    if comparison_path is not None and comparison_windows:
        _save_comparison_gif(comparison_windows, comparison_path, gif_fps)
    return EpisodeResult(success, step, None, action_log)


def _load_initial_states(
    task_suite: Any,
    task_id: int,
    *,
    task_description: str,
    initial_states_path: str,
    episode_idx: int,
) -> np.ndarray | None:
    default_initial_states = task_suite.get_task_init_states(task_id)

    if initial_states_path == "DEFAULT":
        return np.array(default_initial_states[episode_idx])

    with open(initial_states_path, encoding="utf-8") as f:
        all_initial_states = json.load(f)

    task_key = task_description.replace(" ", "_")
    episode_key = f"demo_{episode_idx}"
    if not all_initial_states[task_key][episode_key]["success"]:
        return None
    return np.array(all_initial_states[task_key][episode_key]["initial_state"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LIBERO closed-loop evaluation via Action HTTP server")
    parser.add_argument(
        "--server_url", type=str, required=True, help="Base URL for Action server (e.g., http://host:8000)"
    )
    parser.add_argument("--task_suite", type=str, default="libero_spatial", choices=sorted(TASK_MAX_STEPS.keys()))
    parser.add_argument("--num_trials_per_task", type=int, default=1)
    parser.add_argument("--task_ids", type=str, default="", help="Comma-separated task IDs to evaluate (default: all)")
    parser.add_argument("--image_size", type=int, default=256, help="Model input image size")
    parser.add_argument("--env_image_size", type=int, default=256, help="Environment render resolution")
    parser.add_argument("--action_horizon", type=int, default=0, help="Actions to execute per request (0=full chunk)")
    parser.add_argument("--action_dim", type=int, default=10, help="Action dimension for LIBERO")
    parser.add_argument(
        "--action_space",
        type=str,
        default="frame_wise_relative",
        choices=["relative", "frame_wise_relative"],
        help="Action space expected from the model (relative=anchored, frame_wise_relative=framewise deltas).",
    )
    parser.add_argument(
        "--rotation_space",
        type=str,
        default="auto",
        choices=["auto", "3d", "6d", "9d"],
        help="Rotation representation for anchored actions (auto infers from action_dim).",
    )
    parser.add_argument(
        "--gripper_mode",
        type=str,
        default="zero_one",
        choices=["zero_one", "pm_one", "pm_one_flip"],
        help="Gripper convention of the training data: 'zero_one' = [0,1] (NVIDIA "
        "LIBERO_LeRobot_v3, mapped 1-2g); 'pm_one' = {-1,+1} (community lerobot/libero_*, "
        "pass-through); 'pm_one_flip' = {-1,+1} with inverted sign.",
    )
    parser.add_argument("--domain_name", type=str, default="libero")
    parser.add_argument(
        "--camera",
        type=str,
        default="agentview",
        help="Camera(s) to use. Single camera: 'agentview' or 'wrist'. Multiple cameras: comma-separated, e.g., 'agentview,wrist'.",
    )
    parser.add_argument("--flip_images", action="store_true", help="Flip images vertically before encoding")
    parser.add_argument(
        "--rotate_180",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rotate images by 180 degrees before encoding (default: True; pass --no-rotate-180 to disable)",
    )
    parser.add_argument("--warmup_steps", type=int, default=10, help="Stabilization steps with dummy actions")
    parser.add_argument(
        "--exclude_dummy_time",
        action="store_true",
        help="Exclude measured dummy stabilization and rendering from whole-evaluation model-action throughput.",
    )
    parser.add_argument(
        "--synthetic_warmup_requests",
        type=int,
        default=None,
        help="Separate pre-evaluation requests (default: 2 when advertised by the server)",
    )
    parser.add_argument(
        "--allow_debug_output",
        action="store_true",
        help="Allow dumps/video during evaluation and mark the run as unsuitable for benchmark comparison",
    )
    parser.add_argument("--max_steps", type=int, default=0, help="Override max steps per episode (0=default)")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP request timeout in seconds")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Execute each action as soon as HAS finishes it instead of waiting for the full chunk.",
    )
    parser.add_argument("--wait_timeout", type=float, default=60.0, help="Seconds to wait for server health")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--timing_warmup_requests",
        type=int,
        default=2,
        help="Exclude this many initial completed requests from steady-state metrics; retain raw records and episodes",
    )
    parser.add_argument("--save_gifs", action="store_true", help="Save per-episode GIFs of rendered frames")
    parser.add_argument(
        "--save_comparison",
        action="store_true",
        help="Save side-by-side comparison GIFs (Action prediction vs environment rollout)",
    )
    parser.add_argument("--gif_fps", type=int, default=20, help="Frames per second for saved GIFs")
    parser.add_argument(
        "--mujoco_gl",
        type=str,
        default="auto",
        choices=["auto", "egl", "osmesa", "glfw"],
        help="MuJoCo GL backend (auto picks egl if /dev/dri is accessible, else osmesa).",
    )
    parser.add_argument(
        "--render_gpu_device_id",
        type=int,
        default=-1,
        help="GPU device index for EGL rendering (-1 uses default device).",
    )
    parser.add_argument(
        "--initial_states_path",
        type=str,
        default="DEFAULT",
        help='Path to initial states JSON. Use "DEFAULT" for benchmark defaults.',
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of parallel LIBERO envs (SubprocVectorEnv). >1 runs trials in waves "
        "with one batched /predict_batch or /predict_batch_stream per control step; "
        "streaming routes sample_index to per-env action queues.",
    )
    parser.add_argument("--output_dir", type=str, default="", help="Directory to save evaluation summary JSON")
    return parser.parse_args()


class _LiberoEnvFactory:
    """Picklable env factory for SubprocVectorEnv under the spawn start method.

    spawn pickles each env_fn and re-imports this module in the child, so the
    factory must be a top-level class (lambdas/closures are not picklable). The
    child sets the GL backend and imports OffScreenRenderEnv locally so its EGL
    context is created fresh in the worker process."""

    def __init__(
        self,
        *,
        bddl_file_name: str,
        camera_heights: int,
        camera_widths: int,
        render_gpu_device_id: int,
        mujoco_gl: str,
        cameras: list[str],
    ) -> None:
        self.bddl_file_name = bddl_file_name
        self.camera_heights = camera_heights
        self.camera_widths = camera_widths
        self.render_gpu_device_id = render_gpu_device_id
        self.mujoco_gl = mujoco_gl
        self.cameras = cameras

    def __call__(self) -> Any:
        # Resolve to a concrete GPU; -1 (auto) makes EGL device selection race/fail
        # across spawned workers (EGLError / "'EGLGLContext' object has no attribute
        # '_context'"). Set the GL backend + pin the EGL device BEFORE importing
        # OffScreenRenderEnv (which dlopen's the GL stack at import).
        dev = self.render_gpu_device_id if self.render_gpu_device_id >= 0 else 0
        os.environ["MUJOCO_GL"] = self.mujoco_gl
        if self.mujoco_gl == "egl":
            os.environ["PYOPENGL_PLATFORM"] = "egl"
            os.environ["MUJOCO_EGL_DEVICE_ID"] = str(dev)
            os.environ["EGL_DEVICE_ID"] = str(dev)
        elif self.mujoco_gl == "osmesa":
            os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        from vllm_omni_cosmos3_client._vendor.libero.envs import OffScreenRenderEnv as _OffScreenRenderEnv

        from .rendering.observation import ObservationRenderEnv

        if remote_render_url := os.environ.get("REMOTE_RENDER_URL"):
            from .rendering.remote import install_renderer

            install_renderer(remote_render_url)

        env = _OffScreenRenderEnv(
            bddl_file_name=self.bddl_file_name,
            camera_heights=self.camera_heights,
            camera_widths=self.camera_widths,
            render_gpu_device_id=dev,
            use_camera_obs=False,
        )
        return ObservationRenderEnv(env, self.cameras, self.camera_heights)


def _run_task_vectorized(
    task: Any,
    task_description: str,
    *,
    num_trials: int,
    num_envs: int,
    env_image_size: int,
    seed: int,
    render_gpu_device_id: int,
    client: ActionEnvironmentClient,
    cameras: list[str],
    flip_images: bool,
    rotate_180: bool,
    action_horizon: int,
    action_dim: int,
    rotation_space: str,
    gripper_mode: str,
    max_steps: int,
    warmup_steps: int,
    init_states: list[np.ndarray | None],
    streaming: bool = False,
) -> list[dict[str, Any]]:
    """Run all `num_trials` of one task across `num_envs` parallel LIBERO envs
    (SubprocVectorEnv), in waves. Each control step gathers obs from the ACTIVE
    (not-done) envs, issues ONE batched request, and steps all active envs;
    done envs are masked out. In streaming mode the batch request is
    ``/predict_batch_stream`` and each ``sample_index`` is routed to its own
    action queue. Returns per-trial result dicts in trial order with the same
    shape as the serial path's episode_results."""
    import multiprocessing as _mp

    from vllm_omni_cosmos3_client._vendor.libero.envs.venv import SubprocVectorEnv

    # LIBERO's SubprocVectorEnv defaults to the fork start method; forked children
    # inherit the parent's already-dlopen'd EGL/GL state, which corrupts per-child
    # render-context creation (EGLError / 'EGLGLContext' has no attribute '_context').
    # Force spawn so each env worker starts clean — exactly like the (working) serial
    # single-process path. spawn pickles env_fns, so the factory below is picklable.
    try:
        _mp.set_start_method("spawn", force=True)
    except RuntimeError:  # pragma: no cover - already set
        pass

    resolved_rotation_space = _infer_rotation_space(action_dim, rotation_space)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    results: list[dict[str, Any]] = [None] * num_trials  # type: ignore[list-item]
    for t in range(num_trials):
        if init_states[t] is None:
            results[t] = {
                "episode": t,
                "success": False,
                "steps": 0,
                "error": "Skipped due to failed expert demo",
                "elapsed_s": 0.0,
            }
    runnable = [t for t in range(num_trials) if init_states[t] is not None]
    if not runnable:
        return results

    n = min(num_envs, len(runnable))

    mujoco_gl = os.environ.get("MUJOCO_GL", "egl")
    env_fn = _LiberoEnvFactory(
        bddl_file_name=bddl,
        camera_heights=env_image_size,
        camera_widths=env_image_size,
        render_gpu_device_id=render_gpu_device_id,
        mujoco_gl=mujoco_gl,
        cameras=cameras,
    )
    venv = SubprocVectorEnv([env_fn for _ in range(n)])
    try:
        venv.seed(seed)
        for w0 in range(0, len(runnable), n):
            wave = runnable[w0 : w0 + n]  # trial indices for this wave
            slots = list(range(len(wave)))  # env slots in use
            t_wave0 = time.perf_counter()
            venv.reset(id=slots)
            states = np.stack([np.asarray(init_states[t], dtype=np.float64) for t in wave])
            obs_arr = venv.set_init_state(states, id=slots)
            obs_by_slot = {s: obs_arr[i] for i, s in enumerate(slots)}
            done = {s: False for s in slots}
            succ = {s: False for s in slots}
            err: dict[int, str | None] = {s: None for s in slots}
            nsteps = {s: max_steps for s in slots}
            step = 0

            if warmup_steps > 0:
                with TIMELINE.span("client.dummy_stabilization", task_id=client.task_id, episodes=wave, env_slots=slots):
                    for _ in range(warmup_steps):
                        act = np.stack([_get_libero_dummy_action() for _ in slots])
                        obs_arr, _, _, _ = venv.step(act, id=slots)
                        for i, s in enumerate(slots):
                            obs_by_slot[s] = obs_arr[i]
                        step += 1

            steps_by_slot = {s: step for s in slots}
            while any(not done[s] and steps_by_slot[s] < max_steps for s in slots):
                active = [s for s in slots if not done[s] and steps_by_slot[s] < max_steps]
                if not active:
                    break
                client.begin_window()
                with client.span("client.render_observation", env_slots=active):
                    for slot, observation in zip(active, venv.get_env_attr("policy_observation", id=active)):
                        obs_by_slot[slot] = observation
                obs_batch = [
                    _get_libero_images(obs_by_slot[s], cameras, flip_images=flip_images, rotate_180=rotate_180)
                    for s in active
                ]
                stream_request: _BatchStreamingActionRequest | None = None
                try:
                    if streaming:
                        stream_request = _BatchStreamingActionRequest(client, obs_batch, action_horizon)
                        horizon = action_horizon if action_horizon > 0 else int(client.action_chunk_size or 16)
                    else:
                        with client.span("client.wait_action", env_slots=active):
                            chunks = client.predict_batch(obs_batch)
                        if not chunks or len(chunks) != len(active):
                            raise RuntimeError("bad batch response from server")
                        chunk_by_slot = {s: chunks[k] for k, s in enumerate(active)}
                        horizon = action_horizon if action_horizon > 0 else len(chunks[0])

                    active_index = {s: k for k, s in enumerate(active)}
                    consumed = {s: 0 for s in active}
                    while True:
                        cur = [
                            s for s in active if not done[s] and steps_by_slot[s] < max_steps and consumed[s] < horizon
                        ]
                        if not cur:
                            break
                        ready_actions = None
                        if streaming:
                            assert stream_request is not None
                            with client.span("client.wait_action", env_slots=cur):
                                ready_actions = stream_request.next_ready_actions([active_index[s] for s in cur])
                            cur = [s for s in cur if active_index[s] in ready_actions]
                        env_actions = []
                        for s in cur:
                            if streaming:
                                action_row = ready_actions[active_index[s]]
                            else:
                                action_row = chunk_by_slot[s][consumed[s]]
                            raw = _format_action(action_row, action_dim)
                            a = _framewise_action_to_delta(np.asarray(raw, dtype=np.float32), resolved_rotation_space)
                            env_actions.append(_remap_gripper(a.tolist(), gripper_mode))
                        with client.span("client.execute_action", env_slots=cur):
                            obs_arr, _, d, info = venv.step(np.stack(env_actions), id=cur)
                        for i, s in enumerate(cur):
                            consumed[s] += 1
                            steps_by_slot[s] += 1
                            obs_by_slot[s] = obs_arr[i]
                            di = bool(d[i])
                            ii = info[i] if isinstance(info, list | np.ndarray) else info
                            is_succ = bool(ii.get("success")) if isinstance(ii, dict) else False
                            if is_succ:
                                done[s], succ[s], nsteps[s] = True, True, steps_by_slot[s]
                            elif di:
                                # mirror serial: done w/o explicit success defaults to success
                                done[s] = True
                                succ[s] = ii.get("success", True) if isinstance(ii, dict) else True
                                nsteps[s] = steps_by_slot[s]
                except Exception as e:  # noqa: BLE001
                    for s in active:
                        if not done[s]:
                            done[s] = True
                            err[s] = f"server error: {e}"
                            nsteps[s] = steps_by_slot[s]
                finally:
                    if stream_request is not None:
                        stream_request.finish()
            per_ep_elapsed = round((time.perf_counter() - t_wave0) / max(1, len(wave)), 3)
            rendered_observations = venv.get_env_attr("rendered_observations", id=slots)
            rendered_camera_frames = venv.get_env_attr("rendered_camera_frames", id=slots)
            for s, t in zip(slots, wave):
                results[t] = {
                    "episode": t,
                    "success": bool(succ[s]),
                    "steps": int(nsteps[s]),
                    "error": err[s],
                    "elapsed_s": per_ep_elapsed,
                    "rendered_observations": rendered_observations[s],
                    "rendered_camera_frames": rendered_camera_frames[s],
                }
    finally:
        try:
            venv.close()
        except Exception:  # noqa: BLE001
            pass
    return results


def _latency_stats(values: list[float]) -> dict[str, float | int] | None:
    """Return distribution statistics without changing inference behavior."""
    return latency_stats(values)


def _summarize_request_timings(records: list[dict[str, Any]]) -> dict[str, Any]:
    return summarize_request_timings(records)


def main() -> None:
    args = _parse_args()
    if args.output_dir:
        os.environ.setdefault("COSMOS3_CLIENT_TIMELINE", str(Path(args.output_dir) / "client_timeline.jsonl"))
    if args.exclude_dummy_time and not os.environ.get("COSMOS3_CLIENT_TIMELINE"):
        raise ValueError("--exclude_dummy_time requires --output_dir or COSMOS3_CLIENT_TIMELINE")
    if args.timing_warmup_requests < 0:
        raise ValueError("--timing_warmup_requests must be nonnegative")
    if args.synthetic_warmup_requests is not None and args.synthetic_warmup_requests < 0:
        raise ValueError("--synthetic_warmup_requests must be nonnegative")
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.save_gifs and not args.output_dir:
        raise ValueError("--save_gifs requires --output_dir to be set")
    if args.save_comparison and not args.output_dir:
        raise ValueError("--save_comparison requires --output_dir to be set")
    if args.streaming and args.save_comparison:
        raise ValueError("--streaming cannot currently be combined with --save_comparison")

    # Parse cameras from comma-separated string
    cameras = [c.strip() for c in args.camera.split(",") if c.strip()]
    if not cameras:
        raise ValueError("At least one camera must be specified")
    for cam in cameras:
        if cam not in ("agentview", "wrist"):
            raise ValueError(f"Unsupported camera={cam!r}. Use 'agentview' or 'wrist'.")

    mujoco_backend = _configure_mujoco_env(args.mujoco_gl)
    _import_libero()

    if remote_render_url := os.environ.get("REMOTE_RENDER_URL"):
        from .rendering.remote import install_renderer

        install_renderer(remote_render_url, Path(args.output_dir) if args.output_dir else None)
        print(f"Remote camera renderer: {remote_render_url}; physics runs locally", flush=True)

    client = ActionEnvironmentClient(
        server_url=args.server_url,
        domain_name=args.domain_name,
        prompt="",
        image_size=args.image_size,
        timeout=args.timeout,
    )
    print(f"MuJoCo GL backend: {mujoco_backend}", flush=True)
    print("Waiting for model server...", flush=True)
    _wait_for_server(client, args.wait_timeout)
    server_info = client.get_info()
    diagnostics = server_info.get("diagnostics") or {}
    debug_output = bool(
        diagnostics.get("dump_enabled")
        or diagnostics.get("video_enabled")
        or server_info.get("return_video")
        or args.save_gifs
        or args.save_comparison
    )
    if debug_output and not args.allow_debug_output:
        raise ValueError(
            "Disable server dumps/video and client GIF/comparison output for benchmarking; "
            "use --allow_debug_output only for a labelled debug run"
        )
    client.action_chunk_size = int(server_info.get("action_chunk_size") or 16)
    early_stop_actions = server_info.get("early_stop_actions")
    if early_stop_actions is not None:
        if type(early_stop_actions) is not int or not 1 <= early_stop_actions <= client.action_chunk_size:
            raise ValueError("Server returned an invalid early_stop_actions count")
        if args.action_horizon > early_stop_actions:
            raise ValueError(
                f"Requested action_horizon={args.action_horizon}, but the server returns only "
                f"{early_stop_actions} actions. Refusing to change the execution window implicitly."
            )
        # A default full-response stream consumes the advertised output size.
        # An explicit execution window must remain exactly as requested.
        client.action_chunk_size = early_stop_actions
    print(f"Connected to model server: {server_info}", flush=True)
    if args.streaming and not server_info.get("streaming", {}).get("supported", False):
        raise ValueError("The connected server does not support action streaming for its configured sampler")
    if early_stop_actions is not None:
        print(
            f"Early stop: returning {early_stop_actions} actions; requested action_horizon={args.action_horizon}.",
            flush=True,
        )

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    num_tasks = int(task_suite.n_tasks)

    if args.task_ids:
        selected_task_ids = [int(t) for t in args.task_ids.split(",") if t.strip()]
    else:
        selected_task_ids = list(range(num_tasks))

    max_steps = args.max_steps if args.max_steps > 0 else TASK_MAX_STEPS[args.task_suite]

    total_episodes = 0
    total_successes = 0
    task_results: list[dict[str, Any]] = []

    output_dir = Path(args.output_dir) if args.output_dir else None
    gif_root = output_dir / "gifs" if output_dir and args.save_gifs else None
    comparison_root = output_dir / "comparisons" if output_dir and args.save_comparison else None
    client.timing_warmup_requests = args.timing_warmup_requests
    client.execution_window = args.action_horizon
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        client.timing_path = output_dir / "request_timings.jsonl"
        client.timing_path.write_text("", encoding="utf-8")
    warmup_supported = bool((server_info.get("synthetic_warmup") or {}).get("supported"))
    synthetic_count = args.synthetic_warmup_requests
    if synthetic_count is None:
        synthetic_count = 2 if warmup_supported and not debug_output else 0
    synthetic_warmup = {"requests": 0, "supported": warmup_supported}
    if synthetic_count:
        if not warmup_supported:
            raise ValueError("The server does not advertise synthetic warmup support")
        synthetic_warmup = client.synthetic_warmup(
            requests_count=synthetic_count,
            streaming=args.streaming,
            batch_size=args.num_envs,
        )
        server_info = client.get_info()
    run_started_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    evaluation_start = time.perf_counter()

    for task_id in selected_task_ids:
        client.task_id = task_id
        client.episode = None
        task = task_suite.get_task(task_id)

        # ---- Vectorized path: N parallel envs + one batched /predict_batch per step ----
        if args.num_envs > 1:
            task_description = str(task.language)
            client.prompt = _augment_task_prompt_with_viewpoint(task_description, cameras)
            init_states = [
                _load_initial_states(
                    task_suite,
                    task_id,
                    task_description=task_description,
                    initial_states_path=args.initial_states_path,
                    episode_idx=e,
                )
                for e in range(args.num_trials_per_task)
            ]
            episode_results = _run_task_vectorized(
                task,
                task_description,
                num_trials=args.num_trials_per_task,
                num_envs=args.num_envs,
                env_image_size=args.env_image_size,
                seed=args.seed,
                render_gpu_device_id=args.render_gpu_device_id,
                client=client,
                cameras=cameras,
                flip_images=args.flip_images,
                rotate_180=args.rotate_180,
                action_horizon=args.action_horizon,
                action_dim=args.action_dim,
                rotation_space=args.rotation_space,
                gripper_mode=args.gripper_mode,
                max_steps=max_steps,
                warmup_steps=args.warmup_steps,
                init_states=init_states,
                streaming=args.streaming,
            )
            task_episodes = 0
            task_successes = 0
            for er in episode_results:
                if str(er.get("error", "")).startswith("Skipped"):
                    print(f"Task {task_id} | Episode {er['episode'] + 1} | {er['error']}", flush=True)
                    continue
                task_episodes += 1
                total_episodes += 1
                if er["success"]:
                    task_successes += 1
                    total_successes += 1
                print(
                    f"Task {task_id} | Episode {er['episode'] + 1}/{args.num_trials_per_task} | "
                    f"success={er['success']} steps={er['steps']} elapsed_s={er['elapsed_s']:.1f} | "
                    f"task SR {task_successes}/{task_episodes} ({100.0 * task_successes / max(1, task_episodes):.1f}%) | "
                    f"overall SR {total_successes}/{total_episodes} "
                    f"({100.0 * total_successes / max(1, total_episodes):.1f}%)",
                    flush=True,
                )
            task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
            task_results.append(
                {
                    "task_id": task_id,
                    "task_description": task_description,
                    "episodes": task_episodes,
                    "successes": task_successes,
                    "success_rate": task_success_rate,
                    "episode_results": episode_results,
                }
            )
            print(
                f"Task {task_id} summary: {task_successes}/{task_episodes} ({task_success_rate * 100:.1f}%)",
                flush=True,
            )
            continue

        with TIMELINE.span("client.env_create", task_id=task_id):
            env, task_description = _get_libero_env(
                task,
                resolution=args.env_image_size,
                seed=args.seed,
                render_gpu_device_id=args.render_gpu_device_id,
                cameras=cameras,
            )

        task_episodes = 0
        task_successes = 0
        episode_results: list[dict[str, Any]] = []

        for episode_idx in range(args.num_trials_per_task):
            client.episode = episode_idx
            episode_t0 = time.perf_counter()
            client.prompt = _augment_task_prompt_with_viewpoint(task_description, cameras)
            initial_state = _load_initial_states(
                task_suite,
                task_id,
                task_description=task_description,
                initial_states_path=args.initial_states_path,
                episode_idx=episode_idx,
            )
            if initial_state is None:
                episode_elapsed_s = time.perf_counter() - episode_t0
                episode_results.append(
                    {
                        "episode": episode_idx,
                        "success": False,
                        "steps": 0,
                        "error": "Skipped due to failed expert demo",
                        "elapsed_s": round(episode_elapsed_s, 3),
                    }
                )
                print(
                    f"Task {task_id} | Episode {episode_idx + 1}/{args.num_trials_per_task} | "
                    "success=False steps=0 "
                    f"elapsed_s={episode_elapsed_s:.1f} "
                    "error='Skipped due to failed expert demo'",
                    flush=True,
                )
                continue

            gif_path = (
                gif_root / f"task_{task_id:03d}" / f"episode_{episode_idx:03d}.gif" if gif_root is not None else None
            )
            comparison_path = (
                comparison_root / f"task_{task_id:03d}" / f"episode_{episode_idx:03d}.gif"
                if comparison_root is not None
                else None
            )
            try:
                result = _run_episode(
                    env,
                    client,
                    cameras=cameras,
                    flip_images=args.flip_images,
                    rotate_180=args.rotate_180,
                    action_horizon=args.action_horizon,
                    action_dim=args.action_dim,
                    action_space=args.action_space,
                    rotation_space=args.rotation_space,
                    gripper_mode=args.gripper_mode,
                    max_steps=max_steps,
                    warmup_steps=args.warmup_steps,
                    initial_state=initial_state,
                    gif_path=gif_path,
                    gif_fps=args.gif_fps,
                    comparison_path=comparison_path,
                    streaming=args.streaming,
                )
            except Exception as exc:
                result = EpisodeResult(False, 0, str(exc), [])
            episode_elapsed_s = time.perf_counter() - episode_t0

            task_episodes += 1
            total_episodes += 1
            if result.success:
                task_successes += 1
                total_successes += 1

            episode_results.append(
                {
                    "episode": episode_idx,
                    "success": result.success,
                    "steps": result.steps,
                    "error": result.error,
                    "elapsed_s": round(episode_elapsed_s, 3),
                    "rendered_observations": env.rendered_observations,
                    "rendered_camera_frames": env.rendered_camera_frames,
                }
            )

            # Save per-episode action log as JSON
            if output_dir is not None and result.actions:
                action_log_dir = output_dir / "actions" / f"task_{task_id:03d}"
                action_log_dir.mkdir(parents=True, exist_ok=True)
                action_log_path = action_log_dir / f"episode_{episode_idx:03d}.json"
                action_log_path.write_text(
                    json.dumps(result.actions, indent=2),
                    encoding="utf-8",
                )

            with TIMELINE.span("client.next_episode", task_id=task_id, episode=episode_idx):
                client.notify_next_episode()

            # Persist each completed episode, including its timeline, during long runs.
            TIMELINE.flush()
            if output_dir is not None:
                progress = {
                    "task_id": task_id,
                    "completed_trial": episode_idx + 1,
                    "planned_trials_per_task": args.num_trials_per_task,
                    "total_completed_episodes": total_episodes,
                    "total_successes": total_successes,
                    "completed_tasks": task_results,
                    "current_task_episodes": episode_results,
                }
                pending = output_dir / "progress.tmp"
                pending.write_text(json.dumps(progress, indent=2) + "\n")
                pending.replace(output_dir / "progress.json")

            print(
                f"Task {task_id} | Episode {episode_idx + 1}/{args.num_trials_per_task} | "
                f"success={result.success} steps={result.steps} elapsed_s={episode_elapsed_s:.1f} | "
                f"task SR {task_successes}/{task_episodes} ({100.0 * task_successes / max(1, task_episodes):.1f}%) | "
                f"overall SR {total_successes}/{total_episodes} ({100.0 * total_successes / max(1, total_episodes):.1f}%)",
                flush=True,
            )

        task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
        task_results.append(
            {
                "task_id": task_id,
                "task_description": task_description,
                "episodes": task_episodes,
                "successes": task_successes,
                "success_rate": task_success_rate,
                "episode_results": episode_results,
            }
        )
        print(
            f"Task {task_id} summary: {task_successes}/{task_episodes} ({task_success_rate * 100:.1f}%)",
            flush=True,
        )
        # Close the env (and its EGL/MuJoCo render context) before the next task.
        # Leaving it open leaks one EGL context per task and hangs after ~8 tasks.
        try:
            with TIMELINE.span("client.env_close", task_id=task_id):
                env.close()
        except Exception:
            pass

    evaluation_wall_s = time.perf_counter() - evaluation_start
    # Keep legacy steps as the complete environment count; model throughput
    # excludes the initial dummy stabilization actions in both evaluator paths.
    for task in task_results:
        for episode in task["episode_results"]:
            episode["environment_steps"] = episode["steps"]
            episode["warmup_actions"] = min(args.warmup_steps, episode["steps"])
            episode["executed_actions"] = episode["steps"] - episode["warmup_actions"]
    overall_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    with client._timing_lock:
        request_timings = [dict(row) for row in client.request_timings]
    request_timing_summary = _summarize_request_timings(request_timings)
    report = build_report(
        request_timings,
        task_results,
        evaluation_wall_s=evaluation_wall_s,
        run={
            "started_at": run_started_at,
            "evaluation_scope": "libero_task_loop_v1",
            "synthetic_warmup": synthetic_warmup,
            "debug_output_enabled": debug_output,
            "completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "task_suite": args.task_suite,
            "task_ids": selected_task_ids,
            "num_trials_per_task": args.num_trials_per_task,
            "num_envs": args.num_envs,
            "seed": args.seed,
            "max_steps": max_steps,
            "action_horizon": args.action_horizon,
            "warmup_steps": args.warmup_steps,
            "exclude_dummy_time": args.exclude_dummy_time,
            "streaming": args.streaming,
            "timing_warmup_requests": args.timing_warmup_requests,
            "server_url": args.server_url,
            "render_mode": "remote" if os.environ.get("REMOTE_RENDER_URL") else args.mujoco_gl,
            "env_image_size": args.env_image_size,
            "image_size": args.image_size,
            "numba_disable_jit": os.environ.get("NUMBA_DISABLE_JIT", "0"),
            "server_info": server_info,
            "render_gpu_device_id": args.render_gpu_device_id,
            "client_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "camera": cameras,
            "action_space": args.action_space,
            "rotation_space": _infer_rotation_space(args.action_dim, args.rotation_space),
            "action_dim": args.action_dim,
            "initial_states_path": args.initial_states_path,
        },
    )
    from .e2e_latency import build_e2e_report
    from .render_exclusion import build_render_free_report

    timeline_events = list(TIMELINE.read_events())
    report["latency"] = {"e2e_no_render": build_e2e_report(timeline_events, request_timings)}
    report["throughput"]["render_excluded"] = build_render_free_report(
        timeline_events,
        request_timings,
        (evaluation_start, evaluation_start + evaluation_wall_s),
        report["accuracy"]["overall"]["executed_actions"],
        exclude_dummy_time=args.exclude_dummy_time,
        expected_dummy_intervals=(
            sum(
                episode["warmup_actions"] > 0
                for task in task_results
                for episode in task["episode_results"]
            )
            if args.exclude_dummy_time and args.num_envs == 1
            else None
        ),
    )
    report["primary_throughput"] = (
        "throughput.render_excluded.evaluation_no_render_no_dummy.actions_per_s"
        if args.exclude_dummy_time
        else "throughput.render_excluded.complete_windows_no_render.actions_per_s"
    )
    summary = {
        "latency": report["latency"],
        "primary_throughput": report["primary_throughput"],
        "schema_version": report["schema_version"],
        "timing_contract": report["timing_contract"],
        "task_suite": args.task_suite,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "overall_success_rate": overall_success_rate,
        "num_trials_per_task": args.num_trials_per_task,
        "selected_task_ids": selected_task_ids,
        "action_space": args.action_space,
        "rotation_space": _infer_rotation_space(args.action_dim, args.rotation_space),
        "action_dim": args.action_dim,
        "streaming": args.streaming,
        "rendering": {
            "schedule": "policy_observation_only" if not debug_output else "policy_and_requested_debug_frames",
            "execution_window": args.action_horizon or client.action_chunk_size,
            "rendered_observations": sum(
                episode.get("rendered_observations", 0) for task in task_results for episode in task["episode_results"]
            ),
            "rendered_camera_frames": sum(
                episode.get("rendered_camera_frames", 0) for task in task_results for episode in task["episode_results"]
            ),
        },
        "request_timing_summary": request_timing_summary,
        "task_results": task_results,
        "accuracy": report["accuracy"],
        "throughput": report["throughput"],
        "run_config": report["run"],
    }

    print(
        f"Overall success rate: {total_successes}/{total_episodes} ({overall_success_rate * 100:.1f}%)",
        flush=True,
    )
    print(report_text(report), flush=True)
    print(
        "PRIMARY_NO_RENDER_actions_per_s: "
        + str(report["throughput"]["render_excluded"]["complete_windows_no_render"]["actions_per_s"]),
        flush=True,
    )
    if args.exclude_dummy_time:
        print(
            "PRIMARY_CLOSED_LOOP_NO_RENDER_NO_DUMMY_actions_per_s: "
            + str(report["throughput"]["render_excluded"]["evaluation_no_render_no_dummy"]["actions_per_s"]),
            flush=True,
        )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        timings_path = output_dir / "request_timings.jsonl"
        timings_path.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in request_timings),
            encoding="utf-8",
        )
        print(f"Saved {len(request_timings)} request timings to {timings_path}", flush=True)
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved summary to {summary_path}", flush=True)
        write_report(output_dir, report)
        print(f"Saved metrics to {output_dir / 'metrics.json'} (also metrics.csv and metrics.txt)", flush=True)
    TIMELINE.flush()


if __name__ == "__main__":
    main()
