# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Online/offline request clients and response types for tests."""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import copy
import io
import json
import time
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import quote

import numpy as np
import requests
import soundfile as sf
import torch
from openai import OpenAI, omit
from PIL import Image
from vllm.logger import init_logger

from tests.helpers.assertions import (
    SuccessRateGate,
    assert_audio_speech_response,
    assert_diffusion_response,
    assert_http_error,
    assert_images_generations_response,
    assert_omni_response,
    collect_at_success_rate,
)
from tests.helpers.media import (
    _merge_base64_audio_to_segment,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from tests.helpers.runtime import OmniRunner

logger = init_logger(__name__)


def _parse_response_json(r: requests.Response) -> dict[str, Any] | list[Any] | None:
    try:
        data = r.json()
        if isinstance(data, (dict, list)):
            return data
    except Exception:
        pass
    return None


def _split_request_config_by_per_output_sizes(cfg: dict[str, Any]) -> list[dict[str, Any]] | None:
    """If ``extra_body`` has list ``height``/``width``, return one config per index (scalar h/w, ``num_outputs_per_prompt=1``)."""
    eb = cfg.get("extra_body")
    if not eb:
        return None
    h, w = eb.get("height"), eb.get("width")
    if (isinstance(h, (list, tuple)) or isinstance(w, (list, tuple))) and not (
        isinstance(h, (list, tuple)) and isinstance(w, (list, tuple))
    ):
        raise ValueError("extra_body height and width must both be lists or both be scalars")
    if not (isinstance(h, (list, tuple)) and isinstance(w, (list, tuple))):
        return None
    if len(h) != len(w):
        raise ValueError(f"height and width lists must have equal length; got {len(h)=} {len(w)=}")
    n = len(h)
    n_out = eb.get("num_outputs_per_prompt")
    if n_out is not None:
        n_out = int(n_out)
        if n_out != n:
            raise ValueError(
                "When height/width are lists, num_outputs_per_prompt must equal their length; "
                f"got num_outputs_per_prompt={n_out}, len(lists)={n}"
            )
    splits: list[dict[str, Any]] = []
    for i in range(n):
        sub = copy.deepcopy(cfg)
        sub_eb = dict(sub.get("extra_body") or {})
        sub_eb["height"] = int(h[i])
        sub_eb["width"] = int(w[i])
        sub_eb["num_outputs_per_prompt"] = 1
        sub["extra_body"] = sub_eb
        splits.append(sub)
    return splits


@dataclass
class OmniResponse:
    """Decoded multimodal / chat output from the OpenAI SDK or offline runner (not raw ``requests``)."""

    text_content: str | None = None
    audio_data: list[str] | None = None
    audio_content: str | None = None
    audio_format: str | None = None
    audio_bytes: bytes | None = None
    #: End-to-end wall time in **seconds** (``perf_counter`` delta), from just before the
    #: OpenAI client call through response parsing and local post-process (e.g. audio decode).
    e2e_latency: float | None = None
    success: bool = False
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    multimodal_tokens: dict[str, int] | None = None
    logprobs: list | None = None


@dataclass
class DiffusionResponse:
    """Decoded diffusion output from chat completions or offline runner (not raw ``requests``)."""

    text_content: str | None = None
    images: list[Image.Image] | None = None
    audios: list[Any] | None = None
    videos: list[Any] | None = None
    #: End-to-end wall time in **seconds** (``perf_counter`` delta), from just before
    #: ``chat.completions.create`` through local image / audio decode.
    e2e_latency: float | None = None
    success: bool = False


@dataclass
class HttpResponse:
    """Normalized view of a ``requests`` response from :class:`OnlineOmniClient` HTTP helpers."""

    status_code: int
    success: bool
    error_message: str | None = None
    json_body: dict[str, Any] | list[Any] | None = None


@dataclass
class WebSocketJsonResponse:
    """First JSON object delivered as a text WebSocket frame (streaming endpoints)."""

    json_body: dict[str, Any]


@dataclass
class OpenPIWebSocketResponse:
    """Msgpack WebSocket session against ``/v1/realtime/robot/openpi``."""

    server_metadata: dict[str, Any]
    operation_responses: list[Any]
    actions: dict[str, np.ndarray] | None = None
    action_tensors: list[np.ndarray] | None = None


def build_openpi_droid_observation(*, session_id: str = "gr00t-smoke") -> dict[str, Any]:
    """Build a minimal DROID-style observation payload for the OpenPI robot endpoint."""
    identity_eef_9d = np.zeros((1, 1, 9), dtype=np.float32)
    identity_eef_9d[..., 3:] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    return {
        "session_id": session_id,
        "video": {
            "exterior_image_1_left": np.zeros((1, 2, 256, 256, 3), dtype=np.uint8),
            "wrist_image_left": np.zeros((1, 2, 256, 256, 3), dtype=np.uint8),
        },
        "state": {
            "eef_9d": identity_eef_9d,
            "gripper_position": np.zeros((1, 1, 1), dtype=np.float32),
            "joint_position": np.zeros((1, 1, 7), dtype=np.float32),
        },
        "language": {"annotation.language.language_instruction": [["pick up the object"]]},
    }


DREAMZERO_DEFAULT_PROMPT = (
    "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan"
)
DREAMZERO_ACTION_HORIZON = 24
DREAMZERO_ACTION_DIM = 8
DREAMZERO_CAMERA_FILES = {
    "observation/exterior_image_0_left": "exterior_image_1_left.mp4",
    "observation/exterior_image_1_left": "exterior_image_2_left.mp4",
    "observation/wrist_image_left": "wrist_image_left.mp4",
}


def _require_opencv() -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional e2e dependency
        raise ModuleNotFoundError("DreamZero OpenPI test dependencies are missing: opencv-python") from exc
    return cv2


def load_dreamzero_camera_frames(video_dir: Path) -> dict[str, np.ndarray]:
    cv2 = _require_opencv()
    camera_frames: dict[str, np.ndarray] = {}
    for camera_key, file_name in DREAMZERO_CAMERA_FILES.items():
        video_path = video_dir / file_name
        if not video_path.exists():
            raise FileNotFoundError(f"Missing DreamZero test asset: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise RuntimeError(f"No frames loaded from {video_path}")
        camera_frames[camera_key] = np.stack(frames, axis=0)
    return camera_frames


def build_dreamzero_demo_observations(
    camera_frames: dict[str, np.ndarray],
    *,
    prompt: str,
    session_id: str,
    num_chunks: int = 2,
) -> list[dict[str, Any]]:
    if num_chunks < 1:
        raise ValueError("num_chunks must be at least 1")

    relative_offsets = [-23, -16, -8, 0]
    total_frames = min(frames.shape[0] for frames in camera_frames.values())
    frame_schedules = [[0]]
    current_frame = 23
    for _ in range(num_chunks - 1):
        indices = [max(current_frame + offset, 0) for offset in relative_offsets]
        if indices[-1] >= total_frames:
            break
        frame_schedules.append(indices)
        current_frame += DREAMZERO_ACTION_HORIZON

    observations: list[dict[str, Any]] = []
    for frame_indices in frame_schedules:
        obs: dict[str, Any] = {}
        for camera_key, all_frames in camera_frames.items():
            selected = all_frames[frame_indices]
            obs[camera_key] = selected[0] if len(frame_indices) == 1 else selected
        obs["observation/joint_position"] = np.zeros(7, dtype=np.float32)
        obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
        obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)
        obs["prompt"] = prompt
        obs["session_id"] = session_id
        observations.append(obs)
    return observations


class OpenPIWebSocketSession:
    """Persistent msgpack session for ``/v1/realtime/robot/openpi``."""

    DEFAULT_PING_INTERVAL_SECS = 300
    DEFAULT_PING_TIMEOUT_SECS = 3600

    @staticmethod
    def _require_dependencies() -> tuple[Any, Any]:
        try:
            import websockets.sync.client as websockets_client
        except ImportError as exc:  # pragma: no cover - optional e2e dependency
            raise ModuleNotFoundError("GR00T OpenPI test dependencies are missing: websockets") from exc
        try:
            from openpi_client import msgpack_numpy
        except ImportError as exc:  # pragma: no cover - optional e2e dependency
            raise ModuleNotFoundError("GR00T OpenPI test dependencies are missing: openpi-client") from exc
        return websockets_client, msgpack_numpy

    def __init__(
        self,
        uri: str,
        *,
        ping_interval: float = DEFAULT_PING_INTERVAL_SECS,
        ping_timeout: float = DEFAULT_PING_TIMEOUT_SECS,
        open_timeout: float = 120.0,
        close_timeout: float = 120.0,
    ) -> None:
        websockets_client, msgpack_numpy = self._require_dependencies()
        self._msgpack_numpy = msgpack_numpy
        self._packer = msgpack_numpy.Packer()
        self._conn = websockets_client.connect(
            uri,
            compression=None,
            max_size=None,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
            open_timeout=open_timeout,
            close_timeout=close_timeout,
        )
        server_metadata = msgpack_numpy.unpackb(self._conn.recv())
        if not isinstance(server_metadata, dict):
            raise TypeError(f"Expected dict metadata from {uri}, got {type(server_metadata)!r}")
        self._server_metadata = server_metadata

    def get_server_metadata(self) -> dict[str, Any]:
        return dict(self._server_metadata)

    def infer(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        response = self._send_operation("infer", obs)
        if not isinstance(response, dict):
            raise TypeError(f"Expected dict infer response, got {type(response)!r}")
        return {str(key): np.asarray(value, dtype=np.float32) for key, value in response.items()}

    def reset(self, reset_info: dict[str, Any] | None = None) -> str:
        response = self._send_operation("reset", dict(reset_info or {}))
        return str(response["status"])

    def close(self) -> None:
        self._conn.close()

    def _send_operation(self, endpoint: str, payload: dict[str, Any]) -> Any:
        body = dict(payload)
        body["endpoint"] = endpoint
        self._conn.send(self._packer.pack(body))
        raw = self._conn.recv()
        if isinstance(raw, str):
            raise RuntimeError(f"OpenPI {endpoint!r} failed: {raw}")
        decoded = self._msgpack_numpy.unpackb(raw)
        if isinstance(decoded, dict) and decoded.get("type") == "error":
            raise RuntimeError(f"OpenPI {endpoint!r} failed: {decoded.get('message')}")
        if endpoint == "reset":
            if not isinstance(decoded, dict) or decoded.get("status") != "reset successful":
                raise RuntimeError(f"Unexpected OpenPI reset response: {decoded!r}")
            return decoded
        return decoded


def _merge_http_expectation_kwargs(
    base: dict[str, Any] | None,
    *,
    err_code: int | tuple[int, ...] | list[int] | None = None,
    err_message: str | tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    cfg = dict(base or {})
    if err_code is not None:
        cfg["err_code"] = err_code
    if err_message is not None:
        cfg["err_message"] = err_message
    return cfg


def _merge_ws_expectation_kwargs(
    base: dict[str, Any] | None,
    *,
    err_message: str | tuple[str, ...] | list[str] | None = None,
    ws_json_type: str | None = None,
    ws_error_code: str | None = None,
) -> dict[str, Any]:
    cfg = dict(base or {})
    if err_message is not None:
        cfg["err_message"] = err_message
    if ws_json_type is not None:
        cfg["ws_json_type"] = ws_json_type
    if ws_error_code is not None:
        cfg["ws_error_code"] = ws_error_code
    return cfg


def _run_ws_expectations_from_request_config(cfg: dict[str, Any], resp: WebSocketJsonResponse) -> None:
    jb = resp.json_body
    want_type = cfg.get("ws_json_type")
    if want_type is not None:
        assert jb.get("type") == want_type, (jb, want_type)
    want_code = cfg.get("ws_error_code")
    if want_code is not None:
        assert jb.get("code") == want_code, (jb, want_code)
    err_message = cfg.get("err_message")
    if err_message is not None:
        assert_http_error(resp, err_message=err_message, websocket_json_message=True)


def _merge_diffusion_responses(parts: list[DiffusionResponse]) -> DiffusionResponse:
    """Concatenate images in order; ``e2e_latency`` is wall-clock of the batch (set by caller) or max of parts."""
    merged = DiffusionResponse()
    merged.success = all(p.success for p in parts) and len(parts) > 0
    imgs: list[Image.Image] = []
    for p in parts:
        if p.images:
            imgs.extend(p.images)
    merged.images = imgs if imgs else None
    latencies = [p.e2e_latency for p in parts if p.e2e_latency is not None]
    merged.e2e_latency = max(latencies) if latencies else None
    return merged


class OnlineOmniClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int | None = None,
        api_key: str = "EMPTY",
        run_level: str | None = None,
        *,
        log_stats: bool = True,
    ):
        if port is None:
            from tests.helpers.runtime import get_open_port

            port = get_open_port()
        self.base_url = f"http://{host}:{port}"
        self.client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key=api_key)
        self.run_level = run_level
        self.log_stats = log_stats

    def _print_client_stat(self, message: str) -> None:
        if self.log_stats:
            print(message, flush=True)

    def _process_stream_omni_response(self, chat_completion, *, wall_start: float) -> OmniResponse:
        """Wall clock from *before* ``chat.completions.create`` through stream drain + local decode."""
        result = OmniResponse()
        try:
            text_content = ""
            audio_data = []
            for chunk in chat_completion:
                for choice in chunk.choices:
                    content = getattr(getattr(choice, "delta", None), "content", None)
                    modality = getattr(chunk, "modality", None)
                    if modality == "audio" and content:
                        audio_data.append(content)
                    elif modality == "text" and content:
                        text_content += content
                # Usage is yielded after the last token
                if chunk.usage:
                    result.prompt_tokens = chunk.usage.prompt_tokens
                    if details := getattr(chunk.usage, "prompt_tokens_details", None):
                        result.cached_tokens = details.cached_tokens
                        result.multimodal_tokens = getattr(details, "multimodal_tokens", None)

            if audio_data:
                merged_seg = _merge_base64_audio_to_segment(audio_data)
                wav_buf = BytesIO()
                merged_seg.export(wav_buf, format="wav")
                result.audio_bytes = wav_buf.getvalue()
            result.text_content = text_content
            result.audio_data = audio_data
            result.e2e_latency = time.perf_counter() - wall_start
            result.success = True
        except Exception as e:
            msg = f"Stream processing error: {str(e)}"
            print(f"Error: {msg}")
        return result

    def _process_non_stream_omni_response(self, chat_completion, *, wall_start: float) -> OmniResponse:
        """Wall clock from *before* ``chat.completions.create`` through response parse + local decode."""
        result = OmniResponse()
        try:
            audio_data = None
            text_content = None
            for choice in chat_completion.choices:
                if hasattr(choice.message, "audio") and choice.message.audio is not None:
                    audio_data = choice.message.audio.data
                if hasattr(choice.message, "content") and choice.message.content is not None:
                    text_content = choice.message.content
            # Extract cached & prompt token counts for prefix caching tests
            usage = getattr(chat_completion, "usage", None)
            if usage:
                result.prompt_tokens = usage.prompt_tokens
                if details := getattr(usage, "prompt_tokens_details", None):
                    result.cached_tokens = details.cached_tokens
                    result.multimodal_tokens = getattr(details, "multimodal_tokens", None)
            if audio_data:
                result.audio_bytes = base64.b64decode(audio_data)
            result.text_content = text_content
            result.e2e_latency = time.perf_counter() - wall_start
            if chat_completion.choices and chat_completion.choices[0].logprobs is not None:
                result.logprobs = chat_completion.choices[0].logprobs.content
            result.success = True
        except Exception as e:
            msg = f"Non-stream processing error: {str(e)}"
            print(f"Error: {msg}")
        return result

    def _process_diffusion_response(self, chat_completion, *, wall_start: float) -> DiffusionResponse:
        """Wall clock from *before* ``chat.completions.create`` through image decode."""
        result = DiffusionResponse()
        try:
            images = []
            audios = []
            for choice in chat_completion.choices:
                content = getattr(choice.message, "content", None)
                if isinstance(content, list):
                    for item in content:
                        image_url = None
                        if isinstance(item, dict):
                            image_url = item.get("image_url", {}).get("url")
                        else:
                            image_url_obj = getattr(item, "image_url", None)
                            image_url = getattr(image_url_obj, "url", None) if image_url_obj else None
                        if image_url and image_url.startswith("data:image"):
                            b64_data = image_url.split(",", 1)[1]
                            img = Image.open(io.BytesIO(base64.b64decode(b64_data)))
                            img.load()
                            images.append(img)

                # OpenAI audio responses (e.g. AudioX text-to-audio) populate `message.audio`.
                audio_obj = getattr(choice.message, "audio", None)
                audio_b64 = getattr(audio_obj, "data", None) if audio_obj is not None else None
                if audio_b64:
                    audios.append(
                        {
                            "wav_bytes": base64.b64decode(audio_b64),
                            "id": getattr(audio_obj, "id", None),
                            "expires_at": getattr(audio_obj, "expires_at", None),
                        }
                    )
            result.images = images if images else None
            result.audios = audios if audios else None
            result.e2e_latency = time.perf_counter() - wall_start
            result.success = True
        except Exception as e:
            msg = f"Diffusion response processing error: {str(e)}"
            print(f"Error: {msg}")
        return result

    def _http_response_from_requests(self, r: requests.Response) -> HttpResponse:
        payload = _parse_response_json(r)
        ok = 200 <= r.status_code < 300
        return HttpResponse(
            status_code=r.status_code,
            success=ok,
            error_message=None if ok else (r.text[:8000] if r.text else None),
            json_body=payload,
        )

    def send_health_http_request(
        self,
        request_config: dict[str, Any] | None = None,
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """GET ``/health`` (raw ``requests``).

        ``request_config``: optional ``timeout`` plus optional ``err_code`` / ``err_message`` for
        :func:`~tests.helpers.assertions.assert_http_error` (also as keyword-only args).
        """
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        r = requests.get(self._build_url("/health"), timeout=float(cfg.get("timeout", 120.0)))
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_models_http_request(
        self,
        request_config: dict[str, Any] | None = None,
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """GET ``/v1/models``. Optional ``timeout`` and HTTP assertions (see :func:`~tests.helpers.assertions.assert_http_error`)."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        r = requests.get(
            self._build_url("/v1/models"),
            headers={"Accept": "application/json"},
            timeout=float(cfg.get("timeout", 120.0)),
        )
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_chat_completions_http_request(
        self,
        request_config: dict[str, Any],
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """POST ``/v1/chat/completions`` with ``json`` or ``raw_body`` (malformed-body / contract tests)."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        r = self._post_json_endpoint("/v1/chat/completions", cfg, default_timeout=120.0)
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_batched_chat_completions_http_request(
        self,
        request_config: dict[str, Any],
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """Post to batched chat completions."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        r = self._post_json_endpoint("/v1/chat/completions/batch", cfg, default_timeout=120.0)
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_completions_http_request(
        self,
        request_config: dict[str, Any],
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """POST ``/v1/completions`` with ``json`` or ``raw_body``."""
        # TODO (Alex): A lot of these helpers should be consolidated as they differ only by endpoint
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        r = self._post_json_endpoint("/v1/completions", cfg, default_timeout=120.0)
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_omni_sleep_http_request(
        self,
        request_config: dict[str, Any],
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """POST ``/v1/omni/sleep`` — ``json`` or ``raw_body``, ``timeout``."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        r = self._post_json_endpoint("/v1/omni/sleep", cfg, default_timeout=120.0)
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_omni_wakeup_http_request(
        self,
        request_config: dict[str, Any],
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """POST ``/v1/omni/wakeup``."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        r = self._post_json_endpoint("/v1/omni/wakeup", cfg, default_timeout=120.0)
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_audio_voices_list_http_request(
        self,
        request_config: dict[str, Any] | None = None,
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """GET ``/v1/audio/voices``."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        r = requests.get(
            self._build_url("/v1/audio/voices"),
            headers={"Accept": "application/json"},
            timeout=float(cfg.get("timeout", 120.0)),
        )
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_audio_voices_create_http_request(
        self,
        request_config: dict[str, Any],
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """POST ``/v1/audio/voices`` (multipart): ``data`` / ``files`` / ``timeout``."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        r = self._post_form_endpoint("/v1/audio/voices", cfg, default_timeout=120.0)
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_audio_voices_delete_http_request(
        self,
        request_config: dict[str, Any],
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """DELETE ``/v1/audio/voices/{name}`` — requires ``name``, optional ``timeout``."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        name = cfg["name"]
        timeout = float(cfg.get("timeout", 120.0))
        path = f"/v1/audio/voices/{quote(str(name), safe='')}"
        r = requests.delete(
            self._build_url(path),
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_audio_speech_http_request(
        self,
        request_config: dict[str, Any],
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """POST ``/v1/audio/speech`` with ``json`` or ``raw_body``."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        r = self._post_json_endpoint("/v1/audio/speech", cfg, default_timeout=120.0)
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_audio_speech_batch_http_request(
        self,
        request_config: dict[str, Any],
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """POST ``/v1/audio/speech/batch``."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        r = self._post_json_endpoint("/v1/audio/speech/batch", cfg, default_timeout=120.0)
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_audio_generate_http_request(
        self,
        request_config: dict[str, Any],
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """POST ``/v1/audio/generate``."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        r = self._post_json_endpoint("/v1/audio/generate", cfg, default_timeout=120.0)
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_images_generations_http_request(
        self,
        request_config: dict[str, Any],
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """POST ``/v1/images/generations`` — ``json`` or ``raw_body``, ``timeout``."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        r = self._post_json_endpoint("/v1/images/generations", cfg, default_timeout=300.0)
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        if cfg.get("err_code") is None:
            assert resp.success, resp.error_message
            payload = resp.json_body
            assert isinstance(payload, dict)
            assert_images_generations_response(payload, cfg, run_level=self.run_level)
        return [resp]

    def send_images_edits_http_request(
        self,
        request_config: dict[str, Any],
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """POST ``/v1/images/edits`` — ``data`` / ``files`` / ``timeout``."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        r = self._post_form_endpoint("/v1/images/edits", cfg, default_timeout=300.0)
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_videos_create_http_request(
        self,
        request_config: dict[str, Any],
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """POST ``/v1/videos`` (async job) — multipart ``data`` / ``files``."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        r = self._post_form_endpoint("/v1/videos", cfg, default_timeout=120.0)
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_videos_sync_http_request(
        self,
        request_config: dict[str, Any],
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """POST ``/v1/videos/sync``."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        r = self._post_form_endpoint("/v1/videos/sync", cfg, default_timeout=120.0)
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_videos_list_http_request(
        self,
        request_config: dict[str, Any] | None = None,
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """GET ``/v1/videos`` — optional ``params``, ``timeout``."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        r = requests.get(
            self._build_url("/v1/videos"),
            params=cfg.get("params"),
            headers={"Accept": "application/json"},
            timeout=float(cfg.get("timeout", 120.0)),
        )
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_video_retrieve_http_request(
        self,
        request_config: dict[str, Any],
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """GET ``/v1/videos/{video_id}``."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        video_id = cfg["video_id"]
        timeout = float(cfg.get("timeout", 120.0))
        r = requests.get(
            self._build_url(f"/v1/videos/{quote(str(video_id), safe='')}"),
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_video_delete_http_request(
        self,
        request_config: dict[str, Any],
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """DELETE ``/v1/videos/{video_id}``."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        video_id = cfg["video_id"]
        timeout = float(cfg.get("timeout", 120.0))
        r = requests.delete(
            self._build_url(f"/v1/videos/{quote(str(video_id), safe='')}"),
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_video_content_http_request(
        self,
        request_config: dict[str, Any],
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """GET ``/v1/videos/{video_id}/content``."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        video_id = cfg["video_id"]
        timeout = float(cfg.get("timeout", 120.0))
        r = requests.get(
            self._build_url(f"/v1/videos/{quote(str(video_id), safe='')}/content"),
            timeout=timeout,
        )
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def _build_ws_url(self, path: str) -> str:
        """Turn HTTP ``base_url`` into ``ws`` / ``wss`` for WebSocket helpers."""
        base = self.base_url.rstrip("/")
        suffix = "/" + path.lstrip("/")
        if base.startswith("http://"):
            return "ws://" + base.removeprefix("http://") + suffix
        if base.startswith("https://"):
            return "wss://" + base.removeprefix("https://") + suffix
        raise ValueError(f"Unsupported base_url for WebSocket: {base!r}")

    def _send_websocket_first_json_request(
        self,
        path: str,
        cfg: dict[str, Any],
    ) -> list[WebSocketJsonResponse]:
        """Connect, optionally send text frames, return first JSON text frame as :class:`WebSocketJsonResponse`.

        ``request_config`` keys:

        - ``send_frames``: optional ``str`` or sequence of ``str`` raw WebSocket text frames (omit when the server
          speaks first, e.g. ``/v1/realtime`` rejection path).
        - ``ws_skip_types``: optional event ``type`` strings to ignore while waiting for the first matching frame
          (e.g. ``["session.created"]`` on ``/v1/realtime``).
        - ``timeout``: seconds to wait for the first inbound text frame (default ``120``).
        - ``ws_max_size``: passed through as ``max_size`` to :func:`websockets.connect` when the key is present.
        """
        send_frames_raw = cfg.get("send_frames")
        if send_frames_raw is None:
            frames: list[str] = []
        elif isinstance(send_frames_raw, str):
            frames = [send_frames_raw]
        else:
            frames = list(send_frames_raw)

        timeout = float(cfg.get("timeout", 120.0))
        uri = self._build_ws_url(path)
        skip_types = set(cfg.get("ws_skip_types") or [])

        connect_kw: dict[str, Any] = {}
        if "ws_max_size" in cfg:
            connect_kw["max_size"] = cfg["ws_max_size"]

        async def _recv_first_json_object() -> WebSocketJsonResponse:
            import websockets

            async with websockets.connect(uri, **connect_kw) as ws:
                for frame in frames:
                    await ws.send(frame)
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    if not isinstance(raw, str):
                        raise AssertionError(f"Expected JSON text frame from {uri}, got {type(raw).__name__}")
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise AssertionError(f"Expected JSON text frame from {uri}, body={raw[:500]!r}") from exc
                    if not isinstance(data, dict):
                        raise AssertionError(f"Expected JSON object from {uri}, got {type(data).__name__}")
                    if skip_types and data.get("type") in skip_types:
                        continue
                    return WebSocketJsonResponse(json_body=data)

        resp = asyncio.run(_recv_first_json_object())
        _run_ws_expectations_from_request_config(cfg, resp)
        return [resp]

    def send_audio_speech_stream_ws_request(
        self,
        request_config: dict[str, Any],
        *,
        err_message: str | tuple[str, ...] | list[str] | None = None,
        ws_json_type: str | None = None,
        ws_error_code: str | None = None,
    ) -> list[WebSocketJsonResponse]:
        """WebSocket ``/v1/audio/speech/stream`` — send ``send_frames`` then read first JSON text frame."""
        cfg = _merge_ws_expectation_kwargs(
            request_config,
            err_message=err_message,
            ws_json_type=ws_json_type,
            ws_error_code=ws_error_code,
        )
        return self._send_websocket_first_json_request("/v1/audio/speech/stream", cfg)

    def send_video_chat_stream_ws_request(
        self,
        request_config: dict[str, Any],
        *,
        err_message: str | tuple[str, ...] | list[str] | None = None,
        ws_json_type: str | None = None,
        ws_error_code: str | None = None,
    ) -> list[WebSocketJsonResponse]:
        """WebSocket ``/v1/video/chat/stream`` — send ``send_frames`` then read first JSON text frame."""
        cfg = _merge_ws_expectation_kwargs(
            request_config,
            err_message=err_message,
            ws_json_type=ws_json_type,
            ws_error_code=ws_error_code,
        )
        return self._send_websocket_first_json_request("/v1/video/chat/stream", cfg)

    def send_realtime_ws_request(
        self,
        request_config: dict[str, Any] | None = None,
        *,
        err_message: str | tuple[str, ...] | list[str] | None = None,
        ws_json_type: str | None = None,
        ws_error_code: str | None = None,
    ) -> list[WebSocketJsonResponse]:
        """WebSocket ``/v1/realtime`` — optional outbound frames, then first JSON text frame (often server-initiated)."""
        cfg = _merge_ws_expectation_kwargs(
            request_config,
            err_message=err_message,
            ws_json_type=ws_json_type,
            ws_error_code=ws_error_code,
        )
        return self._send_websocket_first_json_request("/v1/realtime", cfg)

    def send_robot_openpi_ws_request(
        self,
        request_config: dict[str, Any] | None = None,
    ) -> list[OpenPIWebSocketResponse]:
        """WebSocket ``/v1/realtime/robot/openpi`` — msgpack metadata plus optional infer/reset ops.

        ``request_config`` keys:

        - ``operations``: optional sequence of ``{"endpoint": "infer"|"reset", "payload": {...}}``.
        - ``run_default_policy_session``: when true and ``operations`` is omitted, run infer then reset
          using :func:`build_openpi_droid_observation`.
        - ``run_dreamzero_policy_session``: when true and ``operations`` is omitted, run the DreamZero
          infer/reset/infer sequence using :func:`build_dreamzero_demo_observations`.
        - ``video_dir``: required for ``run_dreamzero_policy_session``.
        - ``prompt``: language instruction for DreamZero observations (optional).
        - ``session_id``: used by the default policy session observation builder.
        - ``num_chunks``: DreamZero chunk count (default ``2``).
        - ``timeout``: seconds to wait for each inbound msgpack frame (default ``120``).
        - ``ping_interval`` / ``ping_timeout``: forwarded to :func:`websockets.sync.client.connect`.
        """
        cfg = dict(request_config or {})
        operations_cfg = cfg.get("operations")
        if operations_cfg is not None:
            operations = [dict(op) for op in operations_cfg]
        elif cfg.get("run_dreamzero_policy_session"):
            video_dir = cfg.get("video_dir")
            if video_dir is None:
                raise ValueError("run_dreamzero_policy_session requires video_dir")
            session_id = str(cfg.get("session_id", "dreamzero-smoke"))
            prompt = str(cfg.get("prompt", DREAMZERO_DEFAULT_PROMPT))
            num_chunks = int(cfg.get("num_chunks", 2))
            observations = build_dreamzero_demo_observations(
                load_dreamzero_camera_frames(Path(video_dir)),
                prompt=prompt,
                session_id=session_id,
                num_chunks=num_chunks,
            )
            operations = [{"endpoint": "infer", "payload": obs} for obs in observations]
            operations.append({"endpoint": "reset", "payload": {}})
            operations.append({"endpoint": "infer", "payload": observations[0]})
        elif cfg.get("run_default_policy_session"):
            session_id = str(cfg.get("session_id", "gr00t-smoke"))
            operations = [
                {"endpoint": "infer", "payload": build_openpi_droid_observation(session_id=session_id)},
                {"endpoint": "reset", "payload": {}},
            ]
        else:
            operations = []
        timeout = float(cfg.get("timeout", 120.0))
        ping_interval = float(cfg.get("ping_interval", OpenPIWebSocketSession.DEFAULT_PING_INTERVAL_SECS))
        ping_timeout = float(cfg.get("ping_timeout", OpenPIWebSocketSession.DEFAULT_PING_TIMEOUT_SECS))

        uri = self._build_ws_url("/v1/realtime/robot/openpi")
        session = OpenPIWebSocketSession(
            uri,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
            open_timeout=timeout,
            close_timeout=timeout,
        )
        try:
            operation_responses: list[Any] = []
            actions: dict[str, np.ndarray] | None = None
            action_tensors: list[np.ndarray] = []
            for operation in operations:
                endpoint = str(operation["endpoint"])
                payload = dict(operation.get("payload") or {})
                response = session._send_operation(endpoint, payload)
                operation_responses.append(response)
                if endpoint == "infer":
                    if isinstance(response, dict):
                        actions = {str(key): np.asarray(value, dtype=np.float32) for key, value in response.items()}
                    else:
                        action_tensors.append(np.asarray(response, dtype=np.float32))
        finally:
            session.close()

        return [
            OpenPIWebSocketResponse(
                server_metadata=session.get_server_metadata(),
                operation_responses=operation_responses,
                actions=actions,
                action_tensors=action_tensors or None,
            )
        ]

    def send_omni_request(
        self,
        request_config: dict[str, Any],
        request_num: int = 1,
        *,
        min_successes: int | None = None,
        max_concurrency: int | None = None,
    ) -> list[OmniResponse]:
        """Chat completions via the OpenAI Python SDK (not raw HTTP).

        ``min_successes`` gates the batch on how many of the ``request_num`` requests pass
        rather than requiring each one to, and returns only the successes; ``max_concurrency``
        caps how many are in flight at once. See ``SuccessRateGate``.
        """
        responses: list[OmniResponse] = []
        stream = request_config.get("stream", False)
        modalities = request_config.get("modalities", ["text", "audio"])
        extra_body: dict[str, Any] = {}
        if "speaker" in request_config:
            extra_body["speaker"] = request_config["speaker"]
        if request_config.get("use_audio_in_video"):
            mm = dict(extra_body.get("mm_processor_kwargs") or {})
            mm["use_audio_in_video"] = True
            extra_body["mm_processor_kwargs"] = mm
        if "sampling_params_list" in request_config:
            extra_body["sampling_params_list"] = request_config["sampling_params_list"]
        if request_config.get("extra_body"):
            extra_body.update(request_config["extra_body"])

        create_kwargs: dict[str, Any] = {
            "model": request_config.get("model"),
            "messages": request_config.get("messages"),
            "stream": stream,
            "modalities": modalities,
        }
        if "logprobs" in request_config:
            create_kwargs["logprobs"] = request_config["logprobs"]
        if "top_logprobs" in request_config:
            create_kwargs["top_logprobs"] = request_config["top_logprobs"]
        if "stream_options" in request_config:
            create_kwargs["stream_options"] = request_config["stream_options"]
        if extra_body:
            create_kwargs["extra_body"] = extra_body

        def _one():
            wall_start = time.perf_counter()
            chat_completion = self.client.chat.completions.create(**create_kwargs)
            return (
                self._process_stream_omni_response(chat_completion, wall_start=wall_start)
                if stream
                else self._process_non_stream_omni_response(chat_completion, wall_start=wall_start)
            )

        if min_successes is not None:
            gate = SuccessRateGate(min_successes=min_successes, concurrency=max_concurrency)

            def _sample() -> OmniResponse:
                resp = _one()
                assert_omni_response(resp, request_config, run_level=self.run_level)
                return resp

            # Not _print_client_stat: the count is the gate's verdict, so it has to survive
            # log_stats=False.
            return collect_at_success_rate(
                _sample,
                gate,
                request_num=request_num,
                report=lambda line: print(f"[omni] {line}", flush=True),
            )

        if request_num == 1:
            resp = _one()
            assert_omni_response(resp, request_config, run_level=self.run_level)
            if resp.e2e_latency is not None:
                self._print_client_stat(f"[omni] request#1 success in {resp.e2e_latency:.3f}s")
            else:
                self._print_client_stat("[omni] request#1 completed")
            responses.append(resp)
            return responses

        with concurrent.futures.ThreadPoolExecutor(max_workers=request_num) as executor:
            futures = {executor.submit(_one): i + 1 for i in range(request_num)}
            for future in concurrent.futures.as_completed(futures):
                request_idx = futures[future]
                resp = future.result()
                assert_omni_response(resp, request_config, run_level=self.run_level)
                if resp.e2e_latency is not None:
                    self._print_client_stat(f"[omni] request#{request_idx} success in {resp.e2e_latency:.3f}s")
                else:
                    self._print_client_stat(f"[omni] request#{request_idx} completed")
                responses.append(resp)
        return responses

    def _process_stream_audio_speech_response(
        self, response, *, response_format: str | None = None, wall_start: float
    ) -> OmniResponse:
        """
        Process streaming /v1/audio/speech responses into an OmniResponse.

        This mirrors _process_stream_omni_response but operates on low-level
        audio bytes. Whisper transcription runs in assert_audio_speech_response
        when the run_level requires it.
        """
        result = OmniResponse()

        try:
            # Aggregate all audio bytes from the streaming response.
            data = bytearray()

            # Preferred OpenAI helper.
            if hasattr(response, "iter_bytes") and callable(getattr(response, "iter_bytes")):
                for chunk in response.iter_bytes():
                    if chunk:
                        data.extend(chunk)
            else:
                # Generic iterable-of-bytes fallback (e.g., generator or list of chunks).
                try:
                    iterator = iter(response)
                except TypeError:
                    iterator = None

                if iterator is not None:
                    for chunk in iterator:
                        if not chunk:
                            continue
                        if isinstance(chunk, (bytes, bytearray)):
                            data.extend(chunk)
                        elif hasattr(chunk, "data"):
                            data.extend(chunk.data)  # type: ignore[arg-type]
                        elif hasattr(chunk, "content"):
                            data.extend(chunk.content)  # type: ignore[arg-type]
                        else:
                            raise TypeError(f"Unsupported stream chunk type: {type(chunk)}")
                else:
                    raise TypeError(f"Unsupported audio speech streaming response type: {type(response)}")

            raw_bytes = bytes(data)

            # Populate OmniResponse.
            result.audio_bytes = raw_bytes
            result.e2e_latency = time.perf_counter() - wall_start
            result.success = True
            result.audio_format = getattr(response, "response", None)
            if result.audio_format is not None:
                result.audio_format = result.audio_format.headers.get("content-type", "")

        except Exception as e:
            msg = f"Audio speech stream processing error: {str(e)}"
            print(f"Error: {msg}")

        return result

    def _process_non_stream_audio_speech_response(
        self, response, *, response_format: str | None = None, wall_start: float
    ) -> OmniResponse:
        """
        Process non-streaming /v1/audio/speech responses into an OmniResponse.

        This mirrors _process_non_stream_omni_response but for the binary
        audio payload returned by audio.speech.create.
        """
        result = OmniResponse()

        try:
            # OpenAI non-streaming audio.speech.create returns HttpxBinaryResponseContent (.read() or .content)
            if hasattr(response, "read") and callable(getattr(response, "read")):
                raw_bytes = response.read()
            elif hasattr(response, "content"):
                raw_bytes = response.content  # type: ignore[assignment]
            else:
                raise TypeError(f"Unsupported audio speech response type: {type(response)}")

            result.audio_bytes = raw_bytes
            result.e2e_latency = time.perf_counter() - wall_start
            result.success = True
            result.audio_format = getattr(response, "response", None)
            if result.audio_format is not None:
                result.audio_format = result.audio_format.headers.get("content-type", "")

        except Exception as e:
            msg = f"Audio speech non-stream processing error: {str(e)}"
            print(f"Error: {msg}")

        return result

    def send_audio_speech_request(self, request_config: dict[str, Any], request_num: int = 1) -> list[OmniResponse]:
        """
        Call the /v1/audio/speech endpoint using the same configuration-dict
        style as send_omni_request, but via the OpenAI Python client's
        audio.speech APIs.

        Expected keys in request_config:
          - model: model name/path (required)
          - input: text to synthesize (required)
          - response_format: audio format such as "wav" or "pcm" (optional)
          - task_type, ref_text, ref_audio: TTS-specific extras (optional, passed via extra_body)
          - min_audio_bytes: optional minimum ``len(audio_bytes)`` checked in ``assert_audio_speech_response``
          - transcript_expected_text: local expected spoken text; defaults to ``input``
          - timeout: request timeout in seconds (float, optional, default 120.0)
          - stream: whether to use streaming API (bool, optional, default False)

        For negative / contract checks (expected 4xx), use
        :meth:`send_audio_speech_http_request` with ``err_code`` / ``err_message``.
        """
        timeout = float(request_config.get("timeout", 120.0))

        model = request_config["model"]
        text_input = request_config["input"]
        stream = bool(request_config.get("stream", False))
        voice = request_config.get("voice", None)

        # Standard OpenAI param: use omit when not provided to keep default behavior.
        response_format = request_config.get("response_format", omit)

        # Qwen3-TTS custom fields, forwarded via extra_body.
        extra_body: dict[str, Any] = {}
        # Keep this list aligned with vllm_omni.entrypoints.openai.protocol.audio params.
        for key in (
            "task_type",
            "ref_text",
            "ref_audio",
            "language",
            "max_new_tokens",
            "seed",
            "instructions",
            "speed",
            "stream_format",
            "x_vector_only_mode",
        ):
            if key in request_config:
                extra_body[key] = request_config[key]

        responses: list[OmniResponse] = []

        speech_fmt: str | None = None if response_format is omit else str(response_format).lower()

        print(f"[audio.speech] start model={model}, stream={stream}, request_num={request_num}, timeout={timeout:.1f}s")

        if request_num == 1:
            if stream:
                # Use streaming response helper.
                wall_start = time.perf_counter()
                with self.client.audio.speech.with_streaming_response.create(
                    model=model,
                    input=text_input,
                    response_format=response_format,
                    extra_body=extra_body or None,
                    timeout=timeout,
                    voice=voice,
                ) as resp:
                    omni_resp = self._process_stream_audio_speech_response(
                        resp, response_format=speech_fmt, wall_start=wall_start
                    )
            else:
                # Non-streaming response.
                wall_start = time.perf_counter()
                resp = self.client.audio.speech.create(
                    model=model,
                    input=text_input,
                    response_format=response_format,
                    extra_body=extra_body or None,
                    timeout=timeout,
                    voice=voice,
                )
                omni_resp = self._process_non_stream_audio_speech_response(
                    resp, response_format=speech_fmt, wall_start=wall_start
                )

            assert_audio_speech_response(omni_resp, request_config, run_level=self.run_level)
            if omni_resp.e2e_latency is not None:
                self._print_client_stat(f"[audio.speech] request#1 success in {omni_resp.e2e_latency:.3f}s")
            else:
                self._print_client_stat("[audio.speech] request#1 completed")
            responses.append(omni_resp)
            return responses
        else:
            # request_num > 1: concurrent requests (use same params as single-request path)

            if stream:

                def _stream_task(request_idx: int):
                    wall_start = time.perf_counter()
                    with self.client.audio.speech.with_streaming_response.create(
                        model=model,
                        input=text_input,
                        response_format=response_format,
                        extra_body=extra_body or None,
                        timeout=timeout,
                        voice=voice,
                    ) as resp:
                        result = self._process_stream_audio_speech_response(
                            resp, response_format=speech_fmt, wall_start=wall_start
                        )
                    if result.e2e_latency is not None:
                        self._print_client_stat(
                            f"[audio.speech] request#{request_idx} success in {result.e2e_latency:.3f}s"
                        )
                    else:
                        self._print_client_stat(f"[audio.speech] request#{request_idx} completed")
                    return result

                with concurrent.futures.ThreadPoolExecutor(max_workers=request_num) as executor:
                    futures = {executor.submit(_stream_task, i + 1): i + 1 for i in range(request_num)}
                    for future in concurrent.futures.as_completed(futures):
                        request_idx = futures[future]
                        try:
                            omni_resp = future.result()
                        except Exception as e:
                            print(
                                f"[audio.speech] request#{request_idx} failed "
                                f"(stream={stream}, timeout={timeout:.1f}s): {e!r}"
                            )
                            raise
                        assert_audio_speech_response(omni_resp, request_config, run_level=self.run_level)
                        responses.append(omni_resp)
            else:

                def _non_stream_task(request_idx: int):
                    wall_start = time.perf_counter()
                    r = self.client.audio.speech.create(
                        model=model,
                        input=text_input,
                        response_format=response_format,
                        extra_body=extra_body or None,
                        timeout=timeout,
                        voice=voice,
                    )
                    result = self._process_non_stream_audio_speech_response(
                        r, response_format=speech_fmt, wall_start=wall_start
                    )
                    if result.e2e_latency is not None:
                        self._print_client_stat(
                            f"[audio.speech] request#{request_idx} success in {result.e2e_latency:.3f}s"
                        )
                    else:
                        self._print_client_stat(f"[audio.speech] request#{request_idx} completed")
                    return result

                with concurrent.futures.ThreadPoolExecutor(max_workers=request_num) as executor:
                    futures = {executor.submit(_non_stream_task, i + 1): i + 1 for i in range(request_num)}
                    for future in concurrent.futures.as_completed(futures):
                        request_idx = futures[future]
                        try:
                            omni_resp = future.result()
                        except Exception as e:
                            print(
                                f"[audio.speech] request#{request_idx} failed "
                                f"(stream={stream}, timeout={timeout:.1f}s): {e!r}"
                            )
                            raise
                        assert_audio_speech_response(omni_resp, request_config, run_level=self.run_level)
                        responses.append(omni_resp)

        return responses

    def send_diffusion_request(
        self, request_config: dict[str, Any] | list[dict[str, Any]], request_num: int = 1
    ) -> list[DiffusionResponse]:
        """
        Send OpenAI requests for diffusion models.
        If ``extra_body`` has list ``height``/``width``, sends one chat completion per index in parallel
        (scalar h/w, ``num_outputs_per_prompt=1`` each) and merges images in list order.

        Args:
            request_config: A single request configuration dict, or a list of
                request configuration dicts (one request per element)
            request_num: Number of requests to send concurrently, defaults to 1 (single request)
        Returns:
            list[DiffusionResponse]: List of DiffusionResponse objects containing the response data
        """
        responses: list[DiffusionResponse] = []

        def _create_from_config(cfg: dict[str, Any]) -> tuple[Any, float]:
            stream = cfg.get("stream", False)
            if stream:
                raise NotImplementedError("Streaming is not currently implemented for diffusion model e2e test")
            modalities = cfg.get("modalities", omit)  # Most diffusion models don't require modalities param
            eb = cfg.get("extra_body")
            extra = copy.deepcopy(eb) if eb else None
            wall_start = time.perf_counter()
            chat_completion = self.client.chat.completions.create(
                model=cfg.get("model"),
                messages=cfg.get("messages"),
                extra_body=extra,
                modalities=modalities,
            )
            return chat_completion, wall_start

        if isinstance(request_config, list):
            if not request_config:
                raise ValueError("request_config list must not be empty")
            if request_num != 1:
                raise ValueError("request_num is not supported when request_config is a list")
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(request_config)) as executor:
                futures = {
                    executor.submit(_create_from_config, cfg): (i + 1, cfg) for i, cfg in enumerate(request_config)
                }
                for future in concurrent.futures.as_completed(futures):
                    request_idx, cfg = futures[future]
                    chat_completion, wall_start = future.result()
                    response = self._process_diffusion_response(chat_completion, wall_start=wall_start)
                    assert_diffusion_response(response, cfg, run_level=self.run_level)
                    if response.e2e_latency is not None:
                        self._print_client_stat(
                            f"[diffusion] request#{request_idx} success in {response.e2e_latency:.3f}s"
                        )
                    else:
                        self._print_client_stat(f"[diffusion] request#{request_idx} completed")
                    responses.append(response)
            return responses

        size_splits = _split_request_config_by_per_output_sizes(request_config)
        if size_splits is not None:
            if request_num != 1:
                raise ValueError(
                    "request_num must be 1 when extra_body height/width are lists (split into concurrent per-size calls)"
                )
            t0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(size_splits)) as executor:
                split_futures = [executor.submit(_create_from_config, sub) for sub in size_splits]
                chat_completions = [f.result() for f in split_futures]
            parts = [self._process_diffusion_response(cc, wall_start=ws) for cc, ws in chat_completions]
            merged = _merge_diffusion_responses(parts)
            merged.e2e_latency = time.perf_counter() - t0
            assert_diffusion_response(merged, request_config, run_level=self.run_level)
            if merged.e2e_latency is not None:
                self._print_client_stat(f"[diffusion] request#1 success in {merged.e2e_latency:.3f}s")
            else:
                self._print_client_stat("[diffusion] request#1 completed")
            return [merged]

        if request_num == 1:
            # Send single request
            chat_completion, wall_start = _create_from_config(request_config)
            response = self._process_diffusion_response(chat_completion, wall_start=wall_start)
            assert_diffusion_response(response, request_config, run_level=self.run_level)
            if response.e2e_latency is not None:
                self._print_client_stat(f"[diffusion] request#1 success in {response.e2e_latency:.3f}s")
            else:
                self._print_client_stat("[diffusion] request#1 completed")
            responses.append(response)
            return responses

        # Send concurrent requests for the same request_config
        with concurrent.futures.ThreadPoolExecutor(max_workers=request_num) as executor:
            idx_by_future = {executor.submit(_create_from_config, request_config): i + 1 for i in range(request_num)}
            for future in concurrent.futures.as_completed(idx_by_future):
                request_idx = idx_by_future[future]
                chat_completion, wall_start = future.result()
                response = self._process_diffusion_response(chat_completion, wall_start=wall_start)
                assert_diffusion_response(response, request_config, run_level=self.run_level)
                if response.e2e_latency is not None:
                    self._print_client_stat(f"[diffusion] request#{request_idx} success in {response.e2e_latency:.3f}s")
                else:
                    self._print_client_stat(f"[diffusion] request#{request_idx} completed")
                responses.append(response)
        return responses

    def send_video_diffusion_request(
        self, request_config: dict[str, Any], request_num: int = 1
    ) -> list[DiffusionResponse]:
        """
        Send native /v1/videos requests: multipart ``form_data`` job create, poll until done, download content.

        For raw HTTP to video routes without polling, use ``send_videos_create_http_request``, etc.
        """
        if request_num != 1:
            raise NotImplementedError("Concurrent video diffusion requests are not currently implemented")

        form_data = request_config.get("form_data")
        if not isinstance(form_data, dict):
            raise ValueError("Video request_config must contain 'form_data'")
        normalized_form_data = {key: str(value) for key, value in form_data.items() if value is not None}
        files: dict[str, tuple[str, BytesIO, str]] = {}
        image_reference = request_config.get("image_reference")
        video_reference = request_config.get("video_reference")
        if image_reference and video_reference:
            raise ValueError("Only one of image_reference or video_reference can be provided")
        if image_reference:
            if image_reference.startswith("data:image"):
                header, encoded = image_reference.split(",", 1)
                content_type = header.split(";")[0].removeprefix("data:")
                extension = content_type.split("/")[-1]
                file_data = base64.b64decode(encoded)
                files["input_reference"] = (f"reference.{extension}", BytesIO(file_data), content_type)
            else:
                normalized_form_data["image_reference"] = json.dumps({"image_url": image_reference})
        if video_reference:
            if video_reference.startswith("data:video"):
                header, encoded = video_reference.split(",", 1)
                content_type = header.split(";")[0].removeprefix("data:")
                extension = content_type.split("/")[-1]
                file_data = base64.b64decode(encoded)
                files["input_reference"] = (f"reference.{extension}", BytesIO(file_data), content_type)
            else:
                normalized_form_data["video_reference"] = json.dumps({"video_url": video_reference})

        result = DiffusionResponse()
        create_url = self._build_url("/v1/videos")
        response = requests.post(
            create_url,
            data=normalized_form_data,
            files=files,
            headers={"Accept": "application/json"},
            timeout=60,
        )
        start_time = time.perf_counter()
        response.raise_for_status()
        job_data = response.json()
        video_id = job_data["id"]
        self._wait_until_video_completed(video_id)
        end_time = time.perf_counter()
        video_content = self._download_video_content(video_id)
        result.success = True
        result.videos = [video_content]
        result.e2e_latency = end_time - start_time
        assert_diffusion_response(result, request_config, run_level=self.run_level)
        if result.e2e_latency is not None:
            self._print_client_stat(f"[diffusion] request#1 success in {result.e2e_latency:.3f}s")
        else:
            self._print_client_stat("[diffusion] request#1 completed")
        return [result]

    def _post_json_endpoint(
        self,
        path: str,
        request_config: dict[str, Any],
        *,
        default_timeout: float,
    ) -> requests.Response:
        url = self._build_url(path)
        timeout = float(request_config.get("timeout", default_timeout))
        if "raw_body" in request_config:
            raw = request_config["raw_body"]
            payload = raw.encode("utf-8") if isinstance(raw, str) else raw
            return requests.post(
                url,
                data=payload,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                timeout=timeout,
            )
        if "json" not in request_config:
            raise ValueError(f"{path} request_config must include 'json' or 'raw_body'")
        return requests.post(
            url,
            json=request_config["json"],
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            timeout=timeout,
        )

    def _post_form_endpoint(
        self,
        path: str,
        request_config: dict[str, Any],
        *,
        default_timeout: float = 120.0,
    ) -> requests.Response:
        url = self._build_url(path)
        timeout = float(request_config.get("timeout", default_timeout))
        data = request_config.get("data")
        files = request_config.get("files")
        if data is None and not files:
            data = {}
        return requests.post(
            url,
            data=data,
            files=files,
            headers={"Accept": "application/json"} if not files else {"Accept": "application/json"},
            timeout=timeout,
        )

    def send_streaming_video_diffusion_request(
        self,
        request_config: dict[str, Any],
        request_num: int = 1,
        *,
        timeout_seconds: float = 600.0,
    ) -> list[DiffusionResponse]:
        """
        Send a native ``/v1/realtime/video`` WebSocket request and return one
        finalized MP4 artifact assembled from the streamed binary fragments.
        """
        if request_num != 1:
            raise NotImplementedError("Concurrent streaming video diffusion requests are not currently implemented")

        response = asyncio.run(
            self._send_streaming_video_diffusion_request_once(
                request_config,
                timeout_seconds=timeout_seconds,
            )
        )
        assert_diffusion_response(response, request_config, run_level=self.run_level)
        if response.e2e_latency is not None:
            self._print_client_stat(f"[diffusion.stream] request#1 success in {response.e2e_latency:.3f}s")
        else:
            self._print_client_stat("[diffusion.stream] request#1 completed")
        return [response]

    async def _send_streaming_video_diffusion_request_once(
        self,
        request_config: dict[str, Any],
        *,
        timeout_seconds: float,
    ) -> DiffusionResponse:
        form_data = request_config.get("form_data")
        if not isinstance(form_data, dict):
            raise ValueError("Video request_config must contain 'form_data'")
        payload: dict[str, Any] = {
            "type": "session.start",
            **{key: value for key, value in form_data.items() if value is not None},
        }
        model = request_config.get("model")
        if model is not None:
            payload["model"] = model
        payload.setdefault("format", "m4s")

        fps = float(payload.get("fps") or 16)
        stream_format = payload["format"]
        url = self._build_ws_url("/v1/realtime/video")

        result = DiffusionResponse()
        chunks: list[bytes] = []
        start_time = time.perf_counter()
        deadline = start_time + timeout_seconds

        import websockets

        async with websockets.connect(url, max_size=None) as websocket:
            await websocket.send(json.dumps(payload))

            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError(f"Streaming video request did not complete within {timeout_seconds}s")

                message = await asyncio.wait_for(websocket.recv(), timeout=remaining)
                if isinstance(message, bytes):
                    chunks.append(message)
                    continue

                msg = json.loads(message)
                msg_type = msg.get("type")
                if msg_type == "video.start":
                    stream_format = msg.get("format") or stream_format
                    continue
                if msg_type == "session.done":
                    break
                if msg_type == "error":
                    raise RuntimeError(str(msg.get("message", msg)))

        from vllm_omni.diffusion.utils.media_utils import finalize_streaming_video_bytes
        from vllm_omni.entrypoints.openai.video_api_utils import StreamingVideoFormat

        streamed_bytes = b"".join(chunks)
        if not streamed_bytes:
            raise RuntimeError("Streaming video request completed without binary video chunks")
        result.videos = [
            finalize_streaming_video_bytes(
                streamed_bytes,
                input_format=cast(StreamingVideoFormat, stream_format),
                fps=fps,
            )
        ]
        result.e2e_latency = time.perf_counter() - start_time
        result.success = True
        return result

    def _wait_until_video_completed(
        self, video_id: str, poll_interval_seconds: int = 2, timeout_seconds: int = 300
    ) -> None:
        status_url = self._build_url(f"/v1/videos/{video_id}")
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            status_resp = requests.get(status_url, headers={"Accept": "application/json"}, timeout=30)
            status_resp.raise_for_status()
            status_data = status_resp.json()
            current_status = status_data["status"]
            if current_status == "completed":
                return
            if current_status == "failed":
                error_msg = status_data.get("last_error", "Unknown error")
                raise RuntimeError(f"Job failed: {error_msg}")
            time.sleep(poll_interval_seconds)
        raise TimeoutError(f"Video job {video_id} did not complete within {timeout_seconds}s")

    def _download_video_content(self, video_id: str) -> bytes:
        download_url = self._build_url(f"/v1/videos/{video_id}/content")
        video_resp = requests.get(download_url, stream=True, timeout=60)
        video_resp.raise_for_status()
        video_bytes = BytesIO()
        for chunk in video_resp.iter_content(chunk_size=8192):
            if chunk:
                video_bytes.write(chunk)
        return video_bytes.getvalue()

    def _build_url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"


class OfflineOmniClient:
    def __init__(self, omni_runner: OmniRunner):
        self.runner = omni_runner

    def _process_omni_output(self, outputs: list[OmniRequestOutput]) -> OmniResponse:
        result = OmniResponse()
        try:
            text_content = None
            audio_content = None
            for stage_output in outputs:
                if getattr(stage_output, "final_output_type", None) == "text":
                    text_content = stage_output.outputs[0].text
                if getattr(stage_output, "final_output_type", None) == "audio":
                    audio_content = stage_output.outputs[0].multimodal_output["audio"]
            result.audio_content = audio_content
            result.text_content = text_content
            result.success = True
        except Exception as e:
            msg = f"Output processing error: {str(e)}"
            result.success = False
            print(f"Error: {msg}")
        return result

    def _process_diffusion_output(self, outputs: list[OmniRequestOutput]) -> DiffusionResponse:
        result = DiffusionResponse()
        output = outputs[0]
        if isinstance(output.images[0], list):
            # Returning frames of images as a video
            result.videos = output.images
        else:
            # Returning actual images
            result.images = output.images
        # [TODO] Add audio processing when tests are introduced
        result.success = True
        return result

    def send_omni_request(self, request_config: dict[str, Any] | None = None) -> OmniResponse:
        if request_config is None:
            request_config = {}
        prompts_value = request_config.get("prompts")
        prompts: str | list[str] = prompts_value if isinstance(prompts_value, (str, list)) else ""
        videos = request_config.get("videos")
        images = request_config.get("images")
        audios = request_config.get("audios")
        modalities = request_config.get("modalities", ["text", "audio"])
        outputs = self.runner.generate_multimodal(
            prompts=prompts, videos=videos, images=images, audios=audios, modalities=modalities
        )
        response = self._process_omni_output(outputs)
        assert_omni_response(response, request_config, run_level="core_model")
        return response

    def send_diffusion_request(self, request_config: dict[str, Any]) -> DiffusionResponse:
        prompt = request_config.get("prompt")
        if prompt is None:
            prompts = request_config.get("prompts")
            if not prompts:
                raise ValueError("request_config must contain a prompt")
            if len(prompts) > 1:
                raise ValueError(
                    "In the current internal data structure, "
                    "only one prompt is supported for diffusion requests. "
                    "Because one prompt can contain multiple images or videos, "
                    "the current internal data structure is ambiguous when multiple prompts are provided."
                )
            prompt = prompts[0]

        # Sync previous `extra_body` field with sampling params object
        sampling_params: OmniDiffusionSamplingParams | None = request_config.get("sampling_params")
        if sampling_params is None:
            extra_body = request_config.get("extra_body", {})
            if extra_body:
                sampling_params = OmniDiffusionSamplingParams(**extra_body)
        else:
            extra_body = asdict(sampling_params)
            request_config["extra_body"] = extra_body
        if not extra_body:
            logger.warning("No sampling params provided in request_config, will skip output assertion")

        negative_prompt = extra_body.get("negative_prompt") or request_config.get("negative_prompt")
        videos = request_config.get("videos")
        images = request_config.get("images")
        audios = request_config.get("audios")
        # Full dict (e.g. image + mask_image for inpainting) or partial; merged with top-level image/video keys.
        extra_multi_modal = request_config.get("multi_modal_data")
        modalities = request_config.get("modalities")  # only used by limited models. Do not add default value here

        prompt_object = OmniTextPrompt(prompt=prompt)
        if negative_prompt:
            prompt_object["negative_prompt"] = negative_prompt
        multi_modal: dict = {}
        if extra_multi_modal is not None:
            multi_modal.update(dict(extra_multi_modal))
        if videos is not None:
            multi_modal["video"] = videos
        if images is not None:
            multi_modal["image"] = images
        if audios is not None:
            multi_modal["audio"] = audios
        if multi_modal:
            prompt_object["multi_modal_data"] = multi_modal
        if modalities:
            prompt_object["modalities"] = modalities  # pyright: ignore[reportGeneralTypeIssues]

        start_time = time.perf_counter()
        raw_outputs = self.runner.generate([prompt_object], [sampling_params] if sampling_params else None)
        end_time = time.perf_counter()
        if not isinstance(raw_outputs, list):
            raw_outputs = list(raw_outputs)

        response = self._process_diffusion_output(raw_outputs)
        response.e2e_latency = end_time - start_time
        assert_diffusion_response(response, request_config, run_level="core_model")
        return response

    def send_audio_speech_request(self, request_config: dict[str, Any]) -> OmniResponse:
        """
        Offline TTS: text -> audio via generate_multimodal, then validate with assert_audio_speech_response.

        request_config must contain:
          - 'input' or 'prompts': text to synthesize.
        Optional keys:
          - 'voice'       -> speaker (CustomVoice)
          - 'task_type'   -> task_type in additional_information (default: "CustomVoice")
          - 'language'    -> language in additional_information (default: "Auto")
          - 'max_new_tokens' -> max_new_tokens in additional_information (default: 2048)
          - 'response_format' -> desired audio format (used only for assertion)
        """
        input_text = request_config.get("input") or request_config.get("prompts")
        if input_text is None:
            raise ValueError("request_config must contain 'input' or 'prompts' for TTS")
        if isinstance(input_text, list):
            input_text = input_text[0] if input_text else ""

        mm_processor_kwargs: dict[str, Any] = {}
        if "voice" in request_config:
            mm_processor_kwargs["speaker"] = request_config["voice"]
        if "task_type" in request_config:
            mm_processor_kwargs["task_type"] = request_config["task_type"]
        if "ref_audio" in request_config:
            mm_processor_kwargs["ref_audio"] = request_config["ref_audio"]
        if "ref_text" in request_config:
            mm_processor_kwargs["ref_text"] = request_config["ref_text"]
        if "language" in request_config:
            mm_processor_kwargs["language"] = request_config["language"]
        if "max_new_tokens" in request_config:
            mm_processor_kwargs["max_new_tokens"] = request_config["max_new_tokens"]

        outputs = self.runner.generate_multimodal(
            prompts=input_text,
            modalities=["audio"],
            mm_processor_kwargs=mm_processor_kwargs or None,
        )
        mm_out: dict[str, Any] | None = None
        for stage_out in outputs:
            if getattr(stage_out, "final_output_type", None) == "audio":
                mm_out = stage_out.outputs[0].multimodal_output
                break
        if mm_out is None:
            raise AssertionError("No audio output from pipeline")

        audio_data = mm_out.get("audio")
        if audio_data is None:
            raise AssertionError("No audio tensor in multimodal output")

        sr_raw = mm_out.get("sr")
        sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
        sr = int(sr_val.item() if hasattr(sr_val, "item") else sr_val)
        wav_tensor = torch.cat(audio_data, dim=-1) if isinstance(audio_data, list) else audio_data
        wav_buf = io.BytesIO()
        sf.write(
            wav_buf,
            wav_tensor.float().cpu().numpy().reshape(-1),
            samplerate=sr,
            format="WAV",
            subtype="PCM_16",
        )
        result = OmniResponse(success=True, audio_bytes=wav_buf.getvalue(), audio_format="audio/wav")
        assert_audio_speech_response(result, request_config, run_level="core_model")
        return result

    def start_profile(self, profile_prefix: str | None = None, stages: list[int] | None = None) -> list[Any]:
        return self.runner.start_profile(profile_prefix=profile_prefix, stages=stages)

    def stop_profile(self, stages: list[int] | None = None) -> list[Any]:
        return self.runner.stop_profile(stages=stages)


# Backward-compatible aliases (historical test fixture / type names).
OpenAIClientHandler = OnlineOmniClient
OmniRunnerHandler = OfflineOmniClient


__all__ = [
    "DiffusionResponse",
    "HttpResponse",
    "WebSocketJsonResponse",
    "OpenPIWebSocketResponse",
    "build_openpi_droid_observation",
    "build_dreamzero_demo_observations",
    "DREAMZERO_ACTION_DIM",
    "DREAMZERO_ACTION_HORIZON",
    "DREAMZERO_CAMERA_FILES",
    "DREAMZERO_DEFAULT_PROMPT",
    "load_dreamzero_camera_frames",
    "OpenPIWebSocketSession",
    "OmniResponse",
    "OfflineOmniClient",
    "OnlineOmniClient",
    "OpenAIClientHandler",
    "OmniRunnerHandler",
]
