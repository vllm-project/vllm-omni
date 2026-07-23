# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""L3 real-weight online expansion for ``/v1/realtime/world/camera``."""

from __future__ import annotations

from typing import Any

import numpy as np
import PIL.Image
import pytest

from tests.helpers.lingbot_world_fast import (
    FPS,
    GREAT_WALL_PROMPT,
    HEIGHT,
    LONG_NUM_FRAMES,
    SEED,
    SHORT_NUM_FRAMES,
    SSIM_THRESHOLD,
    WIDTH,
    find_lingbot_world_fast_assets,
    frame_ssim,
    golden_frames_dir,
    load_camera_trajectory,
    slice_camera_chunk,
)
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer

# Optional protocol deps mirror what the connection itself imports lazily.
msgpack_numpy = pytest.importorskip("openpi_client.msgpack_numpy")
ws_sync = pytest.importorskip("websockets.sync.client")

pytestmark = [
    pytest.mark.advanced_model,
    pytest.mark.core_model,
    pytest.mark.diffusion,
]

_CONNECT_KWARGS = {"max_size": None, "ping_interval": None, "ping_timeout": None}


# ---------------------------------------------------------------------------
# Asset / golden fixtures (module-scoped to amortize file IO)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lingbot_world_fast_assets():
    assets = find_lingbot_world_fast_assets()
    if assets is None:
        pytest.skip(
            "Lingbot-World-Fast L3 assets not available. Set LINGBOT_WORLD_FAST_PATH, "
            "LINGBOT_WORLD_FAST_CAMERA_PATH and LINGBOT_WORLD_FAST_IMAGE.",
        )
    return assets


_LINGBOT_SERVER_ARGS = [
    "--model-class-name",
    "LingbotWorldFastPipeline",
    "--ws-max-size",
    "16777216",  # 16 MiB — matches run_server.sh; large enough for a 480×832 image
    "--ws",
    "wsproto",
    "--stage-init-timeout",
    "6000",
    "--init-timeout",
    "6000",
]


@pytest.fixture(scope="module")
def lingbot_world_fast_server(lingbot_world_fast_assets):
    """Module-scoped real-weight server; amortizes the multi-minute cold load
    across the four protocol tests in this file."""
    with OmniServer(
        str(lingbot_world_fast_assets.weights_path),
        list(_LINGBOT_SERVER_ARGS),
        use_omni=True,
    ) as server:
        yield server


def _ws_url(server: OmniServer) -> str:
    return f"ws://{server.host}:{server.port}/v1/realtime/world/camera"


# ---------------------------------------------------------------------------
# WebSocket helpers
# ---------------------------------------------------------------------------


def _drain_handshake(ws) -> dict[str, Any]:
    handshake = msgpack_numpy.unpackb(ws.recv())
    return handshake


def _send_request(ws, req: dict[str, Any]) -> None:
    ws.send(msgpack_numpy.packb(req))


def _drain_frames_or_error(ws) -> tuple[list[np.ndarray] | None, dict[str, Any] | None, str | None]:
    """Return ``(frames, error, text)``. Exactly one of the three is non-None.

    * ``frames``: list of per-chunk uint8 arrays once ``total`` frames arrive.
    * ``error``: parsed ``{"type": "error", "message": ...}`` payload.
    * ``text``: text-frame reply (e.g. ``"reset successful"``).
    """
    chunks: list[np.ndarray] = []
    total: int | None = None
    while total is None or len(chunks) < total:
        msg = ws.recv()
        if isinstance(msg, str):
            return None, None, msg
        decoded = msgpack_numpy.unpackb(msg)
        if isinstance(decoded, dict) and decoded.get("type") == "error":
            return None, decoded, None
        if not isinstance(decoded, dict) or decoded.get("type") != "frame":
            continue  # ignore unknown
        total = decoded["total"]
        chunks.append(np.asarray(decoded["video"]))
    return chunks, None, None


def _build_request(
    *,
    session_id: str,
    image: np.ndarray | None,
    camera_chunk: dict[str, np.ndarray],
    num_frames: int,
) -> dict[str, Any]:
    req: dict[str, Any] = {
        "prompt": GREAT_WALL_PROMPT,
        "camera": camera_chunk,
        "session_id": session_id,
        "extra_body": {
            "num_frames": num_frames,
            "height": HEIGHT,
            "width": WIDTH,
            "fps": FPS,
            "session_id": session_id,
            "seed": SEED,
        },
    }
    if image is not None:
        req["image"] = image
    return req


# ---------------------------------------------------------------------------
# Test 1: Single session generation
# ---------------------------------------------------------------------------


@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": 1})
@pytest.mark.parametrize("num_frames, length", [(SHORT_NUM_FRAMES, "short"), (LONG_NUM_FRAMES, "long")])
def test_lingbot_world_online_video(
    num_frames,
    length,
    lingbot_world_fast_server,
    lingbot_world_fast_assets,
):
    with ws_sync.connect(_ws_url(lingbot_world_fast_server), **_CONNECT_KWARGS) as ws:
        _drain_handshake(ws)

        image = (
            PIL.Image.open(lingbot_world_fast_assets.image_path)
            .convert("RGB")
            .resize((WIDTH, HEIGHT), PIL.Image.Resampling.LANCZOS)
        )
        image = np.asarray(image)
        poses, intrinsics = load_camera_trajectory(lingbot_world_fast_assets.camera_dir)
        poses = poses[:num_frames]
        intrinsics = intrinsics[:num_frames]

        camera = {"poses": poses, "intrinsics": intrinsics}

        req = _build_request(
            session_id=f"SESSION-ID-{length}",
            image=image,
            camera_chunk=camera,
            num_frames=num_frames,
        )

        _send_request(ws, req)

        chunks, error, text = _drain_frames_or_error(ws)
        assert error is None and text is None, f"Got unexpected control reply: error={error} text={text}"
        assert chunks is not None and chunks, "Returned no frames"

        reassembled = np.concatenate(chunks, axis=0)

    assert reassembled.ndim == 4 and reassembled.shape[0] >= 2, (
        f"Reassembled video has too few frames: {reassembled.shape}"
    )

    first_frame = (reassembled[0] * 255.0).round().astype(np.uint8)
    last_frame = (reassembled[-1] * 255.0).round().astype(np.uint8)

    first_path = golden_frames_dir() / f"golden_frame_{length}_first.npy"
    last_path = golden_frames_dir() / f"golden_frame_{length}_last.npy"

    first_golden = np.load(first_path)
    last_golden = np.load(last_path)

    ssim_first = frame_ssim(first_frame, first_golden)
    ssim_last = frame_ssim(last_frame, last_golden)
    print(
        f"[lingbot-world-fast L3 online] SSIM(first)={ssim_first:.4f} "
        f"SSIM(last)={ssim_last:.4f} (threshold {SSIM_THRESHOLD})"
    )
    assert ssim_first >= SSIM_THRESHOLD, (
        f"First-frame SSIM {ssim_first:.4f} below {SSIM_THRESHOLD}: regression in fresh-call path."
    )
    assert ssim_last >= SSIM_THRESHOLD, (
        f"Last-frame SSIM {ssim_last:.4f} below {SSIM_THRESHOLD}: regression in extension-call path."
    )


# ---------------------------------------------------------------------------
# Test 2: Session-id churn mid-stream
# ---------------------------------------------------------------------------


@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": 1})
def test_websocket_session_id_churn_resets_state(
    lingbot_world_fast_server,
    lingbot_world_fast_assets,
):
    """A new ``session_id`` mid-stream resets pipeline state. The next ``infer``
    that omits the image (i.e. an "extension-style" payload) must error
    because the new session is fresh."""
    poses, intrinsics = load_camera_trajectory(lingbot_world_fast_assets.camera_dir)
    camera_a = slice_camera_chunk(poses, intrinsics, call_index=0)
    camera_b = slice_camera_chunk(poses, intrinsics, call_index=1)

    image = (
        PIL.Image.open(lingbot_world_fast_assets.image_path)
        .convert("RGB")
        .resize((WIDTH, HEIGHT), PIL.Image.Resampling.LANCZOS)
    )
    image = np.asarray(image)

    with ws_sync.connect(_ws_url(lingbot_world_fast_server), **_CONNECT_KWARGS) as ws:
        _drain_handshake(ws)

        _send_request(
            ws,
            _build_request(
                session_id="churn-session-a",
                image=image,
                camera_chunk=camera_a,
                num_frames=SHORT_NUM_FRAMES,
            ),
        )
        chunks, error, text = _drain_frames_or_error(ws)
        assert chunks is not None and not error and not text, (
            f"First infer on session-a should succeed; got error={error} text={text}"
        )

        # Switch session_id mid-stream WITHOUT sending an image. The pipeline
        # treats this as a fresh call (new session) and rejects.
        _send_request(
            ws,
            _build_request(
                session_id="churn-session-b",
                image=None,
                camera_chunk=camera_b,
                num_frames=SHORT_NUM_FRAMES,
            ),
        )
        chunks2, error2, text2 = _drain_frames_or_error(ws)
        assert error2 is not None, (
            "Server must reject a fresh session that omits ``image``; got "
            f"frames={None if chunks2 is None else len(chunks2)} text={text2}"
        )
        assert error2.get("type") == "error"


# ---------------------------------------------------------------------------
# Test 3: Mid-session ``reset`` RPC re-initializes
# ---------------------------------------------------------------------------


@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": 1})
def test_websocket_mid_session_reset_reinitializes(
    lingbot_world_fast_server,
    lingbot_world_fast_assets,
):
    """After a ``reset`` text ack, the next ``infer`` with the same ``session_id``
    is a brand-new fresh call. We verify this by asserting that the
    follow-up ``infer`` *without* an image errors (same logic as the
    session-id churn test)."""
    poses, intrinsics = load_camera_trajectory(lingbot_world_fast_assets.camera_dir)
    camera_a = slice_camera_chunk(poses, intrinsics, call_index=0)
    camera_b = slice_camera_chunk(poses, intrinsics, call_index=1)

    image = (
        PIL.Image.open(lingbot_world_fast_assets.image_path)
        .convert("RGB")
        .resize((WIDTH, HEIGHT), PIL.Image.Resampling.LANCZOS)
    )
    image = np.asarray(image)

    with ws_sync.connect(_ws_url(lingbot_world_fast_server), **_CONNECT_KWARGS) as ws:
        _drain_handshake(ws)

        _send_request(
            ws,
            _build_request(
                session_id="reset-session",
                image=image,
                camera_chunk=camera_a,
                num_frames=SHORT_NUM_FRAMES,
            ),
        )
        chunks, error, text = _drain_frames_or_error(ws)
        assert chunks is not None and not error and not text, (
            f"Initial infer must succeed; got error={error} text={text}"
        )

        # Mid-session reset RPC.
        ws.send(msgpack_numpy.packb({"endpoint": "reset"}))
        _, _, reset_text = _drain_frames_or_error(ws)
        assert reset_text == "reset successful", f"Expected 'reset successful' text frame, got {reset_text!r}"

        # Same session_id, no image → fresh-call branch → server error.
        _send_request(
            ws,
            _build_request(
                session_id="reset-session",
                image=None,
                camera_chunk=camera_b,
                num_frames=SHORT_NUM_FRAMES,
            ),
        )
        _, post_reset_error, post_reset_text = _drain_frames_or_error(ws)
        assert post_reset_error is not None, (
            "After mid-session reset the server must treat the next infer as a fresh call; "
            f"missing-image payload should error. Got text={post_reset_text!r}"
        )
