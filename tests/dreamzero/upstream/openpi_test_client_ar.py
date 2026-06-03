#!/usr/bin/env python3
"""
Copied from the DreamZero repository's `test_client_AR.py`:
https://github.com/dreamzero0/dreamzero/blob/main/test_client_AR.py

Kept here for end-to-end compatibility / parity testing against the vLLM
OpenPI server.

Test client for AR_droid policy server using roboarena interface.

Sends real video frames from debug_image/ directory instead of zero dummy images.

Frame schedule (matching debug_inference.py):
  - Step 0 (initial): send frame [0]             (1 frame, H W 3)
  - Step 1: send frames [0, 7, 15, 23]           (4 frames, 4 H W 3)
  - Step 2: send frames [24, 31, 39, 47]         (4 frames)
  - Step 3: send frames [48, 55, 63, 71]         (4 frames)
  - ...

Expected server configuration:
    - image_resolution: (180, 320)
    - n_external_cameras: 2
    - needs_wrist_camera: True
    - action_space: "joint_position"

Usage:
    # Start server with roboarena interface:
    torchrun --nproc_per_node=8 socket_test_optimized_AR.py --port 8000

    # Run against the original DreamZero websocket server (default path):
    python test_client_AR.py --host <server_host> --port 8000

    # Run against the vLLM OpenPI server:
    python test_client_AR.py --host <server_host> --port 8000 --path /v1/realtime/robot/openpi

    # Use zero images instead of real video (old behavior):
    python test_client_AR.py --host <server_host> --port 8000 --use-zero-images
"""

import argparse
import logging
import os
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from openpi_client import msgpack_numpy

_DREAMZERO_REPO_ENV = os.environ.get("DREAMZERO_REPO")
DREAMZERO_REPO = Path(_DREAMZERO_REPO_ENV).expanduser() if _DREAMZERO_REPO_ENV else None


def _import_upstream_policy_modules():
    if DREAMZERO_REPO is None:
        raise ImportError("Set DREAMZERO_REPO to an upstream DreamZero checkout before using this helper.")
    if DREAMZERO_REPO.exists() and str(DREAMZERO_REPO) not in sys.path:
        sys.path.insert(0, str(DREAMZERO_REPO))

    import eval_utils.policy_server as policy_server
    from eval_utils.policy_client import WebsocketClientPolicy

    return policy_server, WebsocketClientPolicy


policy_server, WebsocketClientPolicy = _import_upstream_policy_modules()

VIDEO_DIR = os.environ.get(
    "DREAMZERO_VIDEO_DIR",
    str(DREAMZERO_REPO / "debug_image") if DREAMZERO_REPO is not None else "debug_image",
)

# roboarena key -> video filename
CAMERA_FILES = {
    "observation/exterior_image_0_left": "exterior_image_1_left.mp4",
    "observation/exterior_image_1_left": "exterior_image_2_left.mp4",
    "observation/wrist_image_left": "wrist_image_left.mp4",
}

# Frame schedule constants (matching debug_inference.py)
RELATIVE_OFFSETS = [-23, -16, -8, 0]
ACTION_HORIZON = 24
DEFAULT_WEBSOCKET_PATH = ""


class OpenPIWebsocketClientPolicy(WebsocketClientPolicy):
    """DreamZero websocket client with a configurable path suffix.

    The original DreamZero client connects to ``ws://host:port``.
    vLLM serves the compatible robot policy endpoint at
    ``/v1/realtime/robot/openpi`` when ``path`` is set accordingly.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        path: str = DEFAULT_WEBSOCKET_PATH,
    ) -> None:
        self._uri = f"ws://{host}:{port}{path}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._wait_for_server()


def load_all_frames(video_path: str) -> np.ndarray:
    """Load all frames from a video file. Returns (N, H, W, 3) uint8 array (RGB)."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames loaded from {video_path}")
    return np.stack(frames, axis=0)


def load_camera_frames() -> dict[str, np.ndarray]:
    """Load all video frames for each camera from the debug_image/ directory.

    Returns:
        Dict mapping roboarena camera keys to (N, H, W, 3) uint8 arrays.
    """
    camera_frames: dict[str, np.ndarray] = {}
    for cam_key, fname in CAMERA_FILES.items():
        path = os.path.join(VIDEO_DIR, fname)
        camera_frames[cam_key] = load_all_frames(path)
        logging.info(f"Loaded {cam_key}: {camera_frames[cam_key].shape}")
    return camera_frames


def build_frame_schedule(total_frames: int, num_chunks: int) -> list[list[int]]:
    """Build the frame index schedule for multi-frame chunks.

    Returns a list of frame-index lists. Each inner list has 4 indices.
    """
    chunks: list[list[int]] = []
    current_frame = 23  # first anchor frame
    for _ in range(num_chunks):
        indices = [max(current_frame + off, 0) for off in RELATIVE_OFFSETS]
        if indices[-1] >= total_frames:
            logging.info(f"Frame {indices[-1]} >= {total_frames}, stopping at {len(chunks)} chunks")
            break
        chunks.append(indices)
        current_frame += ACTION_HORIZON
    return chunks


def _make_obs_from_video(
    camera_frames: dict[str, np.ndarray],
    frame_indices: list[int],
    prompt: str,
    session_id: str,
) -> dict:
    """Build an observation dict from real video frames.

    For 1 frame: each image key is (H, W, 3).
    For 4 frames: each image key is (4, H, W, 3).
    """
    obs: dict = {}
    for cam_key, all_frames in camera_frames.items():
        selected = all_frames[frame_indices]  # (T, H, W, 3)
        if len(frame_indices) == 1:
            selected = selected[0]  # (H, W, 3)
        obs[cam_key] = selected

    obs["observation/joint_position"] = np.zeros(7, dtype=np.float32)
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
    obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)
    obs["prompt"] = prompt
    obs["session_id"] = session_id
    return obs


def _make_zero_observation(
    server_config: policy_server.PolicyServerConfig,
    prompt: str = "pick up the object",
    session_id: str | None = None,
) -> dict:
    """Create a dummy observation matching AR_droid expectations.

    AR_droid expects:
        - 2 external cameras (exterior_image_0_left, exterior_image_1_left)
        - 1 wrist camera (wrist_image_left)
        - Image resolution: 180x320 (H x W)
        - joint_position: 7 DoF
        - gripper_position: 1 DoF
    """
    obs = {}

    # Determine image resolution
    if server_config.image_resolution is not None:
        h, w = server_config.image_resolution
    else:
        # Default for AR_droid
        h, w = 180, 320

    # External cameras (0-indexed in roboarena)
    for i in range(server_config.n_external_cameras):
        obs[f"observation/exterior_image_{i}_left"] = np.zeros((h, w, 3), dtype=np.uint8)
        if server_config.needs_stereo_camera:
            obs[f"observation/exterior_image_{i}_right"] = np.zeros((h, w, 3), dtype=np.uint8)

    # Wrist camera
    if server_config.needs_wrist_camera:
        obs["observation/wrist_image_left"] = np.zeros((h, w, 3), dtype=np.uint8)
        if server_config.needs_stereo_camera:
            obs["observation/wrist_image_right"] = np.zeros((h, w, 3), dtype=np.uint8)

    # Session ID - should be passed in to ensure consistency within a session
    if server_config.needs_session_id:
        import uuid

        # Generate unique session ID if not provided
        obs["session_id"] = session_id if session_id else str(uuid.uuid4())

    # State observations (AR_droid: 7 DoF arm + 1 gripper)
    obs["observation/joint_position"] = np.zeros(7, dtype=np.float32)
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
    obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)

    # Language prompt
    obs["prompt"] = prompt

    return obs


def test_ar_droid_policy_server(
    host: str = "localhost",
    port: int = 8000,
    path: str = DEFAULT_WEBSOCKET_PATH,
    num_chunks: int = 15,
    prompt: str = "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan",
    use_zero_images: bool = False,
):
    """Test the AR_droid policy server with roboarena interface.

    When use_zero_images is False (default), loads real video frames from
    debug_image/ and follows the frame schedule from debug_inference.py.
    """
    logging.info(f"Connecting to AR_droid server at ws://{host}:{port}{path} ...")

    client = OpenPIWebsocketClientPolicy(host=host, port=port, path=path)

    # Validate server metadata
    metadata = client.get_server_metadata()
    logging.info(f"Server metadata: {metadata}")
    assert isinstance(metadata, dict), "Metadata should be a dict"

    try:
        server_config = policy_server.PolicyServerConfig(**metadata)
    except Exception as e:
        logging.error(f"Error parsing metadata: {e}")
        raise e

    # Validate expected AR_droid configuration
    logging.info(f"Server config: {server_config}")
    assert server_config.n_external_cameras == 2, f"Expected 2 external cameras, got {server_config.n_external_cameras}"
    assert server_config.needs_wrist_camera, "Expected wrist camera to be enabled"
    assert server_config.action_space == "joint_position", (
        f"Expected joint_position action space, got {server_config.action_space}"
    )

    logging.info("Server configuration validated for AR_droid")

    # Generate unique session ID for this test run
    session_id = str(uuid.uuid4())
    logging.info(f"Session ID: {session_id}")

    # ── Zero-image fallback mode ──────────────────────────────────────
    if use_zero_images:
        logging.info("Using ZERO dummy images (legacy mode)")
        for i in range(num_chunks):
            obs = _make_zero_observation(server_config, prompt=prompt, session_id=session_id)
            logging.info(f"Inference {i + 1}/{num_chunks}: prompt='{prompt}'")
            t0 = time.time()
            actions = client.infer(obs)
            dt = time.time() - t0
            _log_action(actions, dt)

        logging.info("Sending reset...")
        client.reset({})
        logging.info("Done (zero-image mode).")
        return

    # ── Real video frame mode ─────────────────────────────────────────
    logging.info("Loading real video frames from debug_image/ directory")
    camera_frames = load_camera_frames()

    total_frames = min(v.shape[0] for v in camera_frames.values())
    logging.info(f"Total frames available: {total_frames}")

    # Build frame schedule
    chunks = build_frame_schedule(total_frames, num_chunks)

    logging.info("Frame schedule:")
    logging.info("  Initial: [0]")
    for i, indices in enumerate(chunks):
        logging.info(f"  Chunk {i}: {indices}")

    # Step 0: initial single frame
    logging.info("=== Initial: frame [0] ===")
    obs = _make_obs_from_video(camera_frames, [0], prompt, session_id)
    t0 = time.time()
    actions = client.infer(obs)
    dt = time.time() - t0
    _log_action(actions, dt)

    # Subsequent chunks: send 4 frames at a time
    for chunk_idx, frame_indices in enumerate(chunks):
        logging.info(f"=== Chunk {chunk_idx}: frames {frame_indices} ===")
        obs = _make_obs_from_video(camera_frames, frame_indices, prompt, session_id)
        t0 = time.time()
        actions = client.infer(obs)
        dt = time.time() - t0
        _log_action(actions, dt)

    # Reset triggers video save on the server
    logging.info("Sending reset to save video...")
    client.reset({})

    logging.info("Done.")


def _log_action(actions: np.ndarray, dt: float) -> None:
    """Pretty-print action shape, range, and timing."""
    assert isinstance(actions, np.ndarray), f"Expected numpy array, got {type(actions)}"
    assert actions.ndim == 2, f"Expected 2D array, got shape {actions.shape}"
    assert actions.shape[-1] == 8, f"Expected 8 action dims (7 joints + 1 gripper), got {actions.shape[-1]}"
    logging.info(f"  Action shape: {actions.shape}, range: [{actions.min():.4f}, {actions.max():.4f}], time: {dt:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Test AR_droid policy server with real video frames from debug_image/")
    parser.add_argument("--host", default="localhost", help="Server hostname")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument(
        "--path",
        default=DEFAULT_WEBSOCKET_PATH,
        help="WebSocket path suffix (default: empty string for the original DreamZero server)",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=15,
        help="Number of 4-frame chunks to send after the initial frame (default: 15)",
    )
    parser.add_argument(
        "--prompt",
        default="Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan",
        help="Language prompt for the policy",
    )
    parser.add_argument(
        "--use-zero-images",
        action="store_true",
        help="Use zero dummy images instead of real video frames (legacy mode)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    test_ar_droid_policy_server(
        host=args.host,
        port=args.port,
        path=args.path,
        num_chunks=args.num_chunks,
        prompt=args.prompt,
        use_zero_images=args.use_zero_images,
    )


if __name__ == "__main__":
    main()
