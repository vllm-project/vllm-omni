#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal OpenPI client for the π0 (Pi-Zero) VLA policy server.

Connects to a running π0 server (see README.md), sends a single robot
observation (3 cameras + state + language instruction), and prints the returned
continuous action chunk ``[action_horizon, action_dim]``.

The server speaks the OpenPI websocket protocol:
    connect -> server sends msgpack(policy_server_config metadata)
    infer   -> client sends msgpack(obs), server replies msgpack(actions ndarray)

Usage:
    python openpi_client.py --host 127.0.0.1 --port 8000 \
        --prompt "pick up the red block"
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np

try:
    # Avoid shadowing the installed ``openpi_client`` package with this dir.
    example_dir = str(Path(__file__).resolve().parent)
    removed = sys.path and sys.path[0] == example_dir
    if removed:
        sys.path.pop(0)
    try:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
    finally:
        if removed:
            sys.path.insert(0, example_dir)
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError("π0 OpenPI example requires `openpi-client`.") from exc

logger = logging.getLogger("pi0_openpi_client")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_PATH = "/v1/realtime/robot/openpi"
DEFAULT_PROMPT = "pick up the red block and place it in the bin"

# π0 / pi0_base camera identities (must match the server's image_feature_keys).
CAMERA_KEYS = (
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
)
STATE_DIM = 32


def _make_dummy_obs(prompt: str, session_id: str, image_size: int = 224) -> dict[str, Any]:
    """Build a single π0 observation with blank cameras + zero state.

    Replace the zeros with real camera frames (HWC uint8) and proprioceptive
    state to drive an actual robot.
    """
    obs: dict[str, Any] = {cam: np.zeros((image_size, image_size, 3), dtype=np.uint8) for cam in CAMERA_KEYS}
    obs["state"] = np.zeros(STATE_DIM, dtype=np.float32)
    obs["prompt"] = prompt
    obs["session_id"] = session_id
    return obs


def _ws_uri(host: str, port: int, path: str) -> str:
    """Build the full websocket URI. WebsocketClientPolicy takes the whole URI as
    ``host`` (it has no separate path arg)."""
    if host.startswith(("ws://", "wss://")):
        return host
    return f"ws://{host}:{port}{path}"


def _as_action_array(response: Any) -> np.ndarray:
    if isinstance(response, dict) and response.get("type") == "error":
        raise RuntimeError(f"Inference failed: {response.get('message', response)}")
    return np.asarray(response, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="π0 OpenPI client")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--path", default=DEFAULT_PATH)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--num-steps", type=int, default=2, help="Inference observations to send.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    policy = WebsocketClientPolicy(host=_ws_uri(args.host, args.port, args.path))
    metadata = dict(policy.get_server_metadata())
    logger.info("Server metadata: %s", metadata)

    session_id = str(uuid.uuid4())
    for step in range(args.num_steps):
        obs = _make_dummy_obs(args.prompt, session_id)
        actions = _as_action_array(policy.infer(obs))
        logger.info(
            "[step %d] actions shape=%s mean=%.4f std=%.4f",
            step,
            actions.shape,
            float(actions.mean()),
            float(actions.std()),
        )
        if not np.isfinite(actions).all():
            raise RuntimeError("Server returned non-finite actions.")
    logger.info("Done — π0 OpenPI round-trip OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
