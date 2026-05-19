#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Minimal OpenPI client for the GR00T-N1.7 policy server.

Sends one **raw** observation (un-normalized proprio state, raw RGB
images, plain-text prompt) to ``/v1/realtime/robot/openpi`` and validates
that the server returns a per-action-key dict of **absolute** ndarrays
(issue #3553 ``multimodal_output["actions"]`` dict shape).

The client carries no GR00T-specific state-of-the-checkpoint logic — no
q01/q99 stats, no Qwen3VL processor, no SO(3) composition.  Server-side
``transform.encode`` normalizes the obs before inference and
``transform.decode`` denormalizes + reconstructs absolute poses
(SE(3) compose for ``eef_9d``) before responding.

Usage:
    python examples/online_serving/gr00t/openpi_client.py \\
        --host 127.0.0.1 --port 8000 \\
        --embodiment oxe_droid_relative_eef_relative_joint

Dependencies (install separately): openpi-client, websockets.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np

try:
    import websockets.sync.client
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError(
        "GR00T OpenPI example requires `websockets`. "
        "Install with `pip install websockets`."
    ) from exc

# This script's filename ``openpi_client.py`` shadows the same-named
# package on `sys.path[0]`; pop the example dir before importing so the
# real `openpi_client` package wins, then restore it (mirrors the
# DreamZero example).
try:
    example_dir = str(Path(__file__).resolve().parent)
    removed_path = False
    if sys.path and sys.path[0] == example_dir:
        sys.path.pop(0)
        removed_path = True
    try:
        from openpi_client import msgpack_numpy
    finally:
        if removed_path:
            sys.path.insert(0, example_dir)
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError(
        "GR00T OpenPI example requires `openpi-client`. "
        "Install with `pip install openpi-client`."
    ) from exc

PING_INTERVAL_SECS = 300
PING_TIMEOUT_SECS = 3600
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_PATH = "/v1/realtime/robot/openpi"
DEFAULT_PROMPT = "pick up the cube and place it on the plate"
DEFAULT_EMBODIMENT = "oxe_droid_relative_eef_relative_joint"


# DROID-style modality: clients can override per dataset if needed.
DEFAULT_MODALITY_CONFIG = {
    "state": {
        "eef_9d": {"start": 0, "end": 9},
        "gripper_position": {"start": 9, "end": 10},
        "joint_position": {"start": 10, "end": 17},
    },
    "action": {
        "eef_9d": {"start": 0, "end": 9},
        "gripper_position": {"start": 9, "end": 10},
        "joint_position": {"start": 10, "end": 17},
    },
}


def _build_synthetic_observation(
    embodiment: str,
    prompt: str,
    modality_config: dict[str, Any],
    image_resolution: tuple[int, int],
    n_external_cameras: int,
    needs_wrist_camera: bool,
) -> dict[str, Any]:
    """Construct a self-contained dummy observation.

    Replace with a real dataset loader (e.g. LeRobotEpisodeLoader) for
    accuracy-style evaluation; this helper is for smoke testing the
    OpenPI protocol path.
    """
    rng = np.random.default_rng(0)
    h, w = image_resolution
    images: dict[str, np.ndarray] = {}
    for i in range(n_external_cameras):
        images[f"external_{i}"] = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    if needs_wrist_camera:
        images["wrist"] = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    return {
        "session_id": str(uuid.uuid4()),
        "embodiment": embodiment,
        "modality_config": modality_config,
        "prompt": prompt,
        "state": {
            "eef_9d": [0.0] * 9,
            "gripper_position": [0.5],
            "joint_position": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        },
        "images": images,
    }


def _validate_action_response(
    actions: Any,
    expected_horizon: int,
    expected_keys: list[str],
) -> None:
    """Server response must be a dict ``{key: ndarray}`` with the right
    horizon length on each value."""
    if not isinstance(actions, dict):
        raise RuntimeError(
            f"GR00T policy must return a dict of ndarrays per issue #3553; "
            f"got {type(actions).__name__}"
        )
    missing = [key for key in expected_keys if key not in actions]
    if missing:
        raise RuntimeError(
            f"Missing action keys: {missing}; "
            f"server returned {sorted(actions.keys())}"
        )
    for key, value in actions.items():
        if not isinstance(value, np.ndarray):
            raise RuntimeError(
                f"actions[{key}] must be ndarray; got {type(value).__name__}"
            )
        if value.shape[0] != expected_horizon:
            raise RuntimeError(
                f"actions[{key}] has horizon {value.shape[0]}, "
                f"expected {expected_horizon}"
            )
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"actions[{key}] contains non-finite values")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--path", default=DEFAULT_PATH)
    parser.add_argument("--embodiment", default=DEFAULT_EMBODIMENT)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--modality-config",
        type=str,
        default=None,
        help="Optional path to a JSON file with state/action modality config",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("gr00t-openpi-client")

    if args.modality_config:
        with open(args.modality_config) as f:
            modality_config = json.load(f)
    else:
        modality_config = DEFAULT_MODALITY_CONFIG

    url = f"ws://{args.host}:{args.port}{args.path}"
    log.info("Connecting to %s", url)
    with websockets.sync.client.connect(
        url,
        ping_interval=PING_INTERVAL_SECS,
        ping_timeout=PING_TIMEOUT_SECS,
        max_size=64 * 1024 * 1024,
    ) as conn:
        metadata = msgpack_numpy.unpackb(conn.recv())
        log.info("Server metadata: %s", json.dumps(metadata, default=str))
        action_horizon = int(metadata["action_horizon"])
        action_keys = list(metadata["action_keys"])
        image_resolution = tuple(metadata.get("image_resolution", (256, 256)))
        n_external = int(metadata.get("n_external_cameras", 1))
        needs_wrist = bool(metadata.get("needs_wrist_camera", False))

        obs = _build_synthetic_observation(
            embodiment=args.embodiment,
            prompt=args.prompt,
            modality_config=modality_config,
            image_resolution=image_resolution,
            n_external_cameras=n_external,
            needs_wrist_camera=needs_wrist,
        )
        conn.send(msgpack_numpy.packb(obs))
        actions = msgpack_numpy.unpackb(conn.recv())

        _validate_action_response(actions, action_horizon, action_keys)
        for key, value in actions.items():
            log.info(
                "actions[%s]: shape=%s, first row=%s",
                key,
                value.shape,
                np.round(value[0], 4).tolist(),
            )

    log.info("OK: GR00T OpenPI roundtrip completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
