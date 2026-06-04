#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run one or more DROID sim-eval rollouts against the vLLM DreamZero server.

This script is the vLLM/OpenPI adaptation of the upstream DreamZero
`eval_utils/run_sim_eval.py` sim-eval client.

Behavior intentionally kept close to upstream:
- same DROID observation extraction (`external_cam`, `external_cam_2`,
  `wrist_cam`, joint position, gripper position)
- same resize-with-pad preprocessing to `(180, 320)`
- same `open_loop_horizon=8`
- same gripper binarization rule (`> 0.5 -> 1`, else `0`)
- same per-scene language prompts

Unlike upstream DreamZero, vLLM serves the compatible websocket policy endpoint
at `/v1/realtime/robot/openpi`, so this script includes the path suffix in the
client URI.

Run this script through Isaac Lab's launcher from the vLLM-Omni repository
root, for example:

    "${ISAACLAB_LAUNCHER}" -p \
      examples/online_serving/dreamzero/droid_sim_eval_client.py \
      --host 127.0.0.1 \
      --port 8000 \
      --scene 1 \
      --episodes 1 \
      --headless \
      --device cuda:1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import mediapy
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError("Optional DROID sim-eval client requires `mediapy`.") from exc

try:
    from typing import override
except ImportError:
    try:
        from typing_extensions import override
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise ImportError("Optional DROID sim-eval client requires `typing-extensions` on Python < 3.12.") from exc

try:
    import websockets.sync.client
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError("Optional DROID sim-eval client requires `websockets`.") from exc

# NOTE:
# This directory already contains a local file named `openpi_client.py`.
# However, what we want here is the *installed* `openpi_client` package from
# upstream OpenPI, not the sibling example file. When a script is executed
# directly, Python often puts the script directory at `sys.path[0]`, which
# would cause `import openpi_client` to resolve to the local example file and
# create a circular import.
#
# To avoid that ambiguity, temporarily remove the current example directory
# from the front of `sys.path`, import the real package, and then restore the
# path afterwards.
example_dir = str(Path(__file__).resolve().parent)
removed_path = False
if sys.path and sys.path[0] == example_dir:
    sys.path.pop(0)
    removed_path = True
try:
    from openpi_client import image_tools, msgpack_numpy
    from openpi_client.base_policy import BasePolicy
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError("Optional DROID sim-eval client requires the `openpi-client` package.") from exc
finally:
    if removed_path:
        sys.path.insert(0, example_dir)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
#
# These values intentionally mirror the upstream DreamZero sim-eval client
# where possible. The important distinction is:
#
# - ACTION_HORIZON = 24
#     The model returns 24 future actions per inference call.
# - DEFAULT_OPEN_LOOP_HORIZON = 8
#     The sim client only executes the first 8 actions locally before asking
#     the server to replan from a fresh observation.
#
# So a single server call predicts 24x8 actions, but the rollout consumes only
# 8 of them before replanning.
PING_INTERVAL_SECS = 60
PING_TIMEOUT_SECS = 600
DEFAULT_PATH = "/v1/realtime/robot/openpi"
DEFAULT_OPEN_LOOP_HORIZON = 8
ACTION_HORIZON = 24
ACTION_DIM = 8
DEFAULT_OUTPUT_ROOT = Path("runs") / "dreamzero_sim_eval"
SCENE_PROMPTS = {
    1: "put the cube in the bowl",
    2: "pick up the can and put it in the mug",
    3: "put the banana in the bin",
}


def _decode_action_response(response: bytes | str) -> np.ndarray:
    if isinstance(response, str):
        raise RuntimeError(f"Error in inference server:\n{response}")
    decoded = msgpack_numpy.unpackb(response)
    if isinstance(decoded, dict) and decoded.get("type") == "error":
        message = decoded.get("message", decoded)
        raise RuntimeError(f"Error in inference server:\n{message}")
    return np.asarray(decoded, dtype=np.float32)


@dataclass(frozen=True)
class StepRecord:
    """One fully materialized rollout step for later JSON export.

    The `episode_00.json` artifact is intended to be human-readable and
    post-process-friendly. Instead of keeping raw tensors around, each step is
    flattened into plain Python types so it can be serialized directly.
    """

    # Index of this control step within the episode.
    step_index: int
    # Whether this step triggered a fresh model call (as opposed to reusing the
    # cached open-loop chunk from the previous server response).
    used_server_call: bool
    # End-to-end latency of the server call that produced the current chunk.
    # This is `None` on steps that only reuse cached actions.
    chunk_latency_s: float | None
    # The concrete 8-D action sent into the simulator at this step.
    action: list[float]
    # Observed 7-DoF arm joint positions before the next environment step.
    joint_position: list[float]
    # Observed gripper scalar before the next environment step.
    gripper_position: list[float]
    # Reward and termination signals directly returned by the simulator.
    reward: float
    terminated: bool
    truncated: bool
    # Optional scene object positions for downstream debugging / success
    # heuristics. This may be empty if the environment does not expose them.
    scene_objects: dict[str, list[float]]


class OpenPIWebsocketClientPolicy(BasePolicy):
    """Minimal websocket client for the DreamZero/OpenPI policy protocol.

    Protocol shape:
    - connect -> server immediately sends a metadata dict
    - infer   -> send msgpack observation, receive action chunk
    - reset   -> send msgpack reset command, receive confirmation string

    This class intentionally stays very small because the more interesting
    DreamZero-specific behavior lives one layer above, in
    `DreamZeroJointPosClient`.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        path: str = DEFAULT_PATH,
    ) -> None:
        # vLLM serves the robot endpoint under `/v1/realtime/robot/openpi`.
        self._uri = f"ws://{host}:{port}{path}"
        # Upstream protocol uses msgpack with numpy support, not JSON.
        self._packer = msgpack_numpy.Packer()
        # Connect immediately and cache the server handshake metadata.
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> dict[str, Any]:
        """Return a copy of the server handshake metadata."""

        return dict(self._server_metadata)

    def _wait_for_server(self):
        """Connect to the websocket server and read the initial metadata frame."""

        logging.info("Connecting to %s", self._uri)
        conn = websockets.sync.client.connect(
            self._uri,
            compression=None,
            max_size=None,
            ping_interval=PING_INTERVAL_SECS,
            ping_timeout=PING_TIMEOUT_SECS,
        )
        metadata = msgpack_numpy.unpackb(conn.recv())
        if not isinstance(metadata, dict):
            raise TypeError(f"Expected dict metadata from server, got {type(metadata)!r}")
        return conn, metadata

    @override
    def infer(self, obs: dict[str, Any]) -> np.ndarray:
        """Send an inference request and return the decoded action chunk."""

        # Keep the upstream DreamZero/OpenPI convention that the request itself
        # tells the server which logical endpoint is being called.
        payload = dict(obs)
        payload["endpoint"] = "infer"
        self._ws.send(self._packer.pack(payload))
        response = self._ws.recv()
        return _decode_action_response(response)

    @override
    def reset(self, reset_info: dict[str, Any] | None = None) -> str:
        """Tell the server to reset its session-side state."""

        payload = dict(reset_info or {})
        payload["endpoint"] = "reset"
        self._ws.send(self._packer.pack(payload))
        response = self._ws.recv()
        if isinstance(response, str):
            return response
        decoded = msgpack_numpy.unpackb(response)
        if not isinstance(decoded, dict) or decoded.get("status") != "reset successful":
            raise RuntimeError(f"Unexpected reset response: {decoded!r}")
        return str(decoded["status"])

    def close(self) -> None:
        """Close the websocket connection explicitly."""

        self._ws.close()


class DreamZeroJointPosClient:
    """DROID sim-eval client that talks to the vLLM OpenPI websocket server.

    This is the main compatibility layer between:
    - Isaac Lab DROID observations (`obs["policy"][...]`)
    - DreamZero/OpenPI websocket payloads
    - local open-loop action reuse across several simulator steps

    In other words:
        simulator obs -> websocket request -> action chunk -> one action per step
    """

    def __init__(
        self,
        remote_host: str = "127.0.0.1",
        remote_port: int = 8000,
        path: str = DEFAULT_PATH,
        open_loop_horizon: int = DEFAULT_OPEN_LOOP_HORIZON,
    ) -> None:
        # Low-level transport client.
        self.client = OpenPIWebsocketClientPolicy(remote_host, remote_port, path=path)
        # Number of actions to execute locally before replanning.
        self.open_loop_horizon = open_loop_horizon
        # Cursor into the currently cached action chunk.
        self.actions_from_chunk_completed = 0
        # Most recent `(ACTION_HORIZON, ACTION_DIM)` server response.
        self.pred_action_chunk: np.ndarray | None = None
        # Session id is part of the DreamZero serving contract. Changing it
        # causes the server side to treat the rollout as a fresh episode.
        self.session_id = str(uuid.uuid4())
        # Simple runtime stats for reporting.
        self.server_calls = 0
        self.last_chunk_latency_s: float | None = None
        self.last_used_server_call = False

    def metadata(self) -> dict[str, Any]:
        """Expose the server metadata to callers / logs."""

        return self.client.get_server_metadata()

    def reset(self) -> str:
        """Reset local chunk state and remote session state.

        Local reset:
        - drop cached action chunk
        - rewind chunk cursor
        - allocate a fresh session id

        Remote reset:
        - send a websocket `reset` message so the server can clear any
          request/session-side state it associates with this client
        """

        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.session_id = str(uuid.uuid4())
        self.last_chunk_latency_s = None
        self.last_used_server_call = False
        return self.client.reset({})

    def infer(self, obs: dict[str, Any], instruction: str) -> dict[str, Any]:
        """Turn one simulator observation into one executable 8-D action.

        Key behavior:
        - call the server only when the local chunk cache is empty/exhausted
        - otherwise, keep consuming the cached chunk open-loop
        - always return exactly one 8-D action for the current simulator step
        """

        # Convert Isaac Lab observation structure into a plain numpy-friendly
        # record that is easier to serialize and visualize.
        curr_obs = self._extract_observation(obs)
        self.last_used_server_call = False

        # Replan if:
        # 1. this is the first step of a rollout / chunk
        # 2. we already consumed `open_loop_horizon` actions from the current chunk
        # 3. no cached chunk is currently available
        if (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
            or self.pred_action_chunk is None
        ):
            self.actions_from_chunk_completed = 0
            # Build the exact DreamZero/OpenPI payload expected by the server.
            #
            # Notes:
            # - images are resized/padded to the serving contract's 180x320
            # - proprio is cast to float64 to match upstream client behavior
            # - cartesian_position is currently unused by DreamZero DROID, so
            #   a dummy zero vector is sent for protocol completeness
            request_data = {
                "observation/exterior_image_0_left": image_tools.resize_with_pad(curr_obs["right_image"], 180, 320),
                "observation/exterior_image_1_left": image_tools.resize_with_pad(curr_obs["left_image"], 180, 320),
                "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 180, 320),
                "observation/joint_position": curr_obs["joint_position"].astype(np.float64),
                "observation/cartesian_position": np.zeros((6,), dtype=np.float64),
                "observation/gripper_position": curr_obs["gripper_position"].astype(np.float64),
                "prompt": instruction,
                "session_id": self.session_id,
            }

            # Measure end-to-end server latency for this chunk request.
            start = time.perf_counter()
            actions = self.client.infer(request_data)
            self.last_chunk_latency_s = time.perf_counter() - start
            self.last_used_server_call = True
            self.server_calls += 1

            # DreamZero DROID serving is expected to return an action chunk with
            # 24 future actions, each action being 8-D.
            if actions.ndim != 2:
                raise AssertionError(f"Expected 2D action array, got shape {actions.shape}")
            if actions.shape != (ACTION_HORIZON, ACTION_DIM):
                raise AssertionError(f"Expected action shape {(ACTION_HORIZON, ACTION_DIM)}, got {actions.shape}")
            self.pred_action_chunk = actions

        # Consume exactly one action row from the cached chunk for this
        # simulator step.
        action = np.array(self.pred_action_chunk[self.actions_from_chunk_completed], copy=True)
        self.actions_from_chunk_completed += 1

        # Upstream DreamZero sim-eval binarizes the gripper command.
        action[-1] = 1.0 if action[-1].item() > 0.5 else 0.0

        # Produce a human-friendly visualization strip for videos:
        # right external | wrist | left external
        img1 = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
        img2 = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        img3 = image_tools.resize_with_pad(curr_obs["left_image"], 224, 224)
        viz = np.concatenate([img1, img2, img3], axis=1)

        # Return both the executable action and auxiliary debug info.
        return {
            "action": action,
            "viz": viz,
            "joint_position": curr_obs["joint_position"],
            "gripper_position": curr_obs["gripper_position"],
            "used_server_call": self.last_used_server_call,
            "chunk_latency_s": self.last_chunk_latency_s if self.last_used_server_call else None,
        }

    @staticmethod
    def _extract_observation(obs_dict: dict[str, Any]) -> dict[str, np.ndarray]:
        """Extract the pieces DreamZero cares about from Isaac Lab observations.

        `sim-evals` exposes camera frames and robot state inside the
        `obs["policy"]` group. This helper converts those tensors into numpy
        arrays so they can be fed into image preprocessing / websocket packing.
        """

        policy = obs_dict["policy"]
        # Isaac Lab stores camera observations as batched tensors; use env 0.
        right_image = policy["external_cam"][0].clone().detach().cpu().numpy()
        left_image = policy["external_cam_2"][0].clone().detach().cpu().numpy()
        wrist_image = policy["wrist_cam"][0].clone().detach().cpu().numpy()
        # Robot proprioception.
        joint_position = policy["arm_joint_pos"].clone().detach().cpu().numpy()
        gripper_position = policy["gripper_pos"].clone().detach().cpu().numpy()

        return {
            "right_image": right_image,
            "left_image": left_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }


def _scene_instruction(scene: int) -> str:
    """Map a numeric scene id onto the fixed language prompt used for rollout."""

    try:
        return SCENE_PROMPTS[scene]
    except KeyError as exc:
        raise ValueError(f"Unsupported scene {scene}. Available scenes: {sorted(SCENE_PROMPTS)}") from exc


def _capture_scene_objects(env: Any) -> dict[str, list[float]]:
    """Best-effort extraction of scene object root positions.

    This is only for debugging / reporting. The rollout logic itself does not
    depend on these positions.
    """

    objects: dict[str, list[float]] = {}
    scene = getattr(env, "scene", None)
    if scene is None:
        return objects

    # Skip non-task entities such as cameras, the robot, and lighting.
    for name in scene.keys():
        if name in {"robot", "external_cam", "external_cam_2", "wrist_cam", "sphere_light", "scene"}:
            continue
        entity = scene[name]
        data = getattr(entity, "data", None)
        root_pos_w = getattr(data, "root_pos_w", None)
        if root_pos_w is None:
            continue
        # Convert tensors to plain lists for JSON serialization.
        value = root_pos_w[0].detach().cpu().to(torch.float32).tolist()
        objects[str(name)] = [float(x) for x in value]
    return objects


def _maybe_infer_success(scene: int, final_objects: dict[str, list[float]]) -> dict[str, Any]:
    """Best-effort geometric heuristic.

    The simulator itself does not expose a built-in success term; this function
    provides a transparent fallback for human-readable reporting only.
    """

    task_pairs = {
        1: ("cube", "bowl"),
        2: ("can", "mug"),
        3: ("banana", "bin"),
    }
    source_name, target_name = task_pairs.get(scene, (None, None))
    if source_name not in final_objects or target_name not in final_objects:
        return {
            "has_builtin_success": False,
            "heuristic_success": None,
            "reason": "scene object names unavailable for heuristic",
        }

    source = np.asarray(final_objects[source_name], dtype=np.float32)
    target = np.asarray(final_objects[target_name], dtype=np.float32)
    xy_distance = float(np.linalg.norm(source[:2] - target[:2]))
    z_delta = float(source[2] - target[2])
    heuristic_success = bool(xy_distance < 0.12 and z_delta > -0.08)
    return {
        "has_builtin_success": False,
        "heuristic_success": heuristic_success,
        "xy_distance": xy_distance,
        "z_delta": z_delta,
        "source_object": source_name,
        "target_object": target_name,
        "reason": "no built-in env success flag; using final object pose heuristic",
    }


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file with stable UTF-8 formatting."""

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _scalar_from_env_value(value: Any) -> float:
    """Normalize simulator scalar outputs into a plain Python float.

    Isaac Lab / Gym values may come back as tensors, numpy arrays, tuples, or
    direct Python scalars depending on the wrapper stack. Centralizing the
    conversion here makes the rollout loop cleaner and more robust.
    """

    if isinstance(value, torch.Tensor):
        return float(value.reshape(-1)[0].detach().cpu().item())
    if isinstance(value, np.ndarray):
        return float(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        return float(np.asarray(value).reshape(-1)[0])
    return float(value)


def _bool_from_env_value(value: Any) -> bool:
    """Normalize simulator boolean-like outputs into a plain Python bool."""

    if isinstance(value, torch.Tensor):
        return bool(value.reshape(-1)[0].detach().cpu().item())
    if isinstance(value, np.ndarray):
        return bool(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        return bool(np.asarray(value).reshape(-1)[0])
    return bool(value)


def _make_output_dir(output_root: Path, scene: int) -> Path:
    """Create a timestamped output directory for one sim-eval run."""

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / f"scene{scene}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main() -> None:
    """Entry point for one or more DROID simulation rollouts.

    High-level flow:
    1. parse command-line flags
    2. bootstrap Isaac Lab / sim-evals imports
    3. create the DROID environment
    4. connect the DreamZero websocket client
    5. run `episodes` rollouts
    6. export videos + JSON summaries
    """

    # Script-level arguments.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run.")
    parser.add_argument("--scene", type=int, default=1, help="DROID scene id (1/2/3).")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="vLLM DreamZero server host.")
    parser.add_argument("--port", type=int, default=8000, help="vLLM DreamZero server port.")
    parser.add_argument("--path", type=str, default=DEFAULT_PATH, help="Websocket path suffix.")
    parser.add_argument(
        "--open-loop-horizon",
        type=int,
        default=DEFAULT_OPEN_LOOP_HORIZON,
        help="How many actions to consume locally before requesting the next chunk.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where videos and trajectory logs are stored.",
    )

    try:
        from isaaclab.app import AppLauncher
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise ImportError(
            "Optional DROID sim-eval client requires Isaac Lab (`isaaclab`). "
            "Launch it from an Isaac Lab environment, e.g. via `isaaclab.sh -p`."
        ) from exc

    # Let Isaac Lab inject its own runtime flags (e.g. headless, device).
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # DreamZero sim-eval always needs camera observations enabled.
    args.enable_cameras = True
    # Boot Isaac Sim / Isaac Lab.
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    # Set defaults so the `finally` block can clean up safely even if an
    # earlier step fails.
    env = None
    client = None

    # Import simulator modules only *after* the app is launched. This matches
    # Isaac Lab's required import ordering.
    try:
        import gymnasium as gym
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise ImportError("Optional DROID sim-eval client requires `gymnasium`.") from exc

    try:
        import sim_evals.environments  # noqa: F401
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise ImportError(
            "Optional DROID sim-eval client requires the external `sim-evals` package or checkout to be importable."
        ) from exc

    try:
        from isaaclab_tasks.utils import parse_env_cfg
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise ImportError("Optional DROID sim-eval client requires `isaaclab_tasks`.") from exc

    # Resolve output location and scene prompt.
    output_dir = _make_output_dir(args.output_root.expanduser().resolve(), args.scene)
    instruction = _scene_instruction(args.scene)

    # Build the DROID environment configuration from `sim-evals`.
    env_cfg = parse_env_cfg(
        "DROID",
        device=args.device,
        num_envs=1,
        use_fabric=True,
    )
    # Select one of the pre-authored scenes/tasks.
    env_cfg.set_scene(args.scene)
    env = gym.make("DROID", cfg=env_cfg)

    # Upstream sim-evals resets twice so materials / cameras are fully ready.
    obs, _ = env.reset()
    obs, _ = env.reset()

    # Connect the websocket policy client.
    client = DreamZeroJointPosClient(
        remote_host=args.host,
        remote_port=args.port,
        path=args.path,
        open_loop_horizon=args.open_loop_horizon,
    )

    # Aggregated per-run results.
    all_episode_summaries: list[dict[str, Any]] = []
    max_steps = int(env.env.max_episode_length)
    logging.info("DreamZero metadata: %s", client.metadata())
    logging.info("Scene %s prompt: %s", args.scene, instruction)
    logging.info("Writing outputs to %s", output_dir)

    try:
        # No gradients are needed in inference-only rollout mode.
        with torch.no_grad():
            for episode_index in range(args.episodes):
                # Per-episode collectors.
                frames: list[np.ndarray] = []
                step_records: list[StepRecord] = []
                episode_start = time.perf_counter()
                server_time_s = 0.0
                final_reward = 0.0
                terminated = False
                truncated = False

                for step_index in range(max_steps):
                    # Ask the policy for the next action. Internally this may or
                    # may not trigger a real server request depending on whether
                    # the local chunk cache has been exhausted.
                    logging.debug("Episode %d step %d: requesting action", episode_index, step_index)
                    result = client.infer(obs, instruction)
                    logging.debug(
                        "Episode %d step %d: got action (server_call=%s latency=%s)",
                        episode_index,
                        step_index,
                        result["used_server_call"],
                        result["chunk_latency_s"],
                    )
                    # Save one visualization frame per simulator step.
                    frames.append(result["viz"])

                    # Isaac Lab expects batched actions, hence `[None]`.
                    action_tensor = torch.tensor(result["action"], dtype=torch.float32)[None]
                    logging.debug("Episode %d step %d: stepping env", episode_index, step_index)
                    obs, reward, term, trunc, info = env.step(action_tensor)
                    logging.debug("Episode %d step %d: env.step returned", episode_index, step_index)
                    logging.debug("Episode %d step %d: parsing reward/flags", episode_index, step_index)
                    logging.debug(
                        "Episode %d step %d: raw types reward=%s term=%s trunc=%s",
                        episode_index,
                        step_index,
                        type(reward).__name__,
                        type(term).__name__,
                        type(trunc).__name__,
                    )
                    # Normalize environment outputs into plain Python scalars so
                    # the rest of the code does not depend on wrapper-specific types.
                    reward_value = _scalar_from_env_value(reward)
                    term_value = _bool_from_env_value(term)
                    trunc_value = _bool_from_env_value(trunc)
                    logging.debug(
                        "Episode %d step %d: parsed reward=%s term=%s trunc=%s",
                        episode_index,
                        step_index,
                        reward_value,
                        term_value,
                        trunc_value,
                    )
                    # Keep scene-object capture optional. It is useful for
                    # debugging / success heuristics, but the rollout should not
                    # fail if the environment does not expose object roots.
                    scene_objects = _capture_scene_objects(env)

                    # Accumulate total server-side time only on steps that
                    # triggered a fresh chunk inference.
                    if result["chunk_latency_s"] is not None:
                        server_time_s += float(result["chunk_latency_s"])

                    # Materialize one JSON-serializable trajectory record.
                    logging.debug("Episode %d step %d: appending trajectory", episode_index, step_index)
                    step_records.append(
                        StepRecord(
                            step_index=step_index,
                            used_server_call=bool(result["used_server_call"]),
                            chunk_latency_s=(
                                float(result["chunk_latency_s"]) if result["chunk_latency_s"] is not None else None
                            ),
                            action=[float(x) for x in np.asarray(result["action"], dtype=np.float32).tolist()],
                            joint_position=[
                                float(x) for x in np.asarray(result["joint_position"], dtype=np.float32).tolist()
                            ],
                            gripper_position=[
                                float(x) for x in np.asarray(result["gripper_position"], dtype=np.float32).tolist()
                            ],
                            reward=reward_value,
                            terminated=term_value,
                            truncated=trunc_value,
                            scene_objects=scene_objects,
                        )
                    )
                    logging.debug("Episode %d step %d: trajectory appended", episode_index, step_index)

                    # Track final status for summary export.
                    final_reward = reward_value
                    terminated = term_value
                    truncated = trunc_value
                    if term_value or trunc_value:
                        # End the rollout early if the environment terminates.
                        break

                # Episode-level timing and video export.
                episode_wall_time_s = time.perf_counter() - episode_start
                video_path = output_dir / f"episode_{episode_index:02d}.mp4"
                logging.info("Episode %d: writing video to %s", episode_index, video_path)
                mediapy.write_video(video_path, frames, fps=15)

                # Reset the policy server between episodes.
                logging.info("Episode %d: sending reset", episode_index)
                reset_response = client.reset()
                final_objects = step_records[-1].scene_objects if step_records else {}
                success_report = _maybe_infer_success(args.scene, final_objects)

                # Assemble the per-episode summary that is written to
                # `episode_XX.json`.
                episode_summary = {
                    "episode_index": episode_index,
                    "prompt": instruction,
                    "video_path": str(video_path),
                    "steps_executed": len(step_records),
                    "max_steps": max_steps,
                    "terminated": terminated,
                    "truncated": truncated,
                    "final_reward": final_reward,
                    "server_calls": client.server_calls,
                    "server_time_s": server_time_s,
                    "episode_wall_time_s": episode_wall_time_s,
                    "avg_server_time_per_call_s": (
                        server_time_s / client.server_calls if client.server_calls else None
                    ),
                    "reset_response": reset_response,
                    "success_report": success_report,
                    "server_metadata": client.metadata(),
                    "trajectory": [record.__dict__ for record in step_records],
                }
                _dump_json(output_dir / f"episode_{episode_index:02d}.json", episode_summary)
                all_episode_summaries.append(episode_summary)

                logging.info(
                    "Episode %d done: steps=%d wall=%.2fs server_calls=%d heuristic_success=%s",
                    episode_index,
                    len(step_records),
                    episode_wall_time_s,
                    client.server_calls,
                    success_report.get("heuristic_success"),
                )

                # Reset per-episode counters while keeping the client alive.
                client.server_calls = 0

        # Top-level run summary across all episodes.
        summary = {
            "scene": args.scene,
            "prompt": instruction,
            "episodes": args.episodes,
            "host": args.host,
            "port": args.port,
            "path": args.path,
            "device": args.device,
            "output_dir": str(output_dir),
            "server_metadata": client.metadata(),
            "episodes_summary": all_episode_summaries,
        }
        _dump_json(output_dir / "summary.json", summary)
        # Also print the summary to stdout so the caller can capture it in logs.
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        # Best-effort cleanup. Avoid masking the main error if cleanup fails.
        try:
            if client is not None:
                client.client.close()
        except Exception:
            pass
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    # Keep the script-level logs readable. Per-step rollout details are still
    # available via `DEBUG` if needed, but websocket / asyncio internals are
    # usually too noisy for normal usage.
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    main()
