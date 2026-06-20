#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vllm_omni.entrypoints.openpi import dreamzero_async_protocol as protocol

try:
    import cv2
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError("DreamZero async example requires `opencv-python`.") from exc

try:
    import websockets
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError("DreamZero async example requires `websockets`.") from exc

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_PATH = "/v1/realtime/robot/dreamzero-async"
DEFAULT_PROMPT = "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan"
DEFAULT_VIDEO_DIR = Path(__file__).resolve().parents[3] / "outputs" / "dreamzero" / "assets"
ACTION_HORIZON = 24
ACTION_DIM = 8
CONTROL_HZ = 15.0
RELATIVE_OFFSETS = [-23, -16, -8, 0]
CAMERA_FILES = {
    "observation/exterior_image_0_left": "exterior_image_1_left.mp4",
    "observation/exterior_image_1_left": "exterior_image_2_left.mp4",
    "observation/wrist_image_left": "wrist_image_left.mp4",
}


@dataclass
class ActionChunk:
    chunk_index: int
    actions: np.ndarray
    provenance: dict[str, Any]
    monitoring: dict[str, Any]
    received_s: float


@dataclass
class ReplayEvent:
    event: str
    t_s: float
    data: dict[str, Any] = field(default_factory=dict)


def _uri(host: str, port: int, path: str) -> str:
    if host.startswith(("ws://", "wss://")):
        return host
    return f"ws://{host}:{port}{path}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _event_dict(event: ReplayEvent, *, t0_s: float | None = None) -> dict[str, Any]:
    t_s = event.t_s if t0_s is None else event.t_s - t0_s
    return {
        "event": event.event,
        "t_s": round(t_s, 6),
        "data": _jsonable(event.data),
    }


def _find_event_time(events: list[ReplayEvent], event_name: str, **data_match: Any) -> float | None:
    for event in events:
        if event.event != event_name:
            continue
        if all(event.data.get(key) == value for key, value in data_match.items()):
            return event.t_s
    return None


def summarize_replay(result: dict[str, Any]) -> dict[str, Any]:
    events = result["raw_events"]
    t0_s = events[0].t_s if events else 0.0
    t_end_s = events[-1].t_s if events else t0_s
    first_obs_s = _find_event_time(events, "observation_sent", observation_index=1)
    first_action_s = _find_event_time(events, "action_chunk_received", chunk_index=1)
    bootstrap_latency_s = None
    if first_obs_s is not None and first_action_s is not None:
        bootstrap_latency_s = first_action_s - first_obs_s

    q_spec_reject_count = sum(1 for event in events if event.event == "q_spec_rejected")
    server_error_count = sum(1 for event in events if event.event == "server_error")
    action_chunk_count = sum(1 for event in events if event.event == "action_chunk_received")
    received_chunk_indices = sorted(
        {
            int(event.data["chunk_index"])
            for event in events
            if event.event == "action_chunk_received" and "chunk_index" in event.data
        }
    )
    executed_chunk_events = [event for event in events if event.event == "chunk_execution_started"]
    executed_chunk_indices = [int(event.data["chunk_index"]) for event in executed_chunk_events]
    post_bootstrap_executed = [index for index in executed_chunk_indices if index > 1]
    sim_conditioned_post_bootstrap = [
        int(event.data["chunk_index"])
        for event in executed_chunk_events
        if int(event.data["chunk_index"]) > 1
        and int(event.data["chunk_index"]) in event.data.get("sim_observations", [])
    ]
    non_sim_conditioned_post_bootstrap = [
        index for index in post_bootstrap_executed if index not in set(sim_conditioned_post_bootstrap)
    ]
    missing_chunk_indices = [
        int(event.data["chunk_index"])
        for event in events
        if event.event == "chunk_underrun" and "chunk_index" in event.data
    ]
    deadline_miss_indices = [
        int(event.data["chunk_index"])
        for event in events
        if event.event == "chunk_deadline_miss" and "chunk_index" in event.data
    ]
    return {
        "session_id": result["session_id"],
        "total_elapsed_s": round(t_end_s - t0_s, 6),
        "bootstrap_latency_s": None if bootstrap_latency_s is None else round(bootstrap_latency_s, 6),
        "executed_rows": result["executed_rows"],
        "underruns": result["underruns"],
        "action_chunk_count": action_chunk_count,
        "received_chunk_indices": received_chunk_indices,
        "executed_chunk_indices": executed_chunk_indices,
        "missing_chunk_indices": missing_chunk_indices,
        "deadline_miss_indices": deadline_miss_indices,
        "deadline_miss_count": len(deadline_miss_indices),
        "post_bootstrap_executed_chunks": post_bootstrap_executed,
        "sim_conditioned_post_bootstrap_chunks": sim_conditioned_post_bootstrap,
        "non_sim_conditioned_post_bootstrap_chunks": non_sim_conditioned_post_bootstrap,
        "q_spec_reject_count": q_spec_reject_count,
        "server_error_count": server_error_count,
        "metadata": _jsonable(result["metadata"]),
        "config": _jsonable(result.get("config", {})),
    }


def write_replay_artifacts(output_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    events = result["raw_events"]
    t0_s = events[0].t_s if events else None
    event_dicts = [_event_dict(event, t0_s=t0_s) for event in events]
    (output_dir / "client_events.jsonl").write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in event_dicts),
        encoding="utf-8",
    )
    summary = summarize_replay(result)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_result_table(output_dir / "result_table.md", summary)
    return summary


def write_result_table(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total elapsed (s) | {summary['total_elapsed_s']:.3f} |",
        "| Bootstrap latency (s) | {} |".format(
            "n/a" if summary["bootstrap_latency_s"] is None else f"{summary['bootstrap_latency_s']:.3f}"
        ),
        f"| Action chunks received | {summary['action_chunk_count']} |",
        f"| Executed rows | {summary['executed_rows']} |",
        f"| Underrun rows | {summary['underruns']} |",
        f"| Deadline misses | {summary['deadline_miss_count']} |",
        f"| Non-sim post-bootstrap chunks | {len(summary['non_sim_conditioned_post_bootstrap_chunks'])} |",
        f"| q_spec rejects | {summary['q_spec_reject_count']} |",
        f"| Server errors | {summary['server_error_count']} |",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_all_frames(video_path: Path) -> np.ndarray:
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
    return np.stack(frames, axis=0)


def load_camera_frames(video_dir: Path) -> dict[str, np.ndarray]:
    camera_frames: dict[str, np.ndarray] = {}
    for camera_key, file_name in CAMERA_FILES.items():
        video_path = video_dir / file_name
        if not video_path.exists():
            raise FileNotFoundError(f"Missing DreamZero example asset: {video_path}")
        camera_frames[camera_key] = load_all_frames(video_path)
    return camera_frames


def build_frame_schedule(total_frames: int, num_chunks: int) -> list[list[int]]:
    chunks: list[list[int]] = []
    current_frame = 23
    for _ in range(num_chunks):
        indices = [max(current_frame + offset, 0) for offset in RELATIVE_OFFSETS]
        if indices[-1] >= total_frames:
            break
        chunks.append(indices)
        current_frame += ACTION_HORIZON
    return chunks


def make_obs_from_video(
    camera_frames: dict[str, np.ndarray],
    frame_indices: list[int],
    *,
    prompt: str,
    session_id: str,
) -> dict[str, Any]:
    obs: dict[str, Any] = {}
    for camera_key, all_frames in camera_frames.items():
        selected = all_frames[frame_indices]
        obs[camera_key] = selected[0] if len(frame_indices) == 1 else selected

    obs["observation/joint_position"] = np.zeros(7, dtype=np.float32)
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
    obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)
    obs["prompt"] = prompt
    obs["session_id"] = session_id
    return obs


def build_replay_observations(
    camera_frames: dict[str, np.ndarray],
    *,
    prompt: str,
    session_id: str,
    num_chunks: int,
    repeat_last_observation: bool = False,
) -> list[dict[str, Any]]:
    total_frames = min(frames.shape[0] for frames in camera_frames.values())
    observations = [
        make_obs_from_video(
            camera_frames,
            [0],
            prompt=prompt,
            session_id=session_id,
        )
    ]
    chunk_schedule = build_frame_schedule(total_frames, num_chunks - 1)
    if repeat_last_observation and chunk_schedule and len(chunk_schedule) < num_chunks - 1:
        while len(chunk_schedule) < num_chunks - 1:
            chunk_schedule.append(chunk_schedule[-1])

    for indices in chunk_schedule:
        observations.append(
            make_obs_from_video(
                camera_frames,
                indices,
                prompt=prompt,
                session_id=session_id,
            )
        )
    return observations


class DreamZeroAsyncReplayClient:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        path: str,
        session_id: str,
        prompt: str,
        control_hz: float,
        q_spec_tolerance: float | None,
    ) -> None:
        self.uri = _uri(host, port, path)
        self.session_id = session_id
        self.prompt = prompt
        self.control_hz = control_hz
        self.q_spec_tolerance = q_spec_tolerance
        self.session_epoch = 0
        self.metadata: dict[str, Any] = {}
        self.chunks: dict[int, ActionChunk] = {}
        self.events: list[ReplayEvent] = []
        self._chunk_ready = asyncio.Condition()
        self._ws: Any = None
        self._receiver_task: asyncio.Task[None] | None = None
        self._receiver_closed = False

    async def __aenter__(self) -> DreamZeroAsyncReplayClient:
        logging.info("Connecting to %s", self.uri)
        self._ws = await websockets.connect(
            self.uri,
            compression=None,
            max_size=None,
            ping_interval=300,
            ping_timeout=3600,
        )
        self.metadata = protocol.unpack_message(await self._ws.recv())
        self._record("metadata", keys=sorted(self.metadata))
        await self._send(
            {
                "type": "session_start",
                "protocol_version": protocol.PROTOCOL_VERSION,
                "session_id": self.session_id,
                "prompt": self.prompt,
            }
        )
        response = protocol.unpack_message(await self._ws.recv())
        if response.get("type") != "session_started":
            raise RuntimeError(f"Expected session_started, got {response!r}")
        self.session_epoch = int(response["session_epoch"])
        self._record("session_started", session_epoch=self.session_epoch)
        self._receiver_task = asyncio.create_task(self._receive_loop())
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._receiver_task is not None:
            self._receiver_task.cancel()
            await asyncio.gather(self._receiver_task, return_exceptions=True)
        if self._ws is not None:
            await self._ws.close()

    async def send_observation(self, observation_index: int, robot_obs: dict[str, Any]) -> None:
        payload = {
            "type": "observation_real",
            "session_id": self.session_id,
            "session_epoch": self.session_epoch,
            "observation_index": observation_index,
            "timestamp_s": time.monotonic(),
            "robot_obs": robot_obs,
        }
        await self._send(payload)
        self._record("observation_sent", observation_index=observation_index)

    async def wait_for_chunk(self, chunk_index: int, timeout_s: float | None) -> ActionChunk | None:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        async with self._chunk_ready:
            while chunk_index not in self.chunks:
                if self._receiver_closed:
                    return None
                if deadline is None:
                    await self._chunk_ready.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                try:
                    await asyncio.wait_for(self._chunk_ready.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    return None
            return self.chunks[chunk_index]

    def has_chunk(self, chunk_index: int) -> bool:
        return chunk_index in self.chunks

    def chunk_is_executable(self, chunk: ActionChunk, robot_obs: dict[str, Any]) -> bool:
        q_spec = chunk.provenance.get("q_spec")
        if self.q_spec_tolerance is None or not q_spec:
            return True

        expected_joint = q_spec.get("joint_position")
        actual_joint = robot_obs.get("observation/joint_position")
        if expected_joint is not None and actual_joint is not None:
            diff = float(np.max(np.abs(np.asarray(actual_joint) - np.asarray(expected_joint))))
            if diff > self.q_spec_tolerance:
                self._record(
                    "q_spec_rejected",
                    chunk_index=chunk.chunk_index,
                    field="joint_position",
                    max_abs_diff=diff,
                )
                return False
        return True

    async def _send(self, payload: dict[str, Any]) -> None:
        await self._ws.send(protocol.pack_message(payload))

    async def _receive_loop(self) -> None:
        try:
            while True:
                payload = protocol.unpack_message(await self._ws.recv())
                msg_type = payload.get("type")
                if msg_type == "action_chunk":
                    chunk = protocol.validate_action_chunk(payload)
                    action_chunk = ActionChunk(
                        chunk_index=chunk["chunk_index"],
                        actions=np.asarray(chunk["actions"], dtype=np.float32),
                        provenance=chunk["provenance"],
                        monitoring=chunk["monitoring"],
                        received_s=time.monotonic(),
                    )
                    async with self._chunk_ready:
                        self.chunks[action_chunk.chunk_index] = action_chunk
                        self._chunk_ready.notify_all()
                    self._record(
                        "action_chunk_received",
                        chunk_index=action_chunk.chunk_index,
                        shape=list(action_chunk.actions.shape),
                        provenance=action_chunk.provenance,
                    )
                    continue
                if msg_type == "error":
                    self._record("server_error", code=payload.get("code"), message=payload.get("message"))
                    continue
                self._record("server_message", type=str(msg_type), payload=payload)
        except Exception as exc:
            self._record("receiver_closed", error=repr(exc))
            async with self._chunk_ready:
                self._receiver_closed = True
                self._chunk_ready.notify_all()

    def _record(self, event: str, **data: Any) -> None:
        self.events.append(ReplayEvent(event=event, t_s=time.monotonic(), data=data))


async def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    session_id = args.session_id or str(uuid.uuid4())
    camera_frames = load_camera_frames(args.video_dir)
    observations = build_replay_observations(
        camera_frames,
        prompt=args.prompt,
        session_id=session_id,
        num_chunks=args.num_chunks,
        repeat_last_observation=args.repeat_last_observation,
    )
    if not observations:
        raise RuntimeError("No replay observations were built.")

    async with DreamZeroAsyncReplayClient(
        host=args.host,
        port=args.port,
        path=args.path,
        session_id=session_id,
        prompt=args.prompt,
        control_hz=args.control_hz,
        q_spec_tolerance=args.q_spec_tolerance,
    ) as client:
        await client.send_observation(1, observations[0])
        first_chunk = await client.wait_for_chunk(1, timeout_s=args.bootstrap_timeout_s)
        if first_chunk is None:
            raise TimeoutError("Timed out waiting for bootstrap action chunk A1.")

        action_horizon = int(client.metadata.get("model", {}).get("action_horizon", ACTION_HORIZON))
        executed_rows = 0
        underruns = 0
        tick_s = 1.0 / args.control_hz
        for chunk_index in range(1, args.num_chunks + 1):
            boundary_obs = observations[min(chunk_index - 1, len(observations) - 1)]
            client._record("chunk_boundary", chunk_index=chunk_index)
            if not client.has_chunk(chunk_index):
                client._record("chunk_deadline_miss", chunk_index=chunk_index)
            chunk = await client.wait_for_chunk(chunk_index, timeout_s=args.chunk_timeout_s)
            if chunk is None:
                raise TimeoutError(f"Timed out waiting for action chunk A{chunk_index}.")
            if not client.chunk_is_executable(chunk, boundary_obs):
                underruns += action_horizon
                client._record("chunk_underrun", chunk_index=chunk_index)
                if args.realtime:
                    await asyncio.sleep(action_horizon * tick_s)
            else:
                client._record(
                    "chunk_execution_started",
                    chunk_index=chunk_index,
                    real_observations=chunk.provenance.get("real_observations", []),
                    sim_observations=chunk.provenance.get("sim_observations", []),
                )
                for row_index in range(min(action_horizon, chunk.actions.shape[0])):
                    executed_rows += 1
                    client._record(
                        "action_row_executed",
                        chunk_index=chunk_index,
                        row_index=row_index,
                    )
                    if args.realtime:
                        await asyncio.sleep(tick_s)
                client._record("chunk_execution_finished", chunk_index=chunk_index)

            next_observation_index = chunk_index + 1
            if next_observation_index <= len(observations):
                await client.send_observation(
                    next_observation_index,
                    observations[next_observation_index - 1],
                )

        return {
            "metadata": client.metadata,
            "session_id": session_id,
            "executed_rows": executed_rows,
            "underruns": underruns,
            "raw_events": client.events,
            "config": {
                "num_chunks": args.num_chunks,
                "control_hz": args.control_hz,
                "realtime": args.realtime,
                "chunk_timeout_s": args.chunk_timeout_s,
                "repeat_last_observation": args.repeat_last_observation,
            },
            "events": [
                _event_dict(event)
                for event in client.events
            ],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DreamZero async replay client.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--path", default=DEFAULT_PATH)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--num-chunks", type=int, default=2)
    parser.add_argument(
        "--repeat-last-observation",
        action="store_true",
        help="Repeat the last available AR observation when videos are shorter than --num-chunks.",
    )
    parser.add_argument("--control-hz", type=float, default=CONTROL_HZ)
    parser.add_argument("--bootstrap-timeout-s", type=float, default=60.0)
    parser.add_argument("--chunk-timeout-s", type=float, default=0.0)
    parser.add_argument("--q-spec-tolerance", type=float, default=None)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    result = asyncio.run(run_replay(args))
    summary = {
        "session_id": result["session_id"],
        "executed_rows": result["executed_rows"],
        "underruns": result["underruns"],
        "event_count": len(result["events"]),
    }
    if args.output_dir is not None:
        summary.update(write_replay_artifacts(args.output_dir, result))
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        json_result = {key: value for key, value in result.items() if key != "raw_events"}
        args.output_json.write_text(json.dumps(json_result, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
