#!/usr/bin/env python3
"""
Lingbot World Fast realtime camera client.

Talks to the WebSocket endpoint ``/v1/realtime/world/camera`` exposed by
``vllm serve --omni`` when the loaded pipeline is ``LingbotWorldFastPipeline``.

The endpoint speaks the OpenPI policy protocol on the wire:
    1. Connect    -> server sends msgpack(CameraServerConfig)
    2. Client send msgpack(request)
    3. Server send msgpack(ndarray)  # generated frames

The ``request`` payload sent here contains:
    - "image":   numpy array, the input image
    - "prompt":  str, the text prompt describing the desired motion
    - "camera":  {"poses": ndarray, "intrinsics": ndarray}

Usage:
    python openai_chat_client.py \\
        --image path/to/image.png \\
        --camera-path path/to/camera_dir \\
        --prompt "Walk along the Great Wall of China" \\
        --output frames.npy
"""

import argparse
from argparse import Namespace
from pathlib import Path

import numpy as np
import PIL.Image
import websockets.sync.client as ws_sync
from diffusers.utils import export_to_video

try:
    from openpi_client import msgpack_numpy
except ImportError as exc:
    raise SystemExit("This example requires `openpi-client`. Install it with `pip install openpi-client`.") from exc


def _pack(obj):
    return msgpack_numpy.packb(obj)


def _unpack(data):
    return msgpack_numpy.unpackb(data)


def _load_image(path: str | None) -> np.ndarray | None:
    image = PIL.Image.open(path).convert("RGB")
    return np.asarray(image)


def _load_camera(camera_dir: str) -> dict:
    camera_path = Path(camera_dir)
    poses = np.load(camera_path / "poses.npy")
    intrinsics = np.load(camera_path / "intrinsics.npy")
    return {"poses": poses, "intrinsics": intrinsics}


def generate_video(args: Namespace) -> list[np.ndarray]:
    """Send inference requests and return the generated frames."""
    image = _load_image(args.image)
    full_camera = _load_camera(args.camera_path)

    extra_body = {
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "session_id": args.session_id,
        "frames_per_chunk": args.frames_per_chunk,
        "seed": args.seed,
    }

    video = []
    starting_frame = 0

    for i in range(args.num_calls):
        camera = {
            "poses": full_camera["poses"][starting_frame : starting_frame + args.num_frames],
            "intrinsics": full_camera["intrinsics"][starting_frame : starting_frame + args.num_frames],
        }

        request: dict = {"prompt": args.prompt, "camera": camera, "extra_body": extra_body}
        if i == 0:
            request["image"] = image

        request["session_id"] = args.session_id

        endpoint = f"{args.server.rstrip('/')}/v1/realtime/world/camera"
        print(f"Connecting to {endpoint} ...")

        with ws_sync.connect(endpoint, max_size=None, ping_interval=None, ping_timeout=None) as ws:
            # 1. Server sends CameraServerConfig on connect.
            server_config: dict = _unpack(ws.recv())
            print("Server Configuration:")
            for key, val in server_config.items():
                print(f"\t{key}: {val}")

            # 2. Send request.
            print(
                f"Sending request image=  ({str(image.shape) if request.get('image', None) is not None else 'None'}, "
                f"poses={camera['poses'].shape}, intrinsics={camera['intrinsics'].shape})..."
            )
            ws.send(_pack(request))

            # 3. Receive generated frames.
            chunks: list[np.ndarray] = []
            total = None
            while total is None or len(chunks) < total:
                msg = _unpack(ws.recv())
                if isinstance(msg, dict) and msg.get("type") == "error":
                    raise RuntimeError(f"Server error: {msg.get('message')}")
                if not isinstance(msg, dict) or msg.get("type") != "frame":
                    continue  # ignore anything unexpected
                total = msg["total"]
                chunks.append(msg["video"])
                print(f"  received chunk {msg['index'] + 1}/{total}")

            clip = np.concatenate(chunks, axis=0)
            # The first chunk of frames returned was used to condition the video continuation but they are not useful
            if i != 0:
                clip = clip[args.num_skip_frames :]
            for frame in clip:
                video.append(frame)

        starting_frame += args.num_frames

    return video


def main():
    parser = argparse.ArgumentParser(description="Lingbot World Fast realtime camera client")
    parser.add_argument("--image", "-i", required=True, help="Path to input image.")
    parser.add_argument(
        "--camera-path",
        "-c",
        required=True,
        help="Directory containing poses.npy and intrinsics.npy.",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default="Walk along the Great Wall of China",
        help="Text prompt describing the desired motion.",
    )
    parser.add_argument(
        "--server",
        "-s",
        default="ws://localhost:8091",
        help="WebSocket server URL (ws:// or wss://).",
    )
    parser.add_argument("--session-id", default=None, help="Optional session id.")
    parser.add_argument(
        "--output",
        "-o",
        default="lingbot-video.mp4",
        help="Path to save the returned frames (npy).",
    )
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--num-calls", type=int, default=1)
    parser.add_argument("--num-skip-frames", type=int, default=4)
    parser.add_argument(
        "--frames-per-chunk", type=int, default=4, help="How many frames are sent in each package in the response"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    frames = generate_video(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_to_video(frames, str(output_path), fps=args.fps)
    print(f"Saved generated video to {output_path}")


if __name__ == "__main__":
    main()
