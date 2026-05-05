#!/usr/bin/env python3
"""End-to-end WebSocket client for testing realtime video streaming.

Usage:
    # Interactive mode: type prompts while video generates continuously
    python tests/test_realtime_e2e_client.py -i

    # Batch mode with fixed block count:
    python tests/test_realtime_e2e_client.py --blocks 5

    # Batch mode with scheduled prompt change:
    python tests/test_realtime_e2e_client.py --blocks 8 --change-prompt-at 3 \
        --new-prompt "A rocket launching into space"

    # Save output frames to disk:
    python tests/test_realtime_e2e_client.py -i --save-dir /tmp/frames

    # V2V mode with local video input:
    python tests/test_realtime_e2e_client.py --mode v2v --video input.mp4 \
        --blocks 5 --prompt "anime style" --save-dir /tmp/v2v_out

    # V2V mode with random frames (no video file):
    python tests/test_realtime_e2e_client.py --mode v2v --blocks 5

Requires: pip install websockets msgpack Pillow
Optional: pip install opencv-python  (for V2V video file input)
"""

import argparse
import asyncio
import io
import os
import threading
import time

try:
    import msgpack
except ImportError:
    msgpack = None

try:
    import websockets
except ImportError:
    websockets = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import cv2
except ImportError:
    cv2 = None


def pack_msg(msg: dict) -> bytes:
    if msgpack is not None:
        return msgpack.packb(msg, use_bin_type=True)
    import json

    return json.dumps(msg).encode()


def unpack_msg(data: bytes) -> dict:
    if msgpack is not None:
        return msgpack.unpackb(data, raw=False)
    import json

    return json.loads(data)


async def run_t2v_interactive(args):
    """Interactive T2V: generates continuously, user types prompts to steer."""
    uri = f"ws://{args.host}:{args.port}/v1/realtime_video/generate"
    print(f"Connecting to {uri} ...")

    async with websockets.connect(uri, max_size=50 * 1024 * 1024) as ws:
        config = {
            "type": "config",
            "mode": "t2v",
            "prompt": args.prompt,
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.steps,
            "num_frames_per_block": args.frames_per_block,
            "frame_format": args.frame_format,
            "frame_quality": args.frame_quality,
        }
        await ws.send(pack_msg(config))
        print(
            f"Config: prompt='{args.prompt}', "
            f"{args.width}x{args.height}, steps={args.steps}, "
            f"format={args.frame_format}"
        )

        data = await ws.recv()
        msg = unpack_msg(data)
        if not (msg.get("type") == "status" and msg.get("content") == "session_started"):
            print(f"Unexpected response: {msg}")
            return
        print("Session started!")
        print("-" * 60)
        print("Type a new prompt and press Enter to change the scene.")
        print("Type 'quit' or 'exit' to stop.")
        print("-" * 60)

        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)

        # Use a thread to read stdin, push lines into a thread-safe queue
        input_queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        quit_flag = False

        def _read_stdin():
            while True:
                try:
                    line = input()
                except EOFError:
                    break
                loop.call_soon_threadsafe(input_queue.put_nowait, line.strip())

        stdin_thread = threading.Thread(target=_read_stdin, daemon=True)
        stdin_thread.start()

        block_count = 0
        frame_count = 0
        block_frame_idx = 0
        last_frame_time = None
        start_time = time.time()
        BLOCK_GAP_THRESHOLD = 1.0

        try:
            while True:
                # Check for new prompts (non-blocking)
                while not input_queue.empty():
                    line = input_queue.get_nowait()
                    if not line:
                        continue
                    if line.lower() in ("quit", "exit", "q"):
                        quit_flag = True
                        break

                    prompt_msg = {"type": "prompt", "content": line}
                    await ws.send(pack_msg(prompt_msg))
                    print(f">>> Prompt updated: '{line}'")

                if quit_flag:
                    break

                # Receive next frame
                try:
                    data = await asyncio.wait_for(ws.recv(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                msg = unpack_msg(data)
                now = time.time()
                elapsed = now - start_time

                if msg["type"] == "frame":
                    content = msg["content"]
                    if last_frame_time is not None and (now - last_frame_time) > BLOCK_GAP_THRESHOLD:
                        block_count += 1
                        block_frame_idx = 0
                    elif last_frame_time is None:
                        pass
                    last_frame_time = now
                    frame_count += 1
                    block_frame_idx += 1

                    parts = [f"Block {block_count} Frame {block_frame_idx}", f"{len(content)} bytes", f"{elapsed:.1f}s"]

                    if Image is not None and isinstance(content, bytes):
                        try:
                            img = Image.open(io.BytesIO(content))
                            parts.append(f"{img.size[0]}x{img.size[1]}")

                            if args.save_dir:
                                path = os.path.join(
                                    args.save_dir,
                                    f"frame_{frame_count:04d}.png",
                                )
                                img.save(path)
                                parts.append(f"saved:{path}")
                        except Exception as e:
                            parts.append(f"decode err: {e}")

                    print(f"  {' | '.join(parts)}")

                elif msg["type"] == "error":
                    print(f"ERROR: {msg['content']}")
                    break

                elif msg["type"] == "status":
                    print(f"Status: {msg['content']}")

        except (KeyboardInterrupt,):
            print("\nInterrupted.")
        except Exception as e:
            if "close" not in str(e).lower():
                print(f"\nConnection error: {e}")

        total = time.time() - start_time
        if frame_count > 0:
            print(f"\nDone! {frame_count} blocks in {total:.1f}s ({frame_count / total:.2f} blocks/sec)")


async def run_t2v_batch(args):
    """Batch T2V: generate a fixed number of blocks, optionally change prompt."""
    uri = f"ws://{args.host}:{args.port}/v1/realtime_video/generate"
    print(f"Connecting to {uri} ...")

    async with websockets.connect(uri, max_size=50 * 1024 * 1024) as ws:
        config = {
            "type": "config",
            "mode": "t2v",
            "prompt": args.prompt,
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.steps,
            "num_frames_per_block": args.frames_per_block,
            "frame_format": args.frame_format,
            "frame_quality": args.frame_quality,
        }
        await ws.send(pack_msg(config))
        print(
            f"Sent config: mode=t2v, prompt='{args.prompt}', "
            f"{args.width}x{args.height}, steps={args.steps}, "
            f"format={args.frame_format}"
        )

        data = await ws.recv()
        msg = unpack_msg(data)
        assert msg["type"] == "status" and msg["content"] == "session_started", f"Expected session_started, got: {msg}"
        print("Session started!")

        block_count = 0
        frame_count = 0
        start_time = time.time()
        vae_temporal_compression = 4

        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)

        prompt_changed = False

        while block_count < args.blocks:
            if args.change_prompt_at and block_count == args.change_prompt_at and not prompt_changed:
                new_prompt = args.new_prompt or "A rocket launching into space with flames"
                prompt_msg = {"type": "prompt", "content": new_prompt}
                await ws.send(pack_msg(prompt_msg))
                print(f"\n>>> Prompt changed to: '{new_prompt}'")
                prompt_changed = True

            data = await ws.recv()
            msg = unpack_msg(data)

            if msg["type"] == "frame":
                block_count += 1
                block_frames = [msg["content"]]

                # Calculate expected frames for this block:
                # Block 0 (first): (num_frames_per_block - 1) * 4 + 1
                # Block 1+: num_frames_per_block * 4
                if block_count == 1:
                    expected = (args.frames_per_block - 1) * vae_temporal_compression + 1
                else:
                    expected = args.frames_per_block * vae_temporal_compression

                # Collect remaining frames of this block by expected count
                try:
                    while len(block_frames) < expected:
                        data = await asyncio.wait_for(ws.recv(), timeout=30.0)
                        msg2 = unpack_msg(data)
                        if msg2["type"] == "frame":
                            block_frames.append(msg2["content"])
                        elif msg2["type"] == "error":
                            print(f"ERROR from server: {msg2['content']}")
                            return
                except asyncio.TimeoutError:
                    pass

                elapsed = time.time() - start_time
                for content in block_frames:
                    frame_count += 1
                    info = (
                        f"Block {block_count}/{args.blocks} | "
                        f"frame {frame_count} | "
                        f"{len(content)} bytes | "
                        f"elapsed: {elapsed:.1f}s"
                    )

                    if Image is not None and isinstance(content, bytes):
                        try:
                            img = Image.open(io.BytesIO(content))
                            info += f" | {img.size[0]}x{img.size[1]} {img.mode}"
                            if args.save_dir:
                                path = os.path.join(
                                    args.save_dir,
                                    f"frame_{frame_count:04d}.png",
                                )
                                img.save(path)
                                info += f" | saved: {path}"
                        except Exception as e:
                            info += f" | decode err: {e}"

                    print(info)

            elif msg["type"] == "error":
                print(f"ERROR from server: {msg['content']}")
                break

            elif msg["type"] == "status":
                print(f"Status: {msg['content']}")

        total = time.time() - start_time
        print(
            f"\nDone! {block_count} blocks, {frame_count} frames in {total:.1f}s ({block_count / total:.2f} blocks/sec)"
        )


def load_video_frames(video_path: str, height: int, width: int) -> list[bytes]:
    """Load all frames from a video file as JPEG bytes."""
    if cv2 is None:
        print("ERROR: pip install opencv-python")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(
        f"Video: {video_path} | {total} frames | {fps:.1f} fps | "
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
    )

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        frames.append(buf.getvalue())
    cap.release()
    print(f"Loaded {len(frames)} frames, resized to {width}x{height}")
    return frames


def make_random_frames(count: int, height: int, width: int) -> list[bytes]:
    """Generate random dummy frames as JPEG bytes."""
    import numpy as np

    frames = []
    for _ in range(count):
        arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        frames.append(buf.getvalue())
    return frames


async def run_v2v_test(args):
    """V2V mode: send video frames, receive transformed output.

    Sends all input frames upfront to avoid deadlock with the server's
    pipeline mode (which may return 0 frames for intermediate blocks).
    """
    uri = f"ws://{args.host}:{args.port}/v1/realtime_video/generate"
    print(f"Connecting to {uri} (V2V mode) ...")

    if Image is None:
        print("ERROR: Pillow is required for V2V test")
        return

    vae_t = 4  # VAE temporal compression ratio
    n = args.frames_per_block
    first_block_frames = (n - 1) * vae_t + 1
    next_block_frames = n * vae_t

    if args.video:
        all_frames = load_video_frames(args.video, args.height, args.width)
        if not all_frames:
            return
    else:
        print("No --video provided, using random dummy frames")
        total_needed = first_block_frames + next_block_frames * max(args.blocks, 1)
        all_frames = make_random_frames(total_needed, args.height, args.width)

    async with websockets.connect(uri, max_size=50 * 1024 * 1024) as ws:
        config = {
            "type": "config",
            "mode": "v2v",
            "prompt": args.prompt,
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.steps,
            "num_frames_per_block": args.frames_per_block,
            "frame_format": args.frame_format,
            "frame_quality": args.frame_quality,
        }
        if args.strength is not None:
            config["v2v_strength"] = args.strength
        await ws.send(pack_msg(config))
        print(
            f"Sent config: mode=v2v, prompt='{args.prompt}', "
            f"format={args.frame_format}"
            f"{f', strength={args.strength}' if args.strength else ''}"
        )

        data = await ws.recv()
        msg = unpack_msg(data)
        assert msg["type"] == "status" and msg["content"] == "session_started"
        print("Session started!")

        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)

        # Pre-compute frame batches. Pipeline mode needs one extra server
        # block because each pipeline block returns the *previous* block's
        # decoded frames (the first pipeline block returns nothing).
        import numpy as np

        server_blocks = args.blocks + 1
        batches = []
        cursor = 0
        for i in range(server_blocks):
            needed = first_block_frames if i == 0 else next_block_frames
            if cursor + needed <= len(all_frames):
                batch = all_frames[cursor : cursor + needed]
                cursor += needed
            elif cursor < len(all_frames):
                remaining = all_frames[cursor:]
                indices = np.round(np.linspace(0, len(remaining) - 1, needed)).astype(int)
                batch = [remaining[int(idx)] for idx in indices]
                cursor = len(all_frames)
            else:
                cursor = 0
                batch = all_frames[:needed]
                cursor = needed
            batches.append(batch)

        print(f"Prepared {len(batches)} input batches ({sum(len(b) for b in batches)} total frames)")

        send_error = None

        async def send_all_frames():
            nonlocal send_error
            try:
                for i, batch in enumerate(batches):
                    await ws.send(pack_msg({"type": "video", "frames": batch}))
            except Exception as e:
                send_error = e

        async def recv_output():
            block_count = 0
            output_frame_idx = 0
            start = time.time()

            while block_count < args.blocks:
                data = await ws.recv()
                msg = unpack_msg(data)

                if msg["type"] == "frame":
                    block_count += 1
                    block_frames = [msg["content"]]

                    try:
                        while True:
                            data = await asyncio.wait_for(ws.recv(), timeout=1.0)
                            msg2 = unpack_msg(data)
                            if msg2["type"] == "frame":
                                block_frames.append(msg2["content"])
                            elif msg2["type"] == "error":
                                print(f"ERROR: {msg2['content']}")
                                return
                    except asyncio.TimeoutError:
                        pass

                    elapsed = time.time() - start
                    for content in block_frames:
                        output_frame_idx += 1
                        parts = [
                            f"Block {block_count}/{args.blocks}",
                            f"frame {output_frame_idx}",
                            f"{len(content)} bytes",
                            f"{elapsed:.1f}s",
                        ]
                        if Image is not None and isinstance(content, bytes):
                            try:
                                img = Image.open(io.BytesIO(content))
                                parts.append(f"{img.size[0]}x{img.size[1]}")
                                if args.save_dir:
                                    path = os.path.join(
                                        args.save_dir,
                                        f"v2v_frame_{output_frame_idx:04d}.png",
                                    )
                                    img.save(path)
                                    parts.append(f"saved:{path}")
                            except Exception as e:
                                parts.append(f"decode err: {e}")
                        print(f"  {' | '.join(parts)}")

                elif msg["type"] == "error":
                    print(f"ERROR: {msg['content']}")
                    return

            total = time.time() - start
            print(f"\nV2V done! {block_count} blocks, {output_frame_idx} output frames in {total:.1f}s")

        await asyncio.gather(send_all_frames(), recv_output())

        if send_error:
            print(f"Frame send error (non-fatal): {send_error}")


def main():
    parser = argparse.ArgumentParser(description="Realtime video E2E test client")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mode", choices=["t2v", "v2v"], default="t2v")
    parser.add_argument(
        "--prompt",
        default="A beautiful sunset over the ocean, cinematic",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=None,
        help="Number of blocks (default: unlimited in interactive, 5 in batch)",
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--steps", type=int, default=4, help="Denoising steps per block")
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive mode: type prompts while video generates continuously",
    )
    parser.add_argument("--change-prompt-at", type=int, default=None, help="[batch] Change prompt at this block number")
    parser.add_argument("--new-prompt", default=None, help="[batch] New prompt for --change-prompt-at")
    parser.add_argument("--save-dir", default=None, help="Directory to save output frames as PNG")
    parser.add_argument("--video", default=None, help="[v2v] Path to input video file (mp4/avi/...)")
    parser.add_argument(
        "--frames-per-block", type=int, default=3, help="num_frames_per_block (latent frames per block)"
    )
    parser.add_argument(
        "--strength", type=float, default=None, help="[v2v] V2V strength (0.0-1.0, lower preserves more)"
    )
    parser.add_argument(
        "--frame-format", choices=["jpeg", "png"], default="jpeg", help="Output frame encoding format (default: jpeg)"
    )
    parser.add_argument(
        "--frame-quality", type=int, default=95, help="JPEG quality 1-100 (default: 95, ignored for PNG)"
    )

    args = parser.parse_args()

    if websockets is None:
        print("ERROR: pip install websockets")
        return
    if msgpack is None:
        print("WARNING: msgpack not installed, using JSON fallback")

    if args.mode == "v2v":
        if args.blocks is None:
            args.blocks = 5
        asyncio.run(run_v2v_test(args))
    elif args.interactive:
        asyncio.run(run_t2v_interactive(args))
    else:
        if args.blocks is None:
            args.blocks = 5
        asyncio.run(run_t2v_batch(args))


if __name__ == "__main__":
    main()
