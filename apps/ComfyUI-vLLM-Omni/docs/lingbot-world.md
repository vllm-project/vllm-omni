# LingBot World

LingBot World runs in vLLM-Omni; ComfyUI only prepares requests and draws the plan, so it can run on CPU. There are two ways to use it:

- **Live**: the **LingBot World (Realtime)** node opens a window that plays the stream while you steer with W/A/S/D and I/J/K/L, change the prompt mid-run, and save the take.
- **Scripted**: **LingBot World Shot** nodes chain a sequence, each with a camera lane editor inside the node, **LingBot World Render** turns the chain into one video, and **LingBot World Visualizer** draws the plan.

## Install and serve

Install the parent `ComfyUI-vLLM-Omni` folder following its [README](../README.md). The lane editor and the visualizer are DOM widgets, so they need a ComfyUI build with `addDOMWidget` (2024 or later).

Start the backend:

```bash
vllm serve robbyant/lingbot-world-v2-14b-causal-fast-diffusers \
  --omni --deploy-config vllm_omni/deploy/lingbot_world_v2_stepwise.yaml \
  --host 127.0.0.1 --port 8000
```

This is the repository's single-GPU configuration; streaming capability alone does not guarantee realtime speed. Start ComfyUI in its own environment:

```bash
python main.py --cpu --listen 127.0.0.1 --port 8188
```

Do not add `--disable-api-nodes`: its media policy blocks the `blob:` URLs the player needs. A Chromium browser with H.264 Media Source Extensions is the validation target.

## Live: steer a world

Import [vLLM-Omni LingBot World.json](../example_workflows/vLLM-Omni%20LingBot%20World.json).

1. Select one RGB image in **Load Image** and set the initial prompt. The image is resized to the requested output size.
2. Keep `width=832`, `height=480` to match the deploy configuration. The frame count must be `9 + 12k`; the default 129 frames at 16 FPS is about eight seconds.
3. Run the workflow to prepare the request. Run does not start inference.
4. Click **Open interaction panel** below the seed, then **Start**. Click the video once to drive it: **WASD** moves, **IJKL** turns, and the keycaps light up while held.
5. **Apply** sends a new prompt to the running session; it is enabled once the first chunk arrives. Camera speed is 0.5x by default, adjustable from 0.1x to 2x. Releasing the keys, leaving the video, or hiding the page releases movement.
6. **Save** stops the session and downloads the take as an MP4. **Stop** only cancels. Closing the window, pressing Escape, deleting the node, or replacing the workflow stops it too.

Changing node parameters requires another Run. Keep this node in the root workflow, outside subgraphs. Only one panel is active per browser page.

## Scripted: shots and camera moves

Import [vLLM-Omni LingBot World Shots.json](../example_workflows/vLLM-Omni%20LingBot%20World%20Shots.json).

| Node | Takes | Gives |
| --- | --- | --- |
| **LingBot World Shot** | a prompt, a length in chunks, an in-node lane editor, an optional `sequence` | `LINGBOT_SEQUENCE` |
| **LingBot World Render** | a first image, a `sequence`, the server settings | `VIDEO` |
| **LingBot World Visualizer** | a `sequence` | nothing; it draws the plan |

Chain shots through `sequence` and feed the last one to **Render** and to **Visualizer**.

### How long a shot is

A rollout is decoded causally, so its first chunk holds 9 pixel frames and every later chunk holds 12:

- the first shot lasts `9 + 12k` frames;
- every later shot lasts whole chunks;
- the whole sequence is `9 + 12k` frames.

The Shot node's `chunks` field counts chunks, one chunk being 0.75 seconds at 16 FPS, and the node sizes the shot for you.

### The lane editor

Each Shot node draws its own timeline under the prompt: a frame ruler in frames and seconds, the chunk grid as dotted lines, and one row per lane.

- Click an empty lane to add a move; it snaps to a latent frame and lasts a chunk by default.
- Drag a move to slide it, drag its right edge to resize, press Delete to remove the selected one.
- Select a move to edit its lane, direction, start frame and length as numbers below.
- Overlapping moves on one lane, a move shorter than one latent frame, an unknown lane or direction, and a move running past the shot are drawn in red with the reason in the tooltip. A move the shot cuts is drawn with a dashed edge and listed by the visualizer.

| Lane | Values | The server integrates |
| --- | --- | --- |
| `move` | `forward`, `back` | 0.05 world units per latent frame |
| `strafe` | `left`, `right` | 0.05 world units per latent frame |
| `look_v` | `up`, `down` | 4 degrees of pitch per latent frame |
| `look_h` | `left`, `right` | 6 degrees of yaw per latent frame |

There is no speed setting: how far the camera travels is how long the move lasts. The vocabulary has no vertical translation and no absolute pose, so a crane shot or an exact camera path cannot be expressed.

### What renders

**Render** runs one session for the whole sequence. The first shot's prompt starts it, and every later prompt goes out as an interaction as soon as the chunk before its boundary has been generated, so it takes effect exactly at the boundary. That is a prompt change inside one continuous world, not a scene cut: the backend does not expose its KV-cache reset here, so a different world means a separate Render.

**Visualizer** walks the chain you wired and draws it before anything runs: the same ruler and chunk grid, one block per shot with its prompt, prompt-switch markers, one row per lane, a footer with totals and warnings, and a playhead that follows a Render of that same chain.

## Local and remote connections

The browser connects directly to `server_url`, so use a WebSocket endpoint reachable from the browser's machine. `localhost` and `127.0.0.1` refer to that machine, not the ComfyUI server. HTTPS pages require `wss://`.

For SSH access to a machine running both services, forward **8188 and 8000**:

```bash
ssh -N -L 8188:127.0.0.1:8188 -L 8000:127.0.0.1:8000 user@server
```

## Control and playback contract

Control changes apply before the next chunk starts, after any playback wait; they cannot change frames already generated or buffered. Camera motion is generative, without a game engine's collision guarantees.

The player buffers up to one second before starting playback, so the shorter first chunk bridges into the next. Keyboard mode reports the video playhead every 100 ms and asks for a 1.25-second generation window: the backend waits before starting another chunk once generated media is that far ahead, and an in-flight chunk can extend the window by one chunk. Pausing playback therefore stops further generation once the window fills, while controls and Stop stay available.

Sessions are finite. There is no reconnect-and-resume, no live recording node and no open-ended mode; increasing `num_frames` only extends one rollout.

## CPU checks

```bash
pytest tests/e2e/features/comfyui/test_lingbot_world.py \
  tests/e2e/features/comfyui/test_lingbot_moves.py \
  tests/e2e/features/comfyui/test_lingbot_scene.py \
  -m 'core_model and cpu' --run-level core_model
```

They need neither GPU, model weights nor a running service. The frontend tests are optional and need Node.js 20+:

```bash
node --test tests/e2e/features/comfyui/lingbot_player.test.mjs \
  tests/e2e/features/comfyui/lingbot_live.test.mjs \
  tests/e2e/features/comfyui/lingbot_move_editor.test.mjs \
  tests/e2e/features/comfyui/lingbot_timeline.test.mjs
```

## Real-model validation

Start the service above and run both workflows against the attributed [upstream example 03 image](https://github.com/Robbyant/lingbot-world-v2/blob/1895d300d8ac936401689b26389f51cbd36530eb/examples/03/image.jpg). Live must reach `session.done` and show camera motion when keys are held. Render must produce a playable video whose two shots differ in prompt and whose camera follows the lane editor. No speed or quality claim is made by the CPU checks.
