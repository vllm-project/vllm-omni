# DreamZero OpenPI Example

This example shows how to serve DreamZero with `vllm serve --omni` and connect a
compatible OpenPI websocket client using bundled real camera videos.

## Files

- `run_server.sh`: launch DreamZero OpenPI serving
- `openpi_client.py`: websocket client that sends real observations
- `droid_sim_eval_client.py`: DROID `sim-evals` rollout client for the vLLM OpenPI server
- `assets/`: minimal real camera videos used by the example

## Environment requirements

- `run_server.sh`, `vllm serve`, `openpi_client.py`, and the standard example/e2e tests:
  use the local `vllm-omni` environment.
- `openpi_client.py` extra deps:

```bash
pip install openpi-client websockets opencv-python
```

- `droid_sim_eval_client.py` must run in an external Isaac Lab / `sim-evals`
  environment, and also needs:

```bash
pip install openpi-client websockets opencv-python mediapy
```

- On Python `< 3.12`, also install:

```bash
pip install typing-extensions
```

- Optional `tests/dreamzero/upstream/*` parity tests also require:
  - local upstream repo at `~/code/dreamzero`
  - local checkpoint at `~/code/dreamzero/checkpoints/dreamzero`

## Start the server

From the repository root:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
examples/online_serving/dreamzero/run_server.sh
```

If you only want 1 GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
CFG_PARALLEL_SIZE=1 \
examples/online_serving/dreamzero/run_server.sh
```

The websocket endpoint is:

- `ws://127.0.0.1:8000/v1/realtime/robot/openpi`

## Run the client

From the repository root:

Environment:

- run this in the `vllm-omni` repo environment
- if imports are missing, install `openpi-client`, `websockets`, and `opencv-python`

```bash
python examples/online_serving/dreamzero/openpi_client.py \
  --host 127.0.0.1 \
  --port 8000
```

The client sends:

- one initial single-frame observation
- one four-frame observation
- one websocket reset
- one post-reset single-frame observation

It validates:

- DreamZero metadata contract
- action tensor shape `(24, 8)`
- finite action values
- reset response

## Run DROID sim-eval against the vLLM server

This is the closest setup to an end-to-end simulated policy rollout.

### 1. Start the vLLM DreamZero server

From the repository root:

Environment:

- run this in the `vllm-omni` repo environment
- no extra DreamZero-specific client package is needed for the server itself

```bash
CUDA_VISIBLE_DEVICES=0 \
ATTENTION_BACKEND=torch \
DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA \
vllm serve \
  GEAR-Dreams/DreamZero-DROID \
  --omni \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name dreamzero-droid \
  --enforce-eager
```

### 2. Start the DROID simulation client

This step runs from an external `sim-evals` checkout, because Isaac Lab /
Isaac Sim assets and environment registration live there.

Like upstream DreamZero, this client does not guess any local path for
`sim_evals` or Isaac Lab. Run it from an environment where those packages are
already importable (for example, the `sim-evals` project environment).

Environment:

- do **not** run this from the plain `vllm-omni` env unless it already has Isaac Lab and `sim_evals`
- launch it from the Isaac Lab / `sim-evals` environment
- make sure the following imports work there:
  - `isaaclab`
  - `isaaclab_tasks`
  - `sim_evals`
  - `gymnasium`
  - `openpi_client`
  - `websockets`
  - `cv2`
  - `mediapy`

Example command:

```bash
cd /path/to/sim-evals

CUDA_VISIBLE_DEVICES=1 \
./submodules/IsaacLab/isaaclab.sh -p \
  /path/to/vllm-omni/examples/online_serving/dreamzero/droid_sim_eval_client.py \
  --host 127.0.0.1 \
  --port 8000 \
  --scene 1 \
  --episodes 1 \
  --headless \
  --device cuda:0
```

Notes:

- `CUDA_VISIBLE_DEVICES=1` keeps Isaac Sim off the GPU used by the vLLM server.
- `--scene` selects one of the built-in DROID tasks:
  - `1`: `put the cube in the bowl`
  - `2`: `pick up the can and put it in the mug`
  - `3`: `put the banana in the bin`
- The client keeps the upstream DreamZero sim-eval behavior:
  - DROID observation extraction from `external_cam`, `external_cam_2`, and `wrist_cam`
  - `resize_with_pad(..., 180, 320)`
  - `open_loop_horizon=8`
  - 24-step action chunks with 8 action dimensions

### Action chunk vs open-loop horizon

DreamZero predicts longer action chunks than the number of actions the
sim-eval client executes before replanning:

- model output action chunk: `(24, 8)`
  - `24`: predicted future action horizon
  - `8`: action dimension, i.e. 7 arm joints + 1 gripper
- sim-eval execution horizon: `open_loop_horizon=8`
  - after one model call, the client executes only the first `8` actions
  - the remaining `16` predicted actions are not consumed
  - the client then sends a fresh observation and asks the server for a new
    `(24, 8)` chunk

This follows the upstream DreamZero sim-eval client:

- `third_party/dreamzero/eval_utils/run_sim_eval.py` defaults
  `open_loop_horizon` to `8`
- DreamZero action outputs use `action_horizon=24`

The split is intentional: `24` lets the model predict a longer future plan,
while `8` keeps execution closed-loop by replanning after roughly half a second
in the DROID simulator.

## How the sim-eval rollout works

At a high level, one rollout does the following:

1. Isaac Lab loads the DROID scene and resets the environment twice.
2. `droid_sim_eval_client.py` reads the current robot observation:
   - two external cameras
   - one wrist camera
   - 7-DoF arm joint positions
   - 1-DoF gripper position
3. The client converts the observation into the DreamZero/OpenPI websocket payload:
   - `observation/exterior_image_0_left`
   - `observation/exterior_image_1_left`
   - `observation/wrist_image_left`
   - `observation/joint_position`
   - `observation/cartesian_position`
   - `observation/gripper_position`
   - `prompt`
   - `session_id`
4. vLLM DreamZero returns one action chunk with shape `(24, 8)`.
5. The sim client consumes that chunk in open loop for `8` control steps.
6. After the local chunk budget is exhausted, the client requests the next action chunk.
7. This repeats until the environment hits its time limit.

The current DROID sim environment does not expose a built-in task success flag,
so the rollout result should be judged primarily from the video and the final
trajectory JSON.

## How to read the `runs/` outputs

By default the client writes results under:

- `runs/dreamzero_sim_eval/<scene>_<timestamp>/`

The key files are:

- `episode_00.mp4`
  - the rollout video
  - this is the first file to inspect
- `episode_00.json`
  - per-step trace for one episode
  - includes:
    - `prompt`
    - `steps_executed`
    - `server_calls`
    - `episode_wall_time_s`
    - `server_time_s`
    - `trajectory`
- `summary.json`
  - top-level run summary across episodes
  - includes:
    - scene id
    - prompt
    - server metadata
    - per-episode summaries

Inside `episode_00.json`, the `trajectory` list contains one entry per control
step. Each entry records:

- `step_index`: control step index
- `used_server_call`: whether this step triggered a new model chunk request
- `chunk_latency_s`: model latency for that chunk request
- `action`: the 8-D action applied to the simulator
- `joint_position`: observed robot joints before the next step
- `gripper_position`: observed gripper state
- `reward`, `terminated`, `truncated`

Practical reading order:

1. watch `episode_00.mp4`
2. open `summary.json` and check:
   - prompt
   - total steps
   - total wall time
   - total model time
   - number of server calls
3. if the behavior looks odd, inspect `episode_00.json`
   - check whether actions saturate
   - check whether the robot stalls
   - check how often a new chunk was requested

For GitHub issues / PR comments, you can also convert `episode_00.mp4` to a GIF
with `ffmpeg` and attach it directly.

## Optional upstream parity checks

The upstream DreamZero-dependent parity tests are kept under:

- `tests/dreamzero/upstream/`

Those tests require a local upstream DreamZero checkout and are not needed for
the standard vLLM example above.
