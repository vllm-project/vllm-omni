# DreamZero OpenPI Example

This example shows how to serve DreamZero with `vllm serve --omni` and connect a
compatible OpenPI websocket client using real camera videos downloaded from
Hugging Face.

## Files

- `run_server.sh`: launch DreamZero OpenPI serving
- `openpi_client.py`: websocket client that sends real observations
- `export_prediction_video.py`: offline helper that runs vLLM once and decodes DreamZero `video_pred` latents to MP4
- `droid_sim_eval_client.py`: DROID `sim-evals` rollout client for the vLLM OpenPI server

## Environment requirements

- `run_server.sh`, `vllm serve`, `openpi_client.py`,
  `export_prediction_video.py`, and the standard example/e2e tests:
  use the local `vllm-omni` environment.
- `openpi_client.py` extra deps:

```bash
pip install openpi-client websockets opencv-python huggingface-hub
```

- video export helper extra deps:

```bash
pip install opencv-python pillow
```

Optional DROID sim-eval dependencies:

- Plain serving, `openpi_client.py`, and standard e2e tests do **not** require
  Isaac Lab or `sim-evals`.
- `droid_sim_eval_client.py` must run in an external Isaac Lab / `sim-evals`
  environment where these imports already work:
  - `isaaclab`
  - `isaaclab_tasks`
  - `sim_evals`
  - `gymnasium`
- In that simulator environment, also install the OpenPI/client-side helpers:

```bash
pip install openpi-client websockets opencv-python mediapy typing-extensions
```

- `typing-extensions` is only needed on Python `< 3.12`.

- Optional `tests/dreamzero/upstream/*` parity tests also require:
  - `DREAMZERO_REPO` pointing to an upstream DreamZero checkout
  - an upstream checkpoint at `DREAMZERO_REPO/checkpoints/dreamzero`

## Start the server

From the repository root:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
examples/online_serving/dreamzero/run_server.sh
```

If you have 2 GPUs with moderate VRAM (less than 80GB), you can use the following command to start the server with TP=2 configuration files:
```bash
CUDA_VISIBLE_DEVICES=0,1 \
examples/online_serving/dreamzero/run_server_with_tp2_config.sh
```

If you only want 1 GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
DEPLOY_CONFIG=vllm_omni/deploy/dreamzero.yaml \
examples/online_serving/dreamzero/run_server.sh
```
Please note DreamZero requires >=74GB VRAM for single-GPU serving.

The websocket endpoint is:

- `ws://127.0.0.1:8000/v1/realtime/robot/openpi`

## Download example videos

The real camera videos are hosted outside this repository:

- <https://huggingface.co/datasets/YangshenDeng/vllm-omni-dreamzero-assets>

Download them into the default example location:

```bash
hf download YangshenDeng/vllm-omni-dreamzero-assets \
  --repo-type dataset \
  --local-dir outputs/dreamzero/assets
```

The expected files are:

- `outputs/dreamzero/assets/exterior_image_1_left.mp4`
- `outputs/dreamzero/assets/exterior_image_2_left.mp4`
- `outputs/dreamzero/assets/wrist_image_left.mp4`

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

If you keep the videos elsewhere, pass `--video-dir`.

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

## Export prediction videos from example inputs

DreamZero serving returns actions to the websocket client. The model also
produces a latent `video_pred`, but vLLM does **not** auto-save it from the
server path. Use the offline helper below when you want visual debug videos.

This script:

1. loads the downloaded camera videos from `outputs/dreamzero/assets/`
2. builds the same DreamZero/OpenPI observations as the client
3. runs vLLM locally through `Omni`
4. collects `video_pred` latents from `OmniRequestOutput.images`
5. decodes them on the DreamZero worker through `DreamZeroVideoExportWorkerExtension`
6. writes an MP4 under `outputs/dreamzero/generated_predictions/`

Single-config export:

```bash
python examples/online_serving/dreamzero/export_prediction_video.py \
  --model GEAR-Dreams/DreamZero-DROID \
  --deploy-config vllm_omni/deploy/dreamzero.yaml \
  --output-dir outputs/dreamzero/generated_predictions \
  --output-stem tp1_cfg1_vllm_example
```

Optional flags:

- `--save-input-video`: also writes a stitched real-input camera video
- `--save-gif`: also writes GIFs for GitHub comments
- `--save-actions`: also writes action chunks as `.npz`

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

Environment:

- do **not** run this from the plain `vllm-omni` env unless it already has Isaac Lab and `sim_evals`
- launch it from the Isaac Lab / `sim-evals` environment
- see the optional DROID sim-eval dependencies above

From the `vllm-omni` repository root, invoke the client through an external
Isaac Lab launcher, for example:

```bash
CUDA_VISIBLE_DEVICES=1 \
"${ISAACLAB_LAUNCHER}" -p \
  examples/online_serving/dreamzero/droid_sim_eval_client.py \
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

This follows the upstream DreamZero sim-eval client behavior:

- the upstream sim-eval default `open_loop_horizon` is `8`
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


# MolmoSpaces DreamZero Evaluation Demo

This example shows how to use the vllm-host to evaluate DreamZero on molmospaces benchmarks.

## Files

- `molmospace_dreamzero_eval_demo.py`: evaluate DreamZero on molmospaces benchmarks

## Environment requirements

- Install molmospaces in your python environment by following the instructions in [molmospaces/README.md](https://github.com/allenai/molmospaces/blob/main/README.md)
- Prepare the benchmark/assets directory by following the instructions in molmospaces.

## Run the evaluation

From the repository root:

```bash
python examples/online_serving/dreamzero/molmospace_dreamzero_eval_demo.py \
  --benchmark_dir "${MOLMOSPACES_BENCHMARK_DIR}/20260327/ithor/FrankaCloseHardBench/FrankaCloseHardBench_20260206_json_benchmark" \
  --output_dir outputs/dreamzero/molmospaces \
  --max_episodes 1 \
  --task_horizon_steps 240 \
  --episode_idx 1
```
