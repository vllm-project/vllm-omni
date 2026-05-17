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

## Optional Evaluation Demos

The files below are optional external evaluation demos kept with the DreamZero
example for discoverability. They are not required for the basic online serving
flow above, and their simulator dependencies are not vLLM-Omni dependencies.

### DROID Sim-Eval

`droid_sim_eval_client.py` runs a DROID rollout through Isaac Lab / `sim-evals`
against an already running vLLM DreamZero OpenPI server.

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

The client keeps the upstream DreamZero sim-eval behavior: 24-step action
chunks, 8 executed open-loop control steps before replanning, and DROID camera
observation extraction from `external_cam`, `external_cam_2`, and `wrist_cam`.

### MolmoSpaces Evaluation

`molmospace_dreamzero_eval_demo.py` evaluates DreamZero through the same vLLM
OpenPI server on MolmoSpaces benchmarks. Install MolmoSpaces and prepare its
benchmark assets by following the upstream MolmoSpaces documentation.

```bash
python examples/online_serving/dreamzero/molmospace_dreamzero_eval_demo.py \
  --benchmark_dir "${MOLMOSPACES_BENCHMARK_DIR}/20260327/ithor/FrankaCloseHardBench/FrankaCloseHardBench_20260206_json_benchmark" \
  --output_dir outputs/dreamzero/molmospaces \
  --max_episodes 1 \
  --task_horizon_steps 240 \
  --episode_idx 1
```

## Optional upstream parity checks

The upstream DreamZero-dependent parity tests are kept under:

- `tests/dreamzero/upstream/`

Those tests require a local upstream DreamZero checkout and are not needed for
the standard vLLM example above.
