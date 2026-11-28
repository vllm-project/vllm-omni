# Motubrain 2.0 robot policy

Motubrain predicts a chunk of controller actions from synchronized RGB views,
a language instruction and the current robot pose. A vLLM-Omni diffusion worker
loads the Motubrain checkpoint and serves its decoded actions over the
Motubrain WebSocket at `/v1/realtime/robot/motubrain2`. The single-GPU BF16
configuration supports Astribot S1 and RoboTwin.

## Model and controller contract

| Profile | Camera observation keys in order | Controller fields in wire order | Output |
| --- | --- | --- | --- |
| RoboTwin | `cam_high`, `cam_left_wrist`, `cam_right_wrist` | Left end-effector pose, left gripper, right pose, right gripper | 24 × 16 |
| Astribot S1 | `head_camera`, `left_camera`, `right_camera` | Torso pose, two head joints, left pose, left gripper, right pose, right gripper | 24 × 25 |

Each pose contains XYZ followed by an XYZW quaternion. Supply a finite
`state` in the same controller layout, the calibrated 4 × 4 base-from-camera
`camera_extrinsic`, a nonempty `prompt`, and one RGB frame per camera.
The 55 coordinates inside the denoiser are mapped to the wire layout by the
robot profile. Clients receive a float32 array and can read its declared
dimensions and action space from the initial handshake.

The policy reads its sampling schedule from the checkpoint. The released Base
and Distill checkpoints use 30-step Euler and four-step consistency sampling,
respectively. Each named session starts its random generator at the deployment
seed (0 by default) and advances it on each successful unseeded prediction.
Reset restarts that stream.
An explicit `seed` or `sampling_params.seed` applies to one request and
preserves the session's generator position. Anonymous predictions start at
the deployment seed for each request. Set `sampling_params.num_inference_steps`
to override the schedule. Camera clips can be sent
under their profile keys at the top level or inside a `cameras` mapping.
Set the request field `return_native_outputs` to `true` to receive a mapping
with `video` (sampled latent tokens), `action` (model coordinates),
`decoded_action` (controller coordinates) and `decoded_valid` (boolean mask)
from the same inference call. The default reply is the controller action array.
Read `decoded_valid` before using native controller coordinates. A native
response with masked coordinates clears the session's RTC prefix. An ordinary
controller request requires every decoded coordinate to be valid.
For real-time chunking, set `rtc.enabled` and
`motubrain2_metadata.needs_session_id` to true under
`stages[0].engine_extras.model_config` in the deployment YAML.
Set `rtc.execute_prefix` to the number of actions the client executes per chunk.
Use a unique `session_id` for each RTC rollout, e.g. a UUID. Send it with
the first observation and again after a reset. The connection reuses the
ID for subsequent observations. The shared ID `default` is reserved for
anonymous requests. With RTC disabled, clients may omit `session_id`.

A request can also supply `rtc` with a nonnegative integer `delay`, a
`future` array of controller waypoints, and an optional `anchor` waypoint.
The policy uses up to `delay` rows of `future` and uses the anchor as its
controller state for that prediction. An anchor requires at least one future
waypoint. This request overrides the automatic session prefix and also works
with automatic RTC disabled. Send `{"rtc": {"delay": 0}}` for an empty prefix.

## Runtime

Install this vLLM-Omni checkout and its matching vLLM release (0.34.0).
Serving requires access to the model authors' separate native **motubrain2**
source and checkpoints, licensed under CC BY-NC-SA 4.0. Install that runtime
in the server's Python environment. The mock tests run without these assets.
[Motubrain source and license](https://github.com/shengshu-ai/Motubrain-2.0) ·
[Distill checkpoint](https://huggingface.co/GensPI/Motubrain2.0-Distill)

The commands below use a vLLM-Omni checkout at `/opt/vllm-omni`. The DA3
geometry profiles use the DINOv2 backbone from the pinned Depth Anything 3
source. That import needs `einops`, `omegaconf` and `addict`. Adding the source
directory to the worker's Python path lets it share vLLM's NumPy 2 environment.

```bash
git clone https://github.com/shengshu-ai/Motubrain-2.0.git /opt/Motubrain-2.0
git -C /opt/Motubrain-2.0 checkout 308137cd0ce9de48ec5e6ed3bd0619859db22c9f
uv pip install /opt/Motubrain-2.0 einops omegaconf addict
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git /opt/Depth-Anything-3
git -C /opt/Depth-Anything-3 checkout 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
export PYTHONPATH="/opt/Depth-Anything-3/src${PYTHONPATH:+:${PYTHONPATH}}"
python -c 'from motubrain2.inference.policy import RawPolicy; from depth_anything_3.model.dinov2.dinov2 import DinoV2'
```

The two deployments use the Wan2.2 VAE, T5Gemma encoder and DA3-LARGE-1.1.
The default deploy config specifies their Hub IDs. For offline serving, use
local copies in `stages[0].engine_extras.model_config`. For an original Wan
`.pth` file set `wan_config` to the matching VAE config directory and
`wan_subfolder` to null. For a standalone DA3 safetensors file set
`da3_config` to its model config.

When the policy bundle embeds robot profiles, select one with
`model_config.embodiment`; otherwise supply `model_config.profile` or the
serving directory's profile. Set `async_io` for native pinned CUDA transfers.

On Blackwell, set `mxfp8: true` under
`stages[0].engine_extras.model_config` to load the standalone
`model-mxfp8.safetensors` in the native policy bundle. The bundle also
requires the matching `manifest.json`, `config.json` and `sampling.json`.
Keep the stage dtype at `bfloat16`. By default the worker loads
`model.safetensors`, including when both checkpoints are present.

The native MXFP8 extra requires FlashInfer 0.7.5. vLLM 0.34.0 pins
FlashInfer 0.7.3 in its standard CUDA environment. Use a dedicated Motubrain
worker environment with the 0.7.5 CUTLASS MXFP8 kernels for your GPU.
[vLLM CUDA dependency pin](https://github.com/vllm-project/vllm/blob/v0.34.0/requirements/cuda.txt#L13)

Omitting `dit_cache_enabled` (or setting it to null) enables caching for
the released Base recipe and leaves it off for Distill. Base is identified
by the checkpoint's original 30-step Euler recipe with video/action shifts
6/1. Serving overrides such as a shorter request keep that choice.

Set `dit_cache_enabled: true` to enable caching explicitly, or `false` to
disable it for either checkpoint. With caching enabled and no explicit
threshold, the primary and secondary cosine thresholds are 0.95 and 0.93.
A custom primary threshold can also enable caching in automatic mode.
The following deployment overlay keys belong under
`stages[0].engine_extras.model_config`. The default intervals are 4 and 2
steps (including the evaluated step), with damping 0.25.

```yaml
dit_cache_enabled: true
dit_cache_threshold: 0.95
dit_cache_threshold_2: 0.93
dit_cache_skip_steps: 4
dit_cache_skip_steps_2: 2
dit_cache_damping: 0.25
```

Leave the Omni `cache_backend` at its default. These thresholds select
approximate reuse inside the native denoising loop. Compare cached and
uncached controller outputs on the same observations and seeds.

## Prepare and serve on NVIDIA B200

This example serves Astribot. It assumes the native checkpoint has been
downloaded into `/models/Motubrain2.0-Distill` and its robot-specific
normalization statistics are available at `/models/calibration/pt3_stats.json`.
Use the statistics and extrinsics for the robot and checkpoint being evaluated.

```bash
mkdir -p /models/astribot-serving
ln -s /models/Motubrain2.0-Distill /models/astribot-serving/policy
cp /opt/Motubrain-2.0/motubrain2/data/profiles/astribot.json /models/astribot-serving/profile.json
ln -s /models/calibration/pt3_stats.json /models/astribot-serving/pt3_stats.json
printf '%s\n' '{"architectures":["Motubrain2Pipeline"]}' > /models/astribot-serving/config.json
printf '%s\n' '{"_class_name":"Motubrain2Pipeline"}' > /models/astribot-serving/model_index.json

vllm serve /models/astribot-serving --omni \
  --deploy-config /opt/vllm-omni/vllm_omni/deploy/motubrain2-astribot.yaml \
  --port 8000
```

The RoboTwin checkpoint ships as a training snapshot. Use the upstream converter
on a compute node to prepare the four-step DMD policy. The conversion writes
a second copy of the weights and preserves the sampling schedule in the bundle.
[RoboTwin checkpoint](https://huggingface.co/Motubrain/Motubrain2.0-robotwin2.0)

```bash
mkdir -p /models/robotwin-serving
python -m motubrain2.checkpoints.convert \
  --source /models/Motubrain2.0-robotwin2.0/weights/dmd-step1000 \
  --sampler consistency --steps 4 --video-shift 6 --action-shift 1 \
  --output /models/robotwin-serving/policy
cp /opt/Motubrain-2.0/motubrain2/data/profiles/robotwin.json /models/robotwin-serving/profile.json
ln -s /models/Motubrain2.0-robotwin2.0/weights/dmd-step1000/rw_n2_q1.json /models/robotwin-serving/rw_n2_q1.json
printf '%s\n' '{"architectures":["Motubrain2Pipeline"]}' > /models/robotwin-serving/config.json
printf '%s\n' '{"_class_name":"Motubrain2Pipeline"}' > /models/robotwin-serving/model_index.json
vllm serve /models/robotwin-serving --omni \
  --deploy-config /opt/vllm-omni/vllm_omni/deploy/motubrain2.yaml
```

The Astribot split deployment uses one worker that can see two CUDA devices.
Use `/opt/vllm-omni/vllm_omni/deploy/motubrain2-astribot-2gpu.yaml` to put
the denoiser on local GPU 0 and the encoders on local GPU 1. Keep model
parallelism at one. Both deployments use BF16 denoiser weights by default.
The default kernel settings allow the policy's local SwiGLU compilation.
Pass `--enforce-eager` to select eager kernels.
FP32 denoiser weights require more memory. The external encoders apply their
own precision policy.

## Connect a robot client

Connect the client to `ws://localhost:8000/v1/realtime/robot/motubrain2`.
Replace `localhost` with the inference host when connecting from another
machine. The server sends the profile metadata in the first binary MessagePack
frame. Send one observation per binary frame (up to 64 MiB) and wait for its
reply before submitting the next observation. Errors arrive as a mapping with
`type: error` and a `message`. Existing robot clients that use the
`__ndarray__` MessagePack marker with `dtype`, `shape` and `data` fields can
keep their observation and reply formats when pointed at this URL.
For Astribot S1 deployments using a RealWorldInference adapter, set the
adapter's backend URL to the Motubrain endpoint. The adapter still supplies
the calibrated extrinsic, synchronized camera frames and formatted instruction.

The following example replays an Astribot observation saved as a NumPy archive.
It reads the three RGB frames, the state and the extrinsic from the same capture.
Run it in a client environment with the codec from the same vLLM-Omni checkout.
The example checkout path on the client is `/opt/vllm-omni`.

```bash
uv pip install numpy msgpack 'websockets>=12'
export PYTHONPATH="/opt/vllm-omni/vllm_omni/entrypoints${PYTHONPATH:+:${PYTHONPATH}}"
```

```python
from uuid import uuid4

import numpy as np
from motubrain2_codec import pack_message, unpack_message
from websockets.sync.client import connect

uri = "ws://localhost:8000/v1/realtime/robot/motubrain2"
with connect(uri, compression=None, max_size=None) as ws:
    meta = unpack_message(ws.recv())
    assert (meta["action_horizon"], meta["action_dim"]) == (24, 25)
    with np.load("/models/calibration/astribot_observation.npz", allow_pickle=False) as record:
        keys = (*meta["camera_sources"], "state", "camera_extrinsic")
        observation = {key: record[key] for key in keys}
    session = str(uuid4())
    ws.send(pack_message({**observation, "prompt": "Pick up the object", "session_id": session, "seed": 42}))
    actions = unpack_message(ws.recv())
    assert isinstance(actions, np.ndarray) and actions.dtype == np.float32
    assert actions.shape == (24, 25) and np.isfinite(actions).all()
    ws.send(pack_message({"endpoint": "reset", "session_id": session}))
    assert unpack_message(ws.recv())["status"] == "reset successful"
```

## Verification

The CPU suite exercises the Motubrain WebSocket and compares four outputs of
a small policy with the native CLI. The checkpoint-gated WebSocket test
accepts a prepared serving directory and its matching deployment.

```bash
python -m pytest --run-level full_model /opt/vllm-omni/tests/diffusion/models/motubrain2
MOTUBRAIN2_TEST_MODEL=/models/astribot-serving \
MOTUBRAIN2_TEST_DEPLOY_CONFIG=/opt/vllm-omni/vllm_omni/deploy/motubrain2-astribot.yaml \
  python -m pytest --run-level full_model /opt/vllm-omni/tests/e2e/online_serving/test_motubrain2_websocket.py
```
