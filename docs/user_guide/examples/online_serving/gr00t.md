# GR00T-N1.7

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/gr00t>.

NVIDIA Isaac GR00T-N1.7 is a vision-language-action (VLA) model with a
Cosmos-Reason2-2B / Qwen3-VL backbone and a diffusion action head.  This
page documents how to serve it from vllm-omni and how OpenPI clients talk
to it.

Tracking issue: <https://github.com/vllm-project/vllm-omni/issues/3553>.

## Installation

Please refer to [README.md](https://github.com/vllm-project/vllm-omni/tree/main/README.md)
for the base `vllm-omni` install.

The OpenPI client requires two optional dependencies:

```bash
pip install openpi-client websockets
```

## Architecture

GR00T-N1.7 is a single-stage diffusion pipeline:

| Stage | Worker | Description |
| :---- | :----- | :---------- |
| Stage 0 | Diffusion | Qwen3-VL backbone (truncated to `select_layer=16`) + flow-matching DiT action head with 4 Euler steps. |

The model emits an action trajectory of shape `[action_horizon=40,
max_action_dim=132]`.  Per [issue
#3553](https://github.com/vllm-project/vllm-omni/issues/3553) the OpenPI
response is a **dict of ndarrays** — one entry per modality-config action
key — rather than a single ndarray.

## Launch the Server

From the repo root:

```bash
CUDA_VISIBLE_DEVICES=0 examples/online_serving/gr00t/run_server.sh
```

Override defaults via env vars:

| Variable                       | Default                                |
| ------------------------------ | -------------------------------------- |
| `MODEL`                        | `nvidia/GR00T-N1.7-3B` (HF id or local path) |
| `HOST` / `PORT`                | `127.0.0.1` / `8000`                   |
| `DEPLOY_CONFIG`                | `vllm_omni/deploy/gr00t.yaml`          |
| `SERVED_MODEL_NAME`            | `gr00t-n17`                            |
| `ATTENTION_BACKEND`            | `torch`                                |
| `DIFFUSION_ATTENTION_BACKEND`  | `TORCH_SDPA`                           |

The OpenPI websocket endpoint is `ws://127.0.0.1:8000/v1/realtime/robot/openpi`.

## OpenPI Handshake

On connect, the server emits a msgpacked `PolicyServerConfig` payload
populated from the deploy YAML (`vllm_omni/deploy/gr00t.yaml`):

| Field                    | Value                                          |
| ------------------------ | ---------------------------------------------- |
| `image_resolution`       | `[256, 256]`                                   |
| `n_external_cameras`     | `1`                                            |
| `needs_wrist_camera`     | `true`                                         |
| `needs_stereo_camera`    | `false`                                        |
| `needs_session_id`       | `true`                                         |
| `action_horizon`         | `40`                                           |
| `action_keys`            | `[eef_9d, gripper_position, joint_position]`   |
| `supported_embodiments`  | 7-tag list (DROID, R1 Pro, SimplerEnv, LIBERO, ...) |

Clients can read these fields to size their observation batches and to
know which action keys the server will return.

## Run the Client

```bash
python examples/online_serving/gr00t/openpi_client.py \
    --host 127.0.0.1 --port 8000 \
    --embodiment oxe_droid_relative_eef_relative_joint
```

The client sends one observation containing:

- `embodiment`: an embodiment tag from the `supported_embodiments` list.
- `state`: dict of joint names → list of floats
  (e.g. `eef_9d`, `gripper_position`, `joint_position`).
- `modality_config`: per-key `{start, end}` offsets that describe how to
  pack `state` into the 132-dim proprio vector and how to slice the 132-dim
  action output back into the per-key dict.
- `prompt`: language instruction.
- `images`: one or more RGB image batches sized per the handshake's
  `image_resolution`.

The client then validates:

- The server returned a `dict[str, ndarray]` (per #3553).
- All advertised `action_keys` are present.
- Each entry has shape `(action_horizon, key_dim)` with finite values.

For real evaluation (e.g. DROID rollouts) replace the synthetic image batch
with a dataset loader call — Isaac-GR00T's
[`LeRobotEpisodeLoader`](https://github.com/NVIDIA/Isaac-GR00T) is a
drop-in choice for the upstream demo datasets.

## Supported Embodiments

| Tag                                                       | Projector ID |
| --------------------------------------------------------- | ------------ |
| `oxe_droid_relative_eef_relative_joint`                   | 24           |
| `xdof_relative_eef_relative_joint`                        | 27           |
| `real_g1_relative_eef_relative_joints`                    | 25           |
| `real_r1_pro_sharpa_relative_eef`                         | 26           |
| `unitree_g1_full_body_with_waist_height_nav_cmd`          | 25           |
| `simpler_env_google`                                      | 0            |
| `simpler_env_widowx`                                      | 1            |
| `libero_sim`                                              | 2            |
| `new_embodiment`                                          | 10           |

The full mapping is in
`vllm_omni/diffusion/models/gr00t/transform.py::EMBODIMENT_TAG_TO_PROJECTOR_INDEX`,
ported verbatim from Isaac-GR00T upstream.
