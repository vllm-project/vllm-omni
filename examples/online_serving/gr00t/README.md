# GR00T-N1.7 OpenPI Example

This example shows how to serve NVIDIA Isaac GR00T-N1.7 with
`vllm serve --omni` and connect an OpenPI-compatible client.

## Files

- `run_server.sh`: launch the GR00T-N1.7 OpenPI policy server.
- `openpi_client.py`: minimal websocket client that sends one synthetic
  observation and validates the dict-of-ndarrays action response.

## Environment requirements

- `run_server.sh` runs inside the standard `vllm-omni` environment.
- The client requires two optional dependencies:

  ```bash
  pip install openpi-client websockets
  ```

## Start the server

```bash
CUDA_VISIBLE_DEVICES=0 examples/online_serving/gr00t/run_server.sh
```

Overridable env vars: `MODEL` (default `nvidia/GR00T-N1.7-3B`, set to a
local checkout path if you've already downloaded it), `HOST`, `PORT`,
`DEPLOY_CONFIG` (default `vllm_omni/deploy/gr00t.yaml`),
`SERVED_MODEL_NAME`, `ATTENTION_BACKEND`, `DIFFUSION_ATTENTION_BACKEND`.

The backbone (`nvidia/Cosmos-Reason2-2B`, referenced from the GR00T
checkpoint's `config.json`) is a **gated HF repo** — first time you run,
either log in via `huggingface-cli login` or set `HF_TOKEN` to a token
that has been granted access.

The websocket endpoint is `ws://127.0.0.1:8000/v1/realtime/robot/openpi`.

## Run the client

```bash
python examples/online_serving/gr00t/openpi_client.py \
    --host 127.0.0.1 --port 8000 \
    --embodiment oxe_droid_relative_eef_relative_joint
```

The client sends one observation containing:

- `embodiment` tag (mapped server-side via
  `EMBODIMENT_TAG_TO_PROJECTOR_INDEX`).
- `state` dict with `eef_9d` / `gripper_position` / `joint_position` joints.
- `modality_config` so the server knows how to slice the 132-dim action
  trajectory back into the per-key dict shape that issue
  [#3553](https://github.com/vllm-project/vllm-omni/issues/3553) requires.
- A synthetic RGB image batch sized per the `image_resolution` field of the
  server handshake (real deployments swap this for a dataset loader, e.g.
  [Isaac-GR00T's `LeRobotEpisodeLoader`](https://github.com/NVIDIA/Isaac-GR00T)).

It then validates:

- Server returned a dict (per #3553), not a single ndarray.
- All advertised `action_keys` are present.
- Each entry has shape `(action_horizon, key_dim)` with finite values.

For a real evaluation with DROID trajectories, point the client at a
LeRobot dataset loader and replace `_build_synthetic_observation` with the
upstream observation pipeline.

## OpenPI server handshake schema

On connect, the server emits a msgpacked `PolicyServerConfig` payload with
GR00T-specific fields (see `vllm_omni/deploy/gr00t.yaml`):

| Field                   | Value                              |
| ----------------------- | ---------------------------------- |
| `image_resolution`      | `[256, 256]`                       |
| `n_external_cameras`    | `1`                                |
| `needs_wrist_camera`    | `true`                             |
| `action_horizon`        | `40`                               |
| `action_keys`           | `[eef_9d, gripper_position, joint_position]` |
| `supported_embodiments` | 7-tag list (DROID, R1 Pro, SimplerEnv, LIBERO, ...) |
