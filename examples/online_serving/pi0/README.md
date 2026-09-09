# π0 (Pi-Zero) VLA — OpenPI realtime serving

[π0](https://www.physicalintelligence.company/blog/pi0) is a Vision-Language-Action
model from Physical Intelligence: multi-camera images + a language instruction +
robot proprioceptive state → a continuous action chunk via flow-matching denoising
(it does **not** emit text tokens). This example serves π0 over the OpenPI realtime
websocket protocol at `/v1/realtime/robot/openpi`.

## Install extras

The core `pip install -e .` does not include the OpenPI client used here:

- `openpi-client` (from the [openpi](https://github.com/Physical-Intelligence/openpi)
  repo: `pip install -e packages/openpi-client`), `websockets`, `msgpack`, `msgpack-numpy`

## Weights

π0 uses the LeRobot `lerobot/pi0_base` checkpoint (HF, ~13 GB). Either let the server
download it (`MODEL=lerobot/pi0_base`) or point at a local copy
(`MODEL=/path/to/pi0_base`).

## Run the server

```bash
vllm serve lerobot/pi0_base --omni --port 8000 \
    --served-model-name pi0 \
    --deploy-config vllm_omni/deploy/pi0.yaml \
    --enforce-eager --disable-log-stats
```

The deploy config (`vllm_omni/deploy/pi0.yaml`) declares a single diffusion stage
(`Pi0Pipeline`), float32, `max_num_seqs: 1`, and the `policy_server_config` handshake
metadata (3 cameras, `joint_position`, action horizon 50, action dim 32).

### Commonly adjusted `model_config` keys

These live under `stages[0].model_config` in `vllm_omni/deploy/pi0.yaml`:

| Key | Default | Meaning |
|---|---|---|
| `chunk_size` | `50` | Action-chunk length (timesteps) the model predicts per inference. |
| `num_inference_steps` | `10` | Flow-matching Euler denoising steps. |
| `max_action_dim` | `32` | Action dimensionality (state/action are padded to this). |
| `max_state_dim` | `32` | Proprioceptive-state dimensionality (zero-padded to this). |
| `image_resolution` | `[224, 224]` | Per-camera input size (square; SigLIP). |
| `tokenizer_max_length` | `48` | Max PaliGemma prompt tokens. |
| `max_cameras` | `3` | Camera slots the model attends to (real + `-1`-padded). |
| `image_feature_keys` | 3 `observation.images.*` keys | Camera order the model attends to. |
| `image_key_map` | `{}` | Map raw obs camera keys → `image_feature_keys` (empty = verbatim). |

The `policy_server_config` block below them is the OpenPI handshake metadata
advertised to the client; keep its `action_horizon` / `action_dim` /
`image_resolution` in sync with the `model_config` values above. The e2e test
`tests/e2e/online_serving/test_pi0_expansion.py::test_pi0_openpi_online` connects to a
live server and asserts the advertised metadata matches these `pi0.yaml` values.

## Run the client

```bash
python examples/online_serving/pi0/openpi_client.py --host 127.0.0.1 --port 8000 \
    --prompt "pick up the red block and place it in the bin"
```

It connects, prints the server metadata, sends robot observations, and prints the
returned `[action_horizon, action_dim] = [50, 32]` action chunks. Replace the blank
cameras / zero state in `_make_dummy_obs` with real frames (HWC uint8) and
proprioceptive state to drive a robot.

### Observation format

The client sends a flat dict per inference:

```python
{
    "observation.images.base_0_rgb":       np.uint8[H, W, 3],
    "observation.images.left_wrist_0_rgb": np.uint8[H, W, 3],
    "observation.images.right_wrist_0_rgb":np.uint8[H, W, 3],
    "state":   np.float32[state_dim],   # zero-padded to max_state_dim=32 server-side
    "prompt":  "pick up the red block",
    "session_id": "<uuid>",             # accepted but ignored (π0 is stateless)
}
```

Camera keys must match the server's `image_feature_keys` (the checkpoint's
`input_features` order). If your robot uses different camera names, set
`model_config.image_key_map` in `pi0.yaml` to map raw obs keys → feature keys.

## Correctness

π0's flow-matching kernel is bit-for-bit matched to the LeRobot `PI0Policy` reference
(`max|Δ| = 7.15e-07`, CPU/float32, fixed noise; see
`tests/diffusion/models/pi0/test_pi0_parity.py::test_pi0_vllm_omni_vs_lerobot`) and is
version-stable across transformers releases (the version-stability checks in
`tests/diffusion/models/pi0/test_pi0_units.py`). An OpenPI websocket online-serving e2e
lives in `tests/e2e/online_serving/test_pi0_expansion.py::test_pi0_openpi_online`.

## Limitations

- **Normalization stats**: `lerobot/pi0_base` uses identity normalization, which is
  fully supported (state passes through, actions are returned in the model's space).
  Per-dataset `norm_stats` declared in a checkpoint's `config.json` are honored, but
  stats stored only in LeRobot's `policy_preprocessor.json` companion safetensors are
  not yet loaded — a fine-tuned checkpoint that relies on those would need that bridge
  added before its actions are in real-world units.
