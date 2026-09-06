# Pi0 (π0, Pi-Zero) VLA

> Vision-Language-Action policy serving over the OpenPI realtime websocket

## Summary

- Vendor: Physical Intelligence (checkpoint redistributed by LeRobot)
- Model: `lerobot/pi0_base`
- Task: Vision-Language-Action — multi-camera images + a language instruction +
  robot proprioceptive state → a continuous action chunk
  `[action_horizon, action_dim] = [50, 32]` via flow-matching denoising. π0 does
  **not** emit text tokens.
- Mode: Online serving via the OpenPI realtime robot API
  (`/v1/realtime/robot/openpi` websocket)
- Maintainer: Community

## When to use this recipe

Use this recipe to deploy `lerobot/pi0_base` as a robot policy server: a robot
client connects over the OpenPI websocket and, per control step, sends an
observation (camera frames + state + instruction) and receives an action chunk.

A single pipeline class (`Pi0Pipeline`) serves the policy. π0 is stateless across
calls (no KV reuse), so the protocol's `session_id` / `reset` are accepted but
ignored.

## Weights

π0 uses the LeRobot `lerobot/pi0_base` checkpoint (HF, ~13 GB). Either let the
server download it (`MODEL=lerobot/pi0_base`) or point at a local copy.

## Hardware Support

### 1x H200 141GB (Online serving)

#### Environment

- OS: Ubuntu 22.04+
- Python: 3.12+
- Driver / runtime: NVIDIA CUDA environment
- vLLM / vLLM-Omni: use the commit you are deploying from
- OpenPI client extras (for the example client): `openpi-client`
  (`pip install -e packages/openpi-client` from the
  [openpi](https://github.com/Physical-Intelligence/openpi) repo), `websockets`,
  `msgpack`, `msgpack-numpy`

#### Command

```bash
vllm serve lerobot/pi0_base \
  --omni \
  --host 0.0.0.0 --port 8000 \
  --served-model-name pi0 \
  --deploy-config vllm_omni/deploy/pi0.yaml \
  --enforce-eager --disable-log-stats
```

The deploy config (`vllm_omni/deploy/pi0.yaml`) declares a single diffusion stage
(`Pi0Pipeline`), `float32`, `max_num_seqs: 1`, and the `policy_server_config`
handshake metadata (3 cameras, `joint_position`, action horizon 50, action dim
32) that enables the OpenPI realtime serving layer.

#### Verification

Run the example client (sends a synthetic observation and prints the returned
action chunk):

```bash
python examples/online_serving/pi0/openpi_client.py \
  --host 127.0.0.1 --port 8000 \
  --prompt "pick up the red block and place it in the bin"
```

It connects, prints the server metadata, sends robot observations, and prints the
returned `[action_horizon, action_dim] = [50, 32]` action chunks. The observation
is a flat dict per inference:

```python
{
    "observation.images.base_0_rgb":        np.uint8[H, W, 3],
    "observation.images.left_wrist_0_rgb":  np.uint8[H, W, 3],
    "observation.images.right_wrist_0_rgb": np.uint8[H, W, 3],
    "state":   np.float32[state_dim],   # zero-padded to max_state_dim=32 server-side
    "prompt":  "pick up the red block",
    "session_id": "<uuid>",             # accepted but ignored (π0 is stateless)
}
```

Camera keys must match the server's `image_feature_keys` (the checkpoint's
`input_features` order). If your robot uses different camera names, set
`model_config.image_key_map` in `pi0.yaml` to map raw obs keys → feature keys.

## Correctness

π0's flow-matching kernel is bit-for-bit matched to the LeRobot `PI0Policy`
reference (`max|Δ| = 7.15e-07`, CPU/float32, fixed noise; see
`tests/diffusion/models/pi0/test_pi0_parity.py::test_pi0_vllm_omni_vs_lerobot`).

## References

- Model blog: <https://www.physicalintelligence.company/blog/pi0>
- Checkpoint: <https://huggingface.co/lerobot/pi0_base>
- Pipeline: [`vllm_omni/diffusion/models/pi0/pipeline_pi0.py`](../../vllm_omni/diffusion/models/pi0/pipeline_pi0.py)
- Deploy config: [`vllm_omni/deploy/pi0.yaml`](../../vllm_omni/deploy/pi0.yaml)
- Example client: [`examples/online_serving/pi0/`](../../examples/online_serving/pi0/)
- Tests: CPU units + LeRobot parity in [`tests/diffusion/models/pi0/`](../../tests/diffusion/models/pi0/),
  OpenPI websocket e2e in
  [`tests/e2e/online_serving/test_pi0_expansion.py`](../../tests/e2e/online_serving/test_pi0_expansion.py)
