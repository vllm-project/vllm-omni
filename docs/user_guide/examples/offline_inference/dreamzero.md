# DreamZero

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/robot_policy>.

## Overview

DreamZero (`GEAR-Dreams/DreamZero-DROID`) is a Vision-Language-Action world
model: given a language task and three camera observations (two external, one
wrist), it autoregressively predicts joint-position action chunks and the
corresponding video rollout of the scene.

DreamZero runs through the shared robot-policy task example
`examples/offline_inference/robot_policy/robot_policy.py`.
Model-specific behavior is declared in
`vllm_omni/model_extras/dreamzero.py`
and registered in the model-extras registry — there is no per-model example
script.

The rollout is autoregressive: the first request carries the initial frame with
`reset=true` and opens a session; every subsequent request sends one 4-frame
observation chunk under the same `session_id` and receives one action chunk
plus the video latents decoded so far.

## Setup

Download the example camera assets (3 MP4 files from a DROID episode):

```bash
hf download YangshenDeng/vllm-omni-dreamzero-assets --repo-type dataset --local-dir outputs/dreamzero/assets
```

The assets directory must contain:

- `exterior_image_1_left.mp4`
- `exterior_image_2_left.mp4`
- `wrist_image_left.mp4`

## Run the example

```bash
python examples/offline_inference/robot_policy/robot_policy.py \
  --model GEAR-Dreams/DreamZero-DROID \
  --deploy-config vllm_omni/deploy/dreamzero.yaml \
  --data-dir outputs/dreamzero/assets \
  --task "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan" \
  --extra-body '{"session_id": "dreamzero_example", "num_chunks": 15, "repeat_chunk_observations": false}'
```

Outputs:

- `robot_policy_output.npz` — the stacked predicted action sequence, one row
  `[action_horizon, action_dim]` per AR step, plus `num_steps`.
- `robot_policy_output.mp4` — the decoded video rollout, exported by the
  worker extension after the last step.

## Model-specific knobs

DreamZero declares its request params in
`vllm_omni/model_extras/dreamzero.py`; the rollout controls below go through
`--extra-body` and are consumed by the observation builder:

| Knob                        | Type   | Default | Meaning                                                                                   |
| --------------------------- | ------ | ------- | ----------------------------------------------------------------------------------------- |
| `session_id`                | str    | random  | AR session identity; reuse the same id to continue one rollout                            |
| `num_chunks`                | int    | `15`    | Number of 4-frame AR chunks after the initial frame (matches the upstream export default) |
| `repeat_chunk_observations` | bool   | `false` | Repeat the last valid chunk when the assets run out of frames                             |
| `reset`                     | bool   | —       | Set by the task script on the first request; resets the session KV state                  |

Memory: with the default `vllm_omni/deploy/dreamzero.yaml` (TP=1, bf16) peak
VRAM is about 71 GiB — use one 80 GiB GPU. `--enable-cpu-offload` and
`--enable-layerwise-offload` reduce the peak when the weights do not fit.

## Measures and validation

- The standard path is covered by
  `tests/e2e/offline_inference/test_dreamzero.py`
  (AR rollout + actions + video export) and the CPU unit tests in
  `tests/model_extras/test_model_extras.py`.
- For online serving over the OpenPI robot protocol see
  `examples/online_serving/dreamzero/`.

## FAQ

- **Q: Which cameras does the model expect?** Two external left cameras plus
  one left wrist camera, named as in the asset files above. The embodiment
  (`roboarena` DROID joint-space) is resolved from the deploy config.
- **Q: Why is the first request slower?** It carries `reset=true`, which
  rebuilds the session KV state; later chunks reuse it. DreamZero also enables
  `step_cache` by default, which warms up over the first denoising steps.
