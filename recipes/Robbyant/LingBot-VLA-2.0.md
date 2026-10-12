# LingBot-VLA 2.0

LingBot-VLA 2.0 predicts a continuous action chunk from RGB cameras, an
instruction and robot state. This integration uses the LeRobot-format
checkpoints and the existing OpenPI websocket endpoint.

## Supported configuration

One CUDA GPU, FP32 parameters, eager execution and one observation per request.
The RoboTwin checkpoint returns 50 actions with 14 joint/gripper values each.
The processor restores checkpoint statistics and maps the robot's values to
55 canonical slots. Each observation gets a fresh prefix KV cache, reused only
within its flow steps. Clients choose how many actions to execute before the
next observation.

The initial path excludes RTC, observation history, relative actions, reduced
precision, quantization, compilation, CUDA graphs and multi-GPU execution.
Current/future query embeddings remain in inference. Training teachers and
auxiliary prediction heads are not executed.

## Run the server

```bash
hf download lerobot/lingbot_vla_v2_robotwin \
  --revision 54f79a4b6e9166bda91b5010088097ff3191bca2 --local-dir ./lingbot-robotwin

vllm serve ./lingbot-robotwin --omni \
  --deploy-config vllm_omni/deploy/lingbot_vla_v2.yaml \
  --dtype float32 --enforce-eager
```

Keep the checkpoint's processor JSON, normalization tensors and tokenizer/image
processor directories together. For offline use, set the stage's
`model_config.tokenizer_path` to a local Qwen3-VL-4B-Instruct config directory.
The default architecture revision is pinned in the model configuration.

## Send an observation

Use the official `openpi-client` package in a separate client environment. The
validated client revision is `215abfb217dbac7d5f1273282331b9b1866c0479` from
Physical-Intelligence/openpi. Install `typing-extensions` alongside that version.
Its NumPy constraint must not downgrade the serving environment.

A synthetic request:

```python
import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

policy = WebsocketClientPolicy(host="ws://127.0.0.1:8000/v1/realtime/robot/openpi")
observation = {
    name: np.zeros((256, 256, 3), dtype=np.uint8)
    for name in (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    )
}
observation.update(
    state=np.zeros(14, dtype=np.float32),
    prompt="pick up the red block",
    seed=123,
    sampling_params={"num_inference_steps": 10},
)
actions = policy.infer(observation)  # float32[50, 14]
```

Images are RGB uint8 HWC, or floating RGB HWC/CHW arrays in [0, 1]. Missing
configured cameras are masked, with at least one present. Camera names follow
the checkpoint's ordered input features. Set `model_config.image_key_map` for
client aliases. State and returned actions are in robot units, so do not
normalize them again in the client. This example returns actions without
commanding hardware.

## Base checkpoint

`lerobot/lingbot_vla_v2_base` uses 55-D canonical inputs/outputs and has no robot
calibration statistics. It requires `model_config.inputs_normalized: true` and
55-D OpenPI metadata. Do not reuse the 14-D RoboTwin deployment unchanged.

## Validation

```bash
pytest tests/diffusion/models/lingbot_vla_v2
pytest tests/e2e/online_serving/test_lingbot_vla_v2.py
```

The full-checkpoint parity test also requires the pinned LeRobot source and
local weights described in its fixture. The online test compares all 700 action
values to an HF-generated reference, then checks interleaving, invalid-input
recovery, reset and reconnect. Model inference and serving tests do not measure
robot task success.

Match matmul and cuDNN TF32 settings separately. The online reference uses
matmul TF32 off and cuDNN TF32 on, matching the default service. Deterministic
model tests disable both. MoE accumulation can amplify small rounding changes,
so a fixed seed alone does not imply bitwise CUDA repeatability.

Inference modules are adapted from the Apache-2.0 LingBot implementation in
LeRobot, with original copyright notices retained. See the
[upstream model](https://github.com/Robbyant/lingbot-vla-v2).

For a direct-policy latency comparison, install the pinned LeRobot implementation
in the same environment and run:

```bash
python benchmarks/lingbot_vla_v2/bench_policy.py \
  --model ./lingbot-robotwin --backbone-config ./qwen3-vl-config
```

This compares model-only and processor-plus-model time with fixed noise, three
warmups and ten interleaved ABBA blocks. It excludes loading and network time.
