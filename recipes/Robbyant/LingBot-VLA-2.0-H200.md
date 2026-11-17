# LingBot-VLA 2.0 robot policy serving — H200

> Serve the LingBot-VLA 2.0 vision-language-action policy over the OpenPI
> realtime websocket: RGB cameras, an instruction and robot state in, a
> continuous action chunk out, on one H200.

## Summary

- Vendor: Robbyant (LeRobot-format checkpoints published by `lerobot`)
- Model: `lerobot/lingbot_vla_v2_robotwin` (RoboTwin, 14-D joint and gripper
  actions) and `lerobot/lingbot_vla_v2_base` (55-D canonical actions)
- Task: vision-language-action policy. Up to three RGB cameras, an instruction
  and robot state produce a continuous `[50, 14]` action chunk through
  flow-matching denoising.
- Mode: online serving through the OpenPI realtime robot endpoint
  (`/v1/realtime/robot/openpi`)
- Hardware: 1x NVIDIA H200 141GB
- Maintainer: Community

## When to use this recipe

Use this recipe to deploy the RoboTwin checkpoint as a robot policy server. A
robot client connects over the OpenPI websocket and, for each control step,
sends one observation and receives a full action chunk. The client chooses how
many actions to execute before the next observation.

The policy is stateless across observations. Each request builds a fresh prefix
KV cache, reused only within its own flow-matching steps, so `session_id` and
`reset` carry no state.

## Supported model contract

| Property | RoboTwin checkpoint |
| --- | --- |
| Cameras | 1 to 3 of `observation.images.cam_high`, `observation.images.cam_left_wrist` and `observation.images.cam_right_wrist`, in checkpoint order. Missing cameras are masked; at least one is required. |
| Images | RGB uint8 HWC, or floating RGB HWC/CHW in [0, 1], resized to 256 × 256 |
| State | `float32[14]` in robot units; the server applies the checkpoint statistics |
| Instruction | non-empty `prompt` (or `task`) string; the chat-formatted text is truncated to 72 tokens |
| Output | `float32[50, 14]` actions in robot units, one chunk per request |
| Request knobs | `seed`; `sampling_params.num_inference_steps` (default 10 flow steps) |
| Deployment profile | `vllm_omni/deploy/lingbot_vla_v2.yaml`: one stage, FP32, eager, `max_num_seqs: 1` |

`lerobot/lingbot_vla_v2_base` uses 55-D canonical inputs and outputs and ships
no robot calibration statistics. Serve it with `model_config.inputs_normalized: true`
and 55-D OpenPI metadata; do not reuse the 14-D RoboTwin deployment unchanged.

## References

- Upstream model: <https://github.com/Robbyant/lingbot-vla-v2>
- Checkpoints: <https://huggingface.co/lerobot/lingbot_vla_v2_robotwin>,
  <https://huggingface.co/lerobot/lingbot_vla_v2_base>
- Pipeline: `vllm_omni/diffusion/models/lingbot_vla_v2/pipeline_lingbot_vla_v2.py`;
  deploy config: `vllm_omni/deploy/lingbot_vla_v2.yaml`
- Tests: `tests/diffusion/models/lingbot_vla_v2/`,
  `tests/e2e/online_serving/test_lingbot_vla_v2.py`
- Benchmark: [`benchmarks/lingbot_vla_v2`](../../benchmarks/lingbot_vla_v2/README.md)

## Hardware

- Accelerator model and per-device memory: NVIDIA H200 141GB
- Number of devices: 1
- Device interconnect: not applicable
- Host memory: enough to stage the 25 GB FP32 checkpoint
- Qualification scope: full-checkpoint parity with the LeRobot reference
  (1, 2 and 3 cameras; 2 and 10 flow steps), the OpenPI serving tests and the
  direct-policy latency comparison below. Robot task success was not evaluated.

## Software environment

- OS: Linux
- Python: 3.12
- Driver / runtime: NVIDIA driver 595 with the vLLM 0.32.0 wheel
  (PyTorch 2.13.0+cu130); Transformers 5.19.0. The parity test and the
  latency benchmark load the LeRobot reference in the same process and use
  Transformers 5.16.1.
- vLLM version: 0.32.0
- vLLM-Omni version or commit: the commit you deploy from

## Command

```bash
hf download lerobot/lingbot_vla_v2_robotwin \
  --revision 54f79a4b6e9166bda91b5010088097ff3191bca2 --local-dir ./lingbot-robotwin

vllm serve ./lingbot-robotwin --omni \
  --deploy-config vllm_omni/deploy/lingbot_vla_v2.yaml \
  --dtype float32 --enforce-eager
```

Keep the checkpoint's processor JSON, normalization tensors and tokenizer and
image-processor directories together. The Qwen3-VL-4B-Instruct architecture
config is fetched at a pinned revision; for offline use, set the stage's
`model_config.tokenizer_path` to a local copy of that config directory.

## Verification

Use the official `openpi-client` package in a separate client environment. The
validated client revision is `215abfb217dbac7d5f1273282331b9b1866c0479` from
Physical-Intelligence/openpi. Install `typing-extensions` alongside that
version. Its NumPy constraint must not downgrade the serving environment.

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
actions = policy.infer(observation)
print(actions.shape, actions.dtype)
```

Expected output:

```text
(50, 14) float32
```

State and returned actions are in robot units, so do not normalize them again
in the client. Set `model_config.image_key_map` for client camera aliases. This
example returns actions without commanding hardware.

Repository checks:

```bash
pytest tests/diffusion/models/lingbot_vla_v2
pytest tests/e2e/online_serving/test_lingbot_vla_v2.py -m "slow and diffusion and H100 and cards_1" --run-level full_model
```

The full-checkpoint parity test also needs the pinned LeRobot source and local
weights described in `tests/diffusion/models/lingbot_vla_v2/test_parity.py`.
The online tests compare all 700 action values with a reference generated by
the LeRobot implementation, then check interleaved observations, invalid-input
recovery, reset and reconnect.

## Notes

- Memory usage: the FP32 weights take about 25 GB (6.26 B inference
  parameters). Serving the repository's online tests (one to three cameras,
  2 and 10 flow steps) peaked at 30.2 GiB of device memory on this H200.
  Keep 40 GiB free, as the serving tests assume.
- Key flags: `--omni` selects the pipeline; `--deploy-config` carries the
  OpenPI handshake metadata (`policy_server_config`); FP32 and eager execution
  are required.
- Latency: on this H200, FP32, batch 1, three cameras and ten flow steps, the
  direct-policy median is 649.62 ms for the model and 655.43 ms with pre- and
  post-processing, against 654.99 ms and 659.31 ms for the LeRobot reference
  (ten interleaved ABBA blocks, fixed noise; see the benchmark README).
- Numerics: match matmul and cuDNN TF32 settings separately. The online
  reference uses matmul TF32 off and cuDNN TF32 on, matching the default
  service. Deterministic model tests disable both. MoE accumulation can amplify
  small rounding changes, so a fixed seed alone does not imply bitwise CUDA
  repeatability.
- Known limitations: RTC, observation history, relative actions, reduced
  precision, quantization, compilation, CUDA graphs, batching and multi-GPU
  execution are not supported. Checkpoints that declare RTC, history, relative
  actions or reduced precision are rejected at load. Training teachers and
  auxiliary prediction heads are not executed.
- Inference modules are adapted from the Apache-2.0 LingBot implementation in
  LeRobot, with original copyright notices retained.

## Supported features

| Feature | Status | Guide |
| --- | --- | --- |
| OpenPI realtime serving | Supported (validated on H200) | — |
| FP32 | Supported (required) | — |
| BF16 / FP16 | Not supported (rejected at load) | — |
| Quantization | Not supported (rejected at load) | — |
| torch.compile / CUDA graphs | Not supported (eager only) | — |
| Request batching | Not supported (`max_num_seqs: 1`, one observation per request) | — |
| Tensor, sequence or data parallelism, HSDP | Not supported (single GPU only) | — |
| Cache-DiT / TeaCache | Not applicable (no DiT block list; disabled) | — |
| RTC / observation history | Not supported (rejected at load) | — |
