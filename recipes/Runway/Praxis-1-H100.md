# Praxis-1 robot policy inference — H100

## Summary

- Vendor: Runway
- Model: Praxis-1, DROID single-arm (`runwayml/praxis-1-droid`) and YAM two-arm (`runwayml/praxis-1-yam`) checkpoints
- Task: cameras, arm history and a task string to Cartesian pose/gripper targets at 15 Hz
- Mode: BF16 offline `Omni.generate`, OpenPI and Positronic WebSocket serving, one GPU
- Hardware: 1x NVIDIA H100 80 GB
- Maintainer: community contribution

## When to use this recipe

Run a Praxis-1 checkpoint from local weights on one H100 and verify its outputs:
offline calls with synthetic observations, or a closed-loop client (simulator or
robot bridge) over the OpenPI or Positronic protocol. Physical-robot behaviour
and minimum-memory configurations are not qualified here.

## Supported model contract

| Input / output | DROID | YAM |
| --- | --- | --- |
| Views | exterior, wrist | exterior, left wrist, right wrist |
| Canvas | two 224x416 tiles (224x832) | three 256x416 tiles (256x1248) |
| Targets | float32 `(T, 13)`: xyz, R_flat9, gripper | float32 `(T, 26)`: left arm then right arm |
| Horizon | 21 cold-start targets, then 24 with history | same |
| Deployment | `vllm_omni/deploy/praxis1.yaml` | `vllm_omni/deploy/praxis1_yam.yaml` |

See the [model guide](../../docs/models/praxis1.md) for observation keys, pose
frames, settings and the episode lifecycle.

## References

- Reference implementation: [runwayml/praxis-1-oss](https://github.com/runwayml/praxis-1-oss), revision `3219e555fc05b2d5d8e77650f4c266f9e8fd7ebc` (simulation suites and the comparison runs below).
- Deployments: [praxis1.yaml](../../vllm_omni/deploy/praxis1.yaml), [praxis1_yam.yaml](../../vllm_omni/deploy/praxis1_yam.yaml).
- Protocol guide: [OpenPI API](../../docs/serving/openpi_api.md). The Positronic session API is documented with its serving PR.
- No model-specific example: offline requests use the generic `Omni.generate` call shown below.

## Hardware

- Accelerator: 1x NVIDIA H100 80 GB HBM3 (SXM), compute capability 9.0.
- Number of devices: 1; the pipeline rejects every parallel degree above 1.
- Interconnect: not used.
- Host memory: weights stream through the component loader; no CPU offload. Peak host memory was not measured.
- Qualification scope: the offline, OpenPI and Positronic paths below, both checkpoints, bfloat16. B200, GB300 and RTX PRO 6000 run the same code but were not exercised.

## Software environment

| Component | Version used |
| --- | --- |
| OS / Python | Ubuntu 24.04 / 3.12 |
| Driver / CUDA runtime | 580.95 / 13.0 (PyTorch 2.14.0+cu130, cuDNN 9.24) |
| vLLM / vLLM-Omni | 0.33.0 / this recipe's commit |
| Transformers / Diffusers | 5.19.0 / 0.40.0 |
| Optional text-encoder kernels | flash-linear-attention 0.5.0 (causal-conv1d 1.6.1 does not load against PyTorch 2.14) |

## Command

### Weights

Download a checkpoint with the Hugging Face CLI and put the companion weights
beside it (skip the companion downloads if the release bundles them):

```bash
uv pip install huggingface_hub
export PRAXIS_CHECKPOINT="$PWD/weights/praxis-1-droid"
hf download runwayml/praxis-1-droid --local-dir "$PRAXIS_CHECKPOINT"
hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir "$PRAXIS_CHECKPOINT"
hf download Qwen/Qwen3.5-2B --local-dir "$PRAXIS_CHECKPOINT/text_encoder"
```

Use `runwayml/praxis-1-yam` and `PRAXIS_YAM_CHECKPOINT` for the two-arm rig.
Optional text-encoder kernels: `uv pip install flash-linear-attention` and
`uv pip install --no-build-isolation causal-conv1d`.

### Offline example

```bash
cat > /tmp/praxis1_offline.py <<'PYTHON'
import os
import numpy as np
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

image = np.zeros((234, 416, 3), dtype=np.uint8)
image[..., 0] = np.arange(416) % 256
image[..., 1] = np.arange(234)[:, None]
observation = {
    "image.exterior": image,
    "image.wrist": image.copy(),
    "robot_state.ee_pose": np.array([0.4, 0, 0.3, 1, 0, 0, 0], np.float32),
    "grip": np.array([0.5], np.float32),
    "task": "Move the gripper forward.",
    "descriptor": "praxis.sim",  # simulator pose frame; real DROID rigs use the default FR3 hand frame
}
engine = Omni(model=os.environ["PRAXIS_CHECKPOINT"], deploy_config="vllm_omni/deploy/praxis1.yaml")
try:
    outputs = engine.generate(
        {"prompt": observation["task"]},
        OmniDiffusionSamplingParams(
            seed=7, num_inference_steps=2,
            extra_args={"robot_obs": observation, "session_id": "recipe", "end_session": True},
        ),
        use_tqdm=False,
    )
    actions = np.asarray(outputs[0].multimodal_output["actions"])
    print(actions.shape, actions.dtype)  # (21, 13) float32
finally:
    engine.close()
PYTHON
python /tmp/praxis1_offline.py
```

### Serving

```bash
vllm serve "$PRAXIS_CHECKPOINT" --omni --host 127.0.0.1 --port 8123 \
  --served-model-name praxis-1 --deploy-config vllm_omni/deploy/praxis1.yaml \
  --stage-init-timeout 1800
```

- Positronic clients connect to `ws://127.0.0.1:8123/api/v1/session` with binary msgpack
  frames (see the Positronic serving guide once that endpoint is available in your build).
- OpenPI clients connect to `/v1/realtime/robot/openpi` and send a nonempty
  `session_id` with each observation; see the [wire guide](../../docs/serving/openpi_api.md).

## Verification

Weight-free checks (CPU, plus one CUDA GPU for the attention test):

```bash
pytest tests/diffusion/models/praxis1 -q
```

Checkpoint-gated engine checks (one H100; skipped without weights):

```bash
PRAXIS_CHECKPOINT=/path/to/praxis-1-droid PRAXIS_YAM_CHECKPOINT=/path/to/praxis-1-yam \
  pytest tests/e2e/offline_inference/test_praxis1.py --run-level=advanced_model -q
```

### Numerical comparison with the reference

Block-level equivalence with real DROID weights. One block of each kind was
built by this port and by the reference implementation
([runwayml/praxis-1-oss](https://github.com/runwayml/praxis-1-oss)), loaded with the
same checkpoint tensors and fed identical inputs, masks and rotary tables on an
H100. Values are max-normalized relative differences of the block output.

| Block | float32 | bfloat16 |
| --- | ---: | ---: |
| Encoder dense block: attention | 5.6e-7 | 1.6e-3 |
| Encoder dense block: SwiGLU feed-forward | bitwise | bitwise |
| Encoder MoE block: attention | 4.7e-7 | 3.3e-3 |
| Encoder MoE block: routing indices | identical | identical |
| Encoder MoE block: feed-forward, eager fp32 math on the port's weights | 3.0e-7 | 3.4e-3 |
| Encoder MoE block: feed-forward, vLLM fused kernel | 1.2e-3 | 2.9e-3 |
| Decoder block with per-token modulation | 2.7e-6 | 5.1e-3 |
| Action expert cross-attention block | 7.8e-8 | 1.6e-3 |
| Action expert self-attention block | 1.8e-7 | 3.3e-3 |

In float32 every block matches to the order of 1e-6, so weights, layouts,
routing, masks, rotary tables and block math are the same. The one float32
exception, the fused MoE kernel at 1.2e-3, is Triton's TF32 matmul default for
float32 operands; the same weights through an eager fp32 grouped computation
match the reference to 3e-7, and in bfloat16 (the production dtype) the fused
kernel is no further from the reference than an eager bfloat16 computation is.
Every bfloat16 difference is below one bfloat16 unit in the last place
(7.8e-3), that is, kernel accumulation order.

End to end with identical inputs, Qwen3.5 text embeddings are bitwise equal
and the Wan2.2 VAE latent differs by 1.1e-2 relative (bf16 autoencoder versus
the reference's fp16 autocast). After two Euler steps with video guidance 4
those bf16-level differences grow to tens of millimetres on individual targets
of a synthetic observation. The reference itself drifts by 34 mm / 2.5° / 0.09
gripper on the same call between its qualified build and a build with eager
FlexAttention and no FLA kernels, so per-target agreement with one reference
build is not a meaningful acceptance criterion for this model; task success is.

### Task success in simulation

The reference repository's MuJoCo evaluation (unchanged physics, rendering and
task checks) was driven against this server through the Positronic API on the
same host and compared with runs of the reference implementation on the same
scene seeds. Settings: 2 Euler steps, sampler seed 0, context 1, horizon 24;
DROID guidance 1/4 and shift 12.5 with impedance, YAM guidance 1/1 and shift 1
without impedance.

DROID, 7 tasks x 2 scene seeds, reference in its qualified build (PyTorch 2.13,
cuDNN 9.24, FA4, FLA and causal-conv1d kernels):

| Task | Reference | This port |
| --- | ---: | ---: |
| `lift_cube` | 2/2 | 1/2 |
| `stack_block` | 1/2 | 2/2 |
| other five tasks | 2/2 each | 2/2 each |
| **all** | **13/14** | **13/14** |

YAM, 11 tasks x 6 scene seeds. The reference was run twice: in its qualified
build and in this port's environment (vLLM 0.33, PyTorch 2.14, Transformers
5.19, FLA, FA4 b36, no causal-conv1d), so the effect of library versions alone
is visible:

| Task | Reference, qualified build | Reference, port environment | This port |
| --- | ---: | ---: | ---: |
| `apple_in_bag` | 6/6 | 5/6 | 6/6 |
| `bell_in_bag` | 5/6 | 6/6 | 3/6 |
| `cup_by_coaster` | 0/6 | 1/6 | 0/6 |
| `dice_in_bag` | 6/6 | 6/6 | 6/6 |
| `screwdriver_in_bin` | 6/6 | 5/6 | 3/6 |
| `tennis_ball_in_bag` | 6/6 | 6/6 | 6/6 |
| `block_tower`, `bottles_in_bin`, `fruit_in_bowl`, `lego_sort`, `mug_upright` | 0/6 each | 0/6 each | 0/6 each |
| **all** | **29/66** | **29/66** | **24/66** |
| mean progress | 0.61 | 0.64 | 0.56 |

Paired outcomes: the two reference runs differ on 4 episodes (2 wins each,
exact McNemar p = 1.0); the port against the reference in the same environment
is 1 win / 6 losses (p = 0.125) and against the qualified build 0 / 5
(p = 0.06). Running the port's VAE in float16 instead of bfloat16 (closer to the
reference's fp16 autocast) gave 22/66 with the misses redistributed across
tasks, so the VAE dtype is not the cause and bfloat16 is kept.

On identical first-chunk inputs the per-target action distance between the port
and either reference run equals the distance between the two reference runs
(median 1.9 to 2.0 mm, mean 3.0 mm, P90 6.6 to 6.9 mm, mean rotation 0.8°,
mean gripper 0.0016 over 1,638 to 2,772 arm targets each). After the first
chunk, closed-loop trajectories diverge in every pairing, including between the
two reference runs. The YAM success deficit is therefore not visible at the
action level and is not statistically established on 66 episodes; it is
reported here as an open observation concentrated in `bell_in_bag` and
`screwdriver_in_bin`. A larger seed sweep would be needed to confirm or dismiss
it. This is a bounded local regression check on one H100, not physical-robot
or full-suite qualification.

## Notes

- Memory usage: about 57 GB of device memory resident per checkpoint (policy, VAE, text encoder) plus activations; one request at a time (`max_num_seqs: 1`).
- Key flags: `--deploy-config` selects the rig; `model_config.attention_impl` selects `flex` (default) or `backend`; `--stage-init-timeout 1800` covers weight loading and warmup compilation.
- Warm cold-start calls take about 0.4 s on an H100 (two Euler steps, guidance
  on, text embedding cached); the first call after startup compiles the
  FlexAttention kernels for each context depth.
- `deploy_config=` selects the YAML in Python; `deploy_config_path=` does not.
- 15 Hz describes target spacing, not inference latency.
- Known limitations: one GPU only; no parallelism, quantization, CPU offload, sleep mode, LoRA or step execution; cache acceleration is not applied to this pipeline.

## Supported features

| Feature | Status |
| --- | --- |
| Offline `Omni.generate` with session continuation, reset and video diagnostics | Supported, validated here |
| OpenPI serving (`/v1/realtime/robot/openpi`) | Supported, validated here; [guide](../../docs/serving/openpi_api.md) |
| Positronic serving (`/api/v1/session`) | Validated here through the simulation runs below; shipped with the Positronic serving PR |
| Tensor / sequence / CFG parallelism, HSDP | Unsupported (rejected at startup); [parallelism guides](../../docs/user_guide/diffusion_features.md) |
| Quantization, CPU offload, sleep mode, LoRA, step execution | Unsupported (rejected at startup); [feature matrix](../../docs/user_guide/diffusion_features.md) |
| Cache acceleration (Cache-DiT, TeaCache) | Not applied; the pipeline owns its Euler loop |
