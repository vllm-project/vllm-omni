# GO-1-Air

> Source repository: <https://github.com/OpenDriveLab/AgiBot-World>
>
> Weights: <https://huggingface.co/agibot-world/GO-1-Air>

This example runs offline inference for **AgiBot GO-1-Air**, an open-source
Vision-Language-Latent-Action (ViLLA) policy. The released checkpoint ships
with the Latent Planner disabled, so this integration covers the
`Vision → Language → diffusion action` path only; an `[B, 30, 16]` action
chunk is produced with the upstream squared-cosine diffusion schedule and
5-step DPM-Solver sampling.

## Requirements

GO-1-Air offline inference should run on a CUDA GPU with at least 12 GiB of
VRAM. In the validated BF16 single-observation smoke run, vLLM-Omni peaked at
4.95 GiB CUDA allocated and 10.05 GiB CUDA reserved after model load and
inference.

## Quick start

```bash
# (Optional) point at a checkpoint directory containing config.json
# and model.safetensors[.index.json].
export GO1_AIR_MODEL_DIR=/path/to/GO-1-Air

bash examples/offline_inference/go1_air/run.sh
```

The smoke test prints `[smoke] OK action shape=(1, 30, 16)` on success.

The repository-level offline smoke test is:

```bash
pytest -q tests/examples/offline_inference/test_go1_air.py
```

## Input contract

GO-1-Air consumes repository-side tensors through
`sampling_params.extra_args["batch_inputs"]`, or OpenPI robot observations
through `sampling_params.extra_args["robot_obs"]`:

| Key | Shape / type | Notes |
| --- | --- | --- |
| `observation.state` | `torch.Tensor[B, 16]` | Robot state vector. |
| `observation.task` | `str` for `B=1`, or `list[str]` / `tuple[str, ...]` of length `B` | Language instruction. |
| `observation.images.<camera>` | `torch.Tensor[B, history, 3, 448, 448]` | RGB image history, already resized to the model resolution. |
| `observation.images.<camera>_mask` | optional scalar, `torch.Tensor[B]`, or `torch.Tensor[B, history]` | Marks valid cameras/history frames before vision tokens are allocated. |
| `control_freq` | optional `torch.Tensor[B]` or scalar tensor | Defaults to 30 Hz when omitted. |

`extra_args["noise"]` may be provided for deterministic debugging and must
match `[B, 30, 16]`.

## OpenPI robot serving

GO-1-Air can be exposed through the OpenPI-compatible robot endpoint added by
the realtime serving layer:

```bash
vllm serve /path/to/GO-1-Air \
  --omni \
  --deploy-config vllm_omni/deploy/go1_air.yaml
```

The endpoint is `/v1/realtime/robot/openpi`. The deploy file provides the
policy-server metadata, and the pipeline converts OpenPI observation keys such
as `observation/joint_position`, `observation/gripper_position`, and
`observation/exterior_image_0_left` into the same tensor schema used by offline
inference. Actions are returned through `multimodal_output["actions"]`.

## Notes

* The full open-loop evaluation harness (dataset loader, deterministic noise,
  result archiving) is added in a follow-up PR — see
  `examples/offline_inference/internvla_a1/` for the structure that will be
  mirrored.
* The GO-1-Air weights are licensed under **CC BY-NC-SA 4.0**; the
  vllm-omni integration code is Apache-2.0 and contains no upstream model
  code. Downstream commercial deployment of the weights is governed by
  AgiBot's license — see the model card before shipping.
