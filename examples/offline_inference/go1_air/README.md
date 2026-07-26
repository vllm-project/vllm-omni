# GO-1-Air offline inference

Minimal smoke test for the `Go1AirPipeline` registration. The full open-loop
evaluation harness (dataset loader, deterministic noise, result archiving)
lives in a follow-up PR mirroring `examples/offline_inference/internvla_a1/`.

## Run

```bash
# Stub mode (no checkpoint, validates import + pipeline shape contract).
bash run.sh

# With weights (sets the env var; same script).
export GO1_AIR_MODEL_DIR=/path/to/GO-1-Air
bash run.sh
```

The smoke succeeds when it prints `[smoke] OK action shape=(1, 30, 16)`.

Repository tests use the CPU-friendly tiny mode:

```bash
python examples/offline_inference/go1_air/smoke.py --tiny-config --device cpu --dtype float32
pytest -q tests/examples/offline_inference/test_go1_air.py
```

## Input schema

`Go1AirPipeline` expects pre-built tensors in
`sampling_params.extra_args["batch_inputs"]`, or an OpenPI robot observation in
`sampling_params.extra_args["robot_obs"]`:

* `observation.state`: `torch.Tensor[B, 16]`
* `observation.task`: one string for `B=1`, or a string list/tuple of length `B`
* `observation.images.<camera>`: `torch.Tensor[B, history, 3, 448, 448]`
* `observation.images.<camera>_mask`: optional scalar, `(B,)`, or `(B, history)` boolean tensor
* `control_freq`: optional scalar or `(B,)` tensor, defaulting to 30 Hz

`extra_args["noise"]` is optional for deterministic debugging and must match
`[B, 30, 16]`.

For online robot serving, use `vllm_omni/deploy/go1_air.yaml`; it declares the
OpenPI policy-server metadata consumed by `/v1/realtime/robot/openpi`. The
pipeline converts OpenPI-style observation keys such as
`observation/joint_position` and `observation/exterior_image_0_left` into the
same `batch_inputs` schema and returns actions through
`multimodal_output["actions"]`.

## Upstream license note

The GO-1-Air weights on HuggingFace (`agibot-world/GO-1-Air`) are released under
**CC BY-NC-SA 4.0** (NonCommercial + ShareAlike). The vllm-omni integration
code in this repository is Apache-2.0 and contains no upstream model code —
it loads weights at runtime and runs them through clean-room implementations
of the architecture. Downstream commercial use of the weights is governed by
AgiBot's license, not by Apache-2.0; consult the GO-1-Air model card before
deploying.
