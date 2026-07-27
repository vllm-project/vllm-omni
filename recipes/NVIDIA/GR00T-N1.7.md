# GR00T-N1.7

> NVIDIA Isaac GR00T-N1.7-3B robot VLA policy served over the OpenPI WebSocket protocol

## Summary

- Vendor: NVIDIA
- Model: `nvidia/GR00T-N1.7-3B`
- Task: Vision-Language-Action (VLA) inference for robot manipulation
- Mode: Online serving via OpenPI WebSocket endpoint
- Maintainer: timzsu

## When to use this recipe

Use this recipe when you need to serve GR00T-N1.7 as a real-time robot policy
over the OpenPI WebSocket API. It configures the DROID embodiment
(`OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`) and exposes the standard DROID action
keys (`eef_9d`, `gripper_position`, `joint_position`) with action horizon 40.

## References

- Upstream model: <https://huggingface.co/nvidia/GR00T-N1.7-3B>
- Upstream codebase: <https://github.com/NVIDIA/Isaac-GR00T>
- OpenPI client library: <https://github.com/Physical-Intelligence/openpi>
- Pipeline: `vllm_omni.diffusion.models.gr00t.pipeline_gr00t.Gr00tN1d7Pipeline`
- Deploy config: [`vllm_omni/deploy/Gr00tN1d7.yaml`](../../vllm_omni/deploy/Gr00tN1d7.yaml)
- E2E test: [`tests/e2e/online_serving/test_gr00t_openpi.py`](../../tests/e2e/online_serving/test_gr00t_openpi.py)

## Environment

- OS: Linux
- Python: 3.11+
- Driver / runtime: NVIDIA CUDA
- Hardware: 1 NVIDIA GPU. The upstream model card lists 16 GB+ VRAM for inference (e.g. RTX 4090, L40, H100); in practice this serving path uses ~6 GiB peak (bf16, TP=1, `max_num_seqs: 1`).
- vLLM-Omni version or commit: use versions from your current checkout

## Start server

From repository root:

```bash
vllm serve nvidia/GR00T-N1.7-3B \
  --omni \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name gr00t-n1d7 \
  --stage-configs-path vllm_omni/deploy/Gr00tN1d7.yaml
```

Notes:

- Only `max_num_seqs: 1` is supported (configured in the deploy YAML). The
  policy is markovian (`state_history_length=1`, `reset()` returns `{}`) — the
  reason for the cap is that `Gr00tPolicy` does its own per-sample
  (un)batching inside `get_action` and is not integrated with vLLM's
  continuous batching path.
- This pipeline is a thin wrapper around the upstream HF policy
  (`AutoModel.from_pretrained` + `enforce_eager`, `tensor_parallel_size=1`,
  pipeline `load_weights` is a no-op). The standard diffusion accelerators
  (SP, CFG, TeaCache, VAE tiling) do not transfer to a flow-matching action
  policy at batch size 1, so a native-kernel port is intentionally out of
  scope for this recipe — the value is the OpenPI serving integration, not
  kernel-level acceleration.
- The WebSocket endpoint is `ws://127.0.0.1:8000/v1/realtime/robot/openpi`.
  The server handshake message (first frame after connect) is a msgpack-encoded
  dict with `action_horizon`, `action_keys`, `embodiment_tag`, and
  `needs_session_id`.

## Verification

```python
from tests.gr00t.openpi_client_helper import run_policy_session, validate_session_result
validate_session_result(run_policy_session(host="127.0.0.1", port=8000))
```

Or run the e2e test suite:

```bash
python -m pytest tests/e2e/online_serving/test_gr00t_openpi.py -v
```

The test sends a synthetic two-frame DROID observation and checks:

- GR00T metadata contract: `image_resolution`, `action_horizon`, `action_keys`, `embodiment_tag`
- Action shapes: `eef_9d (1,40,9)`, `gripper_position (1,40,1)`, `joint_position (1,40,7)`
- All action values are finite float32
- Reset response is `"reset successful"`

## Notes

- **Do not change the model-specific `policy_server_config` values.** `action_horizon`,
  `action_keys`, and `supported_embodiments` are fixed by the GR00T-N1.7 checkpoint and
  are validated against the loaded policy at startup (the server refuses to start on a
  mismatch). Only `image_resolution` and `needs_session_id` are deployment knobs.
- To switch embodiment, edit `embodiment_tag` under both `model_config` and
  `policy_server_config` in `vllm_omni/deploy/Gr00tN1d7.yaml`. Supported values:
  `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT` (default), `XDOF`, `XDOF_SUBTASK`,
  `REAL_G1`, `REAL_R1_PRO_SHARPA`, `LIBERO_PANDA`, `SIMPLER_ENV_GOOGLE`,
  `SIMPLER_ENV_WIDOWX`.
- GR00T weights are loaded directly by `Gr00tPolicy` via `AutoModel.from_pretrained`;
  the pipeline's `load_weights` is intentionally a no-op.
