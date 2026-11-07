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
- Hardware: 1 NVIDIA GPU. The upstream model card lists 16 GB+ VRAM for inference (e.g. RTX 4090, L40, H100); in practice this serving path uses ~6 GiB peak (bf16, TP=1, `max_num_seqs: 1`), and about 6.3 GiB with `max_num_seqs: 8`.
- vLLM-Omni version or commit: use versions from your current checkout

## Start server

From repository root:

```bash
vllm serve nvidia/GR00T-N1.7-3B \
  --omni \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name gr00t-n1d7 \
  --deploy-config vllm_omni/deploy/Gr00tN1d7.yaml
```

Notes:

- The bundled deploy YAML serves one request at a time (`max_num_seqs: 1`).
  To serve several robots from one GPU, see
  [Serving several robots](#serving-several-robots).
- This pipeline is a thin wrapper around the upstream HF policy
  (`AutoModel.from_pretrained` + `enforce_eager`, `tensor_parallel_size=1`,
  pipeline `load_weights` is a no-op). The standard diffusion accelerators
  (SP, CFG, TeaCache, VAE tiling) do not transfer to a flow-matching action
  policy, so a native-kernel port is intentionally out of
  scope for this recipe — the value is the OpenPI serving integration, not
  kernel-level acceleration.
- The WebSocket endpoint is `ws://127.0.0.1:8000/v1/realtime/robot/openpi`.
  The server handshake message (first frame after connect) is a msgpack-encoded
  dict with `action_horizon`, `action_keys`, `embodiment_tag`, and
  `needs_session_id`.

## Serving several robots

The pipeline supports request-level batching. Requests that arrive together are
served by one policy call, and each request keeps its own seed. Raise
`max_num_seqs` and give the scheduler a short wait so concurrent closed-loop
clients land in the same batch:

```yaml
stages:
  - stage_id: 0
    max_num_seqs: 8
    request_batch_max_wait_ms: 20
```

Measured on 1x RTX PRO 6000 Blackwell with N closed-loop OpenPI clients (eager,
40 requests per client):

| Clients | `max_num_seqs: 1` | `max_num_seqs: 8`, wait 20 ms |
| --- | --- | --- |
| 1 | 9.7 req/s, 103 ms | 9.2 req/s, 109 ms |
| 2 | 9.9 req/s, 201 ms | 15.4 req/s, 130 ms |
| 4 | 9.8 req/s, 406 ms | 23.9 req/s, 167 ms |
| 8 | 9.9 req/s, 798 ms | 28.5 req/s, 277 ms |

Without the wait, two clients alternate and rarely share a batch. With a single
robot, keep the bundled config: the wait only adds latency.

Batched actions are not bit-identical to the same request served alone: bf16
kernels change with the batch size, which moves actions by up to about 0.02 for
`joint_position` and 0.09 for `gripper_position`. Changing the seed moves them
much more (about 0.09 and 0.53).

## Verification

```python
from tests.helpers.runtime import OpenAIClientHandler

handler = OpenAIClientHandler(host="127.0.0.1", port=8000, log_stats=False)
response = handler.send_robot_openpi_ws_request(
    {"run_default_policy_session": True, "session_id": "gr00t-smoke"}
)[0]
assert response.operation_responses[-1]["status"] == "reset successful"
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
