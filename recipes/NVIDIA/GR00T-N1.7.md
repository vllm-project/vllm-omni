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
  --deploy-config vllm_omni/deploy/Gr00tN1d7.yaml
```

Notes:

- The deploy YAML defaults to `max_num_seqs: 1`, serving one request at a
  time. The policy is markovian (`state_history_length=1`, `reset()` returns
  `{}`), and `Gr00tN1d7Pipeline` implements the request-batch contract, so
  raising `max_num_seqs` in an overlay batches concurrent sessions into one
  policy call — see "Optional: batch concurrent sessions" below.
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

## Optional: batch concurrent sessions

The bundled deploy config serves one request at a time (`max_num_seqs: 1`):
with N robots connected, every observation waits for the requests ahead of it,
so latency grows linearly with N while throughput stays flat.
`Gr00tN1d7Pipeline` supports request-level batching — the scheduler groups
in-flight requests into one wave and the pipeline serves the whole wave with a
single `Gr00tPolicy.get_action()` call. Opt in with an overlay:

```yaml
# gr00t_batch.yaml
base_config: vllm_omni/deploy/Gr00tN1d7.yaml
stages:
  - stage_id: 0
    max_num_seqs: 8
```

```bash
vllm serve nvidia/GR00T-N1.7-3B --omni --deploy-config gr00t_batch.yaml
```

Measured on 1xA30 (eager, DP=1, 30 requests/client after 20 warmup requests),
closed-loop OpenPI WebSocket clients:

| concurrent sessions | `max_num_seqs: 1` (default) | `max_num_seqs: 8` |
| --- | --- | --- |
| 1 | 6.56 req/s, 152 ms | 6.76 req/s, 148 ms |
| 2 | 6.68 req/s, 297 ms | 6.94 req/s, 286 ms |
| 4 | 6.71 req/s, 589 ms | 10.89 req/s, 366 ms |
| 8 | 6.77 req/s, 1165 ms | 13.84 req/s, 576 ms |

Notes:

- Batches form opportunistically: requests that arrive while a wave is running
  are grouped into the next wave, so a single client is served exactly as
  before (wave size 1, no added wait). `request_batch_max_wait_ms` (default 0)
  can trade first-request latency for fuller waves.
- Actions for a request depend only on that request's observation — a
  neighbor's content never leaks into your actions. Batched execution does
  change bf16 kernel shapes, which shifts actions on the order of 5e-2 versus
  a solo run (1e-5 when the same comparison is done in fp32) — well inside the
  policy's own seed-to-seed sampling spread (~8e-1). With a fixed
  `GR00T_NOISE_SEED`, per-request noise additionally depends on the wave size,
  so bit-exact reproducibility holds per wave shape, not across load levels.
  The single-session golden test is unaffected (wave size 1).
- Batching grows the per-wave activation footprint only modestly: waves of 7
  peak at ~6.4 GiB reserved vs ~5.9 GiB for single-request serving on A30
  (the model dominates; per-observation activations are small).

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
