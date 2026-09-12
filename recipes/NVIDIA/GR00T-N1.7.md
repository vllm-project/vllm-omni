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

## Optional: torch.compile the action head

The default deploy config runs eager. The pipeline is launch-bound rather than
compute-bound: an Nsight Systems capture of one steady-state request on an A30 shows
5,086 kernel launches with a 6 us median duration, and the GPU busy for only 70 ms of a
144 ms request. 3,446 of those launches (68%) are ~6 us elementwise/layernorm kernels
that account for 29% of GPU time.

`Gr00tN1d7Pipeline.setup_compile()` torch.compiles the action head, which the runner
calls when a stage is not `enforce_eager`. Opt in with an overlay:

```yaml
# gr00t_compile.yaml
base_config: vllm_omni/deploy/Gr00tN1d7.yaml
stages:
  - stage_id: 0
    enforce_eager: false
    compile_mode: default          # or reduce-overhead
```

!!! warning
    Keep `compile_mode` at the stage level, not under `model_config:`. Overlay
    merging is deep only for the keys in `_DEEP_MERGE_KEYS`; a `model_config:`
    block in an overlay **replaces** the base block instead of merging, which
    drops `policy_server_config` and silently disables OpenPI serving
    (`Robot OpenPI serving disabled for model ...`, and the client fails its
    msgpack handshake with `TypeError: a bytes-like object is required`).

```bash
vllm serve nvidia/GR00T-N1.7-3B --omni --deploy-config gr00t_compile.yaml
```

Measured on 1xA30, DP=1, 30 requests/client after 20 warmup requests. Actions were
checked against the golden values in
`tests/e2e/online_serving/test_gr00t_openpi_expansion.py` (atol 1e-2) before each run:

| `compile_mode` | latency | throughput | golden values |
|---|---|---|---|
| (eager, default) | 146.9 ms | 7.49 req/s | PASS |
| `default` | 102.9 ms (-30%) | 10.69 req/s (+43%) | PASS |
| `reduce-overhead` | 99.3 ms (-32%) | 11.06 req/s (+48%) | PASS |

Notes:

- Inductor fusion carries most of the win; CUDA graphs add ~3 ms on top. `default` has
  none of the CUDA graph static-address constraints, so prefer it unless you have
  measured otherwise.
- Only the action head is compiled. Its shapes are fixed by the checkpoint (batch 1,
  `action_horizon` 40, `num_inference_timesteps` 4). The Qwen3-VL backbone is left eager
  on purpose: its per-observation token count defeats CUDA graph capture, and compiling
  it moves the actions past the e2e tolerance (`eef_9d` 1.5e-2 against atol 1e-2) while
  being slower than compiling the action head alone.
- Compilation happens on the first inference, so warm up before measuring.

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
