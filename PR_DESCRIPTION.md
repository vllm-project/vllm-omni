# PR: Add Weight Transfer Support for Multi-Stage RL Training

## Purpose

Enable dynamic model weight updates during inference without engine restart, which is required for RLHF / RL training loops (e.g. verl-omni) that update policy weights every training iteration.

This PR ports the upstream vLLM weight transfer architecture (introduced in vLLM 0.19+, backends: `ipc`, `nccl`, `sparse_nccl`, `sharded_rdt`) to vLLM-Omni's multi-stage orchestration. The four-phase protocol (`init_weight_transfer_engine` → `start_weight_update` → `update_weights` → `finish_weight_update`) is preserved exactly; method names, argument structure, and state machine semantics match upstream so future upstream improvements flow downstream automatically.

Key changes:

- **Config propagation**: `weight_transfer_config` added as a pipeline-wide deploy field (stage-level override via `deploy_override`), propagated through `EngineConfig` to all workers
- **AR stage worker** (`ar_worker.py`): exposes the four-phase lifecycle, delegates to the upstream `GPUExecutor` weight transfer support
- **Diffusion stage worker** (`diffusion_worker.py`): full lifecycle with state machine validation (rejects update-before-start, double-start; sessions reusable after finish)
- **Orchestrator** (`async_omni.py`): `AsyncOmni` exposes the four-phase API and forwards to all stage clients via `collective_rpc`
- **HTTP API** (`entrypoints/serve/weight_transfer_api.py`): four `POST` endpoints registered in `api_server.py` so RL frameworks can drive weight updates over HTTP
- **Docs**: user guide (`docs/user_guide/weight_transfer.md`), config reference updates, and design RFC (`design/weight-transfer-engine-rfc.en.md`)

## Test Plan

**Unit tests** — `tests/test_weight_transfer_omni.py` (22 tests):

```bash
pytest tests/test_weight_transfer_omni.py -v
```

Covers config propagation (pipeline-wide field, deploy override, config reaching workers), worker lifecycle (AR four methods, diffusion state machine, invalid-transition error handling, session reuse), and orchestrator forwarding to all stages.

**E2E: generation verification** — `tests/e2e/features/rlhf_test/test_weight_transfer_changes_output.py`:

```bash
pytest tests/e2e/features/rlhf_test/test_weight_transfer_changes_output.py -v -s
```

Generates an image with original weights → perturbs a weight via `update_weights` → verifies the output changes → restores the weight → verifies output returns to the original. Uses `tiny-random/Qwen-Image` (30MB) on 1 GPU.

**E2E: HTTP API** — `tests/e2e/features/rlhf_test/test_weight_transfer_http_api.py`:

```bash
pytest tests/e2e/features/rlhf_test/test_weight_transfer_http_api.py -v -s
```

Starts a vllm-omni server with `--weight-transfer-config '{"backend": "ipc"}'` via a module-scoped pytest fixture, then drives the full four-phase protocol over HTTP and asserts error handling for missing request fields.

**vLLM Version:** vLLM 0.29

**vLLM-Omni Commit:** d4dba0ddb

## Test Result

All tests pass on a single-GPU remote machine (backend: `ipc`, model: `tiny-random/Qwen-Image`):

- `tests/test_weight_transfer_omni.py`: **22 passed**
- `tests/e2e/features/rlhf_test/test_weight_transfer_changes_output.py`: **passed** — image output changed after weight perturbation and returned to the original after restore, confirming updates reach the live model
- `tests/e2e/features/rlhf_test/test_weight_transfer_http_api.py`: **passed** — server started with `--weight-transfer-config`, all four HTTP endpoints returned 200 in protocol order, and invalid requests (missing `init_info` / `update_info`) returned 400

Notes:

- Unit tests run anywhere (no GPU required); both e2e tests require 1 CUDA GPU and download the 30MB test model
- `nccl` / `sparse_nccl` / `sharded_rdt` backends are inherited from upstream vLLM unchanged; only `ipc` is exercised in e2e because multi-GPU/multi-node setups are not covered by CI here
