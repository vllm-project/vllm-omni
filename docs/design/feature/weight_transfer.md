# Weight Transfer Engine

## Purpose

RL trainers need to publish updated policy weights to a running rollout engine.
vLLM-Omni exposes upstream vLLM's four-phase weight-transfer lifecycle across AR
and diffusion stages. The transport, trainer payloads, and receiving engines come
from upstream vLLM. Omni owns stage routing and diffusion model preparation.

## Architecture

```text
Training model -> upstream trainer WTE
                         |
                 RPC or HTTP control
                 (IPC handles / backend metadata)
                         |
                     AsyncOmni
                         |
              configured stage workers
                  /              \
             AR worker       DiffusionWorker
                  \              /
                  upstream WTE
                         |
              pipeline.load_weights()
                         |
             upstream layerwise reload
                         |
               running rollout model
```

HTTP carries control requests and serialized backend metadata. With IPC, tensor
storage stays on the same node and is accessed through CUDA IPC handles. Choosing
HTTP does not change IPC into a cross-node transport.

## Diffusion loading contract

`diffusion/model_loader/weight_reload.py` is the shared integration point. When
weight transfer is configured, the loader requires an explicit
`supports_weight_reload` capability and records upstream reload metadata after
pipeline construction, before external checkpoint loading and postprocessing.
It selects the target device explicitly, including for custom pipelines built
on CPU. Ordinary inference without weight transfer does not take this path.

The initial supported pipeline is unquantized Qwen-Image, including custom
pipelines that preserve its loading contract. Its `AutoWeightsLoader` and
transformer loaders already dispatch through each parameter's `weight_loader`.
The worker therefore constructs the original upstream engine through
`WeightTransferEngineFactory`; no Omni IPC backend or apply strategy is needed.

For IPC and NCCL, upstream reload saves the live tensors, restores the recorded
loading layout, and wraps parameter loaders. Weights may arrive in several
chunks. Upstream buffers incomplete layers, processes completed layers, and
copies results into the original storage. Finish restores untouched layers and
completes deferred processing. The trainer must send complete weights for the
layers it updates; this is not an element-delta protocol.

To enable another pipeline, validate these contracts before opting in:

- Capture its loading layout before transformations change parameter structure.
- Route incoming weights through the wrapped parameter loaders, including fused
  or renamed parameters; direct `.data.copy_()` paths bypass reload tracking.
- Ensure required post-load transformations run in the upstream layer lifecycle.
  An arbitrary pipeline-level initial-load hook is not automatically rerun.
- Verify repeated updates, chunk boundaries, original storage/device retention,
  and untouched parameters/buffers with real model execution.

Quantized pipelines, inference weight offloading, HSDP, and diffusion cache
acceleration are currently rejected with weight transfer. Supporting those
combinations requires additional validation and model integration. A capability
flag alone does not establish support for every DiT model or backend.

## Lifecycle and ownership

1. `init_weight_transfer_engine(init_info)` initializes the configured transport.
2. `start_weight_update()` begins a reload session.
3. `update_weights(update_info)` receives one chunk; this call may repeat.
4. `finish_weight_update()` completes loading before generation resumes.

AR workers inherit upstream worker methods. Diffusion workers expose the same
methods and delegate to upstream WTE, with guards against invalid call ordering.
The trainer must quiesce generation for the entire update session.

The WTE is constructed after the active pipeline is loaded. Custom pipeline
replacement closes the previous engine and binds a new one to the replacement
model; replacement during an active update is rejected. Normal RL publications
reuse the running pipeline and engine. Trainer topology queries, checksums, and
publication versions belong to the training framework's adapter.

Worker shutdown closes the WTE and releases its pipeline references before
distributed-environment and memory-pool teardown. Repeated shutdown is a no-op;
transport cleanup failures are logged and do not prevent the remaining cleanup.

Updates are not transactional. A failed chunk can leave partially updated or
meta tensors; resetting the target does not roll back model values. Stop training
and recreate the rollout after a failed update. There is no automatic rollback
or atomic commit across stages.

## Configuration and routing

```yaml
weight_transfer_config:
  backend: ipc
```

A pipeline-wide value enables all stages. To select one stage, omit the top-level
value and configure only that stage:

```yaml
stages:
  - stage_id: 0
  - stage_id: 1
    weight_transfer_config:
      backend: ipc
```

The config flows through deploy overrides, diffusion config projection, and the
typed `VllmConfig.weight_transfer_config`. `AsyncOmni` sends lifecycle RPCs only
to configured stages and raises if any selected stage reports an error.
A call with no configured stages also raises.

Upstream registers `ipc`, `nccl`, and `sparse_nccl`. AR retains upstream behavior.
Diffusion validation in this change targets Qwen-Image with `ipc`; registering
other backends does not establish diffusion E2E coverage for them.

## HTTP API

These routes are mounted only when weight transfer is enabled:

```text
POST /init_weight_transfer_engine   {"init_info": <backend handshake>}
POST /start_weight_update
POST /update_weights                {"update_info": <backend payload>}
POST /finish_weight_update
```

The HTTP layer passes payloads to the engine. Invalid JSON, non-object request
bodies, and missing required fields return 400; worker failures propagate as
errors. Build payloads with upstream trainer engines instead of maintaining a
separate Omni wire format. See the
[user guide](../../user_guide/weight_transfer.md) for a trainer example.

## Validation

`tests/test_weight_transfer_omni.py` covers config propagation, AR inheritance,
diffusion lifecycle guards, pipeline replacement, real upstream reload across
chunks and consecutive rounds, and preservation of frozen state and storage.

The RL acceptance workload is UniRL FlowGRPO with PickScore, two FSDP training
ranks colocated with two TP1 Qwen-Image rollouts, no training CPU offload, three
optimizer updates, and two weight publications. Both native RPC + IPC and
HTTP + IPC must preserve weight/sample versions and sender/receiver checksums.
The HTTP harness exercises the production router over TCP and production
AsyncOmni control methods; generation remains native. It does not validate the
full OpenAI server CLI startup or HTTP generation.

## Related files

| File | Responsibility |
| ---- | -------------- |
| `config/stage_config.py`, `config/omni_config.py` | Deploy routing and config projection |
| `diffusion/data.py`, `diffusion/vllm_config.py` | Diffusion and upstream WTE configuration |
| `diffusion/model_loader/weight_reload.py` | Shared reload preparation and support checks |
| `diffusion/model_loader/diffusers_loader.py` | Metadata capture during model construction |
| `diffusion/worker/diffusion_worker.py` | WTE lifecycle and active pipeline binding |
| `entrypoints/async_omni.py` | Stage selection and error propagation |
| `entrypoints/serve/weight_transfer_api.py` | HTTP control routes |
