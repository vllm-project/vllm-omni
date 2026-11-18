# Weight Transfer for RL Training

Weight transfer enables dynamic model weight updates during inference without restarting the engine. This is essential for Reinforcement Learning from Human Feedback (RLHF) and other RL-based training workflows that require frequent weight updates.

## Overview

Traditional approaches restart the inference engine for each weight update, causing significant overhead from model reloading and GPU initialization. vLLM-Omni's weight transfer feature enables in-place weight updates through a four-phase protocol aligned with upstream vLLM, across the pipeline's AR and diffusion stages.

## Configuration

Weight transfer is **opt-in**. Enable it with the pipeline-wide `weight_transfer_config` deploy field (or the `--weight-transfer-config` CLI flag, inherited from upstream vLLM `EngineArgs`):

```yaml
# deploy override
weight_transfer_config:
  backend: ipc
```

Restrict it to specific stages with a stage-level override; lifecycle RPCs are then routed only to the stages where the feature is enabled:

```yaml
stages:
  - stage_id: 0
    # Stage 0 is not configured; it will not receive weight-transfer RPCs
  - stage_id: 1
    weight_transfer_config:
      backend: ipc
```

### Supported transports

Omni uses upstream vLLM's engine factory and payload formats. AR stages retain
upstream backend support (`ipc`, `nccl`, and `sparse_nccl`). The diffusion path
validated here is unquantized Qwen-Image using `ipc` and upstream layerwise reload.
No separate apply-strategy setting is required.

CUDA IPC handles contain serialized Python objects. For HTTP IPC updates and
Python IPC updates to AR stages (including `omni_lora_ipc`), set
`VLLM_ALLOW_INSECURE_SERIALIZATION=1` before creating the client and engine so
the engine-core RPC can carry these handles. Use this opt-in only with trusted
trainers and restrict access to weight-transfer endpoints; it permits
pickle-based deserialization. The in-process diffusion path does not cross this
AR RPC boundary.

Other diffusion pipelines must explicitly enable `supports_weight_reload` after
validating their loading contract. See the [design document](../design/feature/weight_transfer.md#diffusion-loading-contract).

## Usage

### Trainer-side engine (recommended)

Payloads are backend-specific and are built by the trainer-side engines in upstream vLLM (`vllm.distributed.weight_transfer`). Do not hand-craft them:

```python
from vllm.distributed.weight_transfer.base import ModuleSource
from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory
from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo

client = HTTPVLLMWeightSyncClient(base_url="http://localhost:8000")
trainer_engine = WeightTransferTrainerFactory.trainer_init(
    IPCTrainerInitInfo(rank=0, packed=False),  # rank 0 is the sender
    client=client,
    source=ModuleSource(training_model),
)

# Each call drives the full round trip on the serving side:
# start_weight_update -> update_weights -> finish_weight_update
trainer_engine.send_weights()
```

In the upstream trainer API, `trainer_init` performs the HTTP init handshake (e.g. `{"init_info": {"packed": false}}` for IPC), and `send_weights` ships one payload per parameter chunk with CUDA IPC handles.

### HTTP API (what the server exposes)

The four endpoints mirror upstream vLLM's RLHF dev routes and are **mounted only when weight transfer is enabled** (with a security warning in the server log):

```text
POST /init_weight_transfer_engine   {"init_info": <backend handshake>}
POST /start_weight_update
POST /update_weights                {"update_info": <backend payload>}
POST /finish_weight_update
```

The `init_info` / `update_info` bodies are opaque to the HTTP layer; their schema is defined by the chosen backend (see `IPCWeightTransferInitInfo` / `IPCWeightTransferUpdateInfo` in upstream vLLM for the IPC shapes). Use the trainer-side engines above to produce them.

### Python API (orchestrator level)

`AsyncOmni` exposes the same four phases for in-process use (tests, colocated rollouts):

```python
engine = AsyncOmni(
    model="path/to/model",
    weight_transfer_config={"backend": "ipc"},
)
await engine.init_weight_transfer_engine({"packed": False})  # IPC handshake
await engine.start_weight_update()
await engine.update_weights(update_info)   # backend payload, as above
await engine.finish_weight_update()
```

## Error Semantics

- Lifecycle calls fan out to every stage with weight transfer enabled and **fail if any stage reports an error** — a failed HTTP call means the round failed, it never silently succeeds.
- Each worker enforces the protocol state machine: `update_weights` before `start_weight_update`, or a second `start_weight_update` without `finish_weight_update`, are rejected.
- Dense updates are not transactional. A failed update can leave partial values or meta tensors. Recreate the rollout after failure; resetting the update target does not restore the old weights.

## Engine-side LoRA merge

The experimental `omni_lora_ipc` backend accepts adapter-only updates and merges
them into live dense weights at `finish_weight_update`. Generate requests without
a `LoRARequest`. Runtime LoRA remains the default.

For an unquantized Qwen-Image diffusion stage:

```yaml
lora_application: merged
weight_transfer_config:
  backend: omni_lora_ipc
```

For the Qwen3-Omni thinker, configure only the trainable AR stage:

```yaml
stages:
  - stage_id: 0
    enable_lora: false
    weight_transfer_config:
      backend: omni_lora_ipc
```

AR merge supports ordinary linear projections, including separate q/k/v and
gate/up factors mapped into fused weights. Expert-routed MoE adapters and
quantized target layers are rejected. Do not set `enable_lora` for AR merged mode;
it uses the dense graph and has no Punica adapter slots in the forward.

Initialize IPC normally. Build tensor payloads with upstream's IPC trainer, then
attach adapter metadata to each chunk before calling the existing client's
`update_weights` method:

```python
update_info = {
    **ipc_update_info,
    "peft_config": peft_config,
    "lora_int_id": 1,
}
client.update_weights(update_info)
```

`ipc_update_info` is the upstream-produced IPC payload; `peft_config` is the
adapter's PEFT configuration. Send checkpoint-layout named A/B tensors, not full
merged model weights or already TP-sharded factors. Repeat the same configuration
and ID on every chunk. The native in-process representation is
`vllm_omni.lora.request.TensorLoRARequest` (`lora_tensors`, `peft_config`).
`add_lora` remains the runtime-adapter API; hot merged updates use the session.

Pause generation before start and resume only after finish succeeds. A malformed
adapter session leaves the previous policy live; retry with a new session. This
guarantee covers ranks within a stage. Configure one trainable stage per session;
there is no atomic transaction spanning multiple stage engines. A chunk without
`peft_config` starts a full-base session and discards the old adapter bank.

The bank consumes an extra copy of targeted weights, with further temporary
copies during commit. GPU graph replay, actor/rollout numerical correlation,
Qwen-Image TP4 peak memory, and end-to-end step-time measurements are required
before production training. No speedup is claimed without these measurements.

Run the CPU regressions in a vLLM development environment:

```bash
pytest tests/diffusion/lora/test_merged_weights.py tests/diffusion/lora/test_merged_manager.py tests/diffusion/lora/test_merged_weight_transfer.py tests/model_executor/test_merged_lora.py -m 'core_model and cpu'
```

Run the graph-storage check on a supported CUDA/ROCm device:

```bash
pytest tests/diffusion/lora/test_merged_lora_graph.py -m core_model
```

## Limitations

- Pause generation while updating weights. Resume only after every selected stage finishes successfully.
- Send complete weights for each updated layer, including all fused parameter shards. Untouched layers may be omitted.
- Diffusion quantization, HSDP, inference weight offloading, and diffusion cache acceleration are not supported with this reload path yet.
- IPC requires same-node CUDA IPC access and a compatible trainer/receiver GPU topology. HTTP control does not remove that requirement.
- Diffusion NCCL and sparse updates require separate validation; the Qwen-Image RL tests cover IPC.

## See Also

- [Design RFC](../design/feature/weight_transfer.md) - Architecture and design rationale
- [Pipeline Configuration](../configuration/stage_configs.md) - Deploy configuration reference
- Upstream vLLM `vllm/distributed/weight_transfer/` - engine and payload reference
