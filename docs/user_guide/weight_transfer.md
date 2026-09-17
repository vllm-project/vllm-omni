# Weight Transfer for RL Training

Weight transfer enables dynamic model weight updates during inference without restarting the engine. This is essential for Reinforcement Learning from Human Feedback (RLHF) and other RL-based training workflows that require frequent weight updates.

## Overview

Traditional approaches restart the inference engine for each weight update, causing significant overhead from model reloading and GPU initialization. vLLM-Omni's weight transfer feature enables in-place weight updates through a four-phase protocol, reducing update latency to <100ms for large models.

The implementation aligns with upstream vLLM 0.19+ architecture while supporting vLLM-Omni's multi-stage orchestration (AR stage + diffusion stage).

## Configuration

### Enable Weight Transfer

Add `weight_transfer_config` to your engine configuration:

```python
from vllm_omni.entrypoints.async_omni import AsyncOmni

engine = AsyncOmni(
    model="path/to/model",
    weight_transfer_config={"backend": "ipc"},
)
```

### Supported Backends

| Backend | Description | Use Case |
|---------|-------------|----------|
| `ipc` | Inter-process communication via PyTorch RPC | Single-node, lightweight testing |
| `nccl` | NVIDIA Collective Communications Library | Multi-GPU, high bandwidth |
| `sparse_nccl` | Sparse tensor transfer over NCCL | Large models with sparse updates |
| `sharded_rdt` | Sharded reliable data transfer | Distributed training across nodes |

### Deploy Configuration

For multi-stage deployments, add `weight_transfer_config` as a pipeline-wide field in your deploy YAML:

```yaml
# deploy/my_model.yaml
weight_transfer_config:
  backend: nccl

stages:
  - stage_id: 0
    max_num_seqs: 16
  - stage_id: 1
    max_num_seqs: 8
```

You can also override the backend for specific stages:

```yaml
weight_transfer_config:
  backend: nccl

stages:
  - stage_id: 0
    weight_transfer_config:
      backend: ipc  # Stage 0 uses IPC instead
  - stage_id: 1
    # Stage 1 inherits the pipeline-wide nccl backend
```

## Usage

### Python API

The four-phase protocol:

1. **Init**: Initialize the weight transfer backend
2. **Start**: Begin a new weight update session
3. **Update**: Stream weight tensors to all workers
4. **Finish**: Finalize the update and swap in new weights

```python
from vllm_omni.entrypoints.async_omni import AsyncOmni

# Initialize engine with weight transfer enabled
engine = AsyncOmni(
    model="path/to/model",
    weight_transfer_config={"backend": "ipc"},
)

# Initialize weight transfer backend
await engine.init_weight_transfer_engine({"backend": "ipc"})

# Training loop
for epoch in range(num_epochs):
    # Generate rollouts with current weights
    outputs = await engine.generate(prompts, sampling_params)
    
    # Compute weight updates from RL algorithm
    weight_deltas = compute_policy_gradient(outputs)
    
    # Apply weight updates
    await engine.start_weight_update()
    await engine.update_weights({
        "names": ["model.layers.0.weight", "model.layers.1.weight", ...],
        "tensors": [delta_tensor_0, delta_tensor_1, ...],
    })
    await engine.finish_weight_update()
```

### HTTP API

Start the server with weight transfer enabled:

```bash
python -m vllm_omni.entrypoints.openai.api_server \
  --model path/to/model \
  --weight-transfer-config '{"backend": "nccl"}'
```

Execute the four-phase protocol via HTTP:

```bash
# Phase 1: Initialize
curl -X POST http://localhost:8000/init_weight_transfer_engine \
  -H "Content-Type: application/json" \
  -d '{"init_info": {"backend": "nccl"}}'

# Phase 2: Start weight update session
curl -X POST http://localhost:8000/start_weight_update

# Phase 3: Update weights
curl -X POST http://localhost:8000/update_weights \
  -H "Content-Type: application/json" \
  -d '{
    "update_info": {
      "names": ["model.layers.0.weight", "model.layers.1.weight"],
      "tensors": [[...], [...]]
    }
  }'

# Phase 4: Finish weight update
curl -X POST http://localhost:8000/finish_weight_update
```

### Integration with verl-omni

For integration with [verl-omni](https://github.com/volcengine/verl-omni), use the HTTP API:

```python
import requests

server_url = "http://localhost:8000"

# Initialize once at startup
requests.post(
    f"{server_url}/init_weight_transfer_engine",
    json={"init_info": {"backend": "nccl"}}
)

# In your RL training loop
def apply_weight_update(weight_deltas):
    requests.post(f"{server_url}/start_weight_update")
    requests.post(
        f"{server_url}/update_weights",
        json={"update_info": {"names": [...], "tensors": [...]}}
    )
    requests.post(f"{server_url}/finish_weight_update")
```

## Best Practices

### Backend Selection

- **Development/Testing**: Use `ipc` for single-node setups
- **Production Multi-GPU**: Use `nccl` for high bandwidth
- **Sparse Updates**: Use `sparse_nccl` when updating <10% of parameters
- **Multi-Node**: Use `sharded_rdt` for distributed setups

### Session Management

- Call `init_weight_transfer_engine()` once at startup
- Reuse the same session across multiple update cycles
- Always call `finish_weight_update()` to complete a session

### Error Handling

The state machine enforces valid transitions:

- Cannot call `update_weights()` before `start_weight_update()`
- Cannot call `start_weight_update()` twice without `finish_weight_update()`
- Sessions can be reused after `finish_weight_update()`

```python
try:
    await engine.start_weight_update()
    await engine.update_weights(update_info)
    await engine.finish_weight_update()
except Exception as e:
    # Handle invalid state transitions
    logger.error(f"Weight update failed: {e}")
```

## Performance Considerations

### Update Latency

- **IPC backend**: ~50ms for 7B model on single node
- **NCCL backend**: ~100ms for 70B model across 8 GPUs
- **Sparse updates**: 2-5x faster when updating <5% of parameters

### Memory Overhead

- Weight transfer maintains a shadow copy of updatable parameters
- Budget an additional ~10% GPU memory for the transfer buffer
- Use `sparse_nccl` to reduce memory overhead for sparse updates

## Limitations

- Weight transfer is not supported for quantized models (INT8/FP8)
- Currently requires all workers to receive the same weight deltas
- Multi-node distributed transfer requires stable network connectivity

## See Also

- [Design RFC](../design/weight-transfer-engine-rfc.en.md) - Architecture and design rationale
- [Pipeline Configuration](../configuration/stage_configs.md) - Deploy configuration reference
- [verl-omni Documentation](https://github.com/volcengine/verl-omni) - RL training integration
