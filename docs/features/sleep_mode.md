# Sleep Mode & ACK Protocol

vLLM-Omni’s **Sleep Mode** allows you to temporarily release most GPU memory used by a model—such as model weights and key-value (KV) caches—**without stopping the server or unloading the Docker container**.

This feature is inherited from [vLLM’s Sleep Mode](https://blog.vllm.ai/2025/10/26/sleep-mode.html) and extended with the **Omni ACK Protocol** to support multi-stage pipelines and heterogeneous hardware backends (NVIDIA, AMD, Intel, Huawei). It is especially useful in **RLHF**, **dynamic model switching**, or **cost-saving scenarios**.

---

## 1. Feature Documentation

### Overview
Omni Sleep Mode provides a mechanism to "sleep" specific model stages. When a stage enters sleep, its physical VRAM is reclaimed by the system, while the process state is preserved for rapid "wake-up" without full re-initialization.

### Sleep Levels
We support two levels of hibernation to balance recovery speed and memory efficiency:

| Level | Name | Mechanism | Recovery Speed | Memory Freed |
| :--- | :--- | :--- | :--- | :--- |
| **Level 1** | **Weight Offloading** | Offloads weights to Host CPU RAM. | **Fast** (DMA) | Substantial |
| **Level 2** | **Full De-mapping** | Physically releases memory pages via VRAM scavenging. | **Moderate** | **Maximum** (up to 95%+) |

### Supported Platforms

Omni Sleep Mode is optimized for high-performance computing backends:

* **NVIDIA**: Supported via Virtual Memory Management (VMM).
* **AMD (ROCm)**: Fully supported with physical page de-mapping.
* **Intel XPU**: Supported via Level Zero memory management.
* **Huawei NPU**: Supported via Ascend memory scavenging.

### Hardware Requirements
* **Memory Considerations**: System RAM must be sufficient to hold offloaded weights during sleep.
* **TP Support**: Tensor Parallel groups synchronize sleep/wake transitions across all workers.

---


## 2. Usage Examples

### Python API Example
You can programmatically control the lifecycle of stages using the `AsyncOmni` engine.

```python

import asyncio
from vllm_omni.entrypoints.async_omni import AsyncOmni

async def run_sleep_demo():
    # 1. initialization
    engine = AsyncOmni(
        model="ByteDance-Seed/BAGEL-7B-MoT",
        enable_sleep_mode=True
    )

    # 2. sleep mode level2
    acks = await engine.sleep(stage_ids=[0], level=2)
    print(f"Freed {acks[0].freed_bytes / 1024**3:.2f} GiB on Stage 0")

    # 3. wake up
    await engine.wake_up(stage_ids=[0])

if __name__ == "__main__":
    asyncio.run(run_sleep_demo())

```

### server command Example
Start the server with sleep mode enabled:

The first method

```

vllm serve ByteDance-Seed/BAGEL-7B-MoT \
--omni \
--enable-sleep-mode \
--trust-remote-code \
--gpu-memory-utilization 0.7

```

The second method

```

python3 -m vllm_omni.entrypoints.openai.api_server \
    --model ByteDance-Seed/BAGEL-7B-MoT \
    --omni \
    --enable-sleep-mode \
    --trust-remote-code \
--gpu-memory-utilization 0.7

```




### Test Scenarios & Commands

#### Scenario 1: LLM Engine Sleep

Objective: Verify VRAM reclamation for Stage 0 (Thinker).

Trigger sleep (Level 1 or Level 2) via client:

```

curl -X POST http://localhost:8000/v1/omni/sleep \
     -H "Content-Type: application/json" \
     -d '{"stage_ids": [0], "level": 2}'

```

Tip: Open a new terminal and run rocm-smi or nvidia-smi or to observe the immediate drop in VRAM usage.



#### Scenario 2: Diffusion Sleep
Objective: Verify VRAM reclamation for Stage 1 (Diffusion).

Trigger sleep (Level 1 or Level 2) via client:

```

curl -X POST http://localhost:8000/v1/omni/sleep \
     -H "Content-Type: application/json" \
     -d '{"stage_ids": [1], "level": 2}'

```



#### Scenario 3: Multi-Stage Coordinated Stress Test
Objective: Test concurrent sleep and rapid wake-up across multiple stages.

Concurrent Sleep (Stage 0 & 1):

```

curl -X POST http://localhost:8000/v1/omni/sleep \
     -H "Content-Type: application/json" \
     -d '{"stage_ids": [0, 1], "level": 2}'

```


Rapid Wake-up:

```

curl -X POST http://localhost:8000/v1/omni/wakeup \
     -H "Content-Type: application/json" \
     -d '{"stage_ids": [0, 1]}'

```


#### Scenario 4: Full Lifecycle Memory Audit & Functional Integrity
Objective: Audit the complete flow from Sleep to Wake-up followed by an Inference validation.

Check Initial State: Observe baseline VRAM usage.

Trigger Deep Sleep (Level 2):

```

curl -X POST http://localhost:8000/v1/omni/sleep \
     -H "Content-Type: application/json" \
     -d '{"stage_ids": [0], "level": 2}'

```

Wake-up Model:

```

curl -X POST http://localhost:8000/v1/omni/wakeup \
     -H "Content-Type: application/json" \
     -d '{"stage_ids": [0]}'

```

Verify Functional Integrity (Inference):
Ensure the model still generates valid output after reloading weights.

```

curl -X POST http://localhost:8000/v1/images/generations \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "A huge swimming pool, with many people swimming.",
       "model": "ByteDance-Seed/BAGEL-7B-MoT",
       "response_format": "b64_json",
       "extra_body": {"sampling_params": {"num_inference_steps": 4, "seed": 42}}
     }' > post.json

```




## 3. API Reference


### Methods

| Method | Arguments | Return Type | Description |
| :--- | :--- | :--- | :--- |
| **sleep** | `stage_ids: List[int], level: int` | `List[OmniACK]` | Triggers hibernation for specified stages. |
| **wake_up** | `stage_ids: List[int]` | `List[OmniACK]` | Reloads weights and re-maps memory. |



### OmniACK Dataclass Fields

| Field | Type | Description |
| :--- | :--- | :--- |
| **task_id** | `str` | Unique identifier for the operation. |
| **status** | `str` | `SUCCESS` or `ERROR`. |
| **stage_id** | `int` | The ID of the stage that responded. |
| **rank** | `int` | The rank ID within the Tensor Parallel group. |
| **freed_bytes** | `int` | Actual amount of physical VRAM reclaimed. |
| **metadata** | `dict` | Additional platform-specific metrics. |

Metadata Field Analysis
The metadata field is a dynamic dictionary containing hardware-specific telemetry and audit data, primarily used for verifying memory reclamation on various backends (e.g., AMD ROCm, NVIDIA CUDA).

```
"metadata": {
    "source": "Platform_AMD_Instinct_MI300X",
    "total_freed_gib": "78.57",
    "rank_residual_gib": "2.07"
}
```

#### Core Utility:
**VRAM Reclamation Audit (total_freed_gib)**: Converts raw freed_bytes into human-readable GiB. It serves as the primary metric to verify that Level 2 sleep has successfully purged model weights from VRAM.

**Residual & Fragmentation Monitoring (rank_residual_gib)**: Reports the remaining VRAM footprint after memory de-mapping. A low residual value (e.g., 2.07 GiB) confirms a successful "clean" state, ensuring the device is ready for high-memory co-located tasks like training or diffusion pipelines.

**Backend Traceability (source)**: Identifies the underlying hardware driver or audit source. This is critical for debugging synchronization issues in multi-stage, distributed environments.

**Performance Analytics (Roadmap)**: Future updates will include latency_ms (context-switch overhead) and cuda_graph_recalled (graph engine status) to optimize performance in high-frequency sleep/wake scenarios.
