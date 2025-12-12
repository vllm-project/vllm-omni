# Disaggregated Inference for Omni-Modality Models

This guide explains how to configure and use distributed connectors (vllm_omni/distributed/connectors) in vllm-omni for multi-stage pipelines.

## 1. Overview

Connectors enable data transfer between pipeline stages (e.g., Thinker -> Talker).
Currently supported connectors operate in **D2H2D (Device-to-Host-to-Device)** mode:
1. **SharedMemoryConnector**: Uses system shared memory.
2. **MooncakeConnector**: Uses [Mooncake](https://github.com/kvcache-ai/Mooncake).

*   **SharedMemoryConnector (Default)**: Zero-copy (on host), lowest latency. Best for **single-node** deployments. Auto-configured if no connectors are specified.
*   **MooncakeConnector**: TCP/RDMA based. Best for **multi-node** distributed deployments. Requires a Mooncake Master service.

## 2. API Design

The connector system is built around the `OmniConnectorBase` abstraction, which decouples data transport from stage logic.

### Core Interface

```python
class OmniConnectorBase(ABC):
    @abstractmethod
    def put(self, from_stage: str, to_stage: str, request_id: str, data: Any) -> tuple[bool, int, Optional[dict]]:
        """
        Store data.
        Returns: (success, serialized_size, metadata)
        """
        pass

    @abstractmethod
    def get(self, from_stage: str, to_stage: str, request_id: str, metadata: Optional[dict] = None) -> Optional[tuple[Any, int]]:
        """
        Retrieve data.
        Args: metadata - Transport-specific handles returned by put() (e.g., SHM name).
        Returns: (object, serialized_size)
        """
        pass
```

### Key Concept: Metadata Passing
Unlike a pure key-value store, some connectors (like `SharedMemoryConnector`) generate transient resources (e.g., a shared memory block name) during `put()`. This `metadata` **must be passed via the control plane** (e.g., HTTP headers, queue messages) from the producer stage to the consumer stage so `get()` can locate the data.

## 3. Backends & Use Cases

### 3.1 SharedMemoryConnector
**Best for:** Single-node, high-performance IPC.

*   **Mechanism:**
    *   **Small Payloads (< Threshold)**: Data is serialized and passed directly "inline" within the `metadata` dictionary. This avoids the overhead of creating SHM blocks for tiny messages.
    *   **Large Payloads (>= Threshold)**: Data is written to a named System Shared Memory block. The block name is returned in `metadata`.
*   **Configuration:**
    *   `shm_threshold_bytes`: Size in bytes to switch from inline to SHM (default: 64KB).

### 3.2 MooncakeConnector
**Best for:** Multi-node distributed inference.

*   **Mechanism:** Uses Mooncake's distributed KVCache store.
    *   **Data Plane**: TCP or RDMA for high-bandwidth transfer.
    *   **Control Plane**: Uses a centralized Mooncake Master and Metadata Server.
    *   **Keying**: Deterministic keys based on `request_id/from_stage_to_stage`.
*   **Requirements**: Requires a running Mooncake Master service.

## 4. Relationship with vLLM

vLLM provides specialized distributed mechanisms for specific artifacts:
*   **KV Transfer** (`vllm.distributed.kv_transfer`): Optimized for transferring KV caches between prefill and decode instances (using NCCL, Mooncake, etc.).
*   **EC Transfer** (`vllm.distributed.ec_transfer`): Optimized for sharing encoder embeddings.
*   **Device Communicators** (`vllm.distributed.device_communicators`): Low-level primitives (NCCL, SHM) for Tensor/Pipeline Parallelism.

`vllm-omni` complements this by introducing a **Generalized Connector Abstraction** (`OmniConnector`) for multimodal pipelines. While vLLM's connectors are artifact-specific, `vllm-omni`:

1.  **Unifies Transport**: Provides a single API (`put`/`get`) to transport *any* stage artifact (Input Embeddings, Hidden States, Audio/Image Tensors, KV Cache, Final Output) between arbitrary pipeline stages (e.g., AudioEncoder -> LLM -> AudioGenerator).
2.  **Extends Connectivity**: Enables flexible construction of complex DAGs (Directed Acyclic Graphs) where stages can run in the same process, same node, or across nodes, using the most appropriate backend (SHM, Mooncake, etc.) for each edge.
3.  **Wraps & Adapts**: Can internally utilize vLLM's specialized `kv_transfer` for KV paths while using generic transports (SHM/Mooncake) for other data types, presenting a consistent interface to the application layer.

## 5. Installation (Mooncake)

If using `MooncakeConnector`, install the library first:

```bash
# For CUDA-enabled systems (Recommended)
pip install mooncake-transfer-engine

# For non-CUDA systems
pip install mooncake-transfer-engine-non-cuda
```

## 6. Using MooncakeConnector

### 6.1 Start Mooncake Master

Start the master service on your primary node:

```bash
# if you use mooncake SSD storage
mkdir -p ./mc_storage

mooncake_master \
  --rpc_port=50051 \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=0.0.0.0 \
  --http_metadata_server_port=8080 \
  --metrics_port=9003 \
  --root_fs_dir=./mc_storage/ \
  --cluster_id=mc-local-1 &
```

### 6.2 Configuration (YAML)

Edit your stage config (e.g., `qwen2_5_omni.yaml`).

**Step 1: Define Connector in Global Runtime**

```yaml
runtime:
  connectors:
    connector_of_mooncake:
      name: MooncakeConnector
      extra:
        host: "127.0.0.1"           # Local Worker IP
        metadata_server: "http://<MASTER_IP>:8080/metadata"
        master: "<MASTER_IP>:50051"
        segment: 512000000          # 512MB segment
        localbuf: 64000000          # 64MB buffer
        proto: "tcp"                # "tcp" or "rdma"
    ```

**Mooncake Configuration Parameters:**

*   **host**: The hostname or IP address of the local machine (worker). Mooncake uses this to register itself in the metadata server so other nodes can find it.
*   **metadata_server**: The URL of the metadata server. This is used for service discovery and connection establishment (e.g., exchanging QP information for RDMA).
*   **master**: The address of the Mooncake Master Server (e.g., `<MASTER_IP>:50051`). This is used for global state management and control plane operations.
*   **segment**: The size of the global memory segment in bytes (default: ~512MB). This defines the shared memory region accessible by Mooncake for data transfer.
*   **localbuf**: The size of the local buffer in bytes (default: ~64MB). Used for local data buffering during transfer operations.
*   **proto**: The transport protocol to use. Options:
    *   `tcp`: Standard TCP/IP (easier setup, universal compatibility).
    *   `rdma`: Remote Direct Memory Access (higher performance, requires RDMA-capable hardware).

For more details, refer to the [Mooncake Repository](https://github.com/kvcache-ai/Mooncake).

    **Step 2: Reference in Stages**

Explicitly link stages using `input_connectors` and `output_connectors`:

```yaml
stage_args:
  - stage_id: 0
    # ...
    output_connectors:
      to_stage_1: connector_of_mooncake

  - stage_id: 1
    # ...
    input_connectors:
      from_stage_0: connector_of_mooncake
```

## 7. Using SharedMemoryConnector (Auto-Mode)

**Best for single-node.**

The system will automatically create `SharedMemoryConnector`s for any pipeline edge that does not have an explicit connector defined. This is inferred from:
1.  `runtime.edges` list in the config.
2.  `engine_input_source` dependencies defined in `stage_args`.

### Threshold Configuration
By default, payloads larger than **64KB** (default threshold) are transferred via shared memory, while smaller ones use the control queue (inline).

To adjust this threshold (e.g., to 1GB), add the following to your `runtime.connectors`:

```yaml
runtime:
  connectors:
    connector_of_shared_memory:
      name: SharedMemoryConnector
      extra:
        shm_threshold_bytes: 1024 # 1KB threshold
```

## 8. Summary

| Use Case | Recommended Connector | Configuration |
| :--- | :--- | :--- |
| **Single Node** | `SharedMemoryConnector` | **None** (Automatic) or Custom Threshold |
| **Multi Node** | `MooncakeConnector` | Explicit YAML + Mooncake Master |

## 9. Operational Notes (important)

- **Fail-fast config validation**: the loader raises if any expected edge is missing a connector. Define `input_connectors`/`output_connectors` or rely on auto-SHM filling; otherwise startup aborts.
- **Missing payloads halt the stage**: workers expect connector payloads; if metadata or connector config is missing, the stage raises and stops. Verify connector wiring and metadata propagation before production.

## 10. Future Roadmap: Device-to-Device (D2D) Transport

The current implementations (`SharedMemoryConnector`, `MooncakeConnector`) utilize a **D2H2D (Device-to-Host-to-Device)** data path. Tensors are moved to CPU memory (Host) for transport, which incurs PCIe overhead.

As outlined in the design RFC, future versions will introduce **D2D (Device-to-Device)** connectors:

*   **Goal**: Direct GPU-to-GPU transfer (via NCCL, UCX, or IPC) to minimize latency for large tensor payloads.
*   **Mechanism**: The `OmniConnector` API allows `put()` to initiate a transfer and return a lightweight handle (metadata) via the control plane, while the heavy payload flows directly between devices.
