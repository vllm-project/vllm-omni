# VLLM-Omni Distributed Connectors

This guide explains how to configure and use distributed connectors in vllm-omni for multi-stage pipelines.

## 1. Overview

Connectors enable data transfer between pipeline stages (e.g., Thinker -> Talker).
Currently supported connectors are:
1. **SharedMemoryConnector**: Uses system shared memory.
2. **MooncakeConnector**: Uses [Mooncake](https://github.com/kvcache-ai/Mooncake).

*   **SharedMemoryConnector (Default)**: Zero-copy, lowest latency. Best for **single-node** deployments. Auto-configured if no connectors are specified.
*   **MooncakeConnector**: TCP/RDMA based. Best for **multi-node** distributed deployments. Requires a Mooncake Master service.

## 2. Installation (Mooncake)

If using `MooncakeConnector`, install the library first:

```bash
# For CUDA-enabled systems (Recommended)
pip install mooncake-transfer-engine

# For non-CUDA systems
pip install mooncake-transfer-engine-non-cuda
```

## 3. Using MooncakeConnector

### 3.1 Start Mooncake Master

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

### 3.2 Configuration (YAML)

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

## 4. Using SharedMemoryConnector (Auto-Mode)

**Best for single-node.**

The system will automatically create SHM connectors based on `runtime.edges` if no explicit connectors are defined.

### Threshold Configuration
By default, payloads larger than **64KB** (default threshold) are transferred via shared memory, while smaller ones use the control queue.

To adjust this threshold (e.g., to 1GB), add the following to your `runtime.connectors`:

```yaml
runtime:
  connectors:
    connector_of_shared_memory:
      name: SharedMemoryConnector
      extra:
        shm_threshold_bytes: 1024 # 1KB threshold
```

## 5. Summary

| Use Case | Recommended Connector | Configuration |
| :--- | :--- | :--- |
| **Single Node** | `SharedMemoryConnector` | **None** (Automatic) or Custom Threshold |
| **Multi Node** | `MooncakeConnector` | Explicit YAML + Mooncake Master |
