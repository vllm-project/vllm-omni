# Distributed utils

This directory (vllm_omni/distributed/ray_utils) contains utilities for distributed execution in vllm-omni, supporting both **Ray** and **Multiprocessing** backends.
## 1. Installation
```bash
pip install "ray[default]"
```
## 2. Ray Utils

The `ray_utils` module provides helper functions for managing Ray clusters and actors, which is essential for:
*   **Multi-node deployment**: Running pipeline stages across different physical machines.
*   **Resource management**: Efficient GPU/CPU allocation.

### 2.1 Basic Usage

To use the Ray backend, specify `worker_backend="ray"` when initializing the engine.

**Command Line Example:**
```bash
vllm serve Qwen/Qwen2.5-Omni-7B \
  --omni \
  --port 8091 \
  --worker-backend ray \
  --ray-address auto
```

### 2.2 Cluster Setup

**Step 1: Start Head Node**
Run this on your primary machine:
```bash
ray start --head --port=6399
```

**Step 2: Connect Worker Nodes**
Run this on each worker machine:
```bash
ray start --address=<HEAD_NODE_IP>:6399
```

> **Tip**: For a complete cluster setup script, refer to the vLLM example:
> [run_cluster.sh](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/run_cluster.sh)

### 2.3 Distributed Connector Support

When running on Ray, the system automatically adapts its communication strategy:

*   **Cross-Node**: Recommended to use `MooncakeConnector` (requires separate configuration).
*   **Same-Node**: Can still use `SharedMemoryConnector` for efficiency, or Ray's native object store (plasma).
*   **SHM threshold default differs**: when `worker_backend="ray"`, the SharedMemoryConnector default threshold is set to `sys.maxsize`, which forces payloads to go inline (no SHM). Override `shm_threshold_bytes` in the connector config if you want SHM for Ray runs.

### 2.4 Internal Helpers

*   **`initialize_ray_cluster`**: Connects to an existing Ray cluster or starts a local one.

## 3. Troubleshooting

*   **Connection Issues**: Ensure the Ray head node is accessible and ports (default 6399 in this example) are open.
*   **Version Mismatch**: Ensure all nodes run the same version of Ray and Python.
