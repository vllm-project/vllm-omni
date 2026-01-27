# MooncakeConnector

## When to Use

Best for multi-node distributed inference using Mooncake.

## Installation

```bash
# For CUDA-enabled systems (recommended)
pip install mooncake-transfer-engine

# For non-CUDA systems
pip install mooncake-transfer-engine-non-cuda
```

## Start Mooncake Master

```bash
# If you use Mooncake SSD storage
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

## Configuration

Define the connector in runtime:

```yaml
runtime:
  connectors:
    connector_of_mooncake:
      name: MooncakeConnector
      extra:
        host: "127.0.0.1"
        metadata_server: "http://<MASTER_IP>:8080/metadata"
        master: "<MASTER_IP>:50051"
        segment: 512000000
        localbuf: 64000000
        proto: "tcp"
```

Wire stages to the connector:

```yaml
stage_args:
  - stage_id: 0
    output_connectors:
      to_stage_1: connector_of_mooncake

  - stage_id: 1
    input_connectors:
      from_stage_0: connector_of_mooncake
```

Parameters:

- host: local worker IP registered in the metadata server.
- metadata_server: metadata server URL for discovery and setup.
- master: Mooncake Master address.
- segment: global memory segment size in bytes.
- localbuf: local buffer size in bytes.
- proto: transport protocol ("tcp" or "rdma").

For more details, refer to the
[Mooncake repository](https://github.com/kvcache-ai/Mooncake).
