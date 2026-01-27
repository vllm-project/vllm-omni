# YuanrongConnector

## When to Use

Best for multi-node distributed inference using Yuanrong Datasystem.

## Mechanism

Uses Yuanrong Datasystem's distributed KV store (`datasystem.kv_client`).

- Data Plane: TCP or RDMA for high-bandwidth transfer.
- Control Plane: Yuanrong Datasystem workers and etcd.
- Keying: deterministic keys based on `put_key` (often composed as `request_id:fromStage_toStage`).

## Installation

```bash
pip install openyuanrong-datasystem
```

## Start etcd

```bash
# Download and install etcd (v3.5.12 or higher)
ETCD_VERSION="v3.5.12"
ETCD_ARCH="linux-arm64"
wget https://github.com/etcd-io/etcd/releases/download/${ETCD_VERSION}/etcd-${ETCD_VERSION}-${ETCD_ARCH}.tar.gz
tar -xvf etcd-${ETCD_VERSION}-${ETCD_ARCH}.tar.gz
cd etcd-${ETCD_VERSION}-${ETCD_ARCH}
sudo cp etcd etcdctl /usr/local/bin/

# Start etcd
etcd \
  --name etcd-single \
  --data-dir /tmp/etcd-data \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://0.0.0.0:2379 \
  --listen-peer-urls http://0.0.0.0:2380 \
  --initial-advertise-peer-urls http://0.0.0.0:2380 \
  --initial-cluster etcd-single=http://0.0.0.0:2380 &

# Verify etcd is running
etcdctl --endpoints "127.0.0.1:2379" put key "value"
etcdctl --endpoints "127.0.0.1:2379" get key
```

For production environments, refer to the
[official etcd clustering documentation](https://etcd.io/docs/current/op-guide/clustering/).

## Start Datasystem Worker

```bash
# Replace ${ETCD_IP} with etcd node IP, ${WORKER_IP} with local node IP
dscli start -w \
  --worker_address "${WORKER_IP}:31501" \
  --etcd_address "${ETCD_IP}:2379" \
  --shared_memory_size_mb 20480
```

To stop the worker:

```bash
dscli stop --worker_address "${WORKER_IP}:31501"
```

## Configuration

Define the connector in runtime:

```yaml
runtime:
  connectors:
    connector_of_yuanrong:
      name: YuanrongConnector
      extra:
        host: "127.0.0.1"
        port: 31501
        get_sub_timeout_ms: 1000
```

Wire stages to the connector:

```yaml
stage_args:
  - stage_id: 0
    output_connectors:
      to_stage_1: connector_of_yuanrong

  - stage_id: 1
    input_connectors:
      from_stage_0: connector_of_yuanrong
```

Parameters:

- host: datasystem worker host.
- port: datasystem worker port.
- get_sub_timeout_ms: get timeout in milliseconds (0 for no timeout).

For more details, refer to the
[Yuanrong Datasystem repository](https://atomgit.com/openeuler/yuanrong-datasystem).
