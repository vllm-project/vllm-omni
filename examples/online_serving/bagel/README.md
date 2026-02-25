# BAGEL-7B-MoT

## 🛠️ Installation

Please refer to [README.md](../../../README.md)

## Run examples (BAGEL-7B-MoT)

**Note**: These examples work with the default configuration on an **NVIDIA A100 (80GB)**. We also tested on dual **NVIDIA RTX 5000 Ada (32GB each)**. For dual-GPU setups, please modify the stage configuration to distribute the model across devices.

### Launch the Server

```bash
# Use default configuration
vllm serve ByteDance-Seed/BAGEL-7B-MoT --omni --port 8091
```

Or use the convenience script:

```bash
cd /workspace/vllm-omni/examples/online_serving/bagel
bash run_server.sh
```

```bash
vllm serve ByteDance-Seed/BAGEL-7B-MoT --omni --port 8091 --stage-configs-path /path/to/stage_configs_file
```

#### 🚀 Tensor Parallelism (TP)

For larger models or multi-GPU environments, you can enable Tensor Parallelism (TP) for the server.

1. **Modify Stage Config**: Create or modify a stage configuration yaml (e.g., [`bagel.yaml`](../../../vllm_omni/model_executor/stage_configs/bagel.yaml)). Set `tensor_parallel_size` to `2` (or more) and update `devices` to include multiple GPU IDs (e.g., `"0,1"`).

```yaml
    engine_args:
      tensor_parallel_size: 2
      ...
    runtime:
      devices: "0,1"
```

2. **Launch Server**:
```bash
vllm serve ByteDance-Seed/BAGEL-7B-MoT --omni --port 8091 --stage-configs-path /path/to/your/custom_bagel.yaml
```

#### Using Mooncake Connector

By default, BAGEL uses `SharedMemoryConnector` for inter-stage communication. You can use the [Mooncake](https://github.com/kvcache-ai/Mooncake) connector to transfer KV cache between stages, which also enables multi-node deployment.

**1. Install Mooncake**

```bash
# For CUDA-enabled systems (recommended)
pip install mooncake-transfer-engine

# For non-CUDA systems
pip install mooncake-transfer-engine-non-cuda
```

**2. Start Mooncake Master** on the primary node:

```bash
# Optional: enable disk-backed storage by creating a directory and passing --root_fs_dir.
# Without it, Mooncake runs in memory-only mode, which is sufficient for KV cache transfer.
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

**3. Launch the server** with the Mooncake stage config:

```bash
vllm serve ByteDance-Seed/BAGEL-7B-MoT --omni --port 8091 \
    --stage-configs-path vllm_omni/model_executor/stage_configs/bagel_multiconnector.yaml
```

> **Note**: Before launching, edit [`bagel_multiconnector.yaml`](../../../vllm_omni/model_executor/stage_configs/bagel_multiconnector.yaml) and replace the `metadata_server` and `master` addresses with your Mooncake master node's actual IP. For single-node testing, `127.0.0.1` works.

The client-side usage is identical to the default setup -- the Mooncake connector is transparent to the API. See the requests section below.

For more details on the Mooncake connector configuration, see the [Mooncake Store Connector documentation](../../../docs/design/feature/omni_connectors/mooncake_store_connector.md).

#### Multi-Node Deployment

You can deploy each stage on a **separate node** for better resource utilization. In this example, the orchestrator (Stage 0 / Thinker) and Stage 1 (DiT) run on different machines, connected via Mooncake.

Replace `<ORCHESTRATOR_IP>` below with the actual IP address of your orchestrator node (e.g., `10.244.227.244`).

> [!WARNING]
> **Before launching**, edit [`bagel_multiconnector.yaml`](../../../vllm_omni/model_executor/stage_configs/bagel_multiconnector.yaml) and replace the `metadata_server` and `master` addresses with your Mooncake master node's actual IP. Mismatched addresses will cause silent connection failures.

**1. Start Mooncake Master** (on the orchestrator node):

```bash
mooncake_master \
  --rpc_port=50051 \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=<ORCHESTRATOR_IP> \
  --http_metadata_server_port=8080 \
  --metrics_port=9003
```

**2. Launch Stage 0 (Thinker / Orchestrator)** on the orchestrator node:

```bash
vllm serve ByteDance-Seed/BAGEL-7B-MoT --omni \
    --port 8000 \ # API server port for client requests
    --stage-configs-path vllm_omni/model_executor/stage_configs/bagel_multiconnector.yaml \
    --stage-id 0 \
    -oma <ORCHESTRATOR_IP> \
    -omp 8091
```

**3. Launch Stage 1 (DiT)** on the remote node in headless mode:

```bash
vllm serve ByteDance-Seed/BAGEL-7B-MoT --omni \
    --stage-configs-path vllm_omni/model_executor/stage_configs/bagel_multiconnector.yaml \
    --stage-id 1 \
    --headless \
    -oma <ORCHESTRATOR_IP> \
    -omp 8091
```

**Mooncake Master arguments:**

| Argument | Description |
| :------- | :---------- |
| `--rpc_port` | Mooncake RPC port for control-plane coordination between stages |
| `--enable_http_metadata_server` | Enable the HTTP metadata server for service discovery |
| `--http_metadata_server_host` | IP address to bind the metadata server (use the orchestrator node's IP) |
| `--http_metadata_server_port` | Port for the HTTP metadata server |
| `--metrics_port` | Port for Prometheus-compatible metrics endpoint |

**vllm serve arguments:**

| Argument | Description |
| :------- | :---------- |
| `--stage-id` | Which stage this process runs (0 = Thinker, 1 = DiT) |
| `--headless` | Run without the API server (worker-only mode) |
| `-oma` | Orchestrator master address |
| `-omp` | Orchestrator master port for Stage 1 to connect to Stage 0 for task coordination |

> [!IMPORTANT]
> **Startup Order**: Stage 0 (orchestrator) must be launched **before** Stage 1 (headless).
> Stage 0 will appear to hang on startup until Stage 1 (worker) connects — this is expected behavior.

**Network Requirements**

All nodes must have network connectivity to each other. Ensure the following ports are open **between all participating nodes**:

| Port | Protocol | Service | Direction |
| :--- | :------- | :------ | :-------- |
| 50051 | TCP | Mooncake Master RPC | Worker → Orchestrator |
| 8080 | TCP | Mooncake HTTP Metadata Server | Worker → Orchestrator |
| 8091 | TCP | Orchestrator Master (`-omp`) | Worker → Orchestrator |
| 8000 | TCP | API Server (`--port`) | Client → Orchestrator |
| 9003 | TCP | Metrics (optional) | Monitoring → Orchestrator |

> **Tip**: If nodes are behind a firewall or in different VPCs/security groups, make sure the above ports are allowed in ingress/egress rules. All nodes should be reachable via their IP addresses (no NAT). Using nodes on the same subnet or VPC is recommended to minimize latency for Mooncake KV cache transfers.

### Send Multi-modal Request

Get into the bagel folder:

```bash
cd examples/online_serving/bagel
```

Send request via Python

```bash
python openai_chat_client.py --prompt "A cute cat" --modality text2img
```

The Python client supports the following command-line arguments:

- `--prompt` (or `-p`): Text prompt for generation (default: `A cute cat`)
- `--output` (or `-o`): Output file path for image results (default: `bagel_output.png`)
- `--server` (or `-s`): Server URL (default: `http://localhost:8091`)
- `--image-url` (or `-i`): Input image URL or local file path (for img2img/img2text modes)
- `--modality` (or `-m`): Task modality (default: `text2img`). Options: `text2img`, `img2img`, `img2text`, `text2text`
- `--height`: Image height in pixels (default: 512)
- `--width`: Image width in pixels (default: 512)
- `--steps`: Number of inference steps (default: 25)
- `--seed`: Random seed (default: 42)
- `--negative`: Negative prompt for image generation

Example with custom parameters:

```bash
python openai_chat_client.py \
    --prompt "A futuristic city" \
    --modality text2img \
    --height 768 \
    --width 768 \
    --steps 50 \
    --seed 42 \
    --negative "blurry, low quality"
```

## Modality Control

BAGEL-7B-MoT supports **multiple modality modes** for different use cases.

The default yaml configuration deploys Thinker and DiT on the same GPU. You can use the default configuration file: [`bagel.yaml`](../../../vllm_omni/model_executor/stage_configs/bagel.yaml)

| Modality    | Input        | Output | Description                            |
| ----------- | ------------ | ------ | -------------------------------------- |
| `text2img`  | Text         | Image  | Generate images from text prompts      |
| `img2img`   | Image + Text | Image  | Transform images using text guidance   |
| `img2text`  | Image + Text | Text   | Generate text descriptions from images |
| `text2text` | Text         | Text   | Pure text generation                   |

### Text to Image (text2img)

Generate images from text prompts:

**Using Python client**

```bash
python openai_chat_client.py \
    --prompt "A beautiful sunset over mountains" \
    --modality text2img \
    --output sunset.png \
    --steps 50
```

**Using curl**

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": [{"type": "text", "text": "<|im_start|>A beautiful sunset over mountains<|im_end|>"}]}],
    "modalities": ["image"],
    "height": 512,
    "width": 512,
    "num_inference_steps": 50,
    "seed": 42
  }'
```


### Image to Image (img2img)

Transform images based on text prompts:

**Using Python client**

```bash
python openai_chat_client.py \
    --prompt "Make the cat stand up" \
    --modality img2img \
    --image-url /path/to/input.jpg \
    --output transformed.png
```

**Using curl**

```bash
IMAGE_BASE64=$(base64 -w 0 cat.jpg)

cat <<EOF > payload.json
{
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "<|im_start|>Make the cat stand up<|im_end|>"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,${IMAGE_BASE64}"}}
      ]
    }],
    "modalities": ["image"],
    "height": 512,
    "width": 512,
    "num_inference_steps": 50,
    "seed": 42
}
EOF

curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @payload.json

```

### Image to Text (img2text)

Generate text descriptions from images:

**Using Python client**

```bash
python openai_chat_client.py \
    --prompt "Describe this image in detail" \
    --modality img2text \
    --image-url /path/to/image.jpg
```

**Using curl**

```bash
IMAGE_BASE64=$(base64 -w 0 cat.jpg)

cat <<EOF > payload.json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "<|im_start|>user\n<|image_pad|>\nDescribe this image in detail<|im_end|>\n<|im_start|>assistant\n"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,${IMAGE_BASE64}"}}
    ]
  }],
  "modalities": ["text"]
}
EOF

curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @payload.json
```

### Text to Text (text2text)

Pure text generation:

**Using Python client**

```bash
python openai_chat_client.py \
    --prompt "What is the capital of France?" \
    --modality text2text
```

**Using curl**

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": [{"type": "text", "text": "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"}]}]
    "modalities": ["text"]
  }'
```

## FAQ

- If you encounter an error about the backend of librosa, try to install ffmpeg with the command below.

```bash
sudo apt update
sudo apt install ffmpeg
```

- If you don’t know how much VRAM is needed for the model or encounter the OOM error, you can try to decrease the max_model_len.

| Stage               | VRAM                         |
| :------------------ | :--------------------------- |
| Stage-0 (Thinker)   | **15.04 GiB** **+ KV Cache** |
| Stage-1 (DiT)       | **26.50 GiB**                |
| Total               | **~42 GiB + KV Cache**       |
