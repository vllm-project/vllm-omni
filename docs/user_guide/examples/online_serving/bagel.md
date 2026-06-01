# BAGEL-7B-MoT

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/bagel>.

## Installation

Please refer to [README.md](https://github.com/vllm-project/vllm-omni/tree/main/README.md)

## Architecture

BAGEL-7B-MoT is a Mixture-of-Transformers (MoT) model supporting both image generation and understanding. It offers two deployment topologies:

| Topology | Stages | Description |
| :------- | :----- | :---------- |
| **Two-stage** (default) | Stage 0 (Thinker, AR) + Stage 1 (DiT, Diffusion) | Thinker handles text/understanding via vLLM AR engine; DiT handles image generation. KV cache is transferred between stages. |
| **Single-stage** | Stage 0 (DiT, Diffusion) only | The DiT stage contains a full LLM, ViT, VAE, and tokenizer internally. All modalities are handled within a single diffusion process. |

Both topologies support all four modalities: `text2img`, `img2img`, `img2text`, `text2text`.

> **Note**: These examples work with the default configuration on an **NVIDIA A100 (80GB)**. We also tested on dual **NVIDIA RTX 5000 Ada (32GB each)**. For dual-GPU setups, modify the deploy YAML to distribute stages across devices.

## Launch the Server

### Two-Stage (Default)

The default pipeline is auto-detected from the model. No extra flags needed:

```bash
vllm serve ByteDance-Seed/BAGEL-7B-MoT --omni --port 8000
```

Or use the convenience script:

```bash
cd examples/online_serving/bagel
bash run_server.sh

# Launch a single stage per terminal
bash run_server_stage_cli.sh --stage 0
bash run_server_stage_cli.sh --stage 1
```

To use a custom deploy YAML (note: `--stage-configs-path` is deprecated in favor of `--deploy-config`):

```bash
vllm serve ByteDance-Seed/BAGEL-7B-MoT --omni --port 8000 \
    --deploy-config /path/to/deploy_config.yaml
```

See [`bagel.yaml`](https://github.com/vllm-project/vllm-omni/tree/main/vllm_omni/deploy/bagel.yaml) for the default two-stage deploy configuration.

### Single-Stage

The DiT stage contains a full LLM, ViT, VAE, and tokenizer, so it can handle all modalities (text2img, img2img, img2text, text2text, think) without a separate Thinker stage:

```bash
vllm serve ByteDance-Seed/BAGEL-7B-MoT --omni --port 8000 \
    --deploy-config vllm_omni/deploy/bagel_single_stage.yaml
```

See [`bagel_single_stage.yaml`](https://github.com/vllm-project/vllm-omni/tree/main/vllm_omni/deploy/bagel_single_stage.yaml) for configuration. The `pipeline: bagel_single_stage` field selects the single-stage topology from the pipeline registry.

### Tensor Parallelism (TP)

For larger models or multi-GPU environments, enable TP via CLI:

```bash
vllm serve ByteDance-Seed/BAGEL-7B-MoT --omni --port 8000 --tensor-parallel-size 2
```

Or set `tensor_parallel_size` per stage in a custom deploy YAML.

### Multi-Node Deployment

Deploy each stage on a **separate node** for better resource utilization. Replace `<ORCHESTRATOR_IP>` with the actual IP address of your orchestrator node.

**1. Launch Stage 0 (Thinker / Orchestrator)** on the orchestrator node:

```bash
# API server port for client requests: 8000
vllm serve ByteDance-Seed/BAGEL-7B-MoT --omni \
    --port 8000 \
    --stage-id 0 \
    --omni-master-address <ORCHESTRATOR_IP> \
    --omni-master-port 8000
```

**2. Launch Stage 1 (DiT)** on the remote node in headless mode:

```bash
vllm serve ByteDance-Seed/BAGEL-7B-MoT --omni \
    --stage-id 1 \
    --headless \
    --omni-master-address <ORCHESTRATOR_IP> \
    --omni-master-port 8000
```

Or use the convenience script:

```bash
# Terminal 1: Stage 0
bash run_server_stage_cli.sh --stage 0

# Terminal 2: Stage 1
bash run_server_stage_cli.sh --stage 1

# With extra args
bash run_server_stage_cli.sh --stage 0 -- --tensor-parallel-size 2
bash run_server_stage_cli.sh --stage 1 -- --gpu-memory-utilization 0.9
```

**vllm serve arguments:**

| Argument | Description |
| :------- | :---------- |
| `--stage-id` | Which stage this process runs (0 = Thinker, 1 = DiT) |
| `--headless` | Run without the API server (worker-only mode) |
| `-oma` / `--omni-master-address` | Orchestrator master address |
| `-omp` / `--omni-master-port` | Orchestrator master port |

> [!IMPORTANT]
> **Startup Order**: Stage 0 (orchestrator) must be launched **before** Stage 1 (headless).
> Stage 0 will appear to hang on startup until Stage 1 (worker) connects — this is expected behavior.

### Inter-Stage Connectors

When deploying stages across nodes, configure the connector type in the deploy YAML:

- **SharedMemoryConnector** (default): Used for single-node deployments. No explicit configuration needed.
- **MooncakeTransferEngineConnector**: For multi-node setups with RDMA hardware. Defined in [`bagel.yaml`](https://github.com/vllm-project/vllm-omni/tree/main/vllm_omni/deploy/bagel.yaml) under `connectors.rdma_connector`.

To use Mooncake, create a custom deploy YAML that binds `output_connectors` / `input_connectors` on each stage to the `rdma_connector` defined in the `connectors` section.

## Send Requests

```bash
cd examples/online_serving/bagel
```

### Text to Image (text2img)

**Python client:**

```bash
python openai_chat_client.py \
    --prompt "A beautiful sunset over mountains" \
    --modality text2img \
    --output sunset.png \
    --steps 50
```

**curl:**

```bash
curl http://localhost:8000/v1/chat/completions \
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

**Python client:**

```bash
python openai_chat_client.py \
    --prompt "Make the cat stand up" \
    --modality img2img \
    --image-url /path/to/input.jpg \
    --output transformed.png
```

**curl:**

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

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @payload.json
```

### Image to Text (img2text)

**Python client:**

```bash
python openai_chat_client.py \
    --prompt "Describe this image in detail" \
    --modality img2text \
    --image-url /path/to/image.jpg
```

**curl:**

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

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @payload.json
```

### Text to Text (text2text)

**Python client:**

```bash
python openai_chat_client.py \
    --prompt "What is the capital of France?" \
    --modality text2text
```

**curl:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": [{"type": "text", "text": "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"}]}],
    "modalities": ["text"]
  }'
```

### Python Client Arguments

| Argument | Default | Description |
| :------- | :------ | :---------- |
| `--prompt` / `-p` | `A cute cat` | Text prompt |
| `--output` / `-o` | `bagel_output.png` | Output file path |
| `--server` / `-s` | `http://localhost:8000` | Server URL |
| `--image-url` / `-i` | `None` | Input image URL or local path (img2img/img2text) |
| `--modality` / `-m` | `text2img` | `text2img`, `img2img`, `img2text`, `text2text` |
| `--height` | `512` | Image height in pixels |
| `--width` | `512` | Image width in pixels |
| `--steps` | `25` | Number of inference steps |
| `--seed` | `42` | Random seed |
| `--negative` | `None` | Negative prompt for CFG |

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

## Configuration Reference

### Deploy YAML Files

| File | Description |
| :--- | :---------- |
| [`bagel.yaml`](https://github.com/vllm-project/vllm-omni/tree/main/vllm_omni/deploy/bagel.yaml) | Two-stage default (Thinker + DiT on GPU 0) |
| [`bagel_single_stage.yaml`](https://github.com/vllm-project/vllm-omni/tree/main/vllm_omni/deploy/bagel_single_stage.yaml) | Single-stage (DiT only) |

### Key Deploy YAML Fields

| Field | Scope | Description |
| :---- | :---- | :---------- |
| `pipeline` | top-level | Override auto-detected pipeline (e.g. `bagel_single_stage`) |
| `stages[].stage_id` | per-stage | Stage identifier (0, 1, ...) |
| `stages[].devices` | per-stage | GPU device IDs (e.g. `"0"`, `"0,1"`) |
| `stages[].max_num_seqs` | per-stage | Maximum concurrent sequences |
| `stages[].gpu_memory_utilization` | per-stage | Fraction of GPU memory to use |
| `stages[].enforce_eager` | per-stage | Disable CUDA graphs |
| `stages[].tensor_parallel_size` | per-stage | TP degree for this stage |
| `connectors` | top-level | Define available connector instances (SHM, Mooncake) |
| `platforms` | top-level | Platform-specific overrides (e.g. `xpu`) |

## FAQ

- If you encounter OOM errors, try decreasing `max_model_len` or `gpu_memory_utilization` in the deploy YAML.

**Two-stage VRAM usage:**

| Stage | VRAM |
| :---- | :--- |
| Stage 0 (Thinker) | **15.04 GiB + KV Cache** |
| Stage 1 (DiT) | **26.50 GiB** |
| Total | **~42 GiB + KV Cache** |

**Single-stage VRAM usage:** The DiT loads the full model (~42 GiB) in one process.

## Example materials

??? abstract "openai_chat_client.py"
    ``````py
    --8<-- "examples/online_serving/bagel/openai_chat_client.py"
    ``````
??? abstract "run_server.sh"
    ``````sh
    --8<-- "examples/online_serving/bagel/run_server.sh"
    ``````
??? abstract "run_server_stage_cli.sh"
    ``````sh
    --8<-- "examples/online_serving/bagel/run_server_stage_cli.sh"
    ``````
