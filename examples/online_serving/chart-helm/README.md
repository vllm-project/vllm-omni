# vLLM-Omni Helm Chart

Helm chart for deploying [vLLM-Omni](https://github.com/vllm-project/vllm-omni) on Kubernetes. vLLM-Omni extends vLLM with omni-modality model serving, supporting text-to-image, multimodal chat, text-to-speech, and more.

## Prerequisites

- Kubernetes 1.24+
- Helm 3.x
- NVIDIA GPU nodes with [NVIDIA Device Plugin](https://github.com/NVIDIA/k8s-device-plugin)

## Quick Start

```bash
helm install my-release ./chart-helm \
  --set model=Tongyi-MAI/Z-Image-Turbo
```

## Configuration

### Model Selection

Set the `model` value to any supported HuggingFace model ID:

| Model | Type | GPUs | Notes |
|-------|------|------|-------|
| `Tongyi-MAI/Z-Image-Turbo` | text-to-image | 1 | Small, fast (default) |
| `stabilityai/stable-diffusion-3.5-medium` | text-to-image | 1 | ~6GB VRAM |
| `Qwen/Qwen-Image` | text-to-image | 1 | Large, ~40GB+ VRAM |
| `Qwen/Qwen2.5-Omni-7B` | multimodal | 2 | Text + audio + image + video |
| `Qwen/Qwen3-Omni-7B-Chat` | multimodal | 2 | Latest omni model |
| `Qwen/Qwen3-TTS` | text-to-speech | 1 | TTS |

### HuggingFace Token

For gated models that require authentication:

```bash
helm install my-release ./chart-helm \
  --set model=Qwen/Qwen2.5-Omni-7B \
  --set hfToken=hf_xxxxx \
  --set resources.requests."nvidia\.com/gpu"=2 \
  --set resources.limits."nvidia\.com/gpu"=2
```

### Omni-Specific Flags

Enable VAE memory optimizations for diffusion models:

```bash
helm install my-release ./chart-helm \
  --set model=Qwen/Qwen-Image \
  --set omniArgs.vaeUseSlicing=true \
  --set omniArgs.vaeUseTiling=true
```

Enable CPU offloading:

```bash
helm install my-release ./chart-helm \
  --set model=Qwen/Qwen-Image \
  --set omniArgs.enableCpuOffload=true
```

Pass additional raw CLI flags:

```bash
helm install my-release ./chart-helm \
  --set model=Qwen/Qwen-Image \
  --set omniArgs.extraArgs[0]="--enable-layerwise-offload"
```

### Model Cache

By default, a PersistentVolumeClaim is created for the HuggingFace model cache to avoid re-downloading models on pod restarts:

```yaml
modelCache:
  enabled: true
  storageSize: "50Gi"
  storageClassName: ""
```

To use an ephemeral volume instead:

```bash
helm install my-release ./chart-helm \
  --set modelCache.enabled=false
```

### Custom Command Override

To fully override the container command:

```bash
helm install my-release ./chart-helm \
  --set image.command[0]=vllm \
  --set image.command[1]=serve \
  --set image.command[2]=my-model \
  --set image.command[3]=--omni \
  --set image.command[4]=--host \
  --set image.command[5]=0.0.0.0
```

## API Endpoints

Once deployed, vLLM-Omni exposes the following OpenAI-compatible endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Chat completions (text/multimodal) |
| `/v1/images/generations` | POST | Image generation |
| `/v1/images/edits` | POST | Image editing |
| `/v1/audio/speech` | POST | Text-to-speech |

## Files

| File | Description |
|------|-------------|
| `Chart.yaml` | Chart metadata (name, version, maintainers) |
| `values.yaml` | Default configuration values |
| `values.schema.json` | JSON schema for validating values |
| `templates/_helpers.tpl` | Helper templates for common configurations |
| `templates/deployment.yaml` | Kubernetes Deployment |
| `templates/service.yaml` | Kubernetes Service (ClusterIP) |
| `templates/secrets.yaml` | Secrets (generic + HuggingFace token) |
| `templates/pvc.yaml` | PersistentVolumeClaim for model cache |
| `templates/configmap.yaml` | Optional ConfigMap |
| `templates/hpa.yaml` | HorizontalPodAutoscaler |
| `templates/poddisruptionbudget.yaml` | PodDisruptionBudget |
| `templates/custom-objects.yaml` | Custom Kubernetes objects |

## Running Tests

This chart includes unit tests using [helm-unittest](https://github.com/helm-unittest/helm-unittest). Install the plugin and run tests:

```bash
# Install plugin
helm plugin install https://github.com/helm-unittest/helm-unittest

# Run tests
helm unittest .
```
