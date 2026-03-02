# GGUF Quantization

## Goals
1. Reuse vLLM quantization configs and weight loaders as much as possible.
2. Add native GGUF support to diffusion transformers without changing model definitions.
3. Keep user-facing knobs minimal and consistent across offline and online flows.

## Scope
1. Models: Z-Image, and Flux2-klein.
2. Components: diffusion transformer weights, loader paths, and quantization configs.
3. Modes: native GGUF (transformer-only weights).

## Architecture Overview
1. `OmniDiffusionConfig` accepts `quantization` or `quantization_config`.
2. Diffusion quantization wrapper (`DiffusionGgufConfig`) produces vLLM `QuantizationConfig` objects for linear layers.
3. `DiffusersPipelineLoader` branches on quantization method and loads either HF weights or GGUF weights for the transformer.
4. GGUF transformer loading is routed through model-specific adapters (e.g., Flux2Klein).
5. vLLM GGUF path uses `GGUFConfig` and `GGUFLinearMethod` for matmul.

## Call Chain (Offline)
```
CLI (examples/offline_inference/text_to_image/text_to_image.py)
  |
  v
Omni (vllm_omni/entrypoints/omni.py)
  |
  v
OmniStage (diffusion)
  |
  v
DiffusionWorker
  |
  v
DiffusionModelRunner
  |
  v
DiffusersPipelineLoader
  |
  v
Pipeline.forward (Flux2/Qwen/Z-Image)
  |
  v
DiffusionEngine
  |
  v
OmniRequestOutput
  |
  v
Client (saved PNG)
```

## Call Chain (Online)
```
Client
  |
  | POST /v1/images/generations
  v
APIServer (vllm_omni/entrypoints/openai/api_server.py)
  |
  v
_generate_with_async_omni
  |
  v
AsyncOmni
  |
  v
DiffusionEngine
  |
  v
OmniRequestOutput
  |
  v
encode_image_base64
  |
  v
ImageGenerationResponse
  |
  v
Client
```

## Call Chain (GGUF Operator Path)
```
Pipeline.forward (Flux2/Qwen/Z-Image)
  |
  v
Transformer blocks
  |
  v
QKVParallelLinear / ColumnParallelLinear / RowParallelLinear
  |
  v
LinearBase.forward
  |
  v
QuantMethod.apply (GGUFLinearMethod.apply)
  |
  v
fused_mul_mat_gguf
  |
  v
_fused_mul_mat_gguf (custom op)
  |
  v
ops.ggml_dequantize
  |
  v
x @ weight.T
```

## GGUF Weight Loading Path (Transformer-Only)
1. `DiffusersPipelineLoader.load_model` detects `quantization_config.method == "gguf"`.
2. `gguf_model` is resolved as one of: local file, `repo/file.gguf`, or `repo:quant_type`.
3. GGUF weights are routed through adapters in `vllm_omni/diffusion/model_loader/gguf_adapters/`.
4. Name mapping is applied per-architecture (Z-Image, Flux2Klein).
5. GGUF weights are loaded into transformer modules, remaining non-transformer weights come from the HF checkpoint.

## GGUF Adapter Design
1. `GGUFAdapter` is an abstract base class for model-specific adapters.
2. `Flux2KleinGGUFAdapter` implements Flux2-Klein remapping + qkv split + adaLN swap.
3. `ZImageGGUFAdapter` implements Z-Image qkv + ffn shard handling and linear qweight routing.
4. `get_gguf_adapter(...)` strictly selects by model class/config; unsupported models raise an error (no fallback adapter).

Adapter paths:
- Base: `vllm_omni/diffusion/model_loader/gguf_adapters/base.py`
- Z-Image: `vllm_omni/diffusion/model_loader/gguf_adapters/z_image.py`
- Flux2-Klein: `vllm_omni/diffusion/model_loader/gguf_adapters/flux2_klein.py`

## User Usage (Offline)

### Baseline BF16
```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model /workspace/models/black-forest-labs/FLUX.2-klein-4B \
  --prompt "a photo of a forest with mist swirling around the tree trunks. The word 'FLUX.2' is painted over it in big, red brush strokes with visible texture" \
  --height 768 \
  --width 1360 \
  --seed 42 \
  --cfg_scale 4.0 \
  --num_images_per_prompt 1 \
  --num_inference_steps 4 \
  --output outputs/flux2_klein_4b.png
```

### Native GGUF (Transformer Only)
```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model /workspace/models/black-forest-labs/FLUX.2-klein-4B \
  --gguf-model "/workspace/models/unsloth/FLUX.2-klein-4B-GGUF/flux-2-klein-4b-Q8_0.gguf" \
  --quantization gguf \
  --prompt "a photo of a forest with mist swirling around the tree trunks. The word 'FLUX.2' is painted over it in big, red brush strokes with visible texture" \
  --height 768 \
  --width 1360 \
  --seed 42 \
  --cfg_scale 4.0 \
  --num_images_per_prompt 1 \
  --num_inference_steps 4 \
  --output outputs/flux2_klein_4b_gguf.png
```

Notes for GGUF:
1. Many GGUF repos do not ship `model_index.json` and configs. Use the base repo for `--model` and only pass the GGUF file via `--gguf-model`.
2. `gguf_model` supports local path, `repo/file.gguf`, or `repo:quant_type`.

## User Usage (Online)

### Start Server (Native GGUF via CLI)
```bash
vllm serve /workspace/models/black-forest-labs/FLUX.2-klein-4B \
  --omni \
  --port 8000 \
  --quantization-config '{"method":"gguf","gguf_model":"/workspace/models/unsloth/FLUX.2-klein-4B-GGUF/flux-2-klein-4b-Q8_0.gguf"}'
```

### Online Request (Images API)
```bash
curl -X POST http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a dragon laying over the spine of the Green Mountains of Vermont",
    "size": "1024x1024",
    "seed": 42,
    "num_inference_steps": 4
  }'
```
