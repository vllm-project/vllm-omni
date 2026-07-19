# DeepSeek Janus for text-to-image generation

## Summary

- Vendor: DeepSeek
- Model: `deepseek-ai/Janus-1.3B` / `deepseek-ai/Janus-Pro-7B`
- Task: Text-to-image generation
- Mode: Online serving with the OpenAI-compatible API
- Maintainer: Community

## When to use this recipe

Use this recipe when serving DeepSeek Janus text-to-image models with
vLLM-Omni. Janus uses autoregressive image-token prediction plus VQ decode
instead of a classical DiT denoising loop, so it requires an explicit
single-stage deploy config.

## References

- Upstream model card: <https://huggingface.co/deepseek-ai/Janus-1.3B>
- Upstream model card (Pro): <https://huggingface.co/deepseek-ai/Janus-Pro-7B>
- Related example under `examples/`:
  [`examples/offline_inference/deepseek_janus/README.md`](../../examples/offline_inference/deepseek_janus/README.md)

## Hardware Support

This recipe documents CUDA GPU serving. Add ROCm, NPU, or XPU sections when
those configurations are validated.

## GPU

### 1x RTX 5090

#### Environment

- OS: Linux
- Python: 3.10+
- Driver / runtime: NVIDIA CUDA environment with an RTX 5090 GPU
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Command

Install Janus-specific dependencies:

```bash
pip install "addict>=2.4.0" "timm>=0.9.16"
```

```bash
vllm serve deepseek-ai/Janus-Pro-7B --omni \
  --deploy-config vllm_omni/deploy/deepseek_janus_single_stage.yaml \
  --port 8091 \
  --tensor-parallel-size 1
```

#### Verification

```bash
curl -X POST http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A scenic mountain lake at sunset",
    "size": "384x384",
    "guidance_scale": 5.0,
    "seed": 42
  }'
```

#### Notes

- Janus outputs fixed 384 x 384 images through a 24 x 24 VQ latent grid.
- `--num-inference-steps` has no effect because Janus runs a fixed 576-token AR
  loop.
- CPU offload and quantization are applicable. TeaCache, Cache-DiT,
  tensor parallelism, CFG parallelism, VAE patch parallelism, and diffusion step
  execution are not wired for this single-stage Janus implementation.
