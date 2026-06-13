# Ideogram4 text-to-image serving

## Summary

- Vendor: Ideogram
- Model: `ideogram-ai/ideogram-4-nf4` (NF4), `ideogram-ai/ideogram-4-fp8` (FP8)
- Task: Text-to-image generation
- Mode: Online serving with the OpenAI-compatible API
- Maintainer: Community

## When to use this recipe

Use this recipe when you want a practical starting point for serving Ideogram4 with vLLM-Omni, especially the gated NF4 checkpoint and its structured JSON prompting flow. This implementation is adapted from the diffusers Ideogram4 pipeline and the upstream Ideogram AI implementation.

> Note: Ideogram4 requires Hugging Face authentication and license acceptance, so it is not a good candidate for default CI/e2e coverage.

## References

- Upstream or canonical docs:
  - `docs/user_guide/examples/offline_inference/ideogram4.md`
  - `docs/user_guide/examples/online_serving/ideogram4.md`
  - `docs/serving/image_generation_api.md`
- Related issue or discussion:
  - None at the moment

## Hardware Support

This recipe currently documents one CUDA serving path. Extend it with additional platforms as community validation lands.

## GPU

### 1x CUDA GPU

#### Environment

- OS: Linux
- Python: 3.12
- Driver / runtime: NVIDIA CUDA stack
- vLLM: use the version from this checkout
- vLLM-Omni: use this repository checkout
- Hugging Face: accept the license gate for `ideogram-ai/ideogram-4-nf4` or `ideogram-ai/ideogram-4-fp8`

#### Command

```bash
vllm serve ideogram-ai/ideogram-4-nf4 --omni --port 8091
```

To use the fp8 checkpoint, swap the model id:

```bash
vllm serve ideogram-ai/ideogram-4-fp8 --omni --port 8091
```

#### Verification

```bash
curl -s http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A retro-futuristic city skyline at sunset",
    "size": "1024x1024",
    "num_inference_steps": 50,
    "guidance_scale": 7.0,
    "seed": 42
  }' | jq -r '.data[0].b64_json' | base64 -d > ideogram4.png
```

#### Notes

- Ideogram4 prefers structured JSON captions, but plain text prompts also work.
- The model uses asymmetric CFG with separate conditional and unconditional branches.
- The default number of diffusion steps in this implementation is 50.
- The checkpoint is gated on Hugging Face; accept the model license before first use.
