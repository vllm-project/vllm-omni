# OmniGen2 for text-to-image serving on 1x L40S 48GB

## Summary

- Vendor: OmniGen2
- Model: `OmniGen2/OmniGen2`
- Task: Text-to-image generation
- Mode: Online serving with the OpenAI-compatible Images API
- Maintainer: Community

## When to use this recipe

Use this recipe to serve `OmniGen2/OmniGen2` text-to-image generation on a
single NVIDIA L40S 48GB GPU.

## References

- Upstream or canonical docs:
  [`docs/user_guide/examples/online_serving/text_to_image.md`](../../docs/user_guide/examples/online_serving/text_to_image.md)
- Related example under `examples/`:
  [`examples/online_serving/text_to_image/README.md`](../../examples/online_serving/text_to_image/README.md)

## Hardware Support

This recipe documents one validated CUDA GPU configuration. Extend it with
additional hardware sections as more community validation lands.

## GPU

### 1x L40S 48GB

#### Environment

- OS: Linux
- Python: 3.12.5
- GPU: 1x NVIDIA L40S 48GB
- Driver / runtime: NVIDIA driver 575.51.03, CUDA 12.9
- PyTorch: 2.11.0+cu129
- vLLM version: 0.20.0
- vLLM-Omni version or commit: `efd95567`

#### Command

Start the baseline server:

```bash
vllm-omni serve OmniGen2/OmniGen2 --omni --port 8091 \
  --init-timeout 2400 \
  --stage-init-timeout 2400
```

#### Verification

After the server is ready, confirm the model is listed:

```bash
curl -s http://localhost:8091/v1/models
```

Then run a direct Images API smoke test:

```bash
curl -s http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OmniGen2/OmniGen2",
    "prompt": "A small robot painting a watercolor landscape on an easel, studio lighting, highly detailed",
    "size": "1024x1024",
    "num_inference_steps": 28,
    "guidance_scale": 4.0,
    "seed": 42,
    "response_format": "b64_json"
  }' | jq -r '.data[0].b64_json' | base64 -d > /tmp/omnigen2_l40s_recipe.png
```

Confirm that the decoded image exists:

```bash
test -s /tmp/omnigen2_l40s_recipe.png
```

Expected result:

```text
GET /v1/models returned OmniGen2/OmniGen2.
POST /v1/images/generations returned one b64_json image.
Decoded output: 1024x1024 RGB PNG.
```

#### Notes

- Memory usage: observed peak memory was 19854 MiB for a 1024x1024 smoke test.
  `nvidia-smi` showed 20407 MiB used after the online server was initialized
  and idle.
