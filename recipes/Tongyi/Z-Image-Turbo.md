# Z-Image-Turbo for text-to-image

## Summary

- Vendor: Tongyi
- Model: `Tongyi-MAI/Z-Image-Turbo`
- Task: Text-to-image generation
- Mode: Online serving with the OpenAI-compatible image generation API
- Maintainer: Community

## When to use this recipe

Use this recipe when you want a known-good starting point for serving
`Tongyi-MAI/Z-Image-Turbo` with vLLM-Omni on a single NVIDIA GeForce RTX 5090
32 GB GPU and validate the deployment with both the online image-generation API
and the shared offline text-to-image example.

## References

- Upstream model card: <https://huggingface.co/Tongyi-MAI/Z-Image-Turbo>
- Supported models:
  [`docs/models/supported_models.md`](../../docs/models/supported_models.md)
- Image generation API:
  [`docs/serving/image_generation_api.md`](../../docs/serving/image_generation_api.md)
- Related example under `examples/`:
  [`examples/offline_inference/text_to_image/text_to_image.py`](../../examples/offline_inference/text_to_image/text_to_image.py)
- Related docs:
  [`docs/user_guide/examples/offline_inference/text_to_image.md`](../../docs/user_guide/examples/offline_inference/text_to_image.md)
- Related issue or discussion:
  [#2645](https://github.com/vllm-project/vllm-omni/issues/2645)

## Hardware Support

## GPU

### 1x NVIDIA GeForce RTX 5090 32GB

#### Environment

- OS: Linux (AutoDL container)
- Python: `3.12.3`
- PyTorch: `2.7.0+cu128`
- Driver / runtime: NVIDIA driver `580.105.08`, CUDA runtime `13.0`
- GPU: NVIDIA GeForce RTX 5090 32 GB (`32607 MiB`)
- vLLM version: `0.24.0`
- vLLM-Omni version or commit: `0.1.dev2146+g86bdcaf3d` (`86bdcaf3`)

#### Command

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Tongyi-MAI/Z-Image-Turbo \
  --omni \
  --port 8000
```

#### Verification

Health check and image generation:

```bash
curl -s -o /tmp/zimage_health.txt -w '%{http_code}' \
  http://127.0.0.1:8000/health
curl -o /tmp/z_image_turbo_5090.png \
  -X POST http://127.0.0.1:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cup of coffee on the table",
    "size": "1024x1024",
    "num_inference_steps": 9,
    "guidance_scale": 0.0,
    "seed": 42,
    "response_format": "file"
  }'
ls -lh /tmp/z_image_turbo_5090.png
file /tmp/z_image_turbo_5090.png
```
```

Offline smoke test:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Tongyi-MAI/Z-Image-Turbo \
  --prompt "a cup of coffee on the table" \
  --seed 42 \
  --guidance-scale 0.0 \
  --num-images-per-prompt 1 \
  --num-inference-steps 9 \
  --height 1024 \
  --width 1024 \
  --output /root/autodl-tmp/outputs/z_image_coffee.png
```

#### Notes

- Output: online serving returned `HTTP 200 OK`; the request saved a `3.1M`
  PNG at `/tmp/z_image_turbo_5090.png`, and `file(1)` identified it as a valid
  `1024x1024` RGB PNG. The offline example also completed successfully and saved
  `/root/autodl-tmp/outputs/z_image_coffee.png`.
- Performance: the validated online request completed in `5125.196 ms`
  end-to-end (`num_inference_steps=9`, `guidance_scale=0.0`, `1024x1024`).
  The offline example completed in `5.3693 s`.
- Memory usage: observed model loading used `19.1941 GiB`; process-scoped GPU
  memory after loading was `19.96 GiB`; offline peak memory was `24230 MB`
  (~`23.66 GiB`). This is consistent with the user-guide baseline of `24.8 GiB`
  peak VRAM for `1024x1024`.
- Key flags: `--omni` is required. `Tongyi-MAI/Z-Image-Turbo` is a distilled
  model, so start from `4-9` denoising steps with `guidance_scale=0.0` rather
  than using large CFG values or 50-step diffusion defaults.
