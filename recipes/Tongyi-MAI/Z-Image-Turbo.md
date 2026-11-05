# Z-Image Turbo

> Few-step text-to-image with `Tongyi-MAI/Z-Image-Turbo`

## Summary

- Vendor: Tongyi-MAI
- Model: `Tongyi-MAI/Z-Image-Turbo`
- Task: Text-to-image and image editing
- Mode: Offline inference / Online serving
- Maintainer: Community

## When to use this recipe

Use this recipe to generate 1024x1024 images with the distilled Z-Image Turbo
checkpoint through the shared offline text-to-image example, or to serve the
model online through the OpenAI-compatible API for image generation and
editing.

## References

- Model card: <https://huggingface.co/Tongyi-MAI/Z-Image-Turbo>
- Shared runnable example:
  [`examples/offline_inference/text_to_image/text_to_image.py`](../../examples/offline_inference/text_to_image/text_to_image.py)
- CPU offload guide:
  [`docs/user_guide/diffusion/cpu_offload.md`](../../docs/user_guide/diffusion/cpu_offload.md)
- Online serving docs:
  [`docs/user_guide/examples/online_serving/text_to_image.md`](../../docs/user_guide/examples/online_serving/text_to_image.md)
- Related example under `examples/`:
  [`examples/online_serving/text_to_image/`](../../examples/online_serving/text_to_image/)
- Related issue or discussion:
    [RFC: add recipes folder](https://github.com/vllm-project/vllm-omni/issues/2645)

## Hardware Support

This recipe documents a validated Intel XPU offline configuration and a
validated CUDA online-serving configuration. Extend it with more hardware
sections as community validation lands.

## XPU

### 1x Intel Arc Pro B70 (32 GB)

BF16 weights with model-level CPU offload and VAE tiling/slicing, 25 steps.

#### Environment

- OS: Linux
- Python: 3.10+
- torch: 2.13.0+xpu
- vLLM: 0.29.0 (`98dff2a8`)
- vLLM-Omni: `main` at `4c7a98c2`

#### Command

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Tongyi-MAI/Z-Image-Turbo \
  --prompt "a cup of coffee on the table" \
  --num-inference-steps 25 \
  --enable-cpu-offload \
  --vae-use-tiling \
  --vae-use-slicing \
  --enforce-eager \
  --output z_image_turbo_output.png
```

#### Verification

Confirm `z_image_turbo_output.png` is written as a 1024x1024 PNG matching the
prompt.

#### Notes

- Memory usage: peak 15.6 GiB, about 58 s per image.
- Known limitations: only offline generation was qualified. Online serving is
  out of scope for this profile.

## GPU

### 1x RTX 5880 48GB

Online serving with the OpenAI-compatible API for image generation and
editing, with optional FP8 quantization and cache backends.

#### Environment

- OS: Ubuntu-22.04.1
- Python: 3.12.13
- CUDA driver: `580.126.09`
- Runtime: CUDA `cu130` wheel stack
- vLLM version: `0.19.0`
- vLLM-Omni version or commit: `0.19.0rc2.dev198+g78f237e60`
- Related package versions: `torch==2.10.0+cu130`, `transformers==5.5.4`, `accelerate==1.12.0`, `soundfile==0.13.1`
- GPU: `NVIDIA RTX 5880 Ada Generation`, 49140MB reported in `nvidia-smi` samples

#### Command

The user should start the vllm server from the root directory.

```bash
vllm serve Tongyi-MAI/Z-Image-Turbo \
  --omni \
  --host 127.0.0.1 \
  --port 8091
```

#### Verification

Check the server health after startup:

```bash
curl -fsS http://localhost:8091/health
```

If the server is healthy, the serving terminal should show a log entry similar to:

```bash
(APIServer pid=XXX) INFO:     127.0.0.1:XXX - "GET /health HTTP/1.1" 200 OK
```

You are ready to run.

You can validate the image generation function with following command:

```bash
curl -s http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
        "model": "Tongyi-MAI/Z-Image-Turbo",
        "prompt": "A cat is sitting inside a bowl.",
        "size": "1024x1024",
        "num_inference_steps": 9,
        "guidance_scale": 0,
        "seed": 42
      }' | \
  python -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["b64_json"])' | \
  base64 -d > $PWD/output.png
```

After the image is generated, you can further test the image editing part with command:

```bash
curl -s -X POST "http://localhost:8091/v1/images/edits" \
  -F "model=Tongyi-MAI/Z-Image-Turbo" \
  -F "image=@$PWD/output.png" \
  -F "prompt=The cat got a huge necklace." \
  -F "num_inference_steps=9" \
  -F "guidance_scale=0" \
  -F "strength=0.6" \
  -F "seed=42" \
  -F "output_format=png" | \
  python -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["b64_json"])' | \
  base64 -d > $PWD/output_edited.png
```

#### Optional acceleration features

The following acceleration features were also tested with the same server setup. The numbers below are reference measurements from one machine. They may vary depending on
driver version, warmup state, prompt, image size, request payload, and runtime environment.

Unless otherwise noted, image generation used `size=1024x1024`, `num_inference_steps=9`, and `seed=42`.

**Example commands:**

Enable FP8 quantization:

```bash
vllm serve Tongyi-MAI/Z-Image-Turbo \
  --omni \
  --host 127.0.0.1 \
  --port 8091 \
  --quantization fp8
```

Enable FP8 quantization with `tea_cache`:

```bash
vllm serve Tongyi-MAI/Z-Image-Turbo \
  --omni \
  --host 127.0.0.1 \
  --port 8091 \
  --quantization fp8 \
  --cache-backend tea_cache
```

Enable `cache_dit`:

```bash
vllm serve Tongyi-MAI/Z-Image-Turbo \
  --omni \
  --host 127.0.0.1 \
  --port 8091 \
  --cache-backend cache_dit
```

Enable FP8 quantization with `cache_dit`:

```bash
vllm serve Tongyi-MAI/Z-Image-Turbo \
  --omni \
  --host 127.0.0.1 \
  --port 8091 \
  --quantization fp8 \
  --cache-backend cache_dit
```

#### Performance results

##### Image generation

| CLI flag | Latency | Peak GPU Memory (`nvidia-smi`) |
| -------- | ------- | ------ |
| baseline | ~9500ms | 24987 MB |
| `--quantization fp8` | ~5600ms | 19753 MB |
| `--cache-backend tea_cache` | ~9300ms | 25003 MB |
| `--cache-backend cache_dit` | ~9300ms | 24911 MB |
| `--quantization fp8 --cache-backend tea_cache` | ~5600ms | 19835 MB |
| `--quantization fp8 --cache-backend cache_dit` | ~5600ms | 19827 MB |

##### Image editing

| CLI flag | Latency | Peak GPU Memory (`nvidia-smi`) |
| -------- | ------- | ------ |
| baseline | ~34500ms | 24987 MB |
| `--quantization fp8` | ~8200ms | 19410 MB |
| `--cache-backend tea_cache` | ~11000ms | 22652 MB |
| `--cache-backend cache_dit` | ~11000ms | 24525 MB |
| `--quantization fp8 --cache-backend tea_cache` | ~7100ms | 17890 MB |
| `--quantization fp8 --cache-backend cache_dit` | ~7100ms | 19410 MB |

#### Notes

- Memory usage: see the Peak GPU Memory columns in the tables above. The highest peaks across all runs came from the unquantized configurations, while `--quantization fp8` cut the peak to the ~17.9-19.8 GB range in every tested combination.
- Key flags: `--omni` is required.
- Known limitations: Image sizes other than `1024x1024` were not tested in this recipe. Memory usage and latency may vary with image size.
- Steps and CFG: `num_inference_steps=9` suffices because `Z-Image-Turbo` is a distilled few-step model that runs without classifier-free guidance (CFG) by default. The example commands pin `guidance_scale=0` explicitly; setting `guidance_scale > 0` enables CFG, roughly doubles the per-step cost, and brings no benefit for this distilled checkpoint (the shared offline example also uses `--guidance-scale 0.0`). The editing command pins `strength=0.6`, the Z-Image default.
- `--quantization fp8` provided the best overall speed and memory improvements.
- Cache backends had little impact on *image generation* but reduced *image editing* latency substantially (see the tables above). Both requests use the same 9 denoising steps, so the step count alone does not explain the asymmetry; this recipe does not characterize the cause further.
- `seed` was set to 42 in this recipe for reproducibility, setting it to a different value may lead to different results.
