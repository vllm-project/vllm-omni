# Z-Image-Turbo for image generation and editing

## Summary

- Vendor: Tongyi-MAI
- Model: `Tongyi-MAI/Z-Image-Turbo`
- Task: Image generation / editing with text input
- Mode: Online serving with the OpenAI-compatible API
- Maintainer: Community

## When to use this recipe

Use this recipe to serve `Tongyi-MAI/Z-Image-Turbo` on a validated CUDA GPU configuration.

## References

- Upstream or canonical docs:
  [`docs/user_guide/examples/online_serving/text_to_image.md`](../../docs/user_guide/examples/online_serving/text_to_image.md)
- Related example under `examples/`:
  [`examples/online_serving/text_to_image/`](../../examples/online_serving/text_to_image/)
- Related issue or discussion:
    [RFC: add recipes folder](https://github.com/vllm-project/vllm-omni/issues/2645)

## Hardware Support

The following CUDA GPU configuration has been validated for this recipe.

## GPU

### 1x RTX 5880 48GB

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

#### Performance results

**Image generation**
| CLI flag | Latency | Peak GPU Memory (`nvidia-smi`) |
| -------- | ------- | ------ |
| baseline | ~9500.00ms | 24987 MB |
| `--quantization fp8` | ~5600.00ms | 19753 MB |
| `--cache-backend tea_cache` | ~9300.00ms | 25003 MB |
| `--cache-backend cache_dit` | ~9300.00ms | 24911 MB |
| `--quantization fp8 --cache-backend tea_cache` | ~5600.00ms | 19835 MB |
| `--quantization fp8 --cache-backend cache_dit` | ~5600.00ms | 19827 MB |


**Image editing**
| CLI flag | Latency | Peak GPU Memory (`nvidia-smi`) |
| -------- | ------- | ------ |
| baseline | ~34500.00ms | 24987 MB |
| `--quantization fp8` | ~8200.00ms | 19410 MB |
| `--cache-backend tea_cache` | ~11000.00ms | 22652 MB |
| `--cache-backend cache_dit` | ~11000.00ms | 24525 MB |
| `--quantization fp8 --cache-backend tea_cache` | ~7100.00ms | 17890 MB |
| `--quantization fp8 --cache-backend cache_dit` | ~7100.00ms | 19410 MB |


#### Notes

- Memory usage: In both image generation and image editing parts of the example run, `nvidia-smi` samples showed peak GPU memory usage of `24987 MB`, and the image generation request completed in about `9500.00ms`, while the image editing request completed in about `34500.00ms`.
- Key flags: `--omni` is required;
- Known limitations: Image sizes other than `1024x1024` were not tested in this recipe. Memory usage and latency may vary with image size.

- `--quantization fp8` provided the best overall speed and memory improvements.
- Cache backends had little impact on *image generation*, likely because `Z-Image-Turbo` uses only a few inference steps. For *image editing*, cache backends reduced latency substantially.
- `seed` was set to 42 in this recipe for reproducibility, setting it to a different value may lead to different results.
