# Z-Image-Turbo for image generation and editing on 1x RTX 5880 48GB

## Summary

- Vendor: Tongyi
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

This recipe documents one validated reference configuration for CUDA GPU serving.

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
- GPU: `NVIDIA RTX 5880 Ada Generation`, 48GB reported in `nvidia-smi` samples

#### Command

The user should start the vllm server from the root directory.

```bash
vllm serve Tongyi-MAI/Z-Image-Turbo \
  --omni \
  --host 127.0.0.1 \
  --port 8091
```

You can also run `vllm-omni serve` with the same argument

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
  base64 -d > output.png
```

After the image is generated, you can further test the image editing part with command:

```bash
curl -s -X POST "http://localhost:8091/v1/images/edits" \
  -F "model=Tongyi-MAI/Z-Image-Turbo" \
  -F "image=@./output.png" \
  -F "prompt=The cat got a huge necklace." \
  -F "seed=42" \
  -F "output_format=png" | \
  python -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["b64_json"])' | \
  base64 -d > output_editted.png
```

#### Notes

- Memory usage: In both image generation and image editing parts of the example run, `nvidia-smi` samples showed peak GPU memory usage of `23.91 GB`, and the request completed in about `9063.00ms`.
- Key flags: `--omni` is required;
- Known limitations: Image sizes other than `1024x1024` were not tested in this recipe. Memory usage and latency may vary with image size.
