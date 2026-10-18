# Nucleus-Image

This example demonstrates how to serve `NucleusAI/Nucleus-Image` online with
vLLM-Omni and call it through the OpenAI-compatible image generation API.

For H20-specific notes and broader benchmark guidance, see:
`recipes/NucleusAI/Nucleus-Image.md`.

## Start Server

### Basic Start

```bash
vllm serve NucleusAI/Nucleus-Image --omni --port 8091
```

### Start with Parameters

```bash
bash examples/online_serving/nucleus_image/run_server.sh
```

### Start with Parallelism / Quantization

```bash
# Tensor parallelism
vllm serve NucleusAI/Nucleus-Image --omni --port 8091   --tensor-parallel-size <tp_size>

# Ulysses sequence parallelism
vllm serve NucleusAI/Nucleus-Image --omni --port 8091   --usp <ulysses_degree>

# Ring sequence parallelism
vllm serve NucleusAI/Nucleus-Image --omni --port 8091   --ring <ring_degree>

# Optional FP8 example
vllm serve NucleusAI/Nucleus-Image --omni --port 8091   --quantization fp8
```

## API Calls

### Method 1: Using curl

```bash
bash examples/online_serving/nucleus_image/run_curl_text_to_image.sh
```

Or run directly:

```bash
curl -X POST http://localhost:8091/v1/images/generations   -H "Content-Type: application/json"   -d '{
    "prompt": "A weathered lighthouse on a rocky coastline at golden hour.",
    "size": "1024x1024",
    "num_inference_steps": 50,
    "guidance_scale": 4.0,
    "negative_prompt": "blurry, low quality",
    "seed": 42
  }' | jq -r '.data[0].b64_json' | base64 -d > output.png
```

### Method 2: Using Python Client

```bash
python examples/online_serving/nucleus_image/openai_chat_client.py   --prompt "A weathered lighthouse on a rocky coastline at golden hour."   --output nucleus_output.png
```

## Generation Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `size` | str | None | Image size, e.g. `1024x1024` |
| `num_inference_steps` | int | 50 | Number of denoising steps |
| `guidance_scale` | float | 4.0 | Classifier-free guidance scale |
| `negative_prompt` | str | None | Negative prompt |
| `seed` | int | None | Random seed |
| `n` | int | 1 | Number of output images |

## Notes

- This example focuses on the standard `/v1/images/generations` path.
- For offline inference, see `examples/offline_inference/nucleus_image/`.
- For benchmark-oriented commands and H20-specific measurements, see `recipes/NucleusAI/Nucleus-Image.md`.
