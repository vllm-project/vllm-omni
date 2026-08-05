# HiDream-I1-Full

> Text-to-image serving and benchmark

## Summary

- Vendor: HiDream.ai
- Model: `HiDream-ai/HiDream-I1-Full`
- Task: Text-to-image generation
- Mode: Online serving with the OpenAI-compatible API
- Maintainer: Community

## When to use this recipe

Use this recipe when you want a known-good starting point for serving
`HiDream-ai/HiDream-I1-Full` on a single 80 GB A800, validate the normal online-serving path.

## References

- Model: <https://huggingface.co/HiDream-ai/HiDream-I1-Full>
- Upstream or canonical docs:
  [`docs/user_guide/examples/online_serving/text_to_image.md`](../../docs/user_guide/examples/online_serving/text_to_image.md)
- Related example under `examples/`:
  [`examples/online_serving/text_to_image/README.md`](../../examples/online_serving/text_to_image/README.md)


## Hardware Support

This recipe currently documents one CUDA GPU serving configuration.

## GPU

### 1x A800 80GB

#### Environment

- OS: Linux
- Python: 3.10+
- Driver / runtime: NVIDIA CUDA environment with an A100 80 GB GPU
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Command

Start the baseline server:

```bash
export MODEL_NAME_OR_PATH=HiDream-ai/HiDream-I1-Full
vllm serve ${MODEL_NAME_OR_PATH} \
   --omni \
   --port 8092 \
   --auxiliary-text-encoder meta-llama/Llama-3.1-8B-Instruct \
   --tensor-parallel-size 1 \
   --vae_use_slicing \
   --vae_use_tiling
```


You can also use the example launcher and pass the extra flags through:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model HiDream-ai/HiDream-I1-Full \
  --prompt "The setting sun of late autumn dyes the riverside with a warm orange hue" \
  --seed 42 \
  --guidance-scale 5.0 \
  --tensor-parallel-size 1 \
  --num-images-per-prompt 1 \
  --num-inference-steps 50 \
  --auxiliary-text-encoder meta-llama/Llama-3.1-8B-Instruct \
  --output /workspace/mnt/cmss-wmd/result/output.png

```

#### Verification

Run the existing client example after the server is ready:

```bash
python examples/online_serving/text_to_image/openai_chat_client.py \
  --server http://localhost:8092 \
  --prompt "The setting sun of late autumn dyes the riverside with a warm orange hue" \
  --output /tmp/hidream_i1_recipe.png
```

For a direct API smoke test:

```bash
curl -s http://localhost:8092/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "messages": [
      {"role": "user", "content": "The setting sun of late autumn dyes the riverside with a warm orange hue"}
    ],
    "extra_body": {
      "num_inference_steps": 50,
      "seed": 42,
      "height": 1024,
      "width": 1024
    }
  }' | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2- | base64 -d > output_online.png
```



#### Notes

- Key flags: `--auxiliary-text-encoder` designates the path for the auxiliary text model meta-llama/Llama-3.1-8B-Instruct. For HiDream-I1-Full, unspecified use defaults meta-llama/Llama-3.1-8B-Instruct (downloaded from official Hugging Face), and custom paths are supported.
