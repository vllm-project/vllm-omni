# Qwen-Image for text-to-image serving with optional continuous batching on 1x A100 80GB

## Summary

- Vendor: Qwen
- Model: `Qwen/Qwen-Image`
- Task: Text-to-image generation
- Mode: Online serving with optional step-wise continuous batching
- Maintainer: Community

## When to use this recipe

Use this recipe when you want a known-good starting point for serving
`Qwen/Qwen-Image` on a single 80 GB A100, validate the normal online-serving
path, and optionally replay the benefit of step-wise continuous batching with
the benchmark assets already bundled in this repository.

## References

- Upstream or canonical docs:
  [`docs/user_guide/examples/online_serving/text_to_image.md`](../../docs/user_guide/examples/online_serving/text_to_image.md)
- Related example under `examples/`:
  [`examples/online_serving/text_to_image/README.md`](../../examples/online_serving/text_to_image/README.md)
- Related benchmark:
  [`benchmarks/diffusion/diffusion_benchmark_serving.py`](../../benchmarks/diffusion/diffusion_benchmark_serving.py)

## Hardware Support

This recipe currently documents one CUDA GPU serving configuration. Extend it
with more hardware sections as community validation lands.

## GPU

### 1x A100 80GB

#### Environment

- OS: Linux
- Python: 3.10+
- Driver / runtime: NVIDIA CUDA environment with an A100 80 GB GPU
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Command

Start the baseline server:

```bash
vllm serve Qwen/Qwen-Image --omni --port 8091
```

To enable the step-wise runtime without batching:

```bash
vllm serve Qwen/Qwen-Image --omni \
  --port 8091 \
  --step-execution \
  --max-num-seqs 1
```

To enable experimental compatible-request batching:

```bash
vllm serve Qwen/Qwen-Image --omni \
  --port 8091 \
  --step-execution \
  --max-num-seqs 8
```

You can also use the example launcher and pass the extra flags through:

```bash
bash examples/online_serving/text_to_image/run_server.sh --step-execution --max-num-seqs 8
```

#### Verification

Run the existing client example after the server is ready:

```bash
python examples/online_serving/text_to_image/openai_chat_client.py \
  --server http://localhost:8091 \
  --prompt "A ceramic teapot on a wooden table" \
  --output /tmp/qwen_image_recipe.png
```

For a direct API smoke test:

```bash
curl http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen-Image",
    "prompt": "A ceramic teapot on a wooden table",
    "size": "1024x1024",
    "num_inference_steps": 20,
    "seed": 42
  }'
```

To replay the batching benefit with matched warmup:

```bash
python benchmarks/diffusion/diffusion_benchmark_serving.py \
  --backend vllm-omni \
  --dataset vbench \
  --task t2i \
  --model Qwen/Qwen-Image \
  --num-prompts 10 \
  --height 1024 \
  --width 1024 \
  --num-inference-steps 50 \
  --seed 42 \
  --port 8089 \
  --max-concurrency 8 \
  --warmup-requests 8 \
  --warmup-concurrency 8 \
  --warmup-num-inference-steps 3 \
  --disable-tqdm
```

Run that once against `--max-num-seqs 1`, then rerun it against `--max-num-seqs 8`
and compare the output JSON or terminal metrics.

#### Notes

- Memory usage: keep headroom for first-request compile and image decode overhead.
- Key flags: `--step-execution` enables the step-wise runtime; `--max-num-seqs` controls how many compatible requests may stay active together.
- Keep `--max-num-seqs 1` when you want the more conservative path, when traffic is mostly single-request, or when you are debugging correctness before measuring throughput.
- Current batching is still shape-sensitive: different step progress can co-batch, but different resolutions do not yet co-batch.
