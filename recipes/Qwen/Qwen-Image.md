# Qwen-Image

> Text-to-image serving with optional step-wise continuous batching

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
  --endpoint /v1/chat/completions \
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

### 2x B200 ModelOpt mixed FP8/NVFP4

#### Environment

- OS: Linux
- Python: 3.12
- Driver / runtime: NVIDIA CUDA environment with two B200 GPUs
- vLLM version: Match the vLLM-Omni checkout used for deployment
- vLLM-Omni version or commit: Use the commit that contains ModelOpt mixed
  FP8/NVFP4 checkpoint loading for diffusion models

#### Command

This configuration uses a pre-quantized ModelOpt mixed checkpoint for
Qwen-Image-2512. The validated checkpoint keeps boundary, modulation,
normalization, and output layers in BF16, uses FP8 for sensitive attention
projections, and uses NVFP4 for heavier MLP / attention-output linear layers.

Validated checkpoint:

```text
feizhai123/qwen-image-2512-modelopt-mixed-fp8-sensitive-nvfp4-heavy
```

Start the server with the CUTLASS quantized linear backend:

```bash
CUDA_VISIBLE_DEVICES=2,3 \
vllm serve feizhai123/qwen-image-2512-modelopt-mixed-fp8-sensitive-nvfp4-heavy \
  --omni \
  --host 0.0.0.0 \
  --port 8091 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --linear-backend cutlass \
  --force-cutlass-fp8
```

#### Verification

Run a direct image generation request after the server is ready:

```bash
curl http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "feizhai123/qwen-image-2512-modelopt-mixed-fp8-sensitive-nvfp4-heavy",
    "prompt": "A cinematic Warhammer 40,000 Ultramarine warrior in blue power armor, armor stained with alien ichor, swinging a chainsword at a Tyranid creature, gothic battlefield, dramatic lighting, high detail, no text, no watermark",
    "negative_prompt": "blurry, low quality, text, watermark, logo, malformed armor, bad hands",
    "size": "1024x1024",
    "num_inference_steps": 20,
    "true_cfg_scale": 4.0,
    "guidance_scale": 4.0,
    "seed": 48
  }'
```

Check that:

- The server responds on `http://localhost:8091/health`.
- The response contains one generated image.
- Logs show CUTLASS selection for ModelOpt FP8 / mixed-precision linear
  layers instead of falling back to the default automatic backend choice.

#### Benchmark

The numbers below are loaded-once request-level measurements, not an HTTP
concurrency benchmark. Each case loads the model once, runs warmup requests,
then measures 100 sequential requests.

```text
Hardware: 2x B200
Tensor parallel size: 2
Requests: 100
Concurrency: 1
Resolution: 1024x1024
Denoising steps: 20
```

| Config | Mean | P90 | P99 | Peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 2.696s | 2.700s | 2.707s | 99,000 MiB |
| ModelOpt FP8 CUTLASS | 2.663s | 2.671s | 2.676s | 85,760 MiB |
| Mixed FP8/NVFP4 CUTLASS | 2.584s | 2.590s | 2.594s | 84,412 MiB |

The same 2-GPU setup was also used to sweep NVFP4 linear backends. Direct
CUTLASS was the fastest validated backend for this checkpoint.

| Backend | Mean | P90 | P99 | Peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| auto | 2.927s | 2.948s | 3.535s | 85,340 MiB |
| flashinfer-cudnn | 3.523s | 3.528s | 3.766s | 84,500 MiB |
| flashinfer-trtllm | 2.711s | 2.727s | 3.010s | 84,996 MiB |
| flashinfer-cutlass | 2.935s | 2.950s | 3.011s | 85,340 MiB |
| cutlass | 2.578s | 2.584s | 2.592s | 84,412 MiB |

#### Notes

- Use `--linear-backend cutlass` to select the validated quantized linear
  backend from the CLI.
- `--force-cutlass-fp8` keeps ModelOpt FP8 layers on the CUTLASS FP8 path.
- The benchmark table intentionally reports Qwen-Image-2512 as
  concurrency-1 request-level data; do not compare it directly with an online
  HTTP concurrency benchmark.
