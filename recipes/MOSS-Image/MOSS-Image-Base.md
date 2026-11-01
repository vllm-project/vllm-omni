# MOSS-Image-Base text-to-image

## Summary

- Model: [`OpenMOSS-Team/MOSS-Image-Base`](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base)
- Pipeline: `MOSSImagePipeline`, matching the checkpoint's `model_index.json`.
- Task: Text-to-image through the offline engine and `/v1/images/generations`.
- Hardware: NVIDIA H200 and NVIDIA A800-SXM4-80GB.

MOSS-Image-Base combines a Qwen3.5-9B text encoder, a 42-block dense joint image/text DiT (hidden size 3072, 24 heads, FFN size 8192, two text-refinement blocks) and a FLUX.2 VAE. Its 128-channel packed latents have spatial stride 16. DiT/encoder weights use BF16; noise, CFG, Euler updates and VAE decoding use FP32.

## Features

| Feature | Configuration |
| --- | --- |
| Tensor parallelism | `--tensor-parallel-size 2`, 4 or 8 |
| Ulysses sequence parallelism (USP) | `--ulysses-degree 2` |
| CFG parallelism | `--cfg-parallel-size 2` |
| Request batching | `--max-num-seqs 4` |
| Continuous batching | `--step-execution --max-num-seqs 4` |
| DLO | Stream DiT, text-refinement and text-encoder blocks using the config below |
| SLG | Two-pass unconditional skipping or three-pass positive SLG via `extra_params` |

RoPE uses each request's valid text length, so batch padding, requests joining or leaving a continuous batch, and Ulysses alignment padding preserve its positional coordinates.

DLO works on one GPU and supports AllGather across DP/SP groups. SLG layer IDs are zero-based main-DiT indices; `[22]` skips the 23rd block. The active window is `slg_start <= step_index / num_steps < slg_stop`. Optional `slg_rescaling_scale` applies per-image rescaling in denoised space. Ring attention and approximate denoising caches are not supported.

## Kernel fusion

The eager CUDA path fuses RMSNorm/AdaLN, Q/K RMSNorm with RoPE, SiTU and gated residual additions while preserving BF16 rounding boundaries. Full-forward requests reuse immutable text conditioning and RoPE. Matched reductions target PyTorch 2.13; unsupported cases use Torch reductions, unfused kernels or eager operators.

## NVIDIA H200

### Environment

Linux, Python 3.12, PyTorch 2.13.0+cu130, vLLM 0.31.0, Transformers 5.14.1 and Diffusers 0.40.0. See the [installation guide](../../docs/getting_started/installation/README.md).

### Deploy

```bash
vllm serve OpenMOSS-Team/MOSS-Image-Base --omni --enforce-eager --port 8091
```

For DLO, add:

```bash
  --diffusion-offload-config '{"mode":"layer","components":["dit","text_encoder"],"layer_options":{"dit":{"weight_transfer":"allgather"},"text_encoder":{"weight_transfer":"allgather"}}}'
```

### Example request

```bash
curl -sS http://localhost:8091/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"OpenMOSS-Team/MOSS-Image-Base","prompt":"1girl, hands up","size":"1024x1024","num_inference_steps":50,"guidance_scale":3.0,"flow_shift":5.0,"seed":0,"extra_params":{"uncond_skip_layers":[22],"slg_start":0.01,"slg_stop":1.0}}'
```

For baseline generation, omit `extra_params`. For three-pass SLG, use `guidance_scale=5.0` and `extra_params={"slg_layers":[22],"slg_scale":3.0,"slg_start":0.01,"slg_stop":1.0}`.

### Performance

Previous image runs: 1× H200, original nutrition-pagoda prompt, 1024×1024, 50 steps, shift 5, CUDA seed 0, batch 1, eager. Single-run image generation latency, excluding model initialization and image saving.

| Configuration | Resident (s) | DLO (s) |
| --- | ---: | ---: |
| CFG 3, no SLG | 11.735 | 18.711 |
| CFG 3, uncond skip `[22]`, window .01–1 | 11.657 | 18.287 |
| CFG 5, cond skip `[22]`, SLG scale 3, window .01–1 | 18.516 | 26.805 |

Peak PyTorch allocation: 30.31 GiB resident, 8.96 GiB DLO. Peak allocator reservation: 34.44 GiB resident, 13.12 GiB DLO, excluding CUDA contexts and other allocations outside PyTorch.

### Offline inference

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model OpenMOSS-Team/MOSS-Image-Base --prompt "1girl, hands up" \
  --height 1024 --width 1024 --num-inference-steps 50 --guidance-scale 3.0 \
  --seed 0 --timesteps-shift 5.0 --enforce-eager \
  --extra-body '{"uncond_skip_layers":[22],"slg_start":0.01,"slg_stop":1.0}' \
  --output moss-image-base.png
```

## NVIDIA A800-SXM4-80GB

### Environment

Linux, Python 3.12.13, CUDA 13.0, driver 595.45.04, PyTorch 2.13.0+cu130,
vLLM 0.32.0, vLLM-Omni `d6bd0fd16`, Transformers 5.19.0 and Diffusers 0.40.0.

### Deploy

Download the tested checkpoint and start the server on one A800:

```bash
modelscope download --model openmoss/MOSS-Image-Base \
  --revision 1f920e15816c63f25466e4e37cf50647976aa5c2 --local-dir ./MOSS-Image-Base
export MOSS_MODEL_DIR="$PWD/MOSS-Image-Base"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN

vllm serve "$MOSS_MODEL_DIR" --omni \
  --served-model-name moss-image-base --model-class-name MOSSImagePipeline \
  --host 127.0.0.1 --port 8091 --dtype bfloat16 \
  --enforce-eager --max-num-seqs 1 \
  --diffusion-attention-backend FLASH_ATTN
```

Send a test request:

```bash
curl --fail-with-body -sS http://127.0.0.1:8091/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"moss-image-base","prompt":"A ceramic teapot on a wooden table","size":"1024x1024","n":1,"num_inference_steps":50,"guidance_scale":3.0,"negative_prompt":"","flow_shift":5.0,"seed":0,"response_format":"b64_json"}' \
  -o response.json
```

For offline generation, use the [offline example](#offline-inference) with `--model "$MOSS_MODEL_DIR"`.

### Performance

1x A800 80GB, BF16 DiT/text encoder, FP32 VAE, eager, FlashAttention.
1024x1024, 50 steps, CFG 3, no SLG, seed 0, shift 5, concurrency 1,
empty negative prompt; prompt: "A ceramic teapot on a wooden table".
One warmup followed by five measured requests on 2026-10-29.

| Mean latency (ms) | Peak device memory (MiB) | Successful requests |
| ---: | ---: | ---: |
| 23643.5 | 33352 | 5/5 |

Latency covers the HTTP response and JSON parsing, excluding startup, warmup and image saving.
Device memory was sampled through NVML every 100 ms during warmup and measurement, excluding startup.
Optional text-encoder kernels (`causal_conv1d`, `flash-linear-attention`) were not installed.

### Latency benchmark

The local prompt file is repeated for all five requests:

```bash
printf '%s\n' 'A ceramic teapot on a wooden table' > moss-a800-prompts.txt
python benchmarks/diffusion/diffusion_benchmark_serving.py \
  --endpoint /v1/images/generations --dataset vbench \
  --dataset-path moss-a800-prompts.txt --task t2i --model moss-image-base \
  --height 1024 --width 1024 --num-inference-steps 50 --seed 0 \
  --extra-body '{"guidance_scale":3.0,"negative_prompt":"","flow_shift":5.0,"extra_params":{}}' \
  --base-url http://127.0.0.1:8091 --num-prompts 5 --max-concurrency 1 \
  --warmup-requests 1 --warmup-concurrency 1 --warmup-num-inference-steps 50 \
  --output-file moss-a800-results.json --save-dir moss-a800-images --disable-tqdm
```
