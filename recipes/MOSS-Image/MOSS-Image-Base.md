# MOSS-Image-Base text-to-image

## Summary

- Model: [`OpenMOSS-Team/MOSS-Image-Base`](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base)
- Pipeline: `MOSSImagePipeline`, matching the checkpoint's `model_index.json`.
- Task: Text-to-image through the offline engine and `/v1/images/generations`.
- Hardware: NVIDIA H200.

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

## Environment

Linux, Python 3.12, PyTorch 2.13.0+cu130, vLLM 0.31.0, Transformers 5.14.1 and Diffusers 0.40.0. See the [installation guide](../../docs/getting_started/installation/README.md).

## Deploy

```bash
vllm serve OpenMOSS-Team/MOSS-Image-Base --omni --enforce-eager --port 8091
```

For DLO, add:

```bash
  --diffusion-offload-config '{"mode":"layer","components":["dit","text_encoder"],"layer_options":{"dit":{"weight_transfer":"allgather"},"text_encoder":{"weight_transfer":"allgather"}}}'
```

## Example request

```bash
curl -sS http://localhost:8091/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"OpenMOSS-Team/MOSS-Image-Base","prompt":"1girl, hands up","size":"1024x1024","num_inference_steps":50,"guidance_scale":3.0,"flow_shift":5.0,"seed":0,"extra_params":{"uncond_skip_layers":[22],"slg_start":0.01,"slg_stop":1.0}}'
```

For baseline generation, omit `extra_params`. For three-pass SLG, use `guidance_scale=5.0` and `extra_params={"slg_layers":[22],"slg_scale":3.0,"slg_start":0.01,"slg_stop":1.0}`.

## Performance

Previous image runs: 1× H200, original nutrition-pagoda prompt, 1024×1024, 50 steps, shift 5, CUDA seed 0, batch 1, eager. Single-run image generation latency, excluding model initialization and image saving.

| Configuration | Resident (s) | DLO (s) |
| --- | ---: | ---: |
| CFG 3, no SLG | 11.735 | 18.711 |
| CFG 3, uncond skip `[22]`, window .01–1 | 11.657 | 18.287 |
| CFG 5, cond skip `[22]`, SLG scale 3, window .01–1 | 18.516 | 26.805 |

Peak PyTorch allocation: 30.31 GiB resident, 8.96 GiB DLO. Peak allocator reservation: 34.44 GiB resident, 13.12 GiB DLO, excluding CUDA contexts and other allocations outside PyTorch.

## Offline inference

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model OpenMOSS-Team/MOSS-Image-Base --prompt "1girl, hands up" \
  --height 1024 --width 1024 --num-inference-steps 50 --guidance-scale 3.0 \
  --seed 0 --timesteps-shift 5.0 --enforce-eager \
  --extra-body '{"uncond_skip_layers":[22],"slg_start":0.01,"slg_stop":1.0}' \
  --output moss-image-base.png
```
