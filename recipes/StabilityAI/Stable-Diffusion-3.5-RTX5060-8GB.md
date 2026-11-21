# Stable Diffusion 3.5 Medium — RTX 5060 Laptop 8 GB

## Summary

- Vendor: Stability AI
- Model: `stabilityai/stable-diffusion-3.5-medium`
- Task: text-to-image
- Mode: online serving, one request at a time
- Hardware: 1x NVIDIA GeForce RTX 5060 Laptop, 8151 MiB
- Maintainer: community

## When to use this recipe

The three text encoders, the transformer and the VAE are resident in one
process, and the T5-XXL encoder alone is 8.87 GiB in fp16, larger than the
7.53 GiB this GPU exposes to a process. The profile streams the encoders and
the transformer block by block and halves the VAE tile, so a 1024 image fits.

## Supported model contract

| Task | Input | Output | Profile |
| --- | --- | --- | --- |
| Text-to-image | one prompt, optional negative prompt | one 1024×1024 image | `sd35_medium_8gb.yaml` |

The checkpoint's own `sample_size` is 1024. 512 also serves and uses the same
memory, but 1024 is the size the profile was qualified at.

## References

- Model card: [stabilityai/stable-diffusion-3.5-medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
- Deploy profile: [`vllm_omni/deploy/sd35_medium_8gb.yaml`](../../vllm_omni/deploy/sd35_medium_8gb.yaml)
- Layerwise offload: [cpu_offload.md](../../docs/user_guide/diffusion/cpu_offload.md)
- Tracking issue: [vllm-omni#10008](https://github.com/vllm-project/vllm-omni/issues/10008)

## Hardware

- Accelerator: NVIDIA GeForce RTX 5060 Laptop, 8151 MiB, SM120 (Blackwell). The process sees 7.53 GiB.
- Devices: 1. No interconnect.
- Host memory: 30 GB. Layerwise offload stages the encoders and transformer in host RAM.
- Qualification: one machine, one run per cell, measured 2026-11-09.

## Software environment

- OS: Linux 7.0.0-42-generic, container `vllm/vllm-omni:nightly`
- Python: 3.12
- Driver / runtime: NVIDIA driver 595.99.02, CUDA 13.0
- vLLM: 0.32.0
- vLLM-Omni: 0.32.0rc2.dev352+g670aa7ec6

The weights were read from an external HDD at about 100 MB/s, so the cold
start is disk bound and is not a pipeline startup cost.

## Command

```bash
vllm serve stabilityai/stable-diffusion-3.5-medium --omni \
  --deploy-config vllm_omni/deploy/sd35_medium_8gb.yaml --port 8091
```

## Verification

```bash
curl -X POST http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "stabilityai/stable-diffusion-3.5-medium",
    "prompt": "a cute cat sitting on a windowsill, highly detailed, sharp focus",
    "guidance_scale": 4.5,
    "n": 1,
    "size": "1024x1024"
  }'
```

## Notes

Measured with seed 42, guidance 4.5, one image per prompt, three fixed prompts.
Peak is device memory from `nvidia-smi`. Steady is per image.

| Resolution | Steps | Offload | VAE tiling | Peak | Cold start | Steady | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 4 | none | no | — | — | — | OOM loading T5 |
| 512 | 4 | module | no | — | — | — | OOM, encoders staged on GPU before the hooks exist |
| 512 | 4 | layer | no | — | — | — | rejected, no offload plan |
| 512 | 4 | layer | no | 3989 MiB | 145.8 s | 4.1–4.3 s | served |
| 512 | 28 | layer | no | 3989 MiB | 244.8 s | 19.0–19.1 s | served |
| 1024 | 28 | layer | no | 7461 MiB | 246.8 s | 28.2–30.5 s | served |
| 1024 | 28 | layer | yes | 3993 MiB | 236.7 s | 27.9–28.4 s | served |

The last row is the profile. 1024 without tiling leaves about 70 MiB under the
7.53 GiB the process can use, so it has no room for anything else resident.

Tiling against no tiling, same seed: PSNR 34.78, 37.48 and 35.90 dB, all above
the 30 dB bar set before the run. That is the tiling difference only. There is
no comparison against an unmodified BF16 run, because the unmodified pipeline
does not fit on this GPU and the CPU reference was not computed.

## Supported features

| Feature | Status on this hardware |
| --- | --- |
| [Layerwise offload](../../docs/user_guide/diffusion/cpu_offload.md) | Required. `dit` and `text_encoder`, `pin_memory: true`. |
| VAE tiling and slicing | Required for 1024. The tile is halved from the checkpoint default, otherwise a 1024 image is one tile. |
| Quantization, cache backends, sleep/wake | Not measured. |
