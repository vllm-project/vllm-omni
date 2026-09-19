# Sana-Sprint 0.6B

## Summary

- Vendor: NVIDIA / MIT / Tsinghua SANA team.
- Model: `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers`.
- Runtime: native vLLM-Omni; BF16, one worker, TP=SP=CFG=PP=1.
- Task: text-to-image, offline and online.
- Validated hardware: one NVIDIA GeForce RTX 5080, 16 GB, PCIe.
- Maintainer: [@Bezdarnost](https://github.com/Bezdarnost).

## Supported model contract

| Item | Contract |
| --- | --- |
| Input | Text prompt; negative prompts and reference images are unsupported |
| Output | RGB images; `n` / `num_outputs_per_prompt` selects the image count |
| Geometry | Positive height and width divisible by 32; default 1024×1024 |
| Resolution buckets | Enabled by default; generate in the nearest trained bucket, then resize/crop to the requested dimensions |
| Sampling | Default 2 SCM steps, distilled guidance 4.5; 1, 2 and 4 steps validated |
| Text encoding | Gemma2-2B, reference instruction template, final sequence length 300 |
| Checkpoint layout | Use the `_diffusers` repository, which includes all components; the raw `.pth` repository is not supported |

`guidance_scale` conditions the distilled transformer; it does not enable a
second unconditional denoising pass. Keep `use_resolution_binning=true` for
normal use. Advanced offline callers can disable it through
`OmniDiffusionSamplingParams.extra_args`; online callers can send the same
field in the request body.

## References

- [Model card](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers)
- [SANA reference implementation](https://github.com/NVlabs/Sana)
- [Shared offline image example](../../examples/offline_inference/text_to_image/README.md)
- [Supported models](../../docs/models/supported_models.md)
- [Diffusion features](../../docs/user_guide/diffusion_features.md)

## Setup and environment

Install vLLM-Omni using the [installation guide](../../docs/getting_started/installation/README.md).
The model repository contains the tokenizer, text encoder, DC-AE, transformer,
and SCM scheduler. Validation used checkpoint revision
`aa76e7f4f4928f378716b6716a2130fba3caf5b1`.

The locally exercised environment was Linux x86-64, Python 3.12.11,
PyTorch 2.13.0+cu130, NVIDIA driver 595.91.07, Diffusers 0.40.0,
Transformers 5.14.1, and vLLM 0.29.0. The integration was based on
vLLM-Omni commit `77d8de70`. These are qualification details, not new
minimum dependency requirements; use the repository's supported installation.

## Offline generation

From the repository root:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers \
  --prompt "A red panda in a bamboo forest" \
  --height 1024 --width 1024 \
  --num-inference-steps 2 --guidance-scale 4.5 \
  --seed 42 --enforce-eager --output sana-sprint.png
```

Specify the steps and guidance explicitly: the shared example's CLI defaults
serve several model families and differ from Sprint's pipeline defaults.
The command writes a 1024×1024 PNG. Omitting `--enforce-eager` enables the
shared compilation path, also exercised locally.

## Online serving

```bash
vllm serve Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers \
  --omni --enforce-eager --port 8091
```

```bash
curl http://localhost:8091/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
    "prompt": "A red panda in a bamboo forest",
    "size": "1024x1024",
    "num_inference_steps": 2,
    "guidance_scale": 4.5,
    "seed": 42,
    "response_format": "b64_json"
  }'
```

Expect HTTP 200 with a PNG in `data[0].b64_json`. Omitting steps and guidance
uses the pipeline defaults of 2 and 4.5. For two portrait images, set
`"size": "768x1024", "n": 2`.

## Supported features

| Feature | Status on one RTX 5080 |
| --- | --- |
| Native BF16 offline and image API | Validated |
| Multiple images and rectangular output | Validated: two 768×1024 images in one HTTP request |
| Regional compilation | Validated with the shared offline example |
| [CPU offload](../../docs/user_guide/diffusion/cpu_offload.md) | Model-level and layerwise modes validated separately at 1024×1024; add `--enable-cpu-offload` or `--enable-layerwise-offload` |
| TP, SP, CFG parallel, PP, HSDP, distributed VAE | Unsupported by this integration |
| Cache-DiT, TeaCache, quantization, LoRA | Unsupported by this integration |
| Image editing, image-to-image, video | Unsupported by this integration |

CPU offload stages weights in host memory; host peak memory was not measured.
Offload is optional for the tested workload, which fits on the 16 GB GPU
without it. Other accelerator models and platforms were not qualified here.

## Verification and evidence

```bash
python -m pytest tests/diffusion/models/sana_sprint -q
python -m pytest tests/e2e/online_serving/test_sana_sprint.py \
  -m 'advanced_model and diffusion' --run-level advanced_model -v
python -m pytest tests/e2e/accuracy/test_sana_sprint.py \
  -m 'advanced_model and diffusion' --run-level advanced_model -v
```

The first command uses tiny random-weight models on CPU and needs no checkpoint
download. It checks reference transformer and SCM parity, masked text tokens,
request defaults, and input validation. The second loads the full checkpoint
and exercises the image API on one CUDA GPU. The third reproduces the
full-checkpoint latent comparison against Diffusers.

Full-checkpoint BF16 validation on the RTX 5080 compared the native pipeline
against Diffusers 0.40.0 at 1024×1024 using the same prompt and CPU generator
seed 42. Final latent tensors matched exactly at 1, 2, and 4 steps. This is a
bounded correctness check, not a general quality or performance benchmark.
The shared runner uses a device generator by default; matching only the seed
without matching the generator device does not imply identical images.
