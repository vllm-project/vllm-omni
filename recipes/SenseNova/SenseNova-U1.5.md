# SenseNova-U1.5

> Unified image generation and understanding, with an official 8-step distilled LoRA

## Summary

- Vendor: SenseNova
- Model: `sensenova/SenseNova-U1.5-8B-MoT`
- LoRA: `sensenova/SenseNova-U1.5-8B-MoT-LoRAs` (`SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors`)
- Task: text2img, img2img, img2text (visual understanding), text2text (chat)
- Mode: Offline inference, Online serving (OpenAI-compatible API)
- Maintainer: Community

## When to use this recipe

U1.5 runs on the same `SenseNovaU1Pipeline` as U1 — the checkpoint keeps
`model_type: neo_chat`, so it resolves without `--model-class-name`. Relative to U1 the
config flips two fields, `use_pixel_head` to `true` (the flow-matching head becomes a
`ConvDecoder`) and `noise_scale_max_value` from 8.0 to 16.0; both are already read by
`SenseNovaU1Config`. The checkpoint is 13 shards / 50.2 GB on disk, 30.3 GB of that fp32, and
loads to roughly 34 GB in bf16.

Use this recipe for U1.5 specifically, including its distilled few-step LoRA. For U1 see
[SenseNova-U1](SenseNova-U1.md).

## Hardware Support

### GPU

#### 1x A800 80GB

- 1x NVIDIA A800 80GB PCIe with an Intel Xeon Gold 6336Y, driver 595.84, Linux
- Python 3.12.13, vLLM 0.28.0, torch 2.13.0+cu130, CUDA 13.0, diffusers 0.40.0,
  transformers 5.14.1, flashinfer 0.6.16.post3, BF16, TP=1
- Peak GPU memory: 34.3 GB at 1024x1024, 36.4 GB at 1536x2720

##### Text-to-Image

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
    --model sensenova/SenseNova-U1.5-8B-MoT \
    --prompt "Close portrait of an elderly woman by a farmhouse window, warm natural light." \
    --width 1024 --height 1024 \
    --seed 42 --num-inference-steps 50 --cfg-scale 4.0 \
    --extra-body '{"think": false, "cfg_norm": "none", "timestep_shift": 3.0, "t_eps": 0.02}' \
    --output sensenova_u15_t2i.png
```

Think mode (`"think": true`) is recommended for higher image quality.

##### Image-to-Image Editing

```bash
python examples/offline_inference/image_to_image/image_edit.py \
    --model sensenova/SenseNova-U1.5-8B-MoT \
    --prompt "Turn this into an oil painting" \
    --image input.png --resolution 1024 \
    --seed 42 --num-inference-steps 25 --cfg-scale 4.0 \
    --extra-args '{"think": true, "img_cfg_scale": 1.0, "cfg_norm": "none", "timestep_shift": 3.0}' \
    --output sensenova_u15_edit.png
```

##### 8-step distilled LoRA

The distilled LoRA is fused into the generation tower at load time:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
    --model sensenova/SenseNova-U1.5-8B-MoT \
    --lora-path SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors --lora-backend distill \
    --prompt "Close portrait of an elderly woman by a farmhouse window, warm natural light." \
    --width 1024 --height 1024 \
    --seed 42 --num-inference-steps 8 --cfg-scale 1.0 \
    --extra-body '{"think": false, "cfg_norm": "none", "timestep_shift": 3.0, "t_eps": 0.02}' \
    --output sensenova_u15_lora8.png
```

**Use `--cfg-scale 1.0` with this LoRA.** It is distilled with DMD and runs without
classifier-free guidance; the default `4.0` applies guidance twice and produces a
blown-out, posterised image.

##### Online serving

```bash
vllm serve sensenova/SenseNova-U1.5-8B-MoT --omni --port 8091

python examples/online_serving/sensenova_u1/openai_chat_client.py \
    -s http://127.0.0.1:8091 -m img2text -i input.png -p "Describe this image."
```

`-s` takes the base URL; the client appends `/v1` itself.

#### Measured latency (1x A800 80GB, 25 steps, median of 3 after a warmup)

| Resolution | Step latency | Total |
| --- | --- | --- |
| 1024x1024 | 209.5 ms | 5.24 s |
| 1536x1536 | 458.5 ms | 11.46 s |

At 1536x2720 with 50 steps, end-to-end is 43.0 s.

#### Measured latency (1x NVIDIA H200 139GB, single run, no warmup)

Reported by @hsliuustc0106 against PR head `29c090d`, on one reserved CUDA device.

- Python 3.12.13, vLLM 0.28.0, torch 2.13.0+cu130, CUDA 13.0, diffusers 0.40.0,
  transformers 5.14.1, flashinfer 0.6.16.post3, BF16, TP=1
- `sensenova/SenseNova-U1.5-8B-MoT` with `SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors`,
  seed 42, 1024x1024

| Case | Stage latency | Peak GPU memory |
| --- | --- | --- |
| 50 steps, cfg 4.0, think off | 9017 ms | 34384 MiB |
| 8 steps, cfg 1.0, no LoRA control | 624.00 ms, 77.85 ms/step | 34376 MiB |
| 8 steps, cfg 1.0, distilled LoRA | 623.85 ms, 77.98 ms/step | 34364 MiB |
| 50 steps, cfg 4.0, think on, `VLLM_OMNI_SENSENOVA_PAGED_DECODE=1` | 81622 ms | 34594 MiB |

Each row runs the Text-to-Image command above with `--num-inference-steps` and
`--cfg-scale` as listed and `think` set through `--extra-body`; the LoRA row adds
`--lora-path` and `--lora-backend distill`.

Single runs without a preceding warmup, so the think-on row carries the first-request
compile rather than a steady state. The distilled LoRA fused into 168 parameters, and its
image differs from the same-seed no-LoRA control in 1,048,574 of 1,048,576 pixels, MAE 58.62.

```bash
pytest -m "cpu and not cuda" tests/diffusion/models/sensenova_u1/ tests/diffusion/lora/test_loader.py -q
pytest -m "cpu and not cuda" tests/config/test_environment_variables.py tests/diffusion/test_diffusion_worker.py -q
pytest -m cuda tests/diffusion/models/sensenova_u1/test_sensenova_u1_attention_gqa.py \
    tests/diffusion/models/sensenova_u1/test_sensenova_u1_paged_decode.py -q --tb=short
```

71, 13 and 9 passed. Serving reached `/health`; `img2text` returned a 1921-character
description and `text2text` the expected answer.

#### Verification

```bash
pytest -q tests/diffusion/models/sensenova_u1/
```

## Notes

- Both `use_pixel_head` and `noise_scale_max_value` come from `config.json`; no flag is needed.
- The checkpoint carries no `configuration_neo_chat.py`, so the loader logs a few
  "does not appear to have a file named configuration_neo_chat.py" errors during startup and
  then proceeds on the in-tree config. Generation is unaffected.
- The LoRA targets the generation tower only (`*_mot_gen`); understanding-tower weights are
  untouched.
- Autoregressive decode (think, text-to-text and image-to-text) runs on a paged K/V cache under
  a captured CUDA graph. It falls back to the ordinary cache when the device or the bundled
  `flash_attn_varlen_func` cannot support it; set `VLLM_OMNI_SENSENOVA_PAGED_DECODE=0` to force
  that fallback.
- The first request after startup costs about 0.7 s more than the steady state whether the paged
  path is on or off. Measured on one A800 with the inductor, triton and vLLM compile caches all
  cleared, median of three runs: 718 ms above steady with the path on, 679 ms with it off.
- Each request captures its own graphs, and a think request captures twice because the sequence
  grows past the 512 bucket, so the capture cost is paid per request rather than once at
  startup.
