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

- OS: Linux, Python 3.12
- vLLM: 0.27.0
- Peak GPU memory: 34.4 GB at 1024x1024, 36.5 GB at 1536x2720

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
| 1024x1024 | 219.9 ms | 5.50 s |
| 1536x1536 | 472.6 ms | 11.82 s |

At 1536x2720 with 50 steps, end-to-end is 46.5 s.

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
  cleared: 711 ms above steady with the path on, 664 ms with it off.
- Each request captures its own graphs, and a think request captures twice because the sequence
  grows past the 512 bucket: 150 ms for the first capture in a process and about 94 ms after
  that, so roughly 188 ms of every request rather than a one-off startup cost.
