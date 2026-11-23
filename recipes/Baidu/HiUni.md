# HiUni

> One checkpoint for image understanding, generation and editing

## Summary

- Vendor: Baidu (ERNIE Research)
- Model: `ernie-research/HiUni`
- Task: text2img, img2img (instruction editing), img2text (understanding), text2text (chat)
- Mode: Offline inference, Online serving (OpenAI-compatible API)
- Hardware: 1x NVIDIA GPU, BF16
- Maintainer: Community

## When to use this recipe

Use this recipe to run HiUni with vLLM-Omni. HiUni is a small unified
multimodal model: **one checkpoint** for image understanding, generation and
editing. A lightweight pixel-space diffusion head reads the backbone's hidden
states of the noisy image tokens and predicts clean pixels directly, so there
is no VAE and no separate text encoder.

`HiUniPipeline` routes each request by its inputs:

| Request | Task |
| ------- | ---- |
| `modalities` contains `"text"` | Text generation; image understanding when an image is attached, chat otherwise |
| one input image | Instruction-based image editing |
| text only | Text-to-image |

The checkpoint's `config.json` (`model_type: pixel_dit_qwen3_vl`) is detected
automatically, so no `--model-class-name` or deploy YAML is needed.

## References

- Model repository: [ernie-research/HiUni](https://github.com/ernie-research/HiUni)
- HuggingFace model page: [ernie-research/HiUni](https://huggingface.co/ernie-research/HiUni)
- Offline text-to-image:
  [`examples/offline_inference/text_to_image/text_to_image.py`](../../examples/offline_inference/text_to_image/text_to_image.py)
- Offline image editing:
  [`examples/offline_inference/image_to_image/image_edit.py`](../../examples/offline_inference/image_to_image/image_edit.py)
- Request knobs: [`vllm_omni/model_extras/hiuni.py`](../../vllm_omni/model_extras/hiuni.py)

## Supported model contract

| Item | Value |
| ---- | ----- |
| Text-to-image size | Any height/width; rounded up to a multiple of 32. Default 1024x1024 |
| Editing inputs | Exactly one reference image |
| Editing output size | Same as the input image; an explicit `height`/`width` (or `size`) is used as given |
| Understanding inputs | Up to one image per request |
| Sampler | Flow-matching Euler, 25 steps, `shift=4.0`, CFG on the predicted clean image |
| Defaults | `cfg_scale=4.0`, `img_cfg_scale=1.0` (image CFG off), `seed=42` |

## Command

### Text-to-image

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
    --model ernie-research/HiUni \
    --prompt "一只橘猫坐在窗台上，阳光明媚" \
    --height 768 --width 1024 \
    --num-inference-steps 25 --seed 42 --cfg-scale 4.0 \
    --extra-body '{"shift": 4.0}' \
    --output hiuni_t2i.png
```

### Image editing

```bash
python examples/offline_inference/image_to_image/image_edit.py \
    --model ernie-research/HiUni \
    --image input.jpg \
    --prompt "把背景换成雪山" \
    --num-inference-steps 25 --seed 42 --cfg-scale 4.0 \
    --extra-args '{"img_cfg_scale": 1.0}' \
    --output hiuni_edit.png
```

`--cfg-scale` sets text guidance. `img_cfg_scale` sets image guidance; values
other than `1.0` add a third, text-free branch per step.

### Online serving

```bash
vllm serve ernie-research/HiUni --omni --port 8091
```

Text chat and image understanding use `modalities: ["text"]`:

```bash
curl -s http://localhost:8091/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
    {"type": "text", "text": "Describe this image in one sentence."}]}],
  "modalities": ["text"],
  "think_mode": "empty",
  "max_new_tokens": 256
}'
```

Text-to-image uses `modalities: ["image"]`; the image comes back as a base64
data URL in `choices[0].message.content`:

```bash
curl -s http://localhost:8091/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "A red lighthouse on a cliff at sunset"}],
  "modalities": ["image"],
  "height": 1024, "width": 1024, "num_inference_steps": 25, "seed": 42,
  "extra_body": {"cfg_scale": 4.0}
}'
```

The OpenAI-style image endpoints also work. `extra_params` carries the
request knobs listed in the notes below:

```bash
curl -s http://localhost:8091/v1/images/generations -H "Content-Type: application/json" -d '{
  "prompt": "A red lighthouse on a cliff at sunset",
  "size": "1024x1024", "num_inference_steps": 25, "seed": 42,
  "true_cfg_scale": 4.0, "extra_params": {"shift": 4.0}
}'

curl -s http://localhost:8091/v1/images/edits \
  -F image=@input.jpg -F prompt="把背景换成雪山" \
  -F num_inference_steps=25 -F seed=42
```

Both return the image as `data[0].b64_json`.

## Verification

The text-to-image command writes `hiuni_t2i.png` at 1024x768. The editing
command writes `hiuni_edit.png` at the input image's size. Both chat requests
return a non-empty `choices[0].message.content`.

## Notes

- Request knobs (`--extra-body` / `--extra-args` / request body), filtered by
  `HIUNI_EXTRA_BODY_PARAMS`:

  | Key | Applies to | Default | Meaning |
  | --- | ---------- | ------- | ------- |
  | `cfg_scale` | T2I, edit | 4.0 | Text guidance |
  | `img_cfg_scale` | edit | 1.0 | Image guidance; 1.0 disables it |
  | `shift` | T2I, edit | 4.0 | Timestep shift of the sigma schedule |
  | `resize_to_edit_image` | edit | `true` unless a size is given | Resize the output back to the input size |
  | `min_pixels` / `max_pixels` | edit, understanding | 262144 / 262144 (edit), 262144 / 4194304 (understanding) | Image processor pixel range |
  | `think_mode` | text | `"think"` | `"think"` opens a reasoning block, `"empty"` prefills an empty one, `"none"` adds nothing |
  | `max_new_tokens` | text | 1024 | Generated token budget |
  | `temperature` / `top_k` / `top_p` | text | greedy | Sampling; `temperature <= 0` is greedy |
  | `return_raw` | text | `false` | Return the decoded text without stripping the reasoning block |

- Text generation keeps a per-request KV cache inside the pipeline; it is
  intended for single-request understanding and chat, not high-throughput
  LLM serving.
- The dense attention mask makes FlashAttention fall back to SDPA for these
  layers. The warning in the log is expected.

## Supported features

| Feature | Status |
| ------- | ------ |
| Tensor parallelism | Not validated; tested with TP=1 |
| CFG parallel / sequence parallel | Not supported |
| Cache acceleration (TeaCache / Cache-DiT) | Not supported |
| Quantization | Not validated |
