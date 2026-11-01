# Text-only generation with Qwen-Image-Edit-2511 (experimental)

`QwenImageEditPlusPipeline` can also run without a reference image. It uses
text-only conditioning and random output latents with the same checkpoint;
it does not insert a blank image or load a separate text-to-image model.
Image-backed requests keep the existing editing path.

The checkpoint is published for image editing. Text-only image quality is
not yet qualified; successful request handling alone is not a quality guarantee.

Start the server:

```bash
vllm serve Qwen/Qwen-Image-Edit-2511 --omni --port 8092
```

Then send a prompt with no image to the generations endpoint:

```bash
curl --fail-with-body -sS http://localhost:8092/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen-Image-Edit-2511",
    "prompt": "A watercolor painting of a green turtle on a sunny beach",
    "negative_prompt": " ",
    "size": "1024x1024",
    "num_inference_steps": 50,
    "true_cfg_scale": 4.0,
    "seed": 42,
    "response_format": "b64_json"
  }'
```

The response contains generated image bytes in `data[].b64_json`. For offline
requests, omit `multi_modal_data.image`; an explicitly empty image list is an
error. Omitted dimensions default to 1024 by 1024, and requested dimensions are
aligned to the pipeline's VAE packing factor.
