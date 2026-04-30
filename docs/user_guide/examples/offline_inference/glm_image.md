# GLM-Image Offline Inference

GLM-Image is a 2-stage image generation model (AR + Diffusion) supported by vLLM-Omni's
declarative config system. The pipeline topology and stage structure are declared in
`vllm_omni/model_executor/models/glm_image/pipeline.py`; deployment knobs live in
`vllm_omni/deploy/glm_image.yaml`.

## Architecture

```
Stage 0 (AR Model)                Stage 1 (Diffusion)
┌───────────────────┐            ┌─────────────────────┐
│ vLLM-optimized    │  prior     │  GlmImagePipeline   │
│ GlmImageFor       │──tokens──►│  ┌───────────────┐  │
│ Conditional       │            │  │ DiT Denoiser  │  │
│ Generation        │            │  └───────┬───────┘  │
│ (9B AR model)     │            │          ▼          │
└───────────────────┘            │  ┌───────────────┐  │
        ▲                        │  │  VAE Decode   │──┼──► Image
        │                        │  └───────────────┘  │
   Text / Image                  └─────────────────────┘
     Input
```

## Text-to-Image

```python
from vllm_omni.entrypoints.omni import Omni

if __name__ == "__main__":
    omni = Omni(model="zai-org/GLM-Image")
    outputs = omni.generate(
        "A photorealistic mountain landscape at sunset",
        sampling_params={
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 50,
            "guidance_scale": 1.5,
            "seed": 42,
        },
    )
    outputs[0].request_output.images[0].save("output.png")
```

## Image-to-Image (Image Editing)

```python
from vllm_omni.entrypoints.omni import Omni

if __name__ == "__main__":
    omni = Omni(model="zai-org/GLM-Image")
    outputs = omni.generate(
        {
            "prompt": "Convert this image to watercolor style",
            "multi_modal_data": {
                "image": "input.png",
            },
        },
        sampling_params={
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 50,
            "guidance_scale": 1.5,
            "seed": 42,
        },
    )
    outputs[0].request_output.images[0].save("output.png")
```

## Generation Parameters

| Parameter             | Type  | Default | Description                         |
| --------------------- | ----- | ------- | ----------------------------------- |
| `height`              | int   | 1024    | Image height in pixels              |
| `width`               | int   | 1024    | Image width in pixels               |
| `num_inference_steps` | int   | 50      | Number of diffusion denoising steps |
| `guidance_scale`      | float | 1.5     | Classifier-free guidance scale      |
| `seed`                | int   | None    | Optional random seed                |
| `negative_prompt`     | str   | None    | Negative prompt                     |

## VRAM Requirements

| Stage             | VRAM                   |
| :---------------- | :--------------------- |
| Stage-0 (AR)      | **~18 GiB + KV Cache** |
| Stage-1 (DiT+VAE) | **~20 GiB**            |
| Total             | **~38 GiB + KV Cache** |
