# Ideogram4 Offline Inference

Ideogram4 is a single-stream text-to-image diffusion transformer with a Qwen3-VL text encoder and asymmetric CFG. This vLLM-Omni implementation is adapted from the diffusers Ideogram4 pipeline and the upstream Ideogram AI implementation. vLLM-Omni supports the gated `ideogram-ai/ideogram-4-nf4` checkpoint out of the box; if you have the fp8 checkpoint, replace the model id with `ideogram-ai/ideogram-4-fp8`.

## Quick Start

```python
from vllm_omni.entrypoints.omni import Omni

if __name__ == "__main__":
    omni = Omni(model="ideogram-ai/ideogram-4-nf4")
    outputs = omni.generate(
        "A detailed character design sheet for a young explorer-mechanic-inventor in a frozen world.",
        sampling_params={
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 50,
            "guidance_scale": 7.0,
            "seed": 42,
        },
    )
    outputs[0].request_output.images[0].save("ideogram4.png")
```

## Recommended Prompting

Ideogram4 is trained on structured JSON captions. Plain text prompts work, but JSON prompts usually produce more controllable results.

```text
{"subject":"a young explorer-mechanic-inventor","style":"semi-realistic painterly illustration","scene":"frozen mountain village","palette":["#d8d3c4","#6a7f5c","#b48a5b"]}
```

## Generation Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `height` | int | 1024 | Image height in pixels |
| `width` | int | 1024 | Image width in pixels |
| `num_inference_steps` | int | 50 | Diffusion denoising steps |
| `guidance_scale` | float | 7.0 | Classifier-free guidance scale |
| `seed` | int | None | Optional random seed |

## Notes

- Accept the Hugging Face license gate before downloading `ideogram-ai/ideogram-4-nf4` or `ideogram-ai/ideogram-4-fp8`.
- The model is best used with structured JSON captions, but plain text prompts are supported.
- The default number of diffusion steps in this implementation is 50.
