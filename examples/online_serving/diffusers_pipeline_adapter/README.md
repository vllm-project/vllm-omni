# Diffusers Backend Adapter Example

This example demonstrates how to serve any 🤗 Diffusers pipeline through vLLM-Omni
using the `diffusers` load format.

## Supported Models

Any model loadable via `DiffusionPipeline.from_pretrained()` should be supported, including text-to-image, image-to-image, text-to-video, image-to-video, and text-to-audio.

## Limitations

The diffusers backend is a black-box adapter. The following features are NOT yet supported.
It is not guaranteed whether they will be supported in the future.

- CFG parallel execution
- Sequence parallel execution
- TeaCache / Cache-DiT acceleration
- Step-wise execution (continuous batching)

For these features, it is recommended to use natively supported pipelines instead.

## Usage

### Option 1: CLI arguments

```bash
vllm serve "stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --omni \
    --diffusion-load-format diffusers \
    --diffusers-load-kwargs '{"use_safetensors": true}' \
    --diffusers-call-kwargs '{"num_inference_steps": 30, "guidance_scale": 7.5}'
```

`--diffusers-load-kwargs` and `--diffusers-call-kwargs` are only valid together with `--diffusion-load-format diffusers`.

### Option 2: Stage config YAML

```bash
vllm serve stable-diffusion-v1-5/stable-diffusion-v1-5 --stage-configs-path examples/online_serving/diffusers_pipeline_adapter/stage_config.yaml --omni
```

The particular fields of interest are `model`, `diffusion_load_format`, `diffusers_load_kwargs`, and `diffusers_call_kwargs` under `engine_args`. They are the same as the CLI arguments.

## Send a Request

```bash
curl http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "prompt": "a photo of an astronaut riding a horse on mars",
    "n": 1,
    "size": "512x512"
  }'
```

Or refer to other documentation pages on how to request a particular input/output modality, such as `examples/online_serving/text_to_image/openai_chat_client.py`.

## Configuration Reference

For the diffusers adapter, set options under **`engine_args`**:

### `diffusion_load_format: "diffusers"`

This field selects the Hugging Face diffusers adapter path (see `DiffusersPipelineLoader`).

### `diffusers_load_kwargs`

Passed to `DiffusionPipeline.from_pretrained()`.

This is suitable for model-specific configurations not available through the vLLM-Omni interface (such as `Omni.__init__()`, `vllm serve` CLI arguments, and stage config YAML fields outside `diffusers_load_kwargs`).

When a parameter is available in the vLLM-Omni interface, it will be adapted here.
But if that parameter is simultaneously set in both the vLLM-Omni interface and `diffusers_load_kwargs`, the **latter** will take precedence.

### `diffusers_call_kwargs`

Passed to `pipeline.__call__()`.

This is suitable for sampling parameters not available through the vLLM-Omni interface (such as `Omni.generate()` and online serving payloads).

When a parameter is available in the vLLM-Omni interface, it will be adapted here.
But if that parameter is simultaneously set in both the vLLM-Omni interface and `diffusers_call_kwargs`, the **former** will take precedence (because it is set at request time).
