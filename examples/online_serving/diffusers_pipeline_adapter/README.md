# Diffusers Backend Adapter

vLLM-Omni supports running diffusion models with the diffusers backend, directly serving any 🤗 Diffusers pipeline online without implementing them natively.

## Limitations

The diffusers backend is a black-box adapter. Its primary focus is to serve diffusion models online.
Currently, the following features are NOT yet supported.
It is not guaranteed whether they will be supported in the future.

- CFG parallel execution
- Sequence parallel execution
- TeaCache / Cache-DiT acceleration
- Step-wise execution (continuous batching)

For these features, it is recommended to use natively supported pipelines instead.

## Model Support

Any model loadable via `DiffusionPipeline.from_pretrained()` should run, including text-to-image, image-to-image, text-to-video, image-to-video, and text-to-audio.

However, as we strive to ensure output similarity between vLLM-Omni's diffuser backend and plain diffusers library, the following models are particularly verified:

- Qwen/Qwen-Image
- Tongyi-MAI/Z-Image-Turbo
- Wan2.2-I2V-A14B-Diffusers

If you find that a model not listed above also produces different outputs from running diffusers model directly.
Please consider file an issue or submit a PR to fix.

## Usage

```bash
vllm serve "stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --omni \
    --diffusion-load-format diffusers
```

Users turn on the diffusers backend primarily through `--diffusion-load-format diffusers` argument.
There are two more optional arguments, `--diffusers-load-kwargs` and `--diffusers-call-kwargs`,
which are only valid together with `--diffusion-load-format diffusers`.

After launching the model, users send a request as usual. Refer to other documentation pages on how to request a particular input/output modality, such as `examples/online_serving/text_to_image/openai_chat_client.py`.

## Configuration Reference

### `--diffusers-load-kwargs`

Passed as-is to `DiffusionPipeline.from_pretrained()`.

This is suitable for model-specific configurations not available through the vLLM-Omni interface.
For example: `--diffusers-load-kwargs '{"use_safetensors": true}'`.

When a parameter is available in the vLLM-Omni interface, it will be adapted here.
But if that parameter is simultaneously set in both the vLLM-Omni interface and `diffusers_load_kwargs`, the **latter** will take precedence.

### `--diffusers-call-kwargs`

Passed to `pipeline.__call__()`.

This is suitable for sampling parameters not available through the vLLM-Omni interface (such as online serving payloads).

When a parameter is available in the vLLM-Omni interface, it will be adapted here.
But if that parameter is simultaneously set in both the vLLM-Omni interface and `diffusers_call_kwargs`, the **former** will take precedence (because it is set at request time).

!!! note
    In vLLM-Omni, the default values for some sampling parameters may be different from diffusers.
    Consider referring to [`OmniDiffusionSamplingParams`](../../../vllm_omni/inputs/data.py) for the default sampling parameters in the vLLM-Omni interface,
    and the corresponding [diffusers pipeline](https://huggingface.co/docs/diffusers/main/en/api/pipelines/overview)'s `__call__` function documentation.

### Attention Backends

The diffusers backend converts
[vLLM-Omni standard of attention backend setting](../../../docs/user_guide/diffusion/attention_backends.md)
to [diffusers standard](https://huggingface.co/docs/diffusers/optimization/attention_backends#available-backends).

Specifically for `FLASH_ATTN`, it will first attempt to use FlashAttention-3 and then FlashAttention-2.

For each attempted version of `FLASH_ATTN` and `SAGE_ATTN`, it will first try to load the attention backend from HuggingFace `kernels` library, then without.

For unsuccessful attention selection or `TORCH_SDPA`, it will use the PyTorch's default attention backend.

The loaded attention backend and the failed attempts (if any) are logged to console.

### Model Specific Settings

The model loading and inference strictly follows the diffusers library, and they may be different from vLLM-Omni's native interface for some specific models.
Users are encouraged to double-check the model pipeline's interface in [diffusers' official documentation](https://huggingface.co/docs/diffusers/api/pipelines/overview).
Some particular examples are below.

#### Wan Series

The Wan series video generation models takes `boundary_ratio` and `flow_shift` during model initialization ([ref](https://huggingface.co/docs/diffusers/api/pipelines/wan)), not during inference.

Since our `OmniDiffusionConfig` contains these two values ([source](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/data.py)), we can directly pass `--boundary-ratio` and `--flow-shift` arguments to `vllm serve` command.

```bash
vllm serve "Wan2.2-T2V-A14B-Diffusers" \
    --omni \
    --boundary-ratio 0.875 \
    --flow-shift 3 \
    --diffusion-load-format diffusers
```

These extra CLI args will be attempted to pass as-is to the `OmniDiffusionConfig` dataclass and being accessible during model loading time.
Special routines inside the pipeline adapter ensures that they are set properly.
