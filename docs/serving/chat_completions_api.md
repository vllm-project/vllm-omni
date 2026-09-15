# Chat Completions API

Use `/v1/chat/completions` for conversational and multimodal pipelines that
follow the OpenAI Chat API schema. Depending on the loaded model, a request can
contain text, image, audio, or video input and can return text, audio, images,
or other model-specific output.

Some diffusion pipelines also support image generation and editing through
Chat Completions. Prefer the dedicated [Image Generation
API](image_generation_api.md) or [Image Edit API](image_edit_api.md) for
task-specific clients; their request fields and response contracts are more
direct.

## Basic Request

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Describe vLLM-Omni briefly."}
    ],
    "modalities": ["text"]
  }'
```

Use `stream: true` for Server-Sent Events. Input media syntax, supported output
`modalities`, and response choices depend on the model; see the
[model-specific online serving examples](../user_guide/examples/online_serving/qwen3_omni.md).

!!! tip
    Each server hosts one model. Query `GET /v1/models` for its served model
    name when your client requires the `model` field.

## vLLM Extension Parameters

Send standard and vLLM-specific Chat parameters as top-level fields in a
direct HTTP request. When using the OpenAI Python SDK, pass vLLM-specific
fields through the SDK's `extra_body` keyword; the SDK merges them into the
top-level JSON sent to the server.

=== "curl"

    ```bash
    curl http://localhost:8091/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "messages": [{"role": "user", "content": "Write a short haiku."}],
        "top_k": 40
      }'
    ```

=== "OpenAI Python SDK"

    ```python
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8091/v1", api_key="none")

    response = client.chat.completions.create(
        model="your-served-model",
        messages=[{"role": "user", "content": "Write a short haiku."}],
        extra_body={"top_k": 40},
    )
    ```

??? note "Diffusion compatibility"

    Diffusion models exposed through Chat Completions can accept supported
    fields such as `num_inference_steps`, `seed`, `height`, and `width`. Send
    them at the top level with direct HTTP, or in the SDK's `extra_body`
    keyword argument.

    A literal nested `"extra_body"` JSON object is accepted for compatibility,
    but it is not recommended for new direct HTTP clients. Do not provide the
    same parameter in multiple locations; the server returns a `400` error for
    duplicate diffusion parameters. Prefer the dedicated [Image Generation
    API](image_generation_api.md) or [Image Edit API](image_edit_api.md) when
    either endpoint matches the task.

## Batch Requests

`POST /v1/chat/completions/batch` accepts the same shared generation fields,
but `messages` is a list of conversations. The response contains one choice
per conversation in input order.

```bash
curl http://localhost:8091/v1/chat/completions/batch \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      [{"role": "user", "content": "Summarize vLLM in one sentence."}],
      [{"role": "user", "content": "Summarize vLLM-Omni in one sentence."}]
    ],
    "max_tokens": 64
  }'
```

Batch Chat Completions does not support streaming, tools, beam search, or
`n > 1`.

## Model-Specific Examples

For complete examples with model-specific inputs and outputs, see:

- [Qwen3-Omni](../user_guide/examples/online_serving/qwen3_omni.md)
- [Qwen2.5-Omni](../user_guide/examples/online_serving/qwen2_5_omni.md)
- [Text-to-Image (Qwen-Image)](../user_guide/examples/online_serving/text_to_image.md)
- [Image-to-Image (Qwen-Image-Edit, Qwen-Image-Layered)](../user_guide/examples/online_serving/image_to_image.md)
- [GLM-Image](../user_guide/examples/online_serving/glm_image.md)
