# Diffusion Chat Completions API

vLLM-Omni supports generating and editing images via the `/v1/chat/completions`
endpoint using diffusion models. This page explains how to pass generation
parameters (such as `num_inference_steps`, `height`, `width`) to diffusion
models through this endpoint.

!!! tip
    For dedicated endpoints that accept generation parameters as top-level
    fields, see [Image Generation API](image_generation_api.md) and
    [Image Edit API](image_edit_api.md).

## Passing Generation Parameters

The `/v1/chat/completions` endpoint follows the OpenAI Chat API schema, which
does not natively include diffusion-specific fields like `num_inference_steps`
or `height`. How you pass these extra fields depends on your client.

### curl / Python `requests`

Wrap generation parameters inside an `"extra_body"` key in the JSON body:

```bash
curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "A beautiful landscape painting"}
    ],
    "extra_body": {
      "num_inference_steps": 50,
      "seed": 42
    }
  }'
```

### OpenAI Python SDK

Use the `extra_body` **keyword argument**. The SDK automatically merges these
fields into the top-level request body:

```python
response = client.chat.completions.create(
    model="Qwen/Qwen-Image",
    messages=[{"role": "user", "content": "A beautiful landscape painting"}],
    extra_body={
        "num_inference_steps": 50,
        "seed": 42,
    },
)
```

!!! note "SDK `extra_body` vs. JSON `extra_body`"
    These two `extra_body` usages look similar but work differently under the
    hood. The SDK flattens the dict into the top-level request JSON, while the
    curl/requests approach sends it as a nested `"extra_body"` key. Both are
    handled correctly by the server.

!!! note "About the `ignored fields` warning"
    You may see a log message like:

    ```
    WARNING: The following fields were present in the request but ignored: {'height', 'width', ...}
    ```

    This is **harmless**. It is emitted by vLLM's request validation layer
    because these fields are not part of the standard OpenAI
    `ChatCompletionRequest` schema. The fields are still stored internally
    and correctly forwarded to the diffusion pipeline.

## Model-Specific Examples

For complete examples with full request/response details, see the model-specific
guides:

- [Text-to-Image (Qwen-Image)](../user_guide/examples/online_serving/text_to_image.md)
- [Image-to-Image (Qwen-Image-Edit, Qwen-Image-Layered)](../user_guide/examples/online_serving/image_to_image.md)
- [GLM-Image](../user_guide/examples/online_serving/glm_image.md)
