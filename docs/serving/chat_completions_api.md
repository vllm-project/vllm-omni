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

## Streaming Audio

For models that support audio output, request `"stream": true`, include
`"audio"` in `modalities`, and set `"audio": {"format": "wav"}` to receive
streaming WAV audio. Audio events have `modality: "audio"`; each choice's
`delta.content` contains base64-encoded audio bytes.

For each `choice.index`, the first nonempty audio chunk contains a WAV header
followed by PCM16 samples. Later chunks for that choice contain only PCM16
samples. Empty chunks do not consume the header, and each new request starts
a new audio stream.

Audio preserves the model's mono or stereo channel layout. Each choice's
`audio_metadata` reports `channels`, `sample_rate_hz`, and `frame_count` (samples
per channel). The WAV header matches this metadata; stereo PCM16 samples are
interleaved in frame order.

Base64-decode each audio chunk and append its bytes, in arrival order, to one
WAV file per choice in the request. Do not save each chunk as a separate WAV
file: chunks after the first do not have a header. The
[multimodal Chat client](https://github.com/vllm-project/vllm-omni/blob/main/examples/online_serving/openai_chat_completion_client_for_multimodal_generation.py)
demonstrates this pattern.

With `audio.format` set to `"pcm"` or `"pcm16"`, all chunks contain raw PCM16
without a WAV header. A non-streaming WAV response contains a complete WAV file.

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

## Per-Stage Sampling Parameters

Use `sampling_params_list` to override individual stages through the SDK's
`extra_body` or as a top-level HTTP field. For LLM stages, each dictionary is
a partial override. Values are resolved in this order, with later values
taking precedence:

1. The stage's deployment sampling defaults.
2. Explicit top-level request parameters, for the comprehension stage only.
3. Explicit fields in that stage's `sampling_params_list` dictionary.

For example, this request retains `max_tokens=256` and the deployment's
other defaults, while overriding the first stage's temperature:

```json
{
  "messages": [{"role": "user", "content": "Describe this recording."}],
  "max_tokens": 256,
  "temperature": 0.7,
  "sampling_params_list": [{"temperature": 0.2}]
}
```

`max_completion_tokens` takes precedence over top-level `max_tokens` when
both are provided; an explicit stage `max_tokens` overrides either one.
Omitted stages use their deployment defaults, with explicit top-level
parameters still applied to an omitted comprehension stage. An empty list
therefore uses defaults plus the explicit comprehension parameters.
LLM stage `extra_args` dictionaries merge with deployment `extra_args` by
key; setting `extra_args` to `null` clears it.

Previously, providing an LLM stage dictionary replaced its entire sampling
configuration with library defaults plus that dictionary. Clients relying
on those library defaults should now provide the desired values explicitly.
Diffusion stage dictionaries retain their existing replacement semantics;
omitted diffusion stages retain their deployment defaults.

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
