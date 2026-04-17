# Ming-flash-omni 2.0

## Installation

Please refer to [README.md](../../../README.md)

## Run examples (Ming-flash-omni 2.0)

### Launch the Server

```bash
vllm serve Jonathan1909/Ming-flash-omni-2.0 --omni --port 8091
```

If you have custom stage configs file, launch the server with command below
```bash
vllm serve Jonathan1909/Ming-flash-omni-2.0 --omni --port 8091 --stage-configs-path /path/to/stage_configs_file
```

### Send Multi-modal Request

#### Send request via python

```bash
python examples/online_serving/openai_chat_completion_client_for_multimodal_generation.py --model Jonathan1909/Ming-flash-omni-2.0 --query-type use_mixed_modalities --port 8091 --host "localhost" --modalities text
```

The Python client supports the following command-line arguments:

- `--query-type` (or `-q`): Query type. Options: `text`, `use_audio`, `use_image`, `use_video`, `use_mixed_modalities`
- `--video-path` (or `-v`): Path to local video file or URL. If not provided and query-type uses video, uses default video URL. Supports local file paths (automatically encoded to base64) or HTTP/HTTPS URLs. Example: `--video-path /path/to/video.mp4` or `--video-path https://example.com/video.mp4`
- `--image-path` (or `-i`): Path to local image file or URL. If not provided and query-type uses image, uses default image URL. Supports local file paths (automatically encoded to base64) or HTTP/HTTPS URLs and common image formats: JPEG, PNG, GIF, WebP. Example: `--image-path /path/to/image.jpg` or `--image-path https://example.com/image.png`
- `--audio-path` (or `-a`): Path to local audio file or URL. If not provided and query-type uses audio, uses default audio URL. Supports local file paths (automatically encoded to base64) or HTTP/HTTPS URLs and common audio formats: MP3, WAV, OGG, FLAC, M4A. Example: `--audio-path /path/to/audio.wav` or `--audio-path https://example.com/audio.mp3`
- `--prompt` (or `-p`): Custom text prompt/question. If not provided, uses default prompt for the selected query type. Example: `--prompt "What are the main activities shown in this video?"`
- `--modalities`: Output modalities. For now, only `text` is supported. Example: `--modalities text`


#### Send request via curl

```bash
bash run_curl_multimodal_generation.sh text
bash run_curl_multimodal_generation.sh use_image
bash run_curl_multimodal_generation.sh use_audio
bash run_curl_multimodal_generation.sh use_video
bash run_curl_multimodal_generation.sh use_mixed_modalities
```

## Modality control

Ming-flash-omni 2.0 currently supports text output only (thinker stage).

| Modalities | Output |
|------------|--------|
| `["text"]` | Text only |
| Not specified | Text only (default) |

### Using curl

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Jonathan1909/Ming-flash-omni-2.0",
    "messages": [
      {"role": "system", "content": [{"type": "text", "text": "你是一个友好的AI助手。\n\ndetailed thinking off"}]},
      {"role": "user", "content": "请详细介绍鹦鹉的生活习性。"}
    ],
    "modalities": ["text"]
  }'
```

### Using OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8091/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Jonathan1909/Ming-flash-omni-2.0",
    messages=[
        {"role": "system", "content": [{"type": "text", "text": "你是一个友好的AI助手。\n\ndetailed thinking off"}]},
        {"role": "user", "content": "请详细介绍鹦鹉的生活习性。"},
    ],
    modalities=["text"],
)
print(response.choices[0].message.content)
```

### Multi-modal input with OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8091/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Jonathan1909/Ming-flash-omni-2.0",
    messages=[
        {"role": "system", "content": [{"type": "text", "text": "你是一个友好的AI助手。\n\ndetailed thinking off"}]},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg"}},
                {"type": "text", "text": "Describe this image in detail."},
            ],
        },
    ],
    modalities=["text"],
)
print(response.choices[0].message.content)
```

## Streaming Output

To enable streaming output:

```bash
python examples/online_serving/openai_chat_completion_client_for_multimodal_generation.py \
    --query-type use_image \
    --model Jonathan1909/Ming-flash-omni-2.0 \
    --modalities text \
    --stream
```

Or with the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8091/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Jonathan1909/Ming-flash-omni-2.0",
    messages=[
        {"role": "system", "content": [{"type": "text", "text": "你是一个友好的AI助手。\n\ndetailed thinking off"}]},
        {"role": "user", "content": "请详细介绍鹦鹉的生活习性。"},
    ],
    modalities=["text"],
    stream=True,
)
for chunk in response:
    for choice in chunk.choices:
        if hasattr(choice, "delta") and choice.delta.content:
            print(choice.delta.content, end="", flush=True)
print()
```

Or using curl:

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Jonathan1909/Ming-flash-omni-2.0",
    "messages": [
      {"role": "system", "content": [{"type": "text", "text": "你是一个友好的AI助手。\n\ndetailed thinking off"}]},
      {"role": "user", "content": "请详细介绍鹦鹉的生活习性。"}
    ],
    "modalities": ["text"],
    "stream": true,
  }'
```


## Reasoning (Thinking Mode)

To enable reasoning/thinking mode, change `detailed thinking off` to `detailed thinking on` in the system prompt:

### Using curl

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Jonathan1909/Ming-flash-omni-2.0",
    "messages": [
      {"role": "system", "content": [{"type": "text", "text": "你是一个友好的AI助手。\n\ndetailed thinking on"}]},
      {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/math_problem.png"}},
        {"type": "text", "text": "Solve this math problem step by step."}
      ]}
    ],
    "modalities": ["text"]
  }'
```

### Using OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8091/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Jonathan1909/Ming-flash-omni-2.0",
    messages=[
        {"role": "system", "content": [{"type": "text", "text": "你是一个友好的AI助手。\n\ndetailed thinking on"}]},
        {"role": "user", "content": "If a train travels 120 km in 2 hours, what is its average speed?"},
    ],
    modalities=["text"],
)
print(response.choices[0].message.content)
```
