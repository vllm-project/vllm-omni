# Qwen3-Omni

## 🛠️ Installation

Please refer to [README.md](../../../README.md)

## Run examples (Qwen3-Omni)

### Launch the Server

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091
```

If you want to open async chunking for qwen3-omni, launch the server with command below

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_omni_moe_async_chunk.yaml
```

If you have custom stage configs file, launch the server with command below
```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 --stage-configs-path /path/to/stage_configs_file
```

### Send Multi-modal Request

Get into the example folder
```bash
cd examples/online_serving/qwen3_omni
```

####  Send request via python

```bash
python examples/online_serving/openai_chat_completion_client_for_multimodal_generation.py --model Qwen/Qwen3-Omni-30B-A3B-Instruct --query-type use_image --port 8091 --host "localhost"
```

#### Realtime WebSocket client (`openai_realtime_client.py`)

[`openai_realtime_client.py`](./openai_realtime_client.py) connects to **`ws://<host>:<port>/v1/realtime`**, uploads a local audio file as **PCM16 mono @ 16 kHz** chunks (OpenAI-style `input_audio_buffer.append` / `commit`), and prints **streaming transcription** (`transcription.delta` / `transcription.done`).

**Dependencies:**

```bash
pip install websockets numpy
```

**From this directory** (`examples/online_serving/qwen3_omni`):

```bash
python openai_realtime_client.py \
  --host localhost \
  --port 8091 \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --audio_path /path/to/your.wav
```

If `--audio_path` is omitted, the script uses a bundled default clip (`mary_had_lamb` via vLLM assets).

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `localhost` | API server host |
| `--port` | `8000` | API server port (match your `vllm serve` port, e.g. `8091`) |
| `--model` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | Must match the served model (also sent in `session.update`) |
| `--audio_path` | *(optional)* | Path to input audio; resampled to 16 kHz mono inside the client |

Ensure the vLLM-Omni server is running with realtime support for this endpoint, for example:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091
```

The Python client supports the following command-line arguments:

- `--query-type` (or `-q`): Query type (default: `use_video`). Options: `text`, `use_audio`, `use_image`, `use_video`
- `--model` (or `-m`): Model name/path (default: `Qwen/Qwen3-Omni-30B-A3B-Instruct`)
- `--video-path` (or `-v`): Path to local video file or URL. If not provided and query-type is `use_video`, uses default video URL. Supports local file paths (automatically encoded to base64) or HTTP/HTTPS URLs. Example: `--video-path /path/to/video.mp4` or `--video-path https://example.com/video.mp4`
- `--image-path` (or `-i`): Path to local image file or URL. If not provided and query-type is `use_image`, uses default image URL. Supports local file paths (automatically encoded to base64) or HTTP/HTTPS URLs and common image formats: JPEG, PNG, GIF, WebP. Example: `--image-path /path/to/image.jpg` or `--image-path https://example.com/image.png`
- `--audio-path` (or `-a`): Path to local audio file or URL. If not provided and query-type is `use_audio`, uses default audio URL. Supports local file paths (automatically encoded to base64) or HTTP/HTTPS URLs and common audio formats: MP3, WAV, OGG, FLAC, M4A. Example: `--audio-path /path/to/audio.wav` or `--audio-path https://example.com/audio.mp3`
- `--prompt` (or `-p`): Custom text prompt/question. If not provided, uses default prompt for the selected query type. Example: `--prompt "What are the main activities shown in this video?"`
- `--speaker`: TTS speaker/voice for audio output when requesting audio (e.g. `ethan`, `chelsie`, `aiden`). Omit to use the model default. Example: `--speaker "chelsie"`


For example, to use a local video file with custom prompt:

```bash
python examples/online_serving/openai_chat_completion_client_for_multimodal_generation.py \
    --query-type use_video \
    --video-path /path/to/your/video.mp4 \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --prompt "What are the main activities shown in this video?"
```

####  Send request via curl

```bash
bash run_curl_multimodal_generation.sh use_image
```


### FAQ

## Modality control
You can control output modalities to specify which types of output the model should generate. This is useful when you only need text output and want to skip audio generation stages for better performance.

### Supported modalities

| Modalities | Output |
|------------|--------|
| `["text"]` | Text only |
| `["audio"]` | Audio only |
| `["text", "audio"]` | Text + Audio |
| Not specified | Text + Audio (default) |

### Using curl

#### Text only

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "messages": [{"role": "user", "content": "Describe vLLM in brief."}],
    "modalities": ["text"]
  }'
```

#### Text + Audio

```bash
curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "messages": [{"role": "user", "content": "Describe vLLM in brief."}],
    "modalities": ["audio"]
  }' | jq -r '.choices[0].message.audio.data' | base64 -d > output.wav
```

### Using Python client

```bash
python examples/online_serving/openai_chat_completion_client_for_multimodal_generation.py \
    --query-type use_image \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --modalities text
```

### Using OpenAI Python SDK

#### Text only

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8091/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    messages=[{"role": "user", "content": "Describe vLLM in brief."}],
    modalities=["text"]
)
print(response.choices[0].message.content)
```

#### Text + Audio

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8091/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    messages=[{"role": "user", "content": "Describe vLLM in brief."}],
    modalities=["text", "audio"]
)
# Response contains two choices: one with text, one with audio
print(response.choices[0].message.content)  # Text response

# Save audio to file
audio_data = base64.b64decode(response.choices[1].message.audio.data)
with open("output.wav", "wb") as f:
    f.write(audio_data)
```

## Speaker selection

When requesting audio output, you can choose the TTS speaker (voice) used for synthesis. If not specified, the model uses its default speaker.

### Using curl

Pass a `speaker` field in the request body:

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "modalities": ["audio"],
    "speaker": "chelsie"
  }'
```

### Using Python client

Use the `--speaker` argument when generating audio:

```bash
python examples/online_serving/openai_chat_completion_client_for_multimodal_generation.py \
    --query-type use_image \
    --modalities audio \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --speaker "chelsie"
```

### Using OpenAI Python SDK

Pass `speaker` in `extra_body`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8091/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
    modalities=["audio"],
    extra_body={"speaker": "chelsie"}
)
# Audio uses the specified speaker
print(response.choices[1].message.audio)
```

Supported speaker names depend on the model (e.g. `Ethan`, `Chelsie`, `Aiden`). Omit `speaker` to use the default.

## Streaming Output
If you want to enable streaming output, please set the argument as below. The final output will be obtained just after generated by corresponding stage. We support both text streaming output and audio streaming output. Other modalities can output normally.
```bash
python examples/online_serving/openai_chat_completion_client_for_multimodal_generation.py \
    --query-type use_image \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --stream
```

## Run Local Web UI Demo

This Web UI demo allows users to interact with the model through a web browser.

### Running Gradio Demo

The Gradio demo connects to a vLLM API server. You have two options:

#### Option 1: One-step Launch Script (Recommended)

The convenience script launches both the vLLM server and Gradio demo together:

```bash
./run_gradio_demo.sh --model Qwen/Qwen3-Omni-30B-A3B-Instruct --server-port 8091 --gradio-port 7861
```

This script will:
1. Start the vLLM server in the background
2. Wait for the server to be ready
3. Launch the Gradio demo
4. Handle cleanup when you press Ctrl+C

The script supports the following arguments:
- `--model`: Model name/path (default: Qwen/Qwen3-Omni-30B-A3B-Instruct)
- `--server-port`: Port for vLLM server (default: 8091)
- `--gradio-port`: Port for Gradio demo (default: 7861)
- `--stage-configs-path`: Path to custom stage configs YAML file (optional)
- `--server-host`: Host for vLLM server (default: 0.0.0.0)
- `--gradio-ip`: IP for Gradio demo (default: 127.0.0.1)
- `--share`: Share Gradio demo publicly (creates a public link)

#### Option 2: Manual Launch (Two-Step Process)

**Step 1: Launch the vLLM API server**

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091
```

If you have custom stage configs file:
```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 --stage-configs-path /path/to/stage_configs_file
```

**Step 2: Run the Gradio demo**

In a separate terminal:

```bash
python gradio_demo.py --model Qwen/Qwen3-Omni-30B-A3B-Instruct --api-base http://localhost:8091/v1 --port 7861
```

Then open `http://localhost:7861/` on your local browser to interact with the web UI.

The gradio script supports the following arguments:

- `--model`: Model name/path (should match the server model)
- `--api-base`: Base URL for the vLLM API server (default: http://localhost:8091/v1)
- `--ip`: Host/IP for Gradio server (default: 127.0.0.1)
- `--port`: Port for Gradio server (default: 7861)
- `--share`: Share the Gradio demo publicly (creates a public link)
