# MiniCPM-o 4.5 Online Serving

This example serves MiniCPM-o 4.5 through the OpenAI-compatible vLLM-Omni chat
API and sends text, image, or video requests with audio output.

## Setup

Install the MiniCPM Token2Wav dependency:

```bash
pip install --no-build-isolation 'minicpmo-utils[all]'
pip install openai
```

The default stage config places the thinker stage on GPU 0 and the
talker/code2wav stages on GPU 1. Use two visible GPUs:

```bash
export CUDA_VISIBLE_DEVICES=0,1
```

## Launch The Server

From the repository root:

```bash
bash examples/online_serving/minicpmo4_5/run_server.sh
```

Equivalent command:

```bash
vllm-omni serve openbmb/MiniCPM-o-4_5 \
  --omni \
  --port 8091 \
  --stage-configs-path vllm_omni/model_executor/stage_configs/minicpmo.yaml \
  --chat-template vllm_omni/model_executor/models/minicpmo4_5/chat_template.jinja \
  --chat-template-content-format openai \
  --trust-remote-code
```

## Python Client

Text to audio:

```bash
python examples/online_serving/minicpmo4_5/openai_chat_client.py \
  --server http://localhost:8091 \
  --query-type text \
  --output-wav minicpmo45_text_to_audio.wav
```

Image to audio:

```bash
python examples/online_serving/minicpmo4_5/openai_chat_client.py \
  --server http://localhost:8091 \
  --query-type use_image \
  --image-path /path/to/image.png \
  --output-wav minicpmo45_image_to_audio.wav
```

Video to audio:

```bash
python examples/online_serving/minicpmo4_5/openai_chat_client.py \
  --server http://localhost:8091 \
  --query-type use_video \
  --video-path /path/to/video.mp4 \
  --output-wav minicpmo45_video_to_audio.wav
```

## Curl

Text to audio:

```bash
bash examples/online_serving/minicpmo4_5/run_curl_text_to_audio.sh
```

Image to audio:

```bash
bash examples/online_serving/minicpmo4_5/run_curl_image_to_audio.sh /path/to/image.png
```

Video to audio:

```bash
bash examples/online_serving/minicpmo4_5/run_curl_video_to_audio.sh /path/to/video.mp4
```

## Request Shape

For audio output, include both `modalities` and the MiniCPM TTS template kwargs:

```json
{
  "modalities": ["audio"],
  "chat_template_kwargs": {
    "use_tts_template": true,
    "enable_thinking": false
  }
}
```

When using the OpenAI Python SDK, pass the same `chat_template_kwargs` through
the SDK's `extra_body` argument. The SDK merges those fields into the request
body; raw curl JSON should not wrap them in an `extra_body` object.

## Notes

- The examples save the first audio response to a WAV file.
- For local image/video files, the clients send data URLs to the server.
- The default non-async stage config is intended for full-response latency and
  audio generation tests, not true time-to-first-audio-chunk measurements.
