# Online serving Example of vLLM-Omni for Qwen2.5-Omni

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/qwen2_5_omni>.


## 🛠️ Installation

Please refer to [README.md](https://github.com/vllm-project/vllm-omni/tree/main/README.md)

## Run examples (Qwen2.5-Omni)

### Launch the Server

```bash
vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091
```

If you have custom stage configs file, launch the server with command below
```bash
vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091 --stage-configs-path /path/to/stage_configs_file
```

### Send Multi-modal Request

Get into the example folder
```bash
cd examples/online_serving/qwen2_5_omni
```

####  Send request via python

```bash
python openai_chat_completion_client_for_multimodal_generation.py --query-type mixed_modalities
```

The Python client supports the following command-line arguments:

- `--query-type` (or `-q`): Query type (default: `mixed_modalities`)
  - Options: `mixed_modalities`, `use_audio_in_video`, `multi_audios`, `text`
- `--video-path` (or `-v`): Path to local video file or URL
  - If not provided and query-type uses video, uses default video URL
  - Supports local file paths (automatically encoded to base64) or HTTP/HTTPS URLs
  - Example: `--video-path /path/to/video.mp4` or `--video-path https://example.com/video.mp4`
- `--image-path` (or `-i`): Path to local image file or URL
  - If not provided and query-type uses image, uses default image URL
  - Supports local file paths (automatically encoded to base64) or HTTP/HTTPS URLs
  - Supports common image formats: JPEG, PNG, GIF, WebP
  - Example: `--image-path /path/to/image.jpg` or `--image-path https://example.com/image.png`
- `--audio-path` (or `-a`): Path to local audio file or URL
  - If not provided and query-type uses audio, uses default audio URL
  - Supports local file paths (automatically encoded to base64) or HTTP/HTTPS URLs
  - Supports common audio formats: MP3, WAV, OGG, FLAC, M4A
  - Example: `--audio-path /path/to/audio.wav` or `--audio-path https://example.com/audio.mp3`
- `--prompt` (or `-p`): Custom text prompt/question
  - If not provided, uses default prompt for the selected query type
  - Example: `--prompt "What are the main activities shown in this video?"`


For example, to use mixed modalities with all local files:

```bash
python openai_chat_completion_client_for_multimodal_generation.py \
    --query-type mixed_modalities \
    --video-path /path/to/your/video.mp4 \
    --image-path /path/to/your/image.jpg \
    --audio-path /path/to/your/audio.wav \
    --prompt "Analyze all the media content and provide a comprehensive summary."
```

####  Send request via curl

```bash
bash run_curl_multimodal_generation.sh mixed_modalities
```

## Run Local Web UI Demo

This Web UI demo allows users to interact with the model through a web browser.

### Running Gradio Demo

Once vllm and vllm-omni are installed, you can launch the web service built on AsyncOmni by

```bash
python gradio_demo.py  --model Qwen/Qwen2.5-Omni-7B --port 7861
```

Then open `http://localhost:7861/` on your local browser to interact with the web UI.

The gradio script supports the following arguments:

- `--model`: Model name
- `--ip`: Host/IP for Gradio server (default: 127.0.0.1)
- `--port`: Port for Gradio server (default: 7861)
- `--stage-configs-path`: Path to custom stage configs YAML file (optional)
- `--share`: Share the Gradio demo publicly (creates a public link)

### FAQ

If you encounter error about backend of librosa, try to install ffmpeg with command below.
```
sudo apt update
sudo apt install ffmpeg
```

## Example materials

??? abstract "gradio_demo.py"
    ``````py
    --8<-- "examples/online_serving/qwen2_5_omni/gradio_demo.py"
    ``````
??? abstract "openai_chat_completion_client_for_multimodal_generation.py"
    ``````py
    --8<-- "examples/online_serving/qwen2_5_omni/openai_chat_completion_client_for_multimodal_generation.py"
    ``````
??? abstract "run_curl_multimodal_generation.sh"
    ``````sh
    --8<-- "examples/online_serving/qwen2_5_omni/run_curl_multimodal_generation.sh"
    ``````
