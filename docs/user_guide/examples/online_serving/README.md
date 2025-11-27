# Online serving Example of vLLM-omni for Qwen2.5-omni

Source <https://github.com/vllm-project/vllm-omni/blob/main/examples/online_serving/README.md>.


## 🛠️ Installation

Please refer to [README.md](https://github.com/vllm-project/vllm-omni/blob/main/README.md)

## Run examples (Qwen2.5-omni)

Launch the server
```bash
vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091
```

If you have custom stage configs file, launch the server with command below
```bash
vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091 --stage-configs-path /path/to/stage_configs_file
```

Get into the example folder
```bash
cd examples/online_serving
```

Send request via python
```bash
python openai_chat_completion_client_for_multimodal_generation.py --query-type mixed_modalities
```

Send request via curl
```bash
bash run_curl_multimodal_generation.sh mixed_modalities
```

### FAQ

If you encounter error about backend of librosa, try to install ffmpeg with command below.
```
sudo apt update
sudo apt install ffmpeg
```

## Run Local Web UI Demo

This Web UI demo allows users to interact with the model through a web browser.

### Running Gradio Demo

Once vllm and vllm-omni are installed, you can launch the web service built on AsyncOmniLLM by

```bash
python gradio_demo.py  --model Qwen/Qwen2.5-Omni-7B --port 7861
```

Then open `http://localhost:7861/` on your local browser to interact with the web UI.


### Options

The gradio demo also supports running with an existing API server and can be customized with the following arguments.


```bash
python gradio_demo.py \
    --model Qwen/Qwen2.5-Omni-7B \
    --use-api-server \
    --api-base http://localhost:8091/v1 \
    --ip 127.0.0.1 \
    --port 7861
```

- `--model`: Model name
- `--use-api-server`: If set, connect to an existing vLLM HTTP API server instead of running AsyncOmniLLM locally.
- `--api-base`: Base URL for vllm serve (only used when `use-api-server` is set, default: http://localhost:8091/v1)
- `--ip`: Host/IP for Gradio server (default: 127.0.0.1)
- `--port`: Port for Gradio server (default: 7861)
- `--stage-configs-path`: Path to custom stage configs YAML file (optional)
- `--share`: Share the Gradio demo publicly (creates a public link)
