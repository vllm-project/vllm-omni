# Online serving Example of vLLM-Omni for Qwen3-omni

Source <https://github.com/vllm-project/vllm/tree/main/examples/online_serving/qwen3_omni>.


## 🛠️ Installation

Please refer to [README.md](https://github.com/vllm-project/vllm/tree/main/examples/README.md)

## Run examples (Qwen3-Omni)

Launch the server
```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091
```

If you have custom stage configs file, launch the server with command below
```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 --stage-configs-path /path/to/stage_configs_file
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
python gradio_demo.py  --model Qwen/Qwen3-Omni-30B-A3B-Instruct --port 7861
```

Then open `http://localhost:7861/` on your local browser to interact with the web UI.


### Options

The gradio demo also supports running with an existing API server and can be customized with the following arguments.


```bash
python gradio_demo.py \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
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

## Example materials

??? abstract "gradio_demo.py"
    ``````py
    --8<-- "/mnt/vllm_open_release/vllm-omni-cursor/vllm-omni/examples/online_serving/qwen3_omni/gradio_demo.py"
    ``````
??? abstract "openai_chat_completion_client_for_multimodal_generation.py"
    ``````py
    --8<-- "/mnt/vllm_open_release/vllm-omni-cursor/vllm-omni/examples/online_serving/qwen3_omni/openai_chat_completion_client_for_multimodal_generation.py"
    ``````
??? abstract "run_curl_multimodal_generation.sh"
    ``````sh
    --8<-- "/mnt/vllm_open_release/vllm-omni-cursor/vllm-omni/examples/online_serving/qwen3_omni/run_curl_multimodal_generation.sh"
    ``````
